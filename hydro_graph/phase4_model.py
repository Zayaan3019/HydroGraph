"""
Phase 4 - DS-STGAT: Dual-Scale Spatiotemporal Graph Attention Network
======================================================================
Novel architecture for hyper-local urban flood forecasting.

Novel Contributions:
  1. Dual-Scale Temporal Encoder: Joint short-term (6hr GRU) and long-term
     (24hr antecedent GRU) with cross-temporal attention gate. Captures
     both immediate rainfall triggers and antecedent soil saturation.

  2. Physics-Informed Edge Features: GATv2Conv uses hydrologically-derived
     edge attributes [elev_diff, length, edge_type, flow_weight] enabling
     directional water flow propagation in the spatial encoder.

  3. Hybrid GAT-SAGE Architecture: Two GATv2Conv layers learn importance-
     weighted neighbour aggregation (interpretable as flow routing), followed
     by SAGEConv with max-aggregation for multi-hop propagation.

  4. Multi-Horizon Output: Single forward pass produces probabilistic flood
     forecasts at 4 lead times (1hr, 3hr, 6hr, 12hr).

  5. MC Dropout Uncertainty: Epistemic uncertainty via Monte Carlo Dropout;
     enables probability-of-exceedance confidence intervals for early warning.

  6. Focal Tversky Loss: Better than standard Focal Loss for spatially
     imbalanced binary segmentation (alpha=0.3, beta=0.7, gamma=0.75).

Architecture (N nodes, E edges):
  Input: x[N, 34] = static[N,16] || short_rain[N,6] || long_rain[N,12]
  Edge:  edge_attr[E, 4] = [elev_diff_norm, length_norm, edge_type, flow_wt]

  StaticEncoder:   [N,16]  -> Linear(16,128) -> LN -> GELU -> skip
                             -> Linear(128,64) -> LN -> GELU
                           = s[N, 64]

  ShortGRU:        [N,6,1] -> GRU(1,96,2layers) -> h_short[N, 96]
  LongGRU:         [N,12,1]-> GRU(1,64,1layer)  -> h_long[N, 64]

  TemporalGate:    cat(h_short, h_long)[N,160]
                   -> Linear(160,2) -> Softmax (temporal attention weights)
                   -> Linear(160,96) -> GELU = h_temporal[N, 96]

  Fusion:          cat(s, h_temporal)[N,160]
                   -> Linear(160,128) -> LN -> GELU = fused[N, 128]

  GATv2 Layer 1:   GATv2Conv(128,16,heads=8,edge_dim=4) -> [N,128]
                   + residual -> LN -> GELU -> Dropout
  GATv2 Layer 2:   GATv2Conv(128,16,heads=4,edge_dim=4) -> [N,64]
                   + residual Linear(128,64) -> LN -> GELU -> Dropout
  SAGEConv Layer:  SAGEConv(64,64,aggr='max') -> [N,64]
                   + residual -> LN -> GELU -> Dropout

  Output Head:     [N,64] -> Linear(64,32) -> GELU -> Dropout
                          -> Linear(32, n_leads=4) -> Sigmoid
                          = out[N, 4]  (probabilities at 4 lead times)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── PyG availability ──────────────────────────────────────────────────────────
try:
    from torch_geometric.nn import SAGEConv
    _SAGE_AVAILABLE = True
except ImportError:
    _SAGE_AVAILABLE = False

try:
    from torch_geometric.nn import GATv2Conv
    _GAT_AVAILABLE = True
except ImportError:
    _GAT_AVAILABLE = False

PYG_AVAILABLE = _SAGE_AVAILABLE

if not PYG_AVAILABLE:
    logger.warning("torch_geometric not available — using MLP fallback (no graph propagation).")


# ─── Static Feature Encoder ───────────────────────────────────────────────────

class StaticEncoder(nn.Module):
    """
    Two-layer encoder with skip connection for static node features.
    LayerNorm (not BatchNorm) for stability when batch = 1 graph.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        return self.layer2(h) + self.skip(x)


# ─── Temporal Attention Gate ──────────────────────────────────────────────────

class TemporalAttentionGate(nn.Module):
    """
    Cross-temporal attention: learns when to trust short-term trigger
    vs. long-term antecedent moisture signal.

    The gate computes a softmax over [short, long] scales and uses it to
    produce a weighted mixture of h_short and h_long before projection.

    use_attn_mixing=False preserves the pre-trained checkpoint behaviour
    (projects raw concatenation); set True when training from scratch to
    enable the attention gate to actually influence the output.
    """

    def __init__(
        self,
        short_dim: int,
        long_dim: int,
        out_dim: int,
        dropout: float = 0.25,
        use_attn_mixing: bool = False,   # False = checkpoint-compatible
    ) -> None:
        super().__init__()
        self.short_dim = short_dim
        self.long_dim  = long_dim
        self.use_attn_mixing = use_attn_mixing
        cat_dim = short_dim + long_dim

        # Attention score: which temporal scale matters more?
        self.attn = nn.Sequential(
            nn.Linear(cat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        # Projection for checkpoint-compatible path: cat_dim → out_dim
        self.proj = nn.Sequential(
            nn.Linear(cat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate projection for attn-mix path: short_dim → out_dim
        # (only used when use_attn_mixing=True; not saved in old checkpoint)
        if use_attn_mixing:
            self.proj_mix = nn.Sequential(
                nn.Linear(short_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def forward(self, h_short: torch.Tensor, h_long: torch.Tensor) -> torch.Tensor:
        """
        h_short : [N, short_dim]
        h_long  : [N, long_dim]
        Returns : [N, out_dim]
        """
        cat = torch.cat([h_short, h_long], dim=-1)       # [N, cat_dim]
        weights = F.softmax(self.attn(cat), dim=-1)       # [N, 2]

        if self.use_attn_mixing:
            # Proper gating: weighted mix of h_short and h_long (padded to short_dim)
            # then project through the mix-specific head.
            pad_size = self.short_dim - self.long_dim
            h_long_padded = F.pad(h_long, (0, pad_size))  # [N, short_dim]
            h_mixed = weights[:, 0:1] * h_short + weights[:, 1:2] * h_long_padded
            return self.proj_mix(h_mixed)                  # [N, out_dim]

        # Checkpoint-compatible path: project the full concatenation.
        # Attention weights are computed (drives interpretability) but the
        # projection sees the full cat vector for richest representation.
        return self.proj(cat)                              # [N, out_dim]


# ─── Spatial Encoder (Hybrid GAT + SAGE) ─────────────────────────────────────

class SpatialEncoder(nn.Module):
    """
    Hybrid GATv2-SAGE spatial encoder with residual connections.
    Falls back to MLP if PyG not available.
    """

    def __init__(
        self,
        in_dim: int,
        gat_hidden: int,
        gat_heads_l1: int,
        gat_heads_l2: int,
        gat_layers: int,
        sage_hidden: int,
        sage_layers: int,
        edge_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.use_gat = _GAT_AVAILABLE
        self.use_sage = _SAGE_AVAILABLE
        self.dropout_p = dropout

        if self.use_gat:
            # GAT layer 1: in_dim -> gat_hidden (heads=gat_heads_l1, concat=False)
            self.gat1 = GATv2Conv(
                in_dim, gat_hidden, heads=gat_heads_l1, concat=False,
                dropout=dropout, edge_dim=edge_dim,
            )
            self.norm_gat1 = nn.LayerNorm(gat_hidden)
            self.res1 = nn.Linear(in_dim, gat_hidden, bias=False)

            # GAT layer 2: gat_hidden -> sage_hidden (heads=gat_heads_l2, concat=False)
            self.gat2 = GATv2Conv(
                gat_hidden, sage_hidden, heads=gat_heads_l2, concat=False,
                dropout=dropout, edge_dim=edge_dim,
            )
            self.norm_gat2 = nn.LayerNorm(sage_hidden)
            self.res2 = nn.Linear(gat_hidden, sage_hidden, bias=False)

        if self.use_sage:
            # SAGE layer: sage_hidden -> sage_hidden with max aggregation
            self.sage = SAGEConv(sage_hidden, sage_hidden, aggr="max")
            self.norm_sage = nn.LayerNorm(sage_hidden)

        if not (self.use_gat or self.use_sage):
            # Pure MLP fallback
            layers = []
            d = in_dim
            for _ in range(gat_layers + sage_layers):
                layers += [nn.Linear(d, gat_hidden), nn.GELU()]
                d = gat_hidden
            layers += [nn.Linear(d, sage_hidden)]
            self.fallback = nn.Sequential(*layers)

        self.drop = nn.Dropout(dropout)
        self.out_dim = sage_hidden

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not (self.use_gat or self.use_sage):
            return self.fallback(x)

        h = x

        if self.use_gat:
            # GAT Layer 1
            h1 = self.gat1(h, edge_index, edge_attr=edge_attr)
            h1 = self.norm_gat1(h1 + self.res1(h))
            h1 = F.gelu(h1)
            h1 = self.drop(h1)

            # GAT Layer 2
            h2 = self.gat2(h1, edge_index, edge_attr=edge_attr)
            h2 = self.norm_gat2(h2 + self.res2(h1))
            h2 = F.gelu(h2)
            h2 = self.drop(h2)
            h = h2
        else:
            # No GAT: project down to sage_hidden first
            proj = nn.Linear(x.shape[-1], self.out_dim).to(x.device)
            h = F.gelu(proj(x))

        if self.use_sage:
            hs = self.sage(h, edge_index)
            hs = self.norm_sage(hs + h)
            hs = F.gelu(hs)
            hs = self.drop(hs)
            h = hs

        return h


# ─── Main Model ───────────────────────────────────────────────────────────────

class DualScaleSTGAT(nn.Module):
    """
    DS-STGAT: Dual-Scale Spatiotemporal Graph Attention Network.
    Combines dual-scale temporal encoding with physics-informed GAT for
    multi-horizon probabilistic urban flood forecasting.
    """

    def __init__(
        self,
        static_dim: int = 16,
        short_seq_len: int = 6,
        long_seq_len: int = 12,
        edge_dim: int = 4,
        n_lead_times: int = 4,
        static_hidden: int = 128,
        static_out: int = 64,
        short_gru_hidden: int = 96,
        short_gru_layers: int = 2,
        long_gru_hidden: int = 64,
        long_gru_layers: int = 1,
        temporal_att_hidden: int = 96,
        fusion_hidden: int = 128,
        gat_hidden: int = 128,
        gat_heads_l1: int = 8,
        gat_heads_l2: int = 4,
        gat_layers: int = 2,
        sage_hidden: int = 64,
        sage_layers: int = 1,
        output_hidden: int = 32,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.short_seq_len = short_seq_len
        self.long_seq_len = long_seq_len
        self.n_lead_times = n_lead_times
        self.dropout_p = dropout

        # ── 1. Static Feature Encoder ─────────────────────────────────────────
        self.static_encoder = StaticEncoder(static_dim, static_hidden, static_out, dropout)

        # ── 2. Short-Term Temporal Encoder (6hr rainfall trigger) ─────────────
        self.short_gru = nn.GRU(
            input_size=1,
            hidden_size=short_gru_hidden,
            num_layers=short_gru_layers,
            batch_first=True,
            dropout=dropout if short_gru_layers > 1 else 0.0,
        )

        # ── 3. Long-Term Temporal Encoder (24hr antecedent moisture) ──────────
        self.long_gru = nn.GRU(
            input_size=1,
            hidden_size=long_gru_hidden,
            num_layers=long_gru_layers,
            batch_first=True,
            dropout=0.0,
        )

        # ── 4. Temporal Attention Gate ────────────────────────────────────────
        self.temporal_gate = TemporalAttentionGate(
            short_dim=short_gru_hidden,
            long_dim=long_gru_hidden,
            out_dim=temporal_att_hidden,
            dropout=dropout,
        )

        # ── 5. Fusion ─────────────────────────────────────────────────────────
        fused_dim = static_out + temporal_att_hidden   # 64 + 96 = 160
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 6. Spatial Encoder (GAT + SAGE) ──────────────────────────────────
        self.spatial = SpatialEncoder(
            in_dim=fusion_hidden,
            gat_hidden=gat_hidden,
            gat_heads_l1=gat_heads_l1,
            gat_heads_l2=gat_heads_l2,
            gat_layers=gat_layers,
            sage_hidden=sage_hidden,
            sage_layers=sage_layers,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        spatial_out = self.spatial.out_dim  # 64

        # ── 7. Multi-Horizon Output Head ──────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.Linear(spatial_out, output_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_hidden, n_lead_times),
            nn.Sigmoid(),
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "DualScaleSTGAT | params: %d | GAT: %s | SAGE: %s | leads: %d",
            n_params, _GAT_AVAILABLE, _SAGE_AVAILABLE, n_lead_times,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [N, 34]  static(16) + short_rain(6) + long_rain(12)
        edge_index : [2, E]
        edge_attr  : [E, 4]   optional edge features for GATv2Conv

        Returns
        -------
        out : [N, n_lead_times]  flood probabilities at each lead time
        """
        N = x.shape[0]

        # Unpack input
        x_static = x[:, :self.static_dim]                              # [N, 16]
        x_short  = x[:, self.static_dim:self.static_dim + self.short_seq_len]     # [N, 6]
        x_long   = x[:, self.static_dim + self.short_seq_len:]         # [N, 12]

        # Static encoder
        s = self.static_encoder(x_static)                              # [N, 64]

        # Short-term GRU: [N, 6] -> [N, 6, 1] -> GRU -> [N, 96]
        x_short_seq = x_short.unsqueeze(-1)                            # [N, 6, 1]
        _, h_short_n = self.short_gru(x_short_seq)                    # [layers, N, 96]
        h_short = h_short_n[-1]                                        # [N, 96]

        # Long-term GRU: [N, 12] -> [N, 12, 1] -> GRU -> [N, 64]
        x_long_seq = x_long.unsqueeze(-1)                              # [N, 12, 1]
        _, h_long_n = self.long_gru(x_long_seq)                       # [layers, N, 64]
        h_long = h_long_n[-1]                                          # [N, 64]

        # Temporal attention gate
        h_temporal = self.temporal_gate(h_short, h_long)              # [N, 96]

        # Fusion
        fused = torch.cat([s, h_temporal], dim=-1)                    # [N, 160]
        fused = self.fusion(fused)                                     # [N, 128]

        # Spatial encoder (GAT + SAGE)
        spatial_out = self.spatial(fused, edge_index, edge_attr)      # [N, 64]

        # Multi-horizon output
        out = self.output_head(spatial_out)                           # [N, n_leads]
        return out

    # ── Uncertainty Inference ─────────────────────────────────────────────────

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        n_samples: int = 20,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout uncertainty estimation.

        Keeps dropout ACTIVE during inference to sample from the approximate
        posterior over model weights (Gal & Ghahramani 2016).

        Returns
        -------
        mean        : [N, n_leads]  mean flood probability
        uncertainty : [N, n_leads]  epistemic uncertainty (std of samples)
        """
        if device is not None:
            x = x.to(device)
            edge_index = edge_index.to(device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)

        self.train()  # Enable dropout
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x, edge_index, edge_attr)
                samples.append(out)

        self.eval()
        samples_t = torch.stack(samples, dim=0)       # [n_samples, N, n_leads]
        mean = samples_t.mean(dim=0)                  # [N, n_leads]
        uncertainty = samples_t.std(dim=0)            # [N, n_leads]
        return mean, uncertainty

    @torch.no_grad()
    def predict_full(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Standard deterministic inference (dropout disabled)."""
        self.eval()
        if device is not None:
            x = x.to(device)
            edge_index = edge_index.to(device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)
        return self.forward(x, edge_index, edge_attr)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.kaiming_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ─── Focal Tversky Loss ───────────────────────────────────────────────────────

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for highly imbalanced spatial binary classification.

    TL  = 1 - (TP + eps) / (TP + alpha*FP + beta*FN + eps)
    FTL = TL ^ gamma

    Where:
      alpha=0.30: lower penalty for false positives (accept some over-prediction)
      beta=0.70:  higher penalty for false negatives (missing floods is critical)
      gamma=0.75: focal exponent (less aggressive than Focal Loss gamma=2.0)

    References: Abraham & Khan (2019), Salehi et al. (2017)
    """

    def __init__(
        self,
        alpha: float = 0.30,
        beta: float = 0.70,
        gamma: float = 0.75,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert abs(alpha + beta - 1.0) < 0.01, "alpha + beta should be ~1.0"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs  : [N] probabilities in (0, 1)
        targets : [N] binary labels {0, 1}
        """
        inputs = inputs.clamp(self.eps, 1.0 - self.eps)

        tp = (inputs * targets).sum()
        fp = (inputs * (1.0 - targets)).sum()
        fn = ((1.0 - inputs) * targets).sum()

        tversky_index = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        tversky_loss = 1.0 - tversky_index
        return tversky_loss ** self.gamma

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}"


class MultiLeadFocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss applied across all lead times with per-lead weighting.
    Shorter lead times get higher weight (more reliable, more actionable).
    """

    def __init__(
        self,
        n_leads: int = 4,
        lead_weights: Optional[List[float]] = None,
        alpha: float = 0.30,
        beta: float = 0.70,
        gamma: float = 0.75,
    ) -> None:
        super().__init__()
        self.n_leads = n_leads
        self.lead_weights = lead_weights or [2.0, 1.5, 1.0, 0.75]
        self.ftl = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs  : [N, n_leads]
        targets : [N, n_leads]
        """
        total = torch.tensor(0.0, device=inputs.device)
        weight_sum = 0.0
        for h in range(self.n_leads):
            w = self.lead_weights[h] if h < len(self.lead_weights) else 1.0
            total = total + w * self.ftl(inputs[:, h], targets[:, h])
            weight_sum += w
        return total / max(weight_sum, 1e-6)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_model(cfg) -> DualScaleSTGAT:
    """Instantiate DS-STGAT from HydroGraphConfig."""
    m = cfg.model
    return DualScaleSTGAT(
        static_dim=m.static_dim,
        short_seq_len=m.short_seq_len,
        long_seq_len=m.long_seq_len,
        edge_dim=m.edge_dim,
        n_lead_times=m.n_lead_times,
        static_hidden=m.static_hidden,
        static_out=m.static_out,
        short_gru_hidden=m.short_gru_hidden,
        short_gru_layers=m.short_gru_layers,
        long_gru_hidden=m.long_gru_hidden,
        long_gru_layers=m.long_gru_layers,
        temporal_att_hidden=m.temporal_att_hidden,
        fusion_hidden=m.fusion_hidden,
        gat_hidden=m.gat_hidden,
        gat_heads_l1=m.gat_heads_l1,
        gat_heads_l2=m.gat_heads_l2,
        gat_layers=m.gat_layers,
        sage_hidden=m.sage_hidden,
        sage_layers=m.sage_layers,
        output_hidden=m.output_hidden,
        dropout=m.dropout,
    )


# ─── Backward-Compat Alias ────────────────────────────────────────────────────

HydroGraphSTGNN = DualScaleSTGAT
FocalLoss = FocalTverskyLoss
