# Hydro-Graph — DS-STGAT

**Dual-Scale Spatiotemporal Graph Attention Network for urban flood forecasting.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Read this before citing any number from this repository.** Every
> reported metric is either (a) computed by the pipeline described below
> from a **synthetic** benchmark, or (b) explicitly marked otherwise. See
> [`MODEL_CARD.md`](MODEL_CARD.md) for exactly what is and isn't real, and
> [`AUDIT_REPORT.md`](AUDIT_REPORT.md) for the leakage/correctness audit
> this codebase was hardened against.

## What this is

Hydro-Graph extracts a city's street + drainage network as a directed graph
`G = (V, E)`, binds 16 physics-grounded static features to each node
(elevation, slope, TWI, NDVI/NDWI/NDBI, imperviousness, SAR VV, distance to
coast/river, drainage-related terms), encodes rainfall at two temporal
scales (a 6-hour short-term trigger window and a 24-hour antecedent-moisture
window, each via its own GRU), fuses them through a cross-temporal attention
gate, propagates the result over the graph with a 2-layer GATv2 + SAGEConv
spatial encoder that consumes physics-informed edge features (including
enforced downhill drainage direction), and outputs per-node flood
probability at 4 forecast horizons (1h / 3h / 6h / 12h) from a shared
multi-lead sigmoid head.

```
rainfall[t-6:t]  ──► short GRU ──┐
                                  ├─► cross-temporal attention gate ─┐
rainfall[t-24:t:2] ─► long GRU ──┘                                  │
                                                                     ▼
static features[16] ──► static encoder ─────────────────────► fusion (MLP)
                                                                     │
                                                                     ▼
              edge features[E,4] ──► GATv2 ×2 + SAGEConv (directed graph)
                                                                     │
                                                                     ▼
                                          multi-lead sigmoid head → P(flood)
                                                    at 1h / 3h / 6h / 12h
```

## The one real implementation

`hydro_graph/` (phases 1-6) driven by `main.py` is the **only** pipeline
this repository executes. `archive/` holds an earlier v1 scaffold
(`src/`, `pipeline/`, a second `config` loader, `examples/`) that nothing
here imports — see `archive/README.md` if you need to compare the two.
`evaluate_paper.py` and `generate_paper.py` are a divergent, unreconciled
second pipeline used to draft `DS_STGAT_Paper.tex`; both now carry a
deprecation warning — don't use them as a source of results.

| Phase | Module | What it does |
|---|---|---|
| 1 | `hydro_graph/phase1_graph.py` | Graph construction (OSMnx, with a deterministic synthetic fallback) + `orient_drainage_edges()` (enforces downhill-only waterway edges) |
| 2 | `hydro_graph/phase2_features.py` | 16-dim static node features (real SRTM/Sentinel raster support, synthetic fallback) |
| 3 | `hydro_graph/phase3_temporal.py` | Dual-scale rainfall encoding, leakage-free multi-lead flood labels, chronological split |
| 4 | `hydro_graph/phase4_model.py` | `DualScaleSTGAT` (GATv2 + SAGE spatial encoder, dual GRU temporal encoder, cross-attention gate) |
| 5 | `hydro_graph/phase5_training.py` | `HydroGraphDataset`, `Trainer` (Focal Tversky loss, per-lead metrics incl. AUC-PR/CSI/ECE), leakage-safe rain normalisation |
| 6 | `hydro_graph/phase6_inference.py` | Inference, static/interactive risk maps, reliability diagram + ECE |
| — | `hydro_graph/baselines.py` | Persistence, Random Forest, LSTM-only, GCN+GRU, GraphSAGEv1+GRU baselines |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
```

`osmnx` and `srtm.py` are optional — without them (or without network
access), graph construction and elevation both fall back to a deterministic
synthetic model (seed=42), and the pipeline still runs and produces genuine
metrics from that synthetic benchmark. See [Data provenance](#data-provenance)
below before treating those metrics as anything more than that.

## Quick start

```bash
# Full pipeline: graph → features → temporal encoding → training → baselines → inference
python main.py --skip-osm --force-retrain

# Faster demo wrapper (same real pipeline, friendlier banner, baselines off by default)
python demo.py

# Regenerate cross-model comparison figures from data/outputs/*.json
python generate_figures.py

# Interactive dashboard (loads the trained checkpoint + cached graph)
streamlit run streamlit_app.py

# Test suite
python -m pytest tests/test_pipeline.py -v
```

Useful `main.py` flags:

| Flag | Effect |
|---|---|
| `--mode {demo,full}` | `demo` = small bbox for fast iteration; `full` = full Chennai metro bbox (large, GPU recommended) |
| `--skip-osm` | Force the deterministic synthetic graph instead of a live OSMnx/Overpass download |
| `--force-retrain` | Purge all caches (graph/features/temporal/checkpoints) and rebuild from scratch |
| `--skip-train` | Load the best existing checkpoint and run inference only |
| `--skip-baselines` | Skip the 5-model baseline ablation |
| `--epochs N` / `--baseline-epochs N` | Override `config/config.yaml`'s epoch budgets |
| `--config path/to.yaml` | Use an alternate config file |

All paths are resolved from `config/config.yaml`'s `paths:` section
(repo-relative) — there are no hardcoded absolute paths in `hydro_graph/`.

## Data provenance

**By default, every run is entirely synthetic.** `config/config.yaml` sets
`features.use_synthetic: true`, and this repository ships no DEM GeoTIFF,
Sentinel-1/2 imagery, or GPM IMERG rainfall CSV under `data/raw/`. Concretely:

- **Graph topology**: OSMnx download of Chennai's real street/waterway
  network is attempted first; on failure (no network, `osmnx` not
  installed) it falls back to a synthetic grid graph
  (`phase1_graph.py::_build_synthetic_multigraph`, fixed seed).
- **Static features**: procedurally generated from hand-tuned,
  Chennai-shaped functions (`phase2_features.py`), not sampled from real
  rasters, unless real `.tif` files are placed at the paths in
  `config.yaml`'s `features:` section.
- **Rainfall & flood labels**: a hand-authored synthetic storm profile with
  fixed event timing/peaks (`phase3_temporal.py`), not GPM IMERG or IMD
  gauge data. Labels are generated deterministically from that synthetic
  rainfall, not observed flood extents.

This is legitimate for exercising and unit-testing a spatiotemporal GNN's
architecture and training loop. It is **not** a validated flood forecasting
system, and no metric produced by a synthetic-mode run should be described
as measuring real-world skill. Full detail, including a known
train/val/test flood-rate skew from chronological splitting on a single
continuous event, is in [`MODEL_CARD.md`](MODEL_CARD.md).

## Edge direction — real terrain vs. synthetic proxy

`orient_drainage_edges()` (`phase1_graph.py`) enforces that every waterway
edge points from higher to lower elevation (message passing along drains
cannot propagate a flood signal uphill) and removes duplicate/uphill
copies. Road edges are deliberately left bidirectional (two-way streets are
real; road-surface runoff isn't channelised the way a drain is).

**What "elevation" means here depends on your run mode.** With real SRTM
data loaded (`features.dem_tif` pointing at a real GeoTIFF), the downhill
orientation is derived from measured terrain. In the default synthetic
mode, it is derived from a procedurally generated elevation surface, not
measured terrain — a physically-plausible proxy, not ground truth. State
which mode produced a given result before calling its edge directions
"terrain-derived."

## Calibration

The output head is a plain sigmoid trained with a ranking-oriented loss
(Focal Tversky) — there is no a priori reason for it to be calibrated.
**Do not describe this model's output as "probabilistic" or "calibrated"
without checking the measured Expected Calibration Error (ECE)** in
`data/outputs/eval_metrics.json` (`ece_lead*` keys) and the reliability
diagram at `data/outputs/calibration_curve.png`, both produced from real
held-out predictions by `main.py` itself.

## Results

Real, reproducible results from the exact command below live in
`data/outputs/eval_metrics.json` (DS-STGAT, all 4 lead times) and
`data/outputs/baseline_metrics.json` (Persistence, Random Forest,
LSTM-only, GCN+GRU, GraphSAGEv1+GRU, all at lead=1h). See
[`MODEL_CARD.md`](MODEL_CARD.md) for the verbatim numbers from this
session's run, the exact command and config used, and an honest read of
what they do and don't show (including the base rate — under class
imbalance, trust AUC-PR and CSI over ROC-AUC). Cross-model comparison
figures are regenerated from those two JSON files by `generate_figures.py`
— it renders nothing it cannot back with a real number.

```bash
python main.py --skip-osm --force-retrain   # writes eval_metrics.json / baseline_metrics.json
python generate_figures.py                  # renders figures from those JSON files only
```

## Project structure

```
HydroGraph_repo/
├── main.py                  # single pipeline entry point (phases 1-6)
├── demo.py                  # thin CLI wrapper over main.py with a friendly banner
├── generate_figures.py      # cross-model comparison figures from real eval JSONs
├── streamlit_app.py         # interactive dashboard
├── config/
│   └── config.yaml          # single source of truth for all paths/hyperparameters
├── hydro_graph/              # the real pipeline (phases 1-6 + baselines + config loader)
├── tests/
│   └── test_pipeline.py     # pytest suite (run via `pytest tests/test_pipeline.py`)
├── data/
│   ├── raw/                 # real DEM/Sentinel/rainfall inputs go here (empty by default)
│   ├── processed/           # cached graph/features/temporal tensors
│   ├── models/              # checkpoints
│   └── outputs/             # eval_metrics.json, baseline_metrics.json, maps, figures
├── archive/                  # dead v1 scaffold — see archive/README.md
├── evaluate_paper.py         # deprecated, divergent pipeline — do not cite its numbers
├── generate_paper.py         # deprecated .docx generator — do not cite its numbers
├── MODEL_CARD.md             # data provenance, splits, real metrics, limitations
└── AUDIT_REPORT.md           # leakage/correctness audit findings and fixes
```

## Reproducibility

`main.py::seed_everything()` seeds Python `random`, numpy, and torch
(CPU+CUDA) and requests deterministic algorithms once at pipeline start —
see its docstring for the residual nondeterminism this does not eliminate
(non-bitwise-deterministic scatter/segment-reduce kernels behind
`GATv2Conv`/`SAGEConv`; live OSM downloads are inherently non-reproducible,
hence `--skip-osm` for a byte-for-byte reproducible graph).

## License

MIT — see `LICENSE`.
