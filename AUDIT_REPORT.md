# HydroGraph — Leakage Audit and Scientific Hardening

Audit + fix pass against `06-hydrograph.md`. All commands below were actually
run in this repository; every number quoted has a file path attached. Where a
number could not be verified, that is stated explicitly rather than assumed.

---

## 0. Resolving the duplication

The prompt referenced two directories, only one of which (`Downloads\HydroGraph_repo`)
exists on this machine — `Downloads\Hydrograph` (no underscore) does not
exist here; the only trace of it is a stale absolute path baked into
`data/outputs/figures/figure_paths.json` (`C:\Users\Mohamed Zayaan\Downloads\Hydrograph\...`),
left over from wherever the checked-in figures were actually generated.

**A second, more consequential duplication exists inside this one repo**:

| | Canonical | Dead |
|---|---|---|
| Package | `hydro_graph/` (phase1–6) | `src/`, `pipeline/`, top-level `config/` |
| Entry point | `main.py`, `tests/test_pipeline.py`, `evaluate_paper.py` | `examples/*.py`, `deploy.py`, `setup.py` |
| Docs | `docs/ARCHITECTURE.md` (matches `hydro_graph/`) | `README.md` Quick Start, `data/README.md` |
| Evidence | Most recent commits (`885eb87`, `2072be8`); produced `DS_STGAT_Paper.tex` and `data/outputs/figures/` | `README.md`'s own Quick Start imports `from src import ...` — **zero callers from `main.py` or `hydro_graph/`** |

`grep -rn "from src\|import src\|from pipeline\|import pipeline" hydro_graph/ main.py` returns nothing.
Per the preamble rule ("zero callers = dead code = a defect regardless of
code quality"), `src/`, `pipeline/`, top-level `config/`, and `examples/` are
dead. `README.md`'s entire Quick Start section documents an API that nothing
in this repository runs. **This audit targets `hydro_graph/` only**, per
instruction not to audit both. Recommendation: delete the dead tree or move
it to a clearly-labeled `archive/` directory — left untouched here since
deleting ~4,600 LOC wasn't explicitly requested and is easy to get wrong
silently.

---

## PRIORITY 1 — Spatiotemporal leakage

### 1.1 — Random vs. time-blocked splits

**[DEFEND].** Splits are chronological, not random —
`hydro_graph/phase3_temporal.py:530` `get_chronological_split()` returns
contiguous, non-overlapping index ranges (`train_idx` all `<` `val_idx` all
`<` `test_idx`), and `main.py:328-343` uses it directly. Verified by
`tests/test_pipeline.py::TestPhase3Temporal::test_chronological_split` (new,
passing) which asserts `train_idx.max() < val_idx.min()` and
`val_idx.max() < test_idx.min()`.

### 1.2 — Normalisation fit scope — **[FIX] CONFIRMED LEAK, FIXED**

`hydro_graph/phase5_training.py:104-105` (original):
```python
self._rain_norm = max(float(np.percentile(rainfall[rainfall > 0], 95)), 1.0) \
    if (rainfall > 0).any() else 1.0
```
`rainfall` here is the **full** `[T, N]` series — train, val, *and* test
periods — passed in from `main.py`. Every training-time input feature is
`rainfall / self._rain_norm` (`get_snapshot`, same file, line ~121), so the
scale of every feature the model trains on depended on rainfall values from
the held-out future, including the catastrophic flood peak if it falls in
val/test (which it usually does — see 1.1-adjacent finding below).

**Concrete reproduction** (`scratchpad/rain_norm_before_after.py`, run this session):
a 200-step synthetic series, train/held-out boundary at t=150, held-out
rainfall replaced with an implausible spike:

```
BEFORE (original committed HydroGraphDataset, fits on the FULL series):
  rain_norm with ordinary held-out rainfall : 4.742
  rain_norm with SPIKED held-out rainfall    : 500.000
  -> changed by 10444% purely because of rainfall the model never trains on.

AFTER (fixed HydroGraphDataset, rain_norm_fit_end_t=train_end_t):
  rain_norm with ordinary held-out rainfall : 4.737
  rain_norm with SPIKED held-out rainfall    : 4.737
  -> identical: True
```

**Fix**: `hydro_graph/phase5_training.py` — `HydroGraphDataset.__init__` gained
a `rain_norm_fit_end_t` parameter; when given, the percentile is computed on
`rainfall[:fit_end]` only. `main.py:328-381` was reordered so the
chronological split is computed *before* dataset construction, and
`train_end_t = train_idx.max() + 1` is passed through. The cross-event 2018
dataset (never trained on at all) reuses the 2015 training split's
`_rain_norm` rather than fitting its own — calibrating a normaliser against
an event's own extremes before evaluating on it is itself a leak, just a
different one. Regression tests:
`test_rain_norm_fit_only_on_training_window`,
`test_rain_norm_without_split_boundary_uses_full_series` (both passing).

Falling back to a full-series fit is still supported (with a logged warning)
for the one case where it's legitimate: building an inference-only dataset
from an already-trained checkpoint, where there is no "future" to leak into
a training run that already happened.

### 1.3 — Lag-window sign correctness — **[DEFEND], now regression-tested**

Checked every shift in `phase3_temporal.py`:
- `build_input(t)`: short window `rainfall[t-6:t]`, long window
  `rainfall[t-24:t:2]` — both strictly `< t`.
- `_generate_flood_labels`: `cum_rain_6h = rainfall[t-6:t].sum()` — past only;
  spatial propagation uses `label_prob[t-1, ...]` — past only.
- `HydroGraphDataset.get_snapshot(t)` (phase5_training.py): same past-only
  windows for `x`; targets are `labels[t+h]` for `h` in `{1,3,6,12}` —
  **future**, which is correct and intentional (that's the forecast target,
  not a leak).

This was previously *asserted* by comments but not tested. Added
`test_build_input_does_not_peek_forward` (corrupts `rainfall[t:]` and checks
`build_input(t)` is unaffected — it is) and
`test_build_targets_reads_future_not_past` (checks `build_targets(t)[h] ==
labels[t+h]` exactly). Both pass.

### 1.4 — Graph built from information unavailable at prediction time?

**[DEFEND].** Node/edge features (elevation, slope, TWI, NDVI/NDWI/NDBI,
drain capacity, distance-to-coast/river) are static and time-invariant — none
are derived from observed flood extents. `orient_drainage_edges()` (new,
§Priority 3) uses elevation, also static. No leak found here.

### 1.5 — Target leakage via a label-derived feature?

**[DEFEND].** The 16 static features (`phase2_features.py`) are computed once
from terrain/spectral/infrastructure geometry, independent of the temporal
label generator in `phase3_temporal.py`. No feature is a transform of
`labels`. One thing worth flagging as a **methodological caveat, not
leakage**: `_generate_flood_labels()` optionally propagates flood
probability along the same graph `edge_index` the GNN is later given
(`phase3_temporal.py:490-496`, gated by `edge_index is not None`). This means
part of the "ground truth" is literally defined by graph adjacency — so a
GNN's measured advantage over non-graph baselines on this specific synthetic
benchmark is partly a property of how the benchmark was constructed, not
purely evidence the graph helps on independently-labeled data. Documented in
`MODEL_CARD.md`; not a train/test leak since it affects all splits equally,
but real enough to disclose.

### Finding not in the original checklist — split flood-rate skew (HIGH)

Chronological splitting is correct and leak-free, but interacts badly with
this benchmark's single-continuous-event structure. Verified twice this
session, at two different window sizes:

```
[demo bbox, 1406 nodes, full Nov1-Dec5 window]
Split flood rates: train=14.85% | val=99.73% | test=62.82%

[verification bbox, 225 nodes, Nov1-Dec1 window]
Split flood rates: train=2.71%  | val=82.59% | test=85.91%
```
(`before_run.log`, `after_run.log`, both this session.)

Because the storm intensity is backloaded toward the end of the event
timeline, a chronological 85/7.5/7.5 split puts almost all of the flood-rate
mass into val/test and almost none into train. Early stopping is then
computed on an almost-entirely-positive (or, depending on window, wildly
different-base-rate) validation slice — not a representative estimate of
deployment-time class balance, and liable to select checkpoints for the
wrong reason. **[FIX recommended, not implemented this session]**: split
across multiple independent synthetic storm events instead of one
continuous event, or use a purged/embargoed walk-forward scheme across
several events. Flagged prominently in `MODEL_CARD.md`.

---

## PRIORITY 2 — Is there a real evaluation?

### 2.6 — Held-out evaluation and actual metric values

Before this audit: `data/outputs/eval_metrics.json` and
`data/outputs/baseline_metrics.json` were **empty** at session start (verified
by `cat`), `data/outputs/paper_results.json` did not exist, and
`data/outputs/figures/figure_paths.json` pointed at an absolute path on a
machine/directory (`Downloads\Hydrograph`) that does not exist anywhere on
this filesystem. **Every number in `DS_STGAT_Paper.tex` was, at the start of
this session, unreproducible from this repository's committed state.**

**It gets worse.** `generate_figures.py` — the script that produced every PNG
under `data/outputs/figures/`, the ones embedded in `DS_STGAT_Paper.tex` —
imports no `json` module and never opens `eval_metrics.json`,
`baseline_metrics.json`, or `paper_results.json`. Every number it plots is a
Python literal typed directly into the script: `val_f1 = [0.7528, 0.8522,
0.8549, 0.9250, ...]` (line 224), `csi = [0.8273, 0.8049, 0.7140, 0.5457]`
(line 288), `"Lead-1hr: ECE=0.082  ✓"` (line 358, an f-string literal, not a
computed value), `"DS-STGAT (ECE=0.082)", "GCN+GRU (ECE=0.113)", "LSTM
(ECE=0.171)", "Rand. Forest (ECE=0.162)"` (lines 752-755, hardcoded per-model
ECE with no code anywhere that produced them). These are not stale cached
numbers from a real prior run — there is no code path in this repository
connecting these literals to any model, checkpoint, or dataset. The figures
are illustrations of asserted numbers, not measurements. Combined with the
missing/empty metrics JSONs, this means **none of the quantitative claims in
`DS_STGAT_Paper.tex` are currently backed by anything in this repository** —
not the training run, not a cached result, not even a script that could
regenerate them from real inputs. `[DOCUMENT]`, not fixed this session
(rewriting `generate_figures.py` to load real `eval_metrics.json`/
`baseline_metrics.json` is a reasonable follow-up once a full-scale training
run exists to populate them — see §Reproduction).

This audit generated real numbers by running `main.py` end-to-end
(`--skip-osm --force-retrain`, reduced scale for CPU tractability — see
§Reproduction below for exact numbers, file paths, and why the scale was
reduced). That is now the source of truth, not the paper.

### 2.7 — Is there a baseline?

Before this audit: 4 baselines (RandomForest, LSTM-only, GCN+GRU,
GraphSAGEv1+GRU) in `hydro_graph/baselines.py`, no persistence baseline.
Without persistence, F1/CSI numbers are unfalsifiable under temporal
autocorrelation — a node flooded at `t` is very likely still flooded at
`t+1`, so "beating" a competent-looking F1 can mean nothing.

**[FIX]**: added `PersistenceBaseline` (`baselines.py`, top of file) —
`flood(t+h) = flood(t)`, zero parameters, zero training. Wired into
`run_all_baselines()` as baseline 0, before RF. `main.py` and docstrings
updated from "4 baselines" to "5 baselines". A CNN baseline was **not**
added — grepped `README.md` and `DS_STGAT_Paper.tex` for "CNN" / "spatial
blindness" claims and found none, so the conditional trigger in the prompt
("if the README claims to beat CNN spatial blindness") doesn't apply here;
noted so it isn't silently dropped.

---

## PRIORITY 3 — GNN correctness

### 3.8 — Edge direction — **[FIX] CONFIRMED CRITICAL BUG, FIXED**

The paper's own framing (`DS_STGAT_Paper.tex:283-285`) defines the graph as
having "directed edges," and the introduction specifically claims to model
"the directional, capacity-dependent nature of stormwater flow"
(line 188) — i.e., this is the model's core physical claim, not incidental.

**Before this audit, it was false**, in the exact way the prompt predicted.
`hydro_graph/phase1_graph.py::_build_synthetic_multigraph` (the fallback
graph — and the *only* graph this repo can produce, since `osmnx` isn't
installed and there's no `data/raw/` to fall back to for anything else) added
**every** edge in both directions unconditionally, including waterway/drain
edges:
```python
G.add_edge(u, v, ..., _edge_type=etype, _drain_capacity=dcap)
G.add_edge(v, u, ..., _edge_type=etype, _drain_capacity=dcap)
```
PyG's `GATv2Conv`/`SAGEConv` treat `edge_index` as directed (source →
target); with both directions present, message passing propagates flood
signal uphill exactly as readily as downhill. Confirmed empirically this
session (`scratchpad/edge_direction_before_after.py`, loads the original
committed file via `git show HEAD:...` and compares to the fixed module,
same seed):

```
BEFORE (original committed hydro_graph/phase1_graph.py):
  Distinct waterway/drain node-pairs: 36
  ...of which BIDIRECTIONAL: 36  (100%)

AFTER graph construction fix (before orientation):
  Distinct waterway/drain node-pairs: 18
  ...of which BIDIRECTIONAL: 0  (0%)

AFTER orient_drainage_edges() (what the model actually trains on):
  Directed waterway edges fed to the model: 18
  ...of which BIDIRECTIONAL: 0
  ...of which still pointing uphill: 0
```

Two compounding bugs, both fixed:
1. **Symmetrisation** — every waterway edge duplicated in both directions.
2. **Wrong-direction placeholder** — even the single intended direction was
   guessed from a crude proxy (grid row index) *before* real elevation was
   known, and the guess didn't match the code's own later elevation model
   (the "south-flowing" comment doesn't correspond to how
   `phase2_features.py::_synthetic_elevation` actually places high/low
   ground). Fixing (1) alone would still have left edges pointing uphill.

**Fix** (`hydro_graph/phase1_graph.py`, new `orient_drainage_edges(G,
edge_features)` function, ~90 lines, called from `main.py` after Phase 2
elevation refinement): for every waterway edge, orient it from higher to
lower elevation using the real (Phase-2-refined) elevation in
`edge_features[:, 0]`, flipping if stored backwards; if both directions of
the same physical connection exist, keep only the steeper-downhill one.
Road edges are deliberately left bidirectional — two-way streets are
realistic, and surface runoff along a road isn't channelised the way a drain
is; this is a modelling choice, not an oversight, and is documented as such
in the function's docstring. This also removes the need to trust that a live
OSMnx/Overpass download would produce correctly-directed waterway edges —
`orient_drainage_edges` enforces the physical claim structurally, regardless
of data source.

Regression tests (all passing): `test_no_bidirectional_waterway_edges`,
`test_waterway_edges_point_downhill`, `test_road_edges_remain_bidirectional`,
`test_waterway_edges_not_bidirectional_pre_orientation`.

### 3.9 — Self-loops, degree normalisation, over-smoothing

**[DEFEND].** `phase1_graph.py::_convert_to_digraph` explicitly strips
self-loops (`G.remove_edges_from(list(nx.selfloop_edges(G)))`), verified by
`test_no_selfloops`. Depth is shallow — 2 `GATv2Conv` layers + 1 `SAGEConv`
layer (`phase4_model.py::SpatialEncoder`) — well below where over-smoothing
typically becomes visible (usually reported past 4-8 layers); each layer has
a residual connection (`h + res(h)`) plus `LayerNorm`, which further guards
against representation collapse. Not independently measured (e.g. via
Dirichlet energy) — flagged as a LOW-severity gap, not fixed this session,
since 3-layer GAT/SAGE stacks collapsing is not the likely failure mode here.

### 3.10 — GRU hidden state, padding, masking

**[DEFEND].** Both GRUs (`short_gru`, `long_gru` in `DualScaleSTGAT.forward`)
are called fresh on each snapshot with `[N, seq_len, 1]` input and no
initial hidden state passed in — i.e., hidden state is **reset every call**,
never carried across batches/timesteps. This is correct for this
architecture: each snapshot's GRU processes a bounded rainfall lag window
(6h or 24h) independently, it is not an RNN unrolled across the training
loop's timestep sequence, so there is no cross-batch state to leak or need
to reset. Padding: `HydroGraphDataset.get_snapshot` zero-pads the short
window at the start of the series (`short_rain.shape[0] < short_seq_len` ->
pad with zeros, `phase5_training.py:113-117`); these padded zero-timesteps
only occur for `t < short_seq_len`, which the `lookback` guard in
`get_chronological_split` excludes from every split — so no padded timestep
is ever used for training or evaluation. No explicit sequence mask is
needed given this, and none is missing.

---

## PRIORITY 4 — The "probabilistic" claim

### 4.11 — Calibration — **[FIX] no ECE existed in the shipped pipeline**

Before this audit: `phase6_inference.py::plot_calibration` drew a reliability
diagram (visual only) but computed **no number**. The only place ECE was
ever computed in this repository was `evaluate_paper.py` — a separate,
divergent one-off script (see §Reproducibility below) — not the pipeline
`main.py` actually runs. `_compute_metrics()` in `phase5_training.py`
(what `Trainer.evaluate()` actually calls) never reported ECE, Brier was the
closest proxy but is not the same thing.

**[FIX]**: `_expected_calibration_error()` added to `phase5_training.py`
(standard 10-bin |accuracy − confidence| weighted-by-bin-mass estimator),
wired into `_compute_metrics()` (so every `f1_lead*`/`auroc_lead*` report now
has a matching `ece_lead*`), and into
`InferenceEngine.plot_calibration()`, which now prints ECE on the reliability
diagram itself and returns `(path, ece)` instead of just `path`. `main.py`
saves it into `eval_metrics.json` as `calibration_ece_lead1h`. Regression
tests: `test_ece_zero_for_perfect_calibration`,
`test_ece_large_for_confident_wrong_predictions` (both passing, using
synthetic perfectly/badly-calibrated distributions to sanity-check the
estimator itself, not just its presence).

The model's output head is a plain sigmoid trained with Focal Tversky loss —
a ranking-oriented loss with no reason a priori to produce calibrated
probabilities. **Real measured ECE for this session's run is reported in
§Reproduction below — do not use the paper's claimed ECE=0.082 without
re-running the pipeline; it was never verified from this repository's
state (see §2.6).** Until a real, current ECE is below a documented
threshold, "probabilistic" should be qualified ("uncalibrated
sigmoid output; see measured ECE") rather than stated flatly.

### 4.12 — Class imbalance / base rate reporting

**[FIX, partial].** Flood base rate was already logged per-split in
`main.py` ("Split flood rates: ..."). `_compute_metrics()` did not
previously surface `base_rate` as a metric key alongside the others (so it
wasn't in `eval_metrics.json`) — added. AUC-PR was already computed
(`aucpr` key) but under-emphasised relative to AUC-ROC in log lines; updated
`Trainer.evaluate()` and `main.py::_log_lead_metrics` to print AUC-PR
alongside AUC-ROC on every line, with a comment explaining why AUC-PR is the
one to trust under this base rate.

---

## PRIORITY 5 — Reproducibility

### 5.13 — Seeding — **[FIX]**

Before: `torch.manual_seed`/`np.random.seed` were only set immediately
before the *training* call (`main.py`, old line ~414), not at pipeline
start — meaning graph/feature/temporal synthesis, and any baseline that ran
before or without going through that code path, were not guaranteed
reproducible from a single documented seed. No CUDA seeding, no
`torch.use_deterministic_algorithms`, no Python `random` seeding, no
documentation of residual nondeterminism.

**[FIX]**: new `seed_everything(seed)` in `main.py`, called once at the top
of `run_pipeline()`. Seeds Python `random`, numpy, `torch.manual_seed`, and
`torch.cuda.manual_seed_all` when CUDA is available; sets
`CUBLAS_WORKSPACE_CONFIG` and calls
`torch.use_deterministic_algorithms(True, warn_only=True)`. Residual
nondeterminism is documented in the function's docstring rather than
silently hidden: scatter/segment-reduce kernels behind
`GATv2Conv`/`SAGEConv` are not bitwise-deterministic even with this flag
(per PyTorch's own documentation), and a live OSMnx/Overpass download is
inherently non-reproducible run-to-run (use `--skip-osm` for a
byte-for-byte reproducible graph, seed=42 fixed in
`_build_synthetic_multigraph`).

### 5.14 — One command reproduces every number — **[FIX / DOCUMENT]**

`python main.py --skip-osm --force-retrain` is that one command (documented
in `MODEL_CARD.md` and below). Two things stood in its way and are now
fixed or documented:

1. `phase6_inference.py` imported `matplotlib.pyplot` without forcing a
   non-interactive backend. On this machine, matplotlib's default backend
   resolution reached for `TkAgg`, which crashed
   (`_tkinter.TclError: Can't find a usable tk.tcl`) because the installed
   Python doesn't have a complete Tk install — a plotting pipeline that only
   ever calls `plt.savefig()` should never depend on a GUI toolkit being
   present at all. **[FIX]**: `matplotlib.use("Agg")` added before `pyplot`
   is imported. Regression: `test_static_map_generation` (was failing before
   this fix, passes after).
2. **`evaluate_paper.py` is a second, divergent pipeline**, not a thin
   wrapper over `hydro_graph/`. It reimplements its own graph builder
   (`build_chennai_graph`, standalone grid logic, *not*
   `GraphConstructor`), its own feature builder (`build_static_features`,
   not `FeatureEngineer`), and trains with a different loss
   (`_WeightedMultiLeadBCE`, not `MultiLeadFocalTverskyLoss`) and different
   hyperparameters than `main.py`'s `Trainer`. It is where
   `DS_STGAT_Paper.tex`'s numbers came from (its `build_chennai_graph`
   docstring literally cites "$E{=}1{,}881$ directed edges", the same figure
   quoted in the paper), **not from `main.py`**. This means the paper's
   claimed numbers were never reproducible by running the packaged
   pipeline, only by running a separate script with independent bugs and
   fixes of its own (it does, notably, get edge direction *more* right than
   `phase1_graph.py` did before this audit — its `build_chennai_graph` adds
   most edges in only one direction per grid offset). **Not reconciled this
   session** (would mean either porting `evaluate_paper.py` onto the
   `hydro_graph/` classes, or formally deprecating it) — flagged here and in
   `MODEL_CARD.md` so it isn't mistaken for the reproducibility entry point.
   Recommendation: pick one pipeline. If `evaluate_paper.py`'s larger-scale
   setup is preferred for the paper, port its graph/feature construction
   onto `GraphConstructor`/`FeatureEngineer` so `main.py` and the paper
   numbers come from the same code.

### 5.15 — Config management, checkpointing, model card

**[DEFEND]** — no hardcoded paths found in `hydro_graph/`; all I/O paths
route through `cfg.paths.*`, pydantic-validated in `hydro_graph/config.py`.
Checkpointing (`Trainer._save_checkpoint`/`load_best_checkpoint`) already
existed and works. **[FIX]**: `cache_version` in `config/config.yaml`
bumped `2.1.0` → `2.2.0` — the graph-construction and dataset-normalisation
fixes in this audit change what the cached artifacts mean, and a stale
`2.1.0`-tagged cache built under the old buggy code would otherwise be
silently treated as valid. **[FIX]**: `MODEL_CARD.md` added (data honesty,
splits, metrics, calibration, edge direction, reproducibility command,
limitations — see that file for the full writeup, summarised inline above).

---

## Test suite — Priority 0 per the shared preamble

**Before this audit, the test suite did not test the pipeline it claimed to
test.** `tests/test_pipeline.py` and `run_tests.py` called an API that had
been replaced by a "v2" rewrite (dual-scale rainfall, multi-lead output,
physics-informed directed edges): `TemporalEncoder(seq_len=...)` instead of
`short_seq_len`/`long_seq_len`, `enc.build_snapshot()` instead of
`build_input`/`build_targets`, a 17-dim single-lead model input instead of
34-dim/4-lead, `FocalLoss(alpha=0.25, gamma=2.0)` instead of
`FocalTverskyLoss(alpha, beta, gamma)` (whose `__init__` asserts
`alpha+beta≈1.0` — the old call would have raised `AssertionError` on the
very first instantiation), `get_chronological_split(enc.T, enc.seq_len)`
instead of the current 3-required-argument signature. Every fixture-dependent
test would have errored before a single assertion ran. This is exactly
"finding #1" the preamble warns about, and it went undetected because no one
had run `pytest` against the current code — the "35 tests" this repo is
described as having were testing a model architecture that no longer exists.

**[FIX]**: `tests/test_pipeline.py` rewritten against the actual current API
(51 tests, organized by phase, same structure as before for familiarity),
plus new regression coverage specifically for the two Priority 1/3 bugs
found in this audit (see sections above) and for the new ECE estimator.
`run_tests.py` (previously a second, independently-stale hand-rolled
reimplementation of the same checks) reduced to a thin wrapper over
`pytest tests/test_pipeline.py` — there should be exactly one test suite,
which is the failure mode that caused the original drift.

```
$ python -m pytest tests/test_pipeline.py -v --tb=short
...
51 passed, 3 warnings in 77.36s
```
(Full pass, this session, CPU. One failure was hit and fixed along the way —
`test_static_map_generation`, the matplotlib backend bug from §5.14 above.)

**The same API-drift disease exists in a third file, `demo.py`, not fixed
this session** — it's a standalone "CTO demo" driver script, not part of the
test suite, so it wasn't in scope for the pytest fix, but `python demo.py`
(as its own docstring instructs) currently fails immediately:
`TemporalEncoder(seq_len=cfg.features.temporal_seq_len, ...)` — `seq_len` was
replaced by `short_seq_len`/`long_seq_len` in the v2 rewrite, and
`cfg.features.temporal_seq_len` doesn't exist on the current
`FeaturesConfig` pydantic model at all (`AttributeError` before the
`TypeError` is even reached). `[DOCUMENT]`, not fixed — flagging so it isn't
mistaken for a working entry point; recommend either updating it to the v2
API (`short_seq_len`/`long_seq_len`, `HydroGraphDataset`'s current
constructor, multi-lead model output) or removing it in favour of `main.py`
directly, which does the same thing correctly.

**`streamlit_app.py` had a related but distinct bug** (not API drift, a
genuine correctness gap introduced by this audit's own fix): it rebuilds
`edge_index` from a cached graph's raw `list(G.edges())`
(`streamlit_app.py:276-279`) independently of `main.py`, so without also
routing it through `orient_drainage_edges()`, the UI could run inference
using a differently-structured (possibly still-symmetric, e.g. against an
older cached graph or a live-OSM graph) adjacency than what the model was
actually trained on. **[FIX]**: updated to call `orient_drainage_edges(G,
edge_features)`, same as `main.py`.

---

## Reproduction — real numbers, this session

`main.py --mode demo --skip-osm --force-retrain` at the documented demo bbox
(`~1400 nodes`) measured **~66 minutes per training epoch on this machine's
CPU** (`before_run.log`: epoch 1 = 3959.1s, epoch 2 = 2513.3s) — too slow to
run the multiple epochs needed for a before/after trained-model comparison
within this session. This is itself worth flagging: "one command reproduces
every number" (§5.14) is only true in a useful sense if that command
finishes in reasonable time, and on CPU it currently does not at demo scale.
Not fixed this session (would mean profiling/batching across timesteps
rather than one NeighborLoader/`k_hop_subgraph` call per snapshot); noted as
a HIGH-severity practicality gap.

Given that, this audit's evidence splits into two kinds:

1. **Structural before/after proof for the two Priority 1/3 correctness bugs**
   (edge symmetry, rain-norm leakage) — instant, deterministic, isolated
   scripts (`scratchpad/edge_direction_before_after.py`,
   `scratchpad/rain_norm_before_after.py`), quoted in full above. These are
   the load-bearing evidence for those two findings and don't depend on
   training converging.
2. **One real, reduced-scale end-to-end run** of the *fixed* pipeline
   (`--skip-osm --force-retrain`, bbox shrunk to `(80.24, 12.98, 80.26,
   13.00)` — 225 nodes — and the training-event window shortened to
   `2015-11-01`–`2015-12-01`, 3 epochs, baseline_epochs=8) to prove the fixed
   pipeline runs to completion and produces genuine `eval_metrics.json` /
   `baseline_metrics.json` artifacts with real numbers — see below for the
   verbatim result and exact file paths. This is a correctness/plumbing
   proof, not a performance claim: at 3 epochs and 225 nodes it is not
   expected to be competitive with a full run, and isn't presented as such.

### Verbatim result

Full run log: `scratchpad/after_run.log` (this session). Config used:
`scratchpad/config_verify.yaml` (a copy of `config/config.yaml` with
`bbox_demo` shrunk to `(80.24, 12.98, 80.26, 13.00)`, the 2015 training
window shortened to `2015-11-01`–`2015-12-01`, `epochs: 3`,
`baseline_epochs: 8`, and all `paths:` redirected to a scratch directory so
the repo's own `data/` tree was never touched). Total wall time: 1050.2s
(~17.5 min). Graph: 225 nodes, 956 edges, 18 waterway edges all oriented
downhill (§3.8). Split: train=582 steps (flood rate 2.71%), val=51 (82.59%),
test=52 (85.91%) — the split-skew finding from §1.1 reproduced again, a third
time, at a third window size.

**DS-STGAT, test (2015), lead=1hr**
(`scratchpad/verify_data/outputs/eval_metrics.json`):

| Metric | Value |
|---|---|
| F1 | 0.2716 |
| Precision | 0.9975 |
| Recall (POD) | 0.1572 |
| AUC-ROC | 0.9346 |
| AUC-PR | 0.9849 |
| CSI | 0.1571 |
| Brier | 0.7348 |
| **ECE** | **0.7348** |
| base rate | 87.17% |

**Baselines, same test split, lead=1hr**
(`scratchpad/verify_data/outputs/baseline_metrics.json`):

| Model | F1 | AUC-ROC | CSI | ECE |
|---|---|---|---|---|
| Persistence | 0.9906 | 0.9844 | 0.9813 | 0.0163 |
| Random Forest | 0.9908 | 0.9997 | 0.9818 | 0.1316 |
| LSTM-only | 0.7730 | 0.9256 | 0.6300 | 0.3219 |
| GCN+GRU | 0.9852 | 0.9793 | 0.9708 | 0.0247 |
| SAGEv1+GRU | 0.9824 | 0.8814 | 0.9654 | 0.0312 |
| **DS-STGAT** | **0.2716** | 0.9346 | **0.1571** | **0.7348** |

**Honest reading of this result, not spun**: at this reduced training
budget, DS-STGAT loses to every baseline, including the zero-parameter
persistence floor, by a wide margin on F1 and CSI (though its AUC-ROC is
mid-pack, meaning it does rank flooded-vs-not somewhat sensibly — the huge
F1 gap comes from Precision=0.9975 but Recall=0.157: the model is only
confident and correct on a small slice of true positives, at the default
0.50 threshold). This is not evidence the architecture is wrong; it is
evidence that **3 epochs is nowhere near enough** for a 528K-parameter
model trained under a 2.71% positive rate to generalise to an 87%-positive
test split (the split-skew finding, §1.1), while the baselines are either
zero-parameter (persistence), tree-based and fast to fit (RF), or far
smaller networks that converge faster (GRU/GCN/SAGE, ~1-2 order of magnitude
fewer parameters). The checkpoint selected was epoch 1 (val F1 peaked there
and *declined* over epochs 2-3 — `after_run.log`, "Epoch 1 ... Best val
F1=0.2797 saved" then epochs 2-3 both lower) — the model had not converged
and arguably wasn't stable yet. **This is the honest drop the task's
Priority-1 instructions predicted** ("a drop is the expected, honest
outcome") — not from fixing leakage this time (DS-STGAT was never leaking
this comparison), but from finally having a real baseline suite that a
short/small verification run can be seen losing to. It should not be
mistaken for a verdict on the architecture: this run's entire purpose was to
prove the pipeline executes correctly end-to-end and produces genuine,
file-backed numbers — it explicitly was not budgeted for convergence (see
the CPU throughput problem above; a full 120-epoch run at the documented
demo bbox is the correct comparison, and this session could not complete one
in reasonable time).

Cross-event (2018 analogue), DS-STGAT lead=1hr: F1=0.2622, AUC-ROC=0.8364,
CSI=0.1509, ECE=0.3456, base_rate=35.81% — directionally consistent with the
2015 test result (same undertrained checkpoint), included for completeness,
not as a generalisation claim.

**To regenerate at full/documented scale**: `python main.py --skip-osm
--force-retrain` (demo bbox) and budget several hours on CPU per the timing
above, or run on a GPU — then re-run `python main.py --skip-osm
--force-retrain` again with a shorter window if a fast sanity check is
wanted, using `scratchpad/config_verify.yaml` as a template for the
`event_end` truncation and reduced `epochs`/`baseline_epochs` that made this
session's run tractable.

---

## Definition of done — status

- [x] Splits time-blocked and leakage-free — **was already true**, now
      regression-tested (§1.1/1.3).
- [x] Rain-normalisation leakage — **found, fixed, regression-tested**
      (§1.2).
- [x] Held-out metrics reported with baselines — **persistence baseline
      added**; real metrics generated this session, see §Reproduction
      (§2.6/2.7). **At the reduced 3-epoch verification scale run this
      session, DS-STGAT did not beat any of the 5 baselines** (F1=0.27 vs.
      0.77-0.99) — reported honestly rather than omitted; almost certainly a
      convergence artifact of the epoch budget and split skew (§Reproduction
      has the full explanation), not evidence against the architecture. A
      full-scale run is needed before claiming DS-STGAT beats any baseline
      on a resume.
- [x] Edge direction verified respected on the directed graph — **found
      violated (100% of waterway edges bidirectional), fixed, verified 0%
      after fix, regression-tested** (§3.8).
- [x] Calibration measured (ECE + reliability diagram) — **ECE was not
      computed anywhere in the shipped pipeline; now computed, plotted, and
      saved to `eval_metrics.json`** (§4.11). Measured ECE this session was
      0.7348 at the reduced verification scale (badly uncalibrated, but
      confounded by the same undertrained-checkpoint/split-skew issue
      above — baselines on the identical split scored ECE 0.016-0.32, so
      part of this is architecture-specific and part is training budget).
      The word "probabilistic" should stay qualified until a real ECE from
      a full-scale run is on record — the paper's claimed 0.082 was
      unverified and, per §2.6, not even connected to any code path that
      could have produced it.
- [ ] One command reproduces every number in the README — **`main.py
      --skip-osm --force-retrain` now works correctly**, but (a) it is slow
      enough on CPU that "reproduces" is impractical at the documented demo
      scale without a GPU or a multi-hour budget, and (b) the numbers
      actually printed in `DS_STGAT_Paper.tex`/README come from
      `evaluate_paper.py`, a second, divergent pipeline this session did not
      reconcile with `main.py`. Both gaps are documented, neither is
      silently glossed over.

**Also found and fixed, outside the original checklist**: a paper draft
(`DS_STGAT_Paper.tex`) had been edited, before this session, to strip
"synthetic" from its own dataset/limitations language and add a paragraph
implying real-world data sources were used for its reported numbers. That
directly contradicts what every code path in this repository actually does
(`use_synthetic=True`, empty `data/raw/`). Reverted — see the diff on that
file and `MODEL_CARD.md`.

---

## Addendum — follow-up session, Phase 2 completion

A second session picked this repo back up specifically to close the two
items left `[ ]` above and finish what `06-hydrograph.md` (Phase 2) asked
for. Summary — full detail in `MODEL_CARD.md`, which is now the canonical,
up-to-date source for real numbers (this file's §Reproduction numbers above
are superseded, kept only as a historical record of the audit):

- **`generate_figures.py` rewritten** — every hardcoded literal removed
  (`train_loss`/`val_auc`/`val_f1` array, the `f1`/`auc`/`csi` arrays, the
  `dsstgat_f1`/`gcn_f1`/etc. arrays, the fabricated calibration curves, the
  hardcoded `Total Parameters: 528,422` banner text). It now loads only
  `data/outputs/eval_metrics.json` / `baseline_metrics.json` /
  `cross_event_metrics.json` and skips any figure it can't back with a real
  number. The dead `C:\...\Downloads\Hydrograph\...` absolute path (this
  file's §0) is gone — output paths are config-driven and repo-relative.
- **A real, full pipeline run against the repo's actual `data/processed/`
  and `data/outputs/` paths** (not a scratch directory this time):
  `python main.py --skip-osm --force-retrain --epochs 15
  --baseline-epochs 15` at a corrected demo bbox (870 nodes; the original
  ~7,800-node demo bbox measured ~66 min/epoch, so `bbox_demo` was shrunk
  and documented as such in `config/config.yaml` and `MODEL_CARD.md`).
  Real `eval_metrics.json`, `baseline_metrics.json`, and (new)
  `cross_event_metrics.json` now exist at those paths.
- **A genuine bug this run surfaced and fixed**: `plot_calibration()`
  (`phase6_inference.py`) crashed (`ValueError: Too many bins for data
  range`) building a probability histogram when predictions collapse to a
  single value — `ax_hist.hist(...)` now passes an explicit
  `range=(0.0, 1.0)` (predictions are sigmoid-bounded regardless of
  spread), fixed and re-verified.
- **The result itself, reported honestly, not chased**: at this budget,
  DS-STGAT and every other gradient-trained baseline collapsed to a
  trivial "always predict flood" solution (recall=1.0000, AUC-ROC≈0.49-0.50
  at every lead) and lost to Persistence and Random Forest by a wide
  margin. This is the split-skew limitation flagged above (§Priority 1,
  "split flood-rate skew") manifesting concretely, not a new bug — full
  table, raw-JSON evidence, and an honest read of what it does and doesn't
  show is in `MODEL_CARD.md` §Metrics.
- **Dead code archived** (`src/`, `pipeline/`, the second `config` loader,
  `examples/`, `deploy.py` → `archive/`, with `archive/README.md`
  explaining why), **`demo.py` fixed** (was calling a v1 API that no
  longer exists anywhere in this repo; now a thin wrapper over `main.py`'s
  real pipeline), **`evaluate_paper.py`/`generate_paper.py` marked
  deprecated** with a runtime warning, **`README.md` rewritten** (the old
  Quick Start documented the dead `src/` API), **`LICENSE`** and a
  top-level **`.gitignore`** added (neither existed).
- One command now reproduces every number in `data/outputs/`:
  `python main.py --skip-osm --force-retrain --epochs 15
  --baseline-epochs 15` (see `MODEL_CARD.md` §Reproducibility for the
  full-budget alternative and why it wasn't run this session).
