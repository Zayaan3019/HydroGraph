# Model Card — Hydro-Graph DS-STGAT

This card documents the model actually shipped in this repository: the
`hydro_graph/` package (phases 1-6) driven by `main.py`. It does **not**
describe `src/`, `pipeline/`, `config/` (top-level) or `examples/`, which are
a divergent, unmaintained v1 scaffold that nothing in `hydro_graph/` imports
— see "Canonical implementation" below before trusting anything in the
top-level README's Quick Start code blocks, which document that dead path.

## What this model is

DS-STGAT (Dual-Scale Spatiotemporal Graph Attention Network): a GATv2 +
GraphSAGE spatial encoder over a directed street/drainage graph, fused with a
dual-scale GRU temporal encoder (6hr short-term rainfall trigger + 24hr
antecedent-moisture window), trained with a multi-horizon Focal Tversky loss
to predict flood probability at 4 lead times (1h/3h/6h/12h) for every node.

## Data — synthetic, not observed

**Every number this model has ever produced comes from a synthetic Chennai
flood simulation, not observed data.** `config/config.yaml` sets
`features.use_synthetic: true`, and `data/raw/` is empty in this repository
— there is no DEM GeoTIFF, no Sentinel-1/2 imagery, no GPM IMERG rainfall CSV
checked in or downloaded. Concretely:

- **Graph topology**: OSMnx download of Chennai's real street/waterway
  network is attempted (`--skip-osm` to force off); if OSMnx isn't installed
  or the Overpass API is unreachable, it silently falls back to a synthetic
  grid graph (`phase1_graph.py::_build_synthetic_multigraph`, seed=42).
- **Static features** (elevation, slope, TWI, NDVI, NDWI, NDBI, SAR VV, …):
  all 16 are procedurally generated from hand-tuned Chennai-shaped functions
  (`phase2_features.py`), not sampled from real rasters.
- **Rainfall**: a hand-authored intensity profile (`_PROFILE_2015`,
  `_PROFILE_2018` in `phase3_temporal.py`) with fixed event timings/peaks,
  perturbed with fixed-seed noise — not GPM IMERG or IMD gauge data.
- **Flood labels**: derived deterministically from that synthetic rainfall
  via a hand-tuned threshold + spatial-propagation formula
  (`phase3_temporal.py::_generate_flood_labels`) — not observed flood
  extents, not the DFO archive (`data_downloader.py` can fetch DFO event
  metadata for context but nothing in the pipeline uses it as a label
  source).

This is a legitimate way to unit-test a spatiotemporal GNN's architecture and
training loop, and the docstrings are honest about "physics-grounded
synthetic" data throughout. It is **not** a validated flood forecasting
system, and metrics from it say nothing about real-world skill — they
describe how well DS-STGAT can recover a synthetic label function it was
partly informed by (the label generator itself propagates through the same
graph edges the GNN is given, see `AUDIT_REPORT.md` §1.5 for
why that inflates the graph's apparent advantage over non-graph baselines
compared to a benchmark with independently-labeled ground truth).

## Splits

Chronological, not random (`phase3_temporal.py::get_chronological_split`):
`train_idx`, `val_idx`, `test_idx` are contiguous, non-overlapping ranges of
the single event's timeline, in that order — no timestep in `val`/`test`
precedes any timestep in `train`. This is the correct approach for a
spatiotemporal panel (see the leakage audit for why a random split would
leak future information through spatial neighbours at the same timestamp).

**Known limitation**: because the whole panel is one continuous storm event
and the catastrophic peak is concentrated late in the timeline, a
chronological 85/7.5/7.5 split produces highly unequal flood base rates
across splits — verified empirically at demo scale as train≈15%,
val≈100%, test≈63% (see the reproduction run below). Early stopping is
computed on an almost-entirely-positive validation slice, which is not a
representative estimate of deployment-time class balance. Recommended fix
(not yet implemented): split across multiple independent synthetic storm
events rather than one continuous event, or use a purged/embargoed
walk-forward scheme across several events.

## Metrics

**Real numbers, generated in this session**, live in
`data/outputs/eval_metrics.json`, `data/outputs/baseline_metrics.json`, and
`data/outputs/cross_event_metrics.json` — all three written directly by
`python main.py`, not typed in by hand. **Do not cite the numbers baked into
`DS_STGAT_Paper.tex`** or produced by `evaluate_paper.py` /
`generate_paper.py`: both carry a deprecation banner now — they are a
second, unreconciled pipeline whose numbers were never connected to this
repository's real training/evaluation code path (see `AUDIT_REPORT.md`
§2.6 and §5.14). The figures under `data/outputs/figures/` are regenerated
by `generate_figures.py`, which loads only the JSON files above and renders
nothing it can't back with a real number. `AUDIT_REPORT.md` §Reproduction
documents an earlier, smaller (225-node, 3-epoch) scratch verification run
from the prior audit session — the table below **supersedes** it as the
current, canonical result, produced at the repo's real `data/processed/` /
`data/outputs/` paths, not a scratch directory.

Primary metrics: F1, Precision, Recall, AUC-ROC, **AUC-PR** (the one to
trust under this benchmark's class imbalance — ROC-AUC flatters rare-event
classifiers), CSI/FAR/POD (hydrology-standard skill scores), Brier score,
and **ECE** (Expected Calibration Error).

**Exact reproduction command** (also see `--baseline-epochs`, added this
session, and the `bbox_demo` change noted under Reproducibility below):

```bash
python main.py --skip-osm --force-retrain --epochs 15 --baseline-epochs 15
```

Graph: 870 nodes, 3,880 edges (68 waterway edges, all verified oriented
downhill). Split: train=683 steps (flood rate 14.93%), val=60 (99.71%),
test=61 (62.12%) — the chronological split-skew limitation described above,
reproduced again at this scale.

**DS-STGAT, test (2015 event)**:

| Lead | F1 | Precision | Recall | AUC-ROC | AUC-PR | CSI | Brier | ECE | base rate |
|---|---|---|---|---|---|---|---|---|---|
| 1h  | 0.7538 | 0.6048 | 1.0000 | 0.4856 | 0.5989 | 0.6048 | 0.3952 | 0.3952 | 60.48% |
| 3h  | 0.7279 | 0.5721 | 1.0000 | 0.4981 | 0.5712 | 0.5721 | 0.4278 | 0.4278 | 57.21% |
| 6h  | 0.6894 | 0.5260 | 1.0000 | 0.5001 | 0.5260 | 0.5260 | 0.4740 | 0.4740 | 52.60% |
| 12h | 0.6235 | 0.4529 | 1.0000 | 0.5000 | 0.4529 | 0.4529 | 0.5471 | 0.5471 | 45.29% |

**Baselines, same test split, lead=1h**:

| Model | F1 | Precision | Recall | AUC-ROC | AUC-PR | CSI | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| Persistence   | **0.9628** | 0.9501 | 0.9759 | **0.9487** | 0.9418 | **0.9283** | **0.0456** | **0.0456** |
| Random Forest | 0.8912 | **0.9999** | 0.8038 | 0.9776 | **0.9873** | 0.8038 | 0.0914 | 0.1161 |
| LSTM-only     | 0.7538 | 0.6048 | 1.0000 | 0.5000 | 0.6048 | 0.6048 | 0.3952 | 0.3952 |
| GCN+GRU       | 0.7538 | 0.6048 | 1.0000 | 0.5000 | 0.6048 | 0.6048 | 0.3952 | 0.3952 |
| SAGEv1+GRU    | 0.7538 | 0.6048 | 1.0000 | 0.5000 | 0.6048 | 0.6048 | 0.3952 | 0.3952 |
| **DS-STGAT**  | 0.7538 | 0.6048 | 1.0000 | 0.4856 | 0.5989 | 0.6048 | 0.3952 | 0.3952 |

**Honest reading, not spun**: at this training budget, DS-STGAT (and every
other *learned neural* baseline — LSTM-only, GCN+GRU, SAGEv1+GRU) collapsed
to predicting "flood" for essentially every node at every lead time
(`recall_lead*` = 1.0000, `tn`/`fn` = 0 in the raw JSON for all four —
i.e. zero true negatives, zero false negatives, at every lead). AUC-ROC
sits at ~0.48-0.50 for all four — no better than a coin flip at ranking
flooded vs. non-flooded nodes; the F1/CSI scores that look moderate
(0.62-0.75) are an artifact of the test split's high base rate (45-60%
positive, itself a symptom of the chronological split-skew limitation
described above), not evidence of learned skill. **DS-STGAT lost to
Persistence and Random Forest by a wide margin on every metric that
matters under this base rate** (AUC-ROC, AUC-PR, ECE, Brier), and did not
beat the three other learned baselines — they collapsed to the identical
trivial solution. `best_model.pt`'s selected checkpoint was epoch 1 (val
F1 peaked there and never moved again across 15 epochs — see
`real_run.err.log`-style training logs: `loss`, `val_loss`, `F1@1h`, and
`AUC@1h` are frozen to 4 decimal places from epoch 2 onward). This is the
same failure mode the leakage audit already flagged as a HIGH-severity,
not-yet-fixed limitation: a recall-heavy Focal Tversky loss
(`tversky_beta=0.70`) combined with a validation split that is 99.71%
positive rewards a trivial "always predict flood" solution and gives
early stopping nothing better to select. **This is the honest, reportable
outcome the task anticipated ("a drop is the expected outcome") — it is
evidence against this specific training configuration on this specific
benchmark, not necessarily against the architecture.** The two
non-neural baselines (Persistence, a zero-parameter heuristic; Random
Forest, which sees no chronological ordering to exploit and is not
vulnerable to this optimization pathology) were unaffected, which is
itself informative: the failure is specific to gradient-based training
under this loss/split combination, not the underlying benchmark.

**Cross-event (2018 analogue)**, DS-STGAT: F1=0.4628, AUC-ROC=0.44-0.50,
CSI=0.3011, ECE=0.6989, base_rate=30.11% (all 4 leads identical, same
collapsed-checkpoint pattern). Included for completeness, not as a
generalisation claim — full table in `data/outputs/cross_event_metrics.json`.

**What would need to change to get a real result here** (not attempted this
session, per the explicit instruction not to tune away a genuine finding):
address the split skew (multi-event or purged walk-forward splitting, not
a single continuous event), reconsider `tversky_beta` given how easily it
lets the loss get "free" recall at test-base-rate cost, and/or train for
enough epochs/patience to see if the collapse is a transient local optimum
or the loss landscape's actual attractor at this scale.

## Calibration

The output head is a plain sigmoid trained with Focal Tversky loss — a
ranking-oriented loss, not a proper scoring rule. There is no reason a
priori for its outputs to be calibrated probabilities, and prior to this
audit the pipeline never computed a calibration number, only a reliability
diagram to eyeball. `_expected_calibration_error()` (`phase5_training.py`)
now computes ECE alongside every other metric, and
`InferenceEngine.plot_calibration()` prints it on the reliability diagram
(`data/outputs/calibration_curve.png`, real held-out predictions) and
returns it. **Measured this session: ECE=0.3952 at lead=1h** (`ece_lead0`
in `eval_metrics.json`) — badly uncalibrated, though this specific number
is confounded by the same collapsed-checkpoint issue above (a model that
predicts ~1.0 for every node has a degenerate, not meaningfully
"miscalibrated in an interesting way," reliability curve). Persistence's
ECE (0.0456) and Random Forest's (0.1161) on the identical split show what
a non-degenerate baseline's calibration looks like here for comparison.
**Do not call this model's output "probabilistic" or "calibrated"** — the
measured number says it plainly isn't, at this training budget.

## Edge direction

Water flows downhill; a drainage graph where message passing can propagate a
flood signal uphill as easily as downhill is not modelling drainage. Before
this audit, both the synthetic-fallback graph builder and (very likely,
though not independently verified against live OSM data) the OSM path
produced a substantially symmetric edge set for waterway edges — see
`AUDIT_REPORT.md` finding C-1. `phase1_graph.py::orient_drainage_edges()`
now enforces this post hoc, using real (Phase-2) elevation: every waterway
edge is oriented from higher to lower elevation, and if both directions of
the same physical connection existed, only the steeper-downhill one
survives. Road edges are deliberately left bidirectional (two-way streets;
surface runoff is not channelised the way a drain is).

## Reproducibility

One command regenerates every number in `data/outputs/`:

```bash
python main.py --skip-osm --force-retrain --epochs 15 --baseline-epochs 15
```

This is the exact command used to produce the numbers in this document —
`--epochs 15 --baseline-epochs 15` overrides `config/config.yaml`'s
production defaults (120 / 50) for CPU tractability at demo scale (~7-9
min/epoch measured this session; 120 epochs would be several hours). Omit
both flags to use the config's full production budget (recommended on a
GPU, or if you have hours to spare on CPU) — that has **not** been run in
this repository and would very plausibly produce a different (and
hopefully non-collapsed) result than the one documented above.

- `--skip-osm` forces the synthetic-fallback graph (seed=42, deterministic)
  instead of a live OSMnx/Overpass download, which is not reproducible
  run-to-run and depends on network access. Omit it only if you specifically
  want to test against live OSM data and accept that graph topology (and
  therefore every downstream number) will vary between runs.
- `--force-retrain` purges cached graph/feature/temporal/checkpoint files so
  nothing stale from a previous `cache_version` is silently reused.
- Seeding covers Python's `random`, numpy, and torch (CPU+CUDA) — see
  `main.py::seed_everything()` for exactly what is and is not covered
  (residual GPU-kernel nondeterminism is documented there, not hidden).

Config is entirely YAML-driven (`config/config.yaml`, pydantic-validated in
`hydro_graph/config.py`); no hardcoded paths in the pipeline itself. All
regenerable outputs — checkpoint, metrics JSON, baseline JSON, predictions
CSV, all PNGs/HTML — live under `data/models/` and `data/outputs/` per the
`paths:` section of the config, not scattered elsewhere.

**On `bbox_demo`'s size**: the original demo bbox (~13x13 km) builds a
~7,800-node synthetic graph, at which one training epoch measured ~66
minutes on CPU (`AUDIT_REPORT.md` §Reproduction) — not actually fast. It has
been shrunk to a size (~4.5x4.5 km, ~870 nodes) empirically verified to
train in ~11 min/epoch on CPU, so `--mode demo` (the default) matches its
own documented purpose. `--mode full` is unchanged and remains the large
production target — budget a GPU or a multi-day CPU run for it, and see
`--epochs`/`--baseline-epochs` to control the trade-off between wall time
and training budget explicitly.

## Limitations (say these out loud on a resume)

1. **Entirely synthetic** end to end — no real DEM, satellite imagery, or
   rainfall gauge/radar data has ever been loaded through this pipeline in
   this repository. Report it as "a synthetic benchmark for a
   spatiotemporal flood-forecasting architecture," not as validated flood
   forecasting.
2. Flood labels are generated by a formula that propagates through the same
   graph edges the GNN then learns to exploit — the graph's measured
   advantage over non-graph baselines is partly a feature of how the
   benchmark's labels were constructed, not solely evidence the graph would
   help on independently-observed flood extents.
3. Chronological train/val/test flood-rate skew (documented above) means
   early stopping and the reported val curve are not a representative
   estimate of deployment-time performance. **This is not theoretical** —
   at the 15-epoch/870-node scale run this session, it caused DS-STGAT and
   every other gradient-trained baseline (LSTM-only, GCN+GRU, SAGEv1+GRU)
   to collapse to a trivial "always predict flood" solution, selected at
   epoch 1 and never improved on (see Metrics above for the full table and
   raw-JSON evidence: recall=1.0000 and zero true/false negatives at every
   lead, for all four learned neural models). DS-STGAT lost to both
   non-neural baselines (Persistence, Random Forest) by a wide margin.
   Do not cite this session's DS-STGAT numbers as evidence the
   architecture works — they demonstrate the split-skew failure mode, not
   the model's ceiling.
4. Calibration is whatever the measured ECE says it is (0.3952 at lead=1h
   this session, badly uncalibrated — see Calibration above), and do not
   describe the output as "probabilistic" if it isn't.
5. Cross-event ("2018 analogue") evaluation reuses the same procedural
   rainfall/label generator with different event timing/intensity — it is a
   test of robustness to a different synthetic storm shape, not of
   generalisation to a real, independent flood event.
