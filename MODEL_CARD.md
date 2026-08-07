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
graph edges the GNN is given, see `flags/leakage-audit.md` finding L-3 for
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

Real numbers, generated in this session, live in
`data/outputs/eval_metrics.json` and `data/outputs/baseline_metrics.json`
after running the reproduction command below — see `AUDIT_REPORT.md` for the
verbatim before/after values captured for this audit, with file paths.
**Do not cite the figures embedded in `DS_STGAT_Paper.tex` or
`data/outputs/figures/*.png`** without re-running the pipeline first: those
were produced by a separate one-off script (`evaluate_paper.py`) on a
machine/directory (`data/outputs/figures/figure_paths.json` points at
`C:\...\Downloads\Hydrograph\...`) that no longer exists anywhere on this
system, and `data/outputs/eval_metrics.json` / `baseline_metrics.json` were
empty at the start of this audit — those claimed numbers were never
reproducible from this repository's committed state.

Primary metrics: F1, Precision, Recall, AUC-ROC, **AUC-PR** (the one to trust
under the ~15-30% base rate here — ROC-AUC flatters rare-event classifiers),
CSI/FAR/POD (hydrology-standard skill scores), Brier score, and **ECE**
(Expected Calibration Error, added this audit — see Calibration below).
Report at all 4 lead times, plus baseline margins over Persistence
(added this audit), Random Forest, LSTM-only, GCN+GRU, and GraphSAGEv1+GRU.

## Calibration

The output head is a plain sigmoid trained with Focal Tversky loss — a
ranking-oriented loss, not a proper scoring rule. There is no reason a
priori for its outputs to be calibrated probabilities, and prior to this
audit the pipeline never computed a calibration number, only a reliability
diagram to eyeball. `_expected_calibration_error()` (`phase5_training.py`)
now computes ECE alongside every other metric, and
`InferenceEngine.plot_calibration()` prints it on the reliability diagram
and returns it. **Until you have looked at a real ECE number, do not call
this model's output "probabilistic"** — see `AUDIT_REPORT.md` for the
measured value.

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
python main.py --skip-osm --force-retrain
```

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
   estimate of deployment-time performance.
4. Calibration is whatever the measured ECE says it is — see
   `AUDIT_REPORT.md` for the number, and do not describe the output as
   "probabilistic" if it isn't.
5. Cross-event ("2018 analogue") evaluation reuses the same procedural
   rainfall/label generator with different event timing/intensity — it is a
   test of robustness to a different synthetic storm shape, not of
   generalisation to a real, independent flood event.
