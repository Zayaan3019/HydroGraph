# Archive — dead code, not the pipeline

Everything under this directory is **not used by anything that runs**.
`hydro_graph/` (phases 1-6, driven by `main.py`) is the only pipeline this
repository actually executes; nothing in it imports from here.

This was confirmed by `AUDIT_REPORT.md` (see "Resolving the duplication"):

```
grep -rn "from src\|import src\|from pipeline\|import pipeline" hydro_graph/ main.py
```

returns nothing. It was moved here (not deleted) so the history/intent is
preserved and reversible, rather than silently discarded.

## What's here and why it existed

- **`src/`** — an earlier, v1 scaffold (`GraphConstructor`, `FeatureEngineer`,
  `HydroGraphSTGNN`, `Trainer`, single-lead, single-scale GRU + plain
  GraphSAGE) that predates the current `hydro_graph/` v2 rewrite
  (dual-scale GRU, GATv2 + SAGE, multi-lead output, physics-informed
  directed edges). The top-level `README.md`'s old "Quick Start" section
  documented this API — that section has been rewritten to document the
  real `hydro_graph/`/`main.py` entry point instead.
- **`pipeline/`** — standalone CLI scripts (`acquire_data.py`,
  `validate_data.py`, `train.py`, `evaluate.py`, `predict.py`) that operate
  on the `src/` classes above. Dead for the same reason.
- **`config/__init__.py`, `config/config_loader.py`** — a second, pydantic
  config loader for the `src/`/`pipeline/` API. **Not the same file as**
  `config/config.yaml`, which is still live and canonical — it's loaded
  directly by `hydro_graph/config.py::load_config()`, not through this
  package.
- **`examples/`** — usage examples written against the `src/` API
  (`end_to_end_pipeline.py`, `test_core.py`, `test_integration.py`,
  `test_setup.py`). These are not part of the real test suite —
  `tests/test_pipeline.py` (run via `pytest`) is.
- **`deploy.py`** — orchestrates `pipeline/*.py` via subprocess calls; dead
  for the same reason as `pipeline/`.

## If you need something from here

Everything a real run needs has a v2 equivalent in `hydro_graph/`:

| Dead (`archive/`)                          | Live equivalent                                    |
|---------------------------------------------|-----------------------------------------------------|
| `src/graph_construction.py`                 | `hydro_graph/phase1_graph.py`                       |
| `src/feature_engineering.py`                | `hydro_graph/phase2_features.py`                    |
| `src/dataset.py`                             | `hydro_graph/phase3_temporal.py`                    |
| `src/model.py`                               | `hydro_graph/phase4_model.py`                       |
| `src/trainer.py`                             | `hydro_graph/phase5_training.py`                    |
| `src/inference.py`                           | `hydro_graph/phase6_inference.py`                   |
| `config/config_loader.py`                    | `hydro_graph/config.py`                             |
| `pipeline/*.py`, `deploy.py`, `examples/*.py`| `main.py` (single entry point, see repo root README)|
