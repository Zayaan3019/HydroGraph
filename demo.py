# -*- coding: utf-8 -*-
"""
Hydro-Graph DS-STGAT -- Quick Demo Wrapper
============================================
A friendly, fast entry point for a live demo: same real pipeline as
`main.py` (hydro_graph/ phases 1-6), just with demo-oriented defaults
(synthetic data, reduced epochs, baselines skipped by default) and a
banner.

This file used to reimplement all 6 phases against a v1 API
(`TemporalEncoder(seq_len=...)`, single-lead `HydroGraphDataset`,
`cfg.training.focal_alpha`, ...) that no longer exists anywhere in this
repository -- every one of those calls raised `AttributeError` or
`TypeError` before a single phase completed. Rather than re-duplicate
`main.py`'s logic a second time (the exact "second divergent pipeline"
pattern flagged for `evaluate_paper.py` in AUDIT_REPORT.md), this wrapper
calls the real, tested `run_pipeline()` in `main.py` directly, so there is
exactly one implementation of the pipeline to keep correct.

Run with:
    python demo.py
    python demo.py --use-real-osm     # download actual Chennai OSM (requires internet)
    python demo.py --epochs 30        # override training epochs (default: config)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BANNER = """
+======================================================================+
|        HYDRO-GRAPH  DS-STGAT  --  Urban Flood Forecasting            |
|        Dual-Scale Spatiotemporal Graph Attention Network             |
+======================================================================+
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hydro-Graph DS-STGAT -- Quick Demo (thin wrapper over main.py)"
    )
    p.add_argument("--use-real-osm", action="store_true",
                    help="Attempt a live OSMnx/Overpass download instead of the synthetic fallback graph")
    p.add_argument("--epochs", type=int, default=None,
                    help="Override number of training epochs from config")
    p.add_argument("--force-retrain", action="store_true",
                    help="Purge cached graph/feature/temporal/checkpoint files and rebuild from scratch")
    p.add_argument("--with-baselines", action="store_true",
                    help="Also run the 5-model baseline ablation (slower)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(BANNER)

    from hydro_graph.config import load_config
    from main import run_pipeline

    cfg = load_config()
    if args.epochs is not None:
        cfg.training.epochs = args.epochs

    print(f"  Study area : {cfg.study_area.name}")
    print(f"  Mode       : demo bbox = {cfg.get_bbox('demo')}")
    print(f"  Data       : {'live OSM download' if args.use_real_osm else 'synthetic fallback (deterministic, seed=42)'}")
    print(f"  Epochs     : {cfg.training.epochs}")
    print(f"  Baselines  : {'on' if args.with_baselines else 'off (pass --with-baselines to include)'}")
    print()

    t0 = time.time()
    run_pipeline(
        cfg,
        bbox_mode="demo",
        skip_osm=not args.use_real_osm,
        skip_train=False,
        force_retrain=args.force_retrain,
        skip_baselines=not args.with_baselines,
    )
    print(f"\nDemo finished in {time.time() - t0:.1f}s.")
    print(f"Outputs written to: {ROOT / cfg.paths.outputs_dir}")


if __name__ == "__main__":
    main()
