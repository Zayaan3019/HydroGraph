# -*- coding: utf-8 -*-
"""
Quick validation entry point — runs the real pytest suite end-to-end.

This used to be a hand-rolled reimplementation of the phase-by-phase checks
that also live in tests/test_pipeline.py, calling the pipeline's public API
directly. That duplication is exactly how it silently rotted: the pipeline
moved to a v2 API (dual-scale rainfall, multi-lead output, seq_len ->
short_seq_len/long_seq_len, build_snapshot -> build_input/build_targets,
FocalLoss -> FocalTverskyLoss with a required beta) and this script kept
calling the v1 signatures, so every run failed at the first fixture despite
looking like a normal validation pass in its own printed summary.

There should be exactly one test suite. This script now just runs it.
"""
import subprocess
import sys

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_pipeline.py", "-v", "--tb=short"],
        cwd=".",
    )
    sys.exit(result.returncode)
