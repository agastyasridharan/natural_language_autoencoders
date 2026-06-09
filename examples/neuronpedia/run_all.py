#!/usr/bin/env python3
"""Run all Neuronpedia NLA paper-replication demos in sequence.

Usage:
    python examples/neuronpedia/run_all.py [--model llama|gemma]

Extra args after the model flag are passed through to each demo.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMOS = sorted(HERE.glob("demo_*.py"))


def main() -> None:
    passthrough = sys.argv[1:]
    failures = []
    for demo in DEMOS:
        print(f"\n{'#' * 78}\n# {demo.name}\n{'#' * 78}")
        rc = subprocess.run([sys.executable, str(demo), *passthrough]).returncode
        if rc != 0:
            failures.append(demo.name)
    print(f"\n{'=' * 78}")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    print(f"All {len(DEMOS)} demos completed.")


if __name__ == "__main__":
    main()
