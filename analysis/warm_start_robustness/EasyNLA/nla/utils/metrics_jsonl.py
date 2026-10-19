"""Durable local scalar-metric logging independent of external dashboards."""

import json
import math
import numbers
import os
from pathlib import Path
from typing import Any


def append_scalar_metrics(path: str | Path, metrics: dict[str, Any], step: int) -> None:
    record: dict[str, Any] = {"step": int(step)}
    for key, value in metrics.items():
        if isinstance(value, bool) or value is None or isinstance(value, str):
            record[key] = value
        elif isinstance(value, numbers.Integral):
            record[key] = int(value)
        elif isinstance(value, numbers.Real):
            number = float(value)
            record[key] = number if math.isfinite(number) else None
        # Tables, tensors, and rich dashboard objects are intentionally omitted.
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
