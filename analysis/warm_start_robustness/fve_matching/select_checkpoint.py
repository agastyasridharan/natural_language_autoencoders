#!/usr/bin/env python3
"""Select the frozen FVE-matched checkpoint pair from exact result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from .core import (
        CHECKPOINT_INTERVAL,
        FINAL_STEP,
        ContractError,
        atomic_write_json,
        select_fve_matched,
        sha256_file,
        sha256_json,
    )
except ImportError:  # direct ``python select_checkpoint.py``
    from core import (  # type: ignore
        CHECKPOINT_INTERVAL,
        FINAL_STEP,
        ContractError,
        atomic_write_json,
        select_fve_matched,
        sha256_file,
        sha256_json,
    )


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    records = []
    for raw_path in args.result:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ContractError(f"exact-FVE result does not exist: {path}")
        try:
            result = json.loads(path.read_bytes())
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid exact-FVE JSON {path}: {exc}") from exc
        records.append((path, result))
    input_results = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path, _ in sorted(records, key=lambda pair: str(pair[0]))
    ]
    if output.exists():
        existing = json.loads(output.read_bytes())
        declared = existing.get("selection_payload_sha256")
        payload = dict(existing)
        payload.pop("selection_payload_sha256", None)
        if (
            existing.get("input_results") == input_results
            and declared == sha256_json(payload)
        ):
            return existing
        raise ContractError(f"refusing to overwrite existing selection {output}")
    selection = select_fve_matched(
        records,
        final_step=FINAL_STEP,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )
    selection["input_results"] = input_results
    component = Path(__file__).resolve().parent
    selection["selection_implementation"] = [
        {
            "path": name,
            "bytes": (component / name).stat().st_size,
            "sha256": sha256_file(component / name),
        }
        for name in ("protocol.yaml", "core.py", "select_checkpoint.py")
    ]
    selection["selection_payload_sha256"] = sha256_json(selection)
    atomic_write_json(output, selection)
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hold the lower-final-FVE condition at step 780 and select the "
            "higher condition's closest exact-FVE 20-step checkpoint."
        )
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Repeat for every exact-FVE result required by the frozen candidate grid.",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selection = run(args)
    print(
        json.dumps(
            {
                "held_lower_final": selection["held_lower_final"],
                "matched_higher": selection["matched_higher"],
                "output": str(Path(args.output).expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
