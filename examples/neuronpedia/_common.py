"""Shared CLI + pretty-printing helpers for the Neuronpedia NLA demos."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# Make ``np_nla`` importable whether the demo is run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from np_nla import NeuronpediaNLA, Verbalization  # noqa: E402

MODELS = {
    "llama": "llama3.3-70b-it",
    "gemma": "gemma-3-27b-it",
}


def make_arg_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="llama",
        help="which hosted NLA to use (default: llama = llama3.3-70b-it / layer 53)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="AV sampling temperature (default 0.4, matching the demo)",
    )
    return ap


def open_nla(args) -> NeuronpediaNLA:
    return NeuronpediaNLA(MODELS[args.model], temperature=args.temperature)


def banner(title: str, paper_ref: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")
    print(f"Paper: {paper_ref}\n")


def rule(label: str = "") -> None:
    print(f"\n{'-' * 78}")
    if label:
        print(label)


def show(v: Verbalization, *, width: int = 96, highlight: list[str] | None = None) -> None:
    """Print one verbalization: token, fidelity, and wrapped description."""
    desc = v.description.strip()
    if highlight:
        marks = [h for h in highlight if h.lower() in desc.lower()]
        flag = f"   <<< mentions: {', '.join(sorted(set(marks)))}" if marks else ""
    else:
        flag = ""
    print(
        f"\n  pos {v.position:>3}  token={v.token!r:<14}  "
        f"cos={v.cosine_similarity:.3f}  mse={v.mse:.3f}{flag}"
    )
    for line in textwrap.wrap(desc, width=width):
        print(f"      {line}")


def run(main_fn) -> None:
    """Run a demo's ``main``, printing API/outage errors cleanly (no traceback)."""
    try:
        main_fn()
    except RuntimeError as e:
        print(f"\n[error] {e}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)


def cache_link(result, model_id: str, base: str = "https://www.neuronpedia.org") -> None:
    """Print the URL that re-opens this run in the web demo (?id=<cacheId>)."""
    if getattr(result, "cache_id", None):
        print(f"\n  [re-open in demo: {base}/{model_id}/nla?id={result.cache_id}]")
