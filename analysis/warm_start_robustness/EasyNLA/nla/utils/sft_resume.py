"""Lightweight SFT resume/config helpers (no model dependencies)."""

from __future__ import annotations

import argparse


def stable_run_config(args: argparse.Namespace) -> dict:
    """Recipe fields shared by fresh and resumed W&B/config validation."""
    return {
        key: value
        for key, value in vars(args).items()
        if key not in {"config", "resume_from"}
    }


def assert_resume_config_matches(previous: dict, args: argparse.Namespace) -> None:
    current = stable_run_config(args)
    previous_stable = {
        key: value for key, value in previous.items() if key != "resume_from"
    }
    for key in sorted(set(previous_stable) | set(current)):
        if previous_stable.get(key) != current.get(key):
            raise RuntimeError(
                f"resume config drift for {key}: "
                f"original={previous_stable.get(key)!r}, current={current.get(key)!r}"
            )


def wandb_init_kwargs(args: argparse.Namespace, wandb_id: str) -> dict:
    """Build W&B kwargs without the resume-local checkpoint path."""
    return {
        "project": args.wandb_project,
        "name": args.wandb_name,
        "group": args.wandb_group,
        "tags": args.wandb_tags.split(",") if args.wandb_tags else None,
        "config": stable_run_config(args),
        "id": wandb_id,
        "resume": "must" if args.resume_from is not None else "allow",
    }
