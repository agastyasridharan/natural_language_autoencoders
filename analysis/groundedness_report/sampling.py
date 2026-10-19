"""Keyed, order-independent sampling for activations, claims, and controls."""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence


def _key(seed: int, namespace: str, item_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{namespace}\0{item_id}".encode()).digest()


def sha256_rank(
    values: Iterable[Any],
    *,
    seed: int,
    namespace: str,
    identity=str,
) -> list[Any]:
    """Return a stable ranking independent of input ordering and process count."""
    rows = list(values)
    identities = [str(identity(value)) for value in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("sampling identities must be unique")
    return [
        value
        for _, _, value in sorted(
            (_key(seed, namespace, item_id), item_id, value)
            for value, item_id in zip(rows, identities)
        )
    ]


def sample_claims(
    claims: Sequence[Mapping[str, Any]],
    *,
    activation_id: str,
    seed: int,
    maximum: int = 3,
) -> list[dict[str, Any]]:
    """Uniformly sample at most three claims and record Horvitz--Thompson weights."""
    eligible = [
        (index, claim)
        for index, claim in enumerate(claims)
        if claim.get("scored", True) and claim.get("located", True)
    ]
    population = len(eligible)
    if population == 0:
        return []
    take = min(maximum, population)
    ranked = sha256_rank(
        eligible,
        seed=seed,
        namespace=f"claim:{activation_id}",
        identity=lambda row: row[0],
    )[:take]
    probability = take / population
    return [
        {
            "claim_i": int(index),
            "sampling_population": population,
            "sampling_n": take,
            "inclusion_probability": probability,
            "analysis_weight": 1.0 / probability,
        }
        for index, _ in sorted(ranked, key=lambda row: row[0])
    ]


def sample_positions(
    eligible_positions: Sequence[int],
    *,
    trajectory_id: str,
    seed: int,
    maximum: int = 4,
) -> list[int]:
    positions = sorted(set(int(value) for value in eligible_positions))
    return sorted(
        sha256_rank(
            positions,
            seed=seed,
            namespace=f"position:{trajectory_id}",
            identity=str,
        )[:maximum]
    )


def cyclic_derangement(
    ids: Sequence[str],
    *,
    seed: int,
    namespace: str = "negative-control",
) -> dict[str, str]:
    """Create the preregistered one-cycle, fixed-point-free activation map."""
    ranked = sha256_rank(ids, seed=seed, namespace=namespace)
    if len(ranked) < 2:
        raise ValueError("a derangement requires at least two unique ids")
    return {str(item): str(ranked[(index + 1) % len(ranked)]) for index, item in enumerate(ranked)}

