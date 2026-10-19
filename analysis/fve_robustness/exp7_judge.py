"""Exp 7 — Step 7.0: meaning-judge SANITY CHECK (API, tiny; hard check-in gate).

The whole experiment rests on one instrument: an LLM that partitions AV claims into MEANING
equivalence-classes across resamples. The single failure that would silently fabricate a "stable core"
is collapsing distinct same-TOPIC claims into one cluster (e.g. the 'robotic vacuum' / 'owner's husband'
/ 'cleaning staff' continuation guesses). This script runs the clustering judge on a small LABELLED set
(paraphrase -> merge; same-topic-different-claim -> split; distinct entities / negation -> split) and
scores it by pairwise agreement, flagging FALSE MERGES (dangerous) vs false splits.

Usage:  python3 exp7_judge.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODEL = "gpt-5.6-luna"
PROVIDER = "openai"
MAX_DIRECT_CLAIMS = 256
MAX_REP_CLAIMS = 256
CACHE_VERSION = "meaning-v6-openai-gpt56luna-exact-fields"
INPUT_USD_PER_MTOK = 0.20
OUTPUT_USD_PER_MTOK = 1.20
BUDGET_CAP_USD = 90.0  # hard internal stop, leaving $10 below the user's $100 ceiling
BUDGET_PATH = HERE / "results" / "exp7_openai_budget.json"
PRICING_SOURCE = "https://developers.openai.com/api/docs/models/compare"
_stats_lock = threading.Lock()
_stats = {"direct_requests": 0, "direct_failures": 0, "failure_examples": []}

SYSTEM = (
    "You group short CLAIMS made about one token of a document into clusters that assert the SAME thing "
    "BY MEANING. Two claims belong together ONLY if one could replace the other without changing what is "
    "asserted -- paraphrases, synonyms, reworded restatements of the same proposition. Put claims in "
    "DIFFERENT clusters if they name different entities, predict different continuations, or make "
    "different assertions, EVEN IF they share a topic or some words.\n"
    "Crucial: same TOPIC is not same CLAIM. Example: for what follows 'the floor was vacuumed by', the "
    "claims 'a robotic vacuum', \"his owner's husband\", and 'the family's cleaning staff' are THREE "
    "DIFFERENT claims (three different predicted continuations), NOT one, despite sharing the cleaning "
    "topic. But 'a robotic vacuum' and 'an automated cleaning device' ARE the same claim. Named-entity "
    "identity is always part of the claim: 'the rose family is important' and 'the Solanaceae family "
    "is important' are DIFFERENT claims because they name different plant families. Never erase or "
    "generalize a named-entity difference merely because the predicate is shared. Negations/opposites "
    "are different claims. When unsure, SPLIT rather than merge.\n"
    "Return one non-negative integer cluster id for EVERY indexed claim. Claims with the same meaning "
    "receive the same integer; different claims receive different integers. Cluster-id values are "
    "arbitrary. Preserve the schema's exact c0..cN mapping to claim indices 0..N."
)

def _output_format(n_claims: int) -> dict:
    """Build an exact-cardinality schema; array item schemas cannot enforce claim alignment."""
    properties = {f"c{i}": {"type": "integer", "minimum": 0} for i in range(n_claims)}
    return {
        "type": "json_schema",
        "name": "claim_clusters",
        "description": "Exactly one semantic-equivalence cluster id for every indexed claim.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


def _env_value(name: str) -> str:
    """Read one dotenv value without adding a non-approved runtime dependency."""
    if os.environ.get(name):
        return os.environ[name]
    path = REPO / ".env"
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                return value
    raise RuntimeError(f"{name} is not set in the process environment or {path}")


class BudgetLedger:
    """Thread-safe usage ledger with conservative pre-request reservation and a hard spend stop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reserved_usd = 0.0
        if BUDGET_PATH.exists():
            self.data = json.loads(BUDGET_PATH.read_text())
            if self.data.get("model") != MODEL or self.data.get("provider") != PROVIDER:
                raise RuntimeError("existing Exp-7 budget ledger belongs to a different judge")
        else:
            self.data = dict(provider=PROVIDER, model=MODEL, hard_cap_usd=BUDGET_CAP_USD,
                             input_usd_per_mtok=INPUT_USD_PER_MTOK,
                             output_usd_per_mtok=OUTPUT_USD_PER_MTOK,
                             pricing_source=PRICING_SOURCE, input_tokens=0, output_tokens=0,
                             estimated_cost_usd=0.0, responses=0)

    def _write(self) -> None:
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = BUDGET_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(BUDGET_PATH)

    def reserve(self, prompt: str, max_output_tokens: int) -> float:
        # UTF-8 bytes / 2 deliberately overestimates English-token input; max output is fully reserved.
        input_upper = math.ceil(len((SYSTEM + prompt).encode("utf-8")) / 2)
        amount = (input_upper * INPUT_USD_PER_MTOK + max_output_tokens * OUTPUT_USD_PER_MTOK) / 1_000_000
        with self.lock:
            projected = self.data["estimated_cost_usd"] + self.reserved_usd + amount
            if projected > BUDGET_CAP_USD:
                raise RuntimeError(f"Exp-7 OpenAI budget stop: projected ${projected:.4f} > ${BUDGET_CAP_USD:.2f}")
            self.reserved_usd += amount
        return amount

    def settle(self, reservation: float, usage: dict | None) -> None:
        usage = usage or {}
        inp, out = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        actual = (inp * INPUT_USD_PER_MTOK + out * OUTPUT_USD_PER_MTOK) / 1_000_000
        with self.lock:
            self.reserved_usd -= reservation
            self.data["input_tokens"] += inp
            self.data["output_tokens"] += out
            self.data["estimated_cost_usd"] = round(self.data["estimated_cost_usd"] + actual, 8)
            self.data["responses"] += 1
            if self.data["estimated_cost_usd"] > BUDGET_CAP_USD:
                raise RuntimeError("Exp-7 OpenAI actual usage exceeded the internal budget cap")
            self._write()

    def release(self, reservation: float) -> None:
        with self.lock:
            self.reserved_usd -= reservation

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.data)


BUDGET_PATH.parent.mkdir(exist_ok=True)
budget = BudgetLedger()
client = httpx.Client(
    base_url="https://api.openai.com/v1",
    headers={"Authorization": f"Bearer {_env_value('OPENAI_API_KEY_2')}",
             "Content-Type": "application/json"},
    timeout=httpx.Timeout(120.0),
)


def valid_partition(partition: list[list[int]], n: int) -> bool:
    flat = [idx for cluster in partition for idx in cluster]
    return sorted(flat) == list(range(n)) and len(flat) == len(set(flat))


def cache_key(claims: list[str]) -> str:
    """Versioned cache key: judge batching/prompt changes must never reuse stale partitions."""
    payload = {"version": CACHE_VERSION, "claims": claims}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _output_text(response: dict) -> str:
    texts = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
            elif content.get("type") == "refusal":
                raise RuntimeError(f"judge refusal: {content.get('refusal', '')}")
    if not texts:
        raise ValueError(f"response contained no output_text (status={response.get('status')})")
    return "".join(texts)


def _ids_partition(ids: list[int], n: int) -> list[list[int]]:
    if len(ids) != n or any(not isinstance(value, int) or value < 0 for value in ids):
        raise ValueError(f"judge returned {len(ids)} cluster ids for {n} claims")
    groups = defaultdict(list)
    for idx, cluster_id in enumerate(ids):
        groups[cluster_id].append(idx)
    return [group for group in groups.values()]


def _cluster_direct(claims: list[str]) -> list[list[int]]:
    if len(claims) <= 1:
        return [[0]] if claims else []
    prompt = ("Assign semantic-equivalence cluster ids to this JSON array of claims. The output "
              "schema fields c0..cN map to claims 0..N in the same order.\n\nCLAIMS:\n" +
              json.dumps([{"index": i, "claim": claim} for i, claim in enumerate(claims)],
                         ensure_ascii=False))
    max_output_tokens = min(8192, max(256, 32 + 16 * len(claims)))
    last_error = None
    for attempt in range(3):
        reservation = 0.0
        try:
            with _stats_lock:
                _stats["direct_requests"] += 1
            reservation = budget.reserve(prompt, max_output_tokens)
            response = client.post("/responses", json={
                "model": MODEL,
                "instructions": SYSTEM,
                "input": prompt,
                "reasoning": {"effort": "none"},
                "text": {"format": _output_format(len(claims)), "verbosity": "low"},
                "max_output_tokens": max_output_tokens,
                "store": False,
            })
            if response.status_code >= 400:
                budget.release(reservation)
                reservation = 0.0
                response.raise_for_status()
            body = response.json()
            settled_reservation = reservation
            reservation = 0.0
            budget.settle(settled_reservation, body.get("usage"))
            parsed = json.loads(_output_text(body))
            ids = [parsed[f"c{i}"] for i in range(len(claims))]
            partition = _ids_partition(ids, len(claims))
            if valid_partition(partition, len(claims)):
                return partition
            raise ValueError("judge returned a non-exhaustive or overlapping partition")
        except Exception as exc:  # noqa: BLE001
            if reservation:
                budget.release(reservation)
            last_error = repr(exc)
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403):
                break
            if "budget stop" in str(exc).lower() or "budget cap" in str(exc).lower():
                raise
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            break
    # Conservative failure mode: singletons cannot fabricate a stable semantic core.
    with _stats_lock:
        _stats["direct_failures"] += 1
        if len(_stats["failure_examples"]) < 20:
            _stats["failure_examples"].append({"n_claims": len(claims), "error": last_error})
    return [[i] for i in range(len(claims))]


def judge_stats() -> dict:
    """Process-local request/failure counters for experiment provenance."""
    with _stats_lock:
        out = dict(_stats)
    out["budget"] = budget.snapshot()
    return out


def cluster_claims(claims: list[str]) -> list[list[int]]:
    """Cluster claims directly; oversized pools use conservative representative merging."""
    if len(claims) <= MAX_DIRECT_CLAIMS:
        return _cluster_direct(claims)
    local_groups: list[list[int]] = []
    for start in range(0, len(claims), MAX_DIRECT_CLAIMS):
        chunk = claims[start:start + MAX_DIRECT_CLAIMS]
        for group in _cluster_direct(chunk):
            local_groups.append([start + idx for idx in group])
    if len(local_groups) > MAX_REP_CLAIMS:
        # Keep the tool output bounded. This fallback preserves within-chunk equivalences but cannot
        # create a false semantic core across chunks.
        return local_groups
    representatives = [claims[group[0]] for group in local_groups]
    rep_partition = (_cluster_direct(representatives) if len(representatives) > MAX_DIRECT_CLAIMS
                     else cluster_claims(representatives))
    merged = [sorted(idx for group_id in rep_group for idx in local_groups[group_id])
              for rep_group in rep_partition]
    return merged if valid_partition(merged, len(claims)) else [[i] for i in range(len(claims))]


# labelled eval cases: (name, claims, gold_partition)
CASES = [
    ("PC same-topic trap (must SPLIT the 3 guesses; merge the 2 vacuum paraphrases)",
     ["a robotic vacuum", "his owner's husband", "the family's cleaning staff",
      "an automated cleaning device", "a Roomba"],
     [[0, 3, 4], [1], [2]]),
    ("genre paraphrases (merge 3, split the formal one)",
     ["Blog-style pet article with playful tone", "a casual, humorous pet blog post",
      "a lighthearted blog about pets", "a formal scientific report"],
     [[0, 1, 2], [3]]),
    ("LCR restatement vs confabulation",
     ["a bulldog lay still while the floor was vacuumed",
      "an English bulldog stayed motionless during vacuuming",
      "the dog was excited and knocked over the vacuum"],
     [[0, 1], [2]]),
    ("distinct entities (rose vs Solanaceae)",
     ["the rose family is a top crop family", "the Solanaceae family is a top crop family",
      "roses are economically important plants", "the rose family is one of the most important crop families"],
     [[0, 2, 3], [1]]),
    ("exact duplicate + one distinct",
     ["expecting a noun phrase", "expecting a noun phrase", "expecting a verb"],
     [[0, 1], [2]]),
    ("all distinct (no merges)",
     ["a preposition like 'of'", "a comma", "the end of the sentence", "a proper noun"],
     [[0], [1], [2], [3]]),
    ("all same (full merge)",
     ["a professional cleaner", "a cleaning professional", "someone who cleans professionally",
      "a professional cleaning person"],
     [[0, 1, 2, 3]]),
    ("negation / opposite (must split)",
     ["the dog was moving around", "the dog remained a fixed object"],
     [[0], [1]]),
    ("large bounded partition (merge paraphrases, split topic-neighbors)",
     ["a robotic vacuum", "an automated cleaning device", "a Roomba", "a robot that vacuums floors",
      "heavy rain is expected tomorrow", "tomorrow will bring intense rainfall",
      "the forecast calls for downpours the next day", "significant precipitation is due tomorrow",
      "a formal scientific report", "a technical paper in an academic register",
      "scholarly research-report style", "a formally written scientific study",
      "his owner's husband", "the family's cleaning staff", "a comma follows", "a proper noun follows",
      "the dog remained still", "the dog moved around", "the rose family is important",
      "the Solanaceae family is important", "the sentence ends", "a verb follows",
      "the result increased", "the result did not increase"],
     [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]] + [[i] for i in range(12, 24)]),
]


def cluster_id(partition, i):
    for k, cl in enumerate(partition):
        if i in cl:
            return k
    return -1


def score_case(claims, gold, pred):
    n = len(claims)
    fm, fs, ok = [], [], 0
    tot = 0
    for i, j in combinations(range(n), 2):
        tot += 1
        g = cluster_id(gold, i) == cluster_id(gold, j)
        p = cluster_id(pred, i) == cluster_id(pred, j)
        if g == p:
            ok += 1
        elif p and not g:
            fm.append((i, j))       # false merge (dangerous)
        else:
            fs.append((i, j))       # false split
    return ok, tot, fm, fs


def main():
    print(f"Exp 7 Step 7.0 — meaning-judge sanity ({MODEL})\n" + "=" * 66)
    T_ok = T_tot = T_fm = T_fs = 0
    for name, claims, gold in CASES:
        pred = cluster_claims(claims)
        ok, tot, fm, fs = score_case(claims, gold, pred)
        T_ok += ok; T_tot += tot; T_fm += len(fm); T_fs += len(fs)
        flag = "OK" if not fm and not fs else ("FALSE-MERGE" if fm else "false-split")
        print(f"\n[{flag}] {name}")
        print(f"  gold {gold}")
        print(f"  pred {[c for c in pred]}")
        if fm:
            print("  FALSE MERGES (dangerous): " + ", ".join(f'\"{claims[i]}\"~\"{claims[j]}\"' for i, j in fm))
        if fs:
            print("  false splits: " + ", ".join(f'\"{claims[i]}\"|\"{claims[j]}\"' for i, j in fs))
    print("\n" + "=" * 66)
    print(f"pairwise agreement {T_ok}/{T_tot} = {100*T_ok/T_tot:.1f}%   "
          f"false-merges={T_fm}  false-splits={T_fs}")
    print("GATE: 0 false-merges + >=90% pairwise agreement -> meaning-judge trustworthy.")


if __name__ == "__main__":
    main()
