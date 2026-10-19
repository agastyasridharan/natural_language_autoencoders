"""Build the held-out Ultra-FineWeb doc set for the FVE baseline.

Held-out definition. The released NLAs' datagen recipe
(configs/datagen/qwen7b_ultrafineweb_100k.yaml) draws from
openbmb/Ultra-FineWeb, split 'en', text column 'content', docs [0:100_000] — a
FRONT slice. The exact released-checkpoint training subset was a ~1B-token slice
that is no longer hosted, so a provably-disjoint holdout is impossible. We
approximate it by sampling from a LATE 'en' shard (well past the front the recipe
draws from) with a fixed seed, dedup by content hash, and a minimum length so a
read position >= 50 always exists. Residual overlap with the delisted training
subset is negligible.

Writes:
  data/ufw_docs.json  = [{"id": <sha1[:16]>, "text": <content>}, ...]  (N docs)
  data/ufw_meta.json  = provenance (dataset, split, shard, seed, filters)

Run on the cluster (needs the RUB proxy + HF). CPU-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

REPO = "openbmb/Ultra-FineWeb"
OUT = Path("/data/agastyas/nla_fve_ufw/data")
N = 2000
SEED = 42
MIN_CHARS = 600          # ~>=100 tokens for every tokenizer => a pos >= 50 exists
LATE_FRAC = 0.75         # pick an 'en' shard ~75% through the shard list


def _pick_en_shard() -> str:
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    parquet = [f for f in files if f.endswith(".parquet")]
    en = sorted(f for f in parquet if "en" in Path(f).parts or "/en/" in f
                or Path(f).name.startswith("en") or "_en" in f or "-en" in f)
    if not en:                                   # fall back: any parquet with 'en'
        en = sorted(f for f in parquet if "en" in f.lower())
    assert en, f"no 'en' parquet shards among {len(parquet)} files; sample: {parquet[:5]}"
    print(f"[build] {len(parquet)} parquet files, {len(en)} 'en' shards")
    print(f"[build] en first={en[:2]} last={en[-2:]}")
    pick = en[int(len(en) * LATE_FRAC)]
    print(f"[build] held-out shard (late, ~{int(LATE_FRAC*100)}%): {pick}")
    return pick


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shard = _pick_en_shard()
    # Stream the shard: read only enough row-groups to collect N valid docs
    # (a UFW shard can be multi-GB; we never need the whole thing). The shard is
    # itself an arbitrary slice of the corpus, so the first N valid docs are a
    # fair held-out sample. Fixed shard + fixed order => reproducible.
    ds = load_dataset("parquet", data_files=f"hf://datasets/{REPO}/{shard}",
                      split="train", streaming=True)
    text_col = None
    docs, seen, scanned = [], set(), 0
    for rec in ds:
        scanned += 1
        if text_col is None:
            text_col = "content" if "content" in rec else "text"
            print(f"[build] streaming; cols={list(rec)} text_col={text_col!r}")
        text = rec.get(text_col)
        if not text or len(text) < MIN_CHARS:
            continue
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        docs.append({"id": h[:16], "text": text})
        if len(docs) >= N:
            break
    assert len(docs) >= N, f"only {len(docs)} valid docs after scanning {scanned}"
    print(f"[build] collected {len(docs)} from {scanned} scanned rows")

    (OUT / "ufw_docs.json").write_text(json.dumps(docs, ensure_ascii=False),
                                       encoding="utf-8")
    (OUT / "ufw_meta.json").write_text(json.dumps(dict(
        dataset=REPO, split="en", text_column=text_col, shard=shard,
        n=len(docs), min_chars=MIN_CHARS, late_frac=LATE_FRAC,
        procedure="first N valid docs (>=min_chars, deduped) streamed from a late en shard",
        held_out_note=("late en shard, disjoint from the training recipe's front "
                       "slice [0:100k]; exact released-checkpoint subset is "
                       "delisted so residual overlap is negligible, not zero"),
    ), indent=2), encoding="utf-8")
    lens = [len(d["text"]) for d in docs]
    print(f"[build] wrote {len(docs)} docs -> {OUT/'ufw_docs.json'} "
          f"(chars min={min(lens)} med={sorted(lens)[len(lens)//2]} max={max(lens)})")


if __name__ == "__main__":
    main()
