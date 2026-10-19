"""Structural validation for completed Experiment-7 artifacts."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from exp7_cluster import TAGS, _outdir
from exp7_judge import valid_partition

EXPECTED = {"main": 200 + 4 * 200 * 24, "floor": 200 * 24, "shuffle": 200 * 24}


def validate(key: str) -> None:
    od = _outdir(key)
    for arm, expected in EXPECTED.items():
        raw = json.loads((od / f"resample_{arm}.json").read_text())
        clustered = json.loads((od / f"exp7_clusters_{arm}.json").read_text())
        assert raw["key"] == clustered["key"] == key
        assert raw["arm"] == clustered["arm"] == arm
        assert raw["n_docs"] == 200 and len(raw["samples"]) == expected

        counts = defaultdict(lambda: [0, 0])
        docs_with_valid = set()
        for sample in raw["samples"].values():
            pair = (sample["idx"], float(sample["T"]))
            bad = sample["cjk"] or sample["noexp"] or sample.get("empty", not sample["text"].strip())
            counts[pair][0] += 1
            counts[pair][1] += int(bad)
            if not bad:
                docs_with_valid.add(sample["idx"])
        assert len(clustered["blocks"]) == len(docs_with_valid) * len(raw["grid"])

        for block in clustered["blocks"].values():
            total, filtered = counts[(block["idx"], float(block["T"]))]
            assert block["n_total"] == total
            assert block["n_filtered"] == filtered
            assert block["effective_N"] == total - filtered
            for tag in TAGS:
                tagblock = block["tags"][tag]
                n = len(tagblock["claims"])
                for kind in ("meaning", "surface"):
                    partition = [cluster["members"] for cluster in tagblock[kind]]
                    assert valid_partition(partition, n), (key, arm, block["idx"], block["T"], tag, kind)
        stats = clustered.get("judge_stats", {})
        print(f"[validate:{key}:{arm}] samples={len(raw['samples'])} blocks={len(clustered['blocks'])} "
              f"direct_requests={stats.get('direct_requests', 0)} "
              f"direct_failures={stats.get('direct_failures', 0)}")

    h5 = json.loads((od / "exp7_h5.json").read_text())
    assert h5["key"] == key and h5["greedy_T"] == 0.0 and h5["core_T"] == 0.4
    for idx, tagrows in h5["rows"].items():
        for tag, row in tagrows.items():
            assert tag in TAGS and int(idx) == row["idx"]
            assert all(0 <= i < row["n_greedy"] for i in row["matched_greedy_indices"])
            assert row["n_matched"] == len(row["matched_greedy_indices"])
    print(f"[validate:{key}:h5] rows={len(h5['rows'])} "
          f"direct_failures={h5.get('judge_stats', {}).get('direct_failures', 0)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=["qwen7b", "gemma12b", "gemma27b"])
    args = ap.parse_args()
    for key in (args.keys or ["qwen7b", "gemma12b", "gemma27b"]):
        validate(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
