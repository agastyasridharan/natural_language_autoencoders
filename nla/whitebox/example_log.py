"""Parser for the `examples/*.txt` per-token NLA decode dumps. STDLIB ONLY.

These dumps are the ground truth the Phase-0 validation gate reproduces. Each
records, for one extraction conversation, the per-token-position round trip:

    [ 71]  REPLY   token=' of'  ||v||=122.6  mse_nrm=0.197  cos=0.902  fve_nrm=0.732
        <multi-paragraph AV explanation text ...>

followed by a summary block (positions 24+) and a sampling-variance section.

This module has NO heavy deps (no torch / transformers) so the gate's parser
and its unit test run anywhere. The numbers parsed here are what the white-box
harness must match.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path

# Header line, e.g.:
#   "[  0]  PROMPT  token='<|im_start|>'  ||v||=235.7  mse_nrm=1.962  cos=0.019  fve_nrm=-1.674"
# The token is Python repr()'d: single-quoted normally, but DOUBLE-quoted when
# it contains a single quote (e.g. "'t" from "don't", position 45). Match either
# quote style via a backreference, non-greedily up to the closing quote that
# precedes "  ||v||=".
_HEADER_RE = re.compile(
    r"^\[\s*(?P<idx>\d+)\]\s+"
    r"(?P<region>PROMPT|REPLY)\s+"
    r"token=(?P<q>['\"])(?P<token>.*?)(?P=q)\s+"
    r"\|\|v\|\|=(?P<raw_norm>[-\d.]+)\s+"
    r"mse_nrm=(?P<mse>[-\d.]+)\s+"
    r"cos=(?P<cos>[-\d.]+)\s+"
    r"fve_nrm=(?P<fve>[-\d.]+)\s*$"
)

# A run of box-drawing chars marks a section boundary (ends an explanation body).
_DIVIDER_RE = re.compile(r"^[─═]{3,}\s*$")

_EXPLANATION_TAGS_RE = re.compile(r"</?explanation>")

# "  → MSE=0.143  cos=0.928"  (sampling-variance section)
_SAMPLE_SCORE_RE = re.compile(r"→\s*MSE=(?P<mse>[-\d.]+)\s+cos=(?P<cos>[-\d.]+)")


@dataclass(frozen=True)
class PositionRecord:
    idx: int
    region: str          # "PROMPT" | "REPLY"
    token: str           # the token string at this position
    raw_norm: float      # ||v|| — RAW (un-normalized) L2 norm of the activation
    mse_nrm: float       # AR reconstruction MSE of L2-normalized vectors = 2(1-cos)
    cos: float
    fve_nrm: float       # 1 - mse_nrm / baseline
    explanation: str     # the AV decode text (verbatim, including any stray tags)

    @property
    def explanation_clean(self) -> str:
        """Explanation with stray <explanation> tags stripped — what you feed
        the AR. Most positions are already tag-free; the last position in the
        Qwen dump leaks a literal '<explanation>' from a truncated decode."""
        return _EXPLANATION_TAGS_RE.sub("", self.explanation).strip()


@dataclass(frozen=True)
class Summary:
    """The dump's own summary block (positions >= min_position)."""

    min_position: int
    n_positions: int
    mse_mean: float
    mse_median: float
    mse_min: float
    mse_max: float
    cos_mean: float
    fve_mean: float
    fve_median: float
    fve_baseline: float
    n_fve_gt_0_5: int
    n_fve_gt_0_8: int
    n_fve_lt_0: int


@dataclass(frozen=True)
class ParsedLog:
    positions: list[PositionRecord]
    summary: Summary | None

    def in_region(self, *regions: str) -> list[PositionRecord]:
        return [p for p in self.positions if p.region in regions]

    def from_position(self, min_position: int) -> list[PositionRecord]:
        return [p for p in self.positions if p.idx >= min_position]


def fve_nrm_from_mse(mse_nrm: float, baseline: float) -> float:
    """The dump's FVE definition: 1 - mse_nrm / Var(v_nrm).

    baseline is the predict-the-mean MSE on the normalized training
    distribution (0.7335 for Qwen-2.5-7B L20). FVE=0 -> predict-mean; FVE=1 ->
    perfect; FVE<0 -> worse than predict-mean.
    """
    return 1.0 - mse_nrm / baseline


def parse_log(path: str | Path) -> ParsedLog:
    """Parse a per-token decode dump into structured records + its summary."""
    text = Path(path).read_text()
    lines = text.splitlines()

    positions: list[PositionRecord] = []
    i = 0
    n = len(lines)
    while i < n:
        m = _HEADER_RE.match(lines[i])
        if m is None:
            i += 1
            continue
        # Accumulate the explanation body until the next header or a divider.
        body: list[str] = []
        j = i + 1
        while j < n and _HEADER_RE.match(lines[j]) is None and not _DIVIDER_RE.match(lines[j]):
            body.append(lines[j])
            j += 1
        positions.append(
            PositionRecord(
                idx=int(m["idx"]),
                region=m["region"],
                token=m["token"],
                raw_norm=float(m["raw_norm"]),
                mse_nrm=float(m["mse"]),
                cos=float(m["cos"]),
                fve_nrm=float(m["fve"]),
                explanation="\n".join(body).strip(),
            )
        )
        i = j

    return ParsedLog(positions=positions, summary=_parse_summary(lines))


def _parse_summary(lines: list[str]) -> Summary | None:
    """Parse the '─── Summary (positions N+...) ───' block, if present."""
    start = None
    min_pos = 24
    for k, ln in enumerate(lines):
        mm = re.search(r"Summary\s*\(positions\s*(\d+)\+", ln)
        if mm:
            start = k
            min_pos = int(mm.group(1))
            break
    if start is None:
        return None

    block = "\n".join(lines[start : start + 12])

    def _grab(pattern: str, default: float | None = None) -> float | None:
        mm = re.search(pattern, block)
        return float(mm.group(1)) if mm else default

    mse_mean = _grab(r"mse_nrm:\s*mean=([-\d.]+)")
    mse_median = _grab(r"mse_nrm:.*?median=([-\d.]+)")
    mse_min = _grab(r"mse_nrm:.*?min=([-\d.]+)")
    mse_max = _grab(r"mse_nrm:.*?max=([-\d.]+)")
    cos_mean = _grab(r"cos:\s*mean=([-\d.]+)")
    fve_mean = _grab(r"fve_nrm:\s*mean=([-\d.]+)")
    fve_median = _grab(r"fve_nrm:.*?median=([-\d.]+)")
    fve_baseline = _grab(r"denom=([-\d.]+)") or 0.0

    def _count(pattern: str) -> int:
        mm = re.search(pattern, block)
        return int(mm.group(1)) if mm else 0

    n_gt5 = _count(r"fve_nrm>0\.5:\s*(\d+)")
    n_gt8 = _count(r"fve_nrm>0\.8:\s*(\d+)")
    n_lt0 = _count(r"fve_nrm<0:\s*(\d+)")
    n_total = _count(r"fve_nrm>0\.5:\s*\d+/(\d+)")

    if mse_mean is None or fve_mean is None:
        return None
    return Summary(
        min_position=min_pos,
        n_positions=n_total,
        mse_mean=mse_mean,
        mse_median=mse_median if mse_median is not None else float("nan"),
        mse_min=mse_min if mse_min is not None else float("nan"),
        mse_max=mse_max if mse_max is not None else float("nan"),
        cos_mean=cos_mean if cos_mean is not None else float("nan"),
        fve_mean=fve_mean,
        fve_median=fve_median if fve_median is not None else float("nan"),
        fve_baseline=fve_baseline,
        n_fve_gt_0_5=n_gt5,
        n_fve_gt_0_8=n_gt8,
        n_fve_lt_0=n_lt0,
    )


@dataclass(frozen=True)
class SummaryStats:
    """Summary statistics recomputed from a list of records — lets the gate
    aggregate fresh measurements the same way the dump's summary block did."""

    n: int
    mse_mean: float
    mse_median: float
    mse_min: float
    mse_max: float
    cos_mean: float
    fve_mean: float
    fve_median: float
    n_fve_gt_0_5: int
    n_fve_gt_0_8: int
    n_fve_lt_0: int


def summarize(records: list[PositionRecord]) -> SummaryStats:
    """Recompute the dump's summary statistics from records (for parser
    self-check and for aggregating fresh gate measurements)."""
    mse = [r.mse_nrm for r in records]
    cos = [r.cos for r in records]
    fve = [r.fve_nrm for r in records]
    return SummaryStats(
        n=len(records),
        mse_mean=statistics.fmean(mse),
        mse_median=statistics.median(mse),
        mse_min=min(mse),
        mse_max=max(mse),
        cos_mean=statistics.fmean(cos),
        fve_mean=statistics.fmean(fve),
        fve_median=statistics.median(fve),
        n_fve_gt_0_5=sum(1 for x in fve if x > 0.5),
        n_fve_gt_0_8=sum(1 for x in fve if x > 0.8),
        n_fve_lt_0=sum(1 for x in fve if x < 0.0),
    )


@dataclass(frozen=True)
class Conversation:
    """The extraction conversation a dump was produced from."""

    user_message: str
    reply_text: str
    n_prompt_tokens: int | None
    n_reply_tokens: int | None


def parse_conversation(path: str | Path) -> Conversation | None:
    """Pull the user message + verbatim greedy reply out of the dump's §1/§2.

    The gate uses these to rebuild the exact extraction conversation (so the
    re-extracted activations line up position-for-position with the logged
    metrics).
    """
    text = Path(path).read_text()
    um = re.search(r"User message:\s*'(.*?)'", text)
    if um is None:
        return None
    reply = ""
    rh = text.find("BASE MODEL REPLY")
    if rh >= 0:
        rm = re.search(r'"([^"]+)"', text[rh:])
        if rm:
            reply = rm.group(1)
    npt = re.search(r"Tokenized:\s*(\d+)\s*tokens", text)
    nrt = re.search(r"Reply tokens:\s*(\d+)", text)
    return Conversation(
        user_message=um.group(1),
        reply_text=reply,
        n_prompt_tokens=int(npt.group(1)) if npt else None,
        n_reply_tokens=int(nrt.group(1)) if nrt else None,
    )


@dataclass(frozen=True)
class SampleScore:
    mse: float
    cos: float


def parse_sampling_variance(path: str | Path) -> list[SampleScore]:
    """Parse the '4. SAMPLING VARIANCE' section's '→ MSE=.. cos=..' lines.

    Used as a soft upper bound on AV-decode tightness (greedy + temp=1 samples
    of the same activation). Optional — returns [] if the section is absent.
    """
    text = Path(path).read_text()
    idx = text.find("SAMPLING VARIANCE")
    if idx < 0:
        return []
    return [
        SampleScore(mse=float(m["mse"]), cos=float(m["cos"]))
        for m in _SAMPLE_SCORE_RE.finditer(text[idx:])
    ]
