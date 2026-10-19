"""Build the activation-space audit (Δcos) data file for the corpus viewer.

Joins athena out/results.json (cos_real / cos_mimic / cos_control / delta per
position, plus the validation + alignment gates) with run_input.json (the real
AV read and the text-only mimic for each position), computes every aggregate the
viewer shows (headline cos+MSE, by-genre, by-cosine-band, Δ distribution, top
examples), and emits a per-model JS file that PUSHES one entry onto
window.ACT_AUDIT.models. Each model gets its own act_audit_<key>.js, so the
replication runs (qwen7b, gemma12b, gemma27b) drop in by adding one script tag.

Usage:
  python build_act_audit_view.py [model_key] [results.json] [run_input.json]

Defaults to the llama70b run already on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"

# model key -> (label, meta) ; extend as replications land
REGISTRY = {
    "llama70b": dict(label="Llama-3.3-70B", base="meta-llama/Llama-3.3-70B-Instruct",
                     ar="kitft/Llama-3.3-70B-NLA-L53-ar", layer=53, d_model=8192,
                     mse_scale=90.51, date="2026-06-17"),
    "qwen7b": dict(label="Qwen2.5-7B", base="Qwen/Qwen2.5-7B-Instruct",
                   ar="kitft/nla-qwen2.5-7b-L20-ar", layer=20, d_model=3584,
                   mse_scale=59.87, date=""),
    "gemma12b": dict(label="Gemma-3-12B", base="google/gemma-3-12b-it",
                     ar="kitft/nla-gemma3-12b-L32-ar", layer=32, d_model=3840,
                     mse_scale=61.97, date=""),
    "gemma27b": dict(label="Gemma-3-27B", base="google/gemma-3-27b-it",
                     ar="kitft/nla-gemma3-27b-L41-ar", layer=41, d_model=5376,
                     mse_scale=73.32, date=""),
}

GOAL = ("This view addresses a single question. When the model reads one of its own "
        "internal vectors and writes a description of it, how much of that "
        "description originates in the vector itself, and how much could be inferred "
        "from the surrounding text alone? At each position the original vector is "
        "reconstructed from three different descriptions, and the accuracy of each "
        "reconstruction is measured. The three descriptions are the genuine one "
        "written from the vector, a text-only one written without the vector, and a "
        "mismatched one taken from another position. If the genuine description "
        "reconstructs the vector substantially more accurately than the text-only one, "
        "the vector carried information that the text alone did not provide.")

# Per-model reference numbers: the shape of the internal vectors, the one or two
# oversized dimensions, the bootstrap and significance results, and the comparison
# against the publicly hosted copies of each model. Sources: RESULTS_cross_model.md
# (vector shape and oversized dimensions), VETTING_REPORT.md (the 20,000-sample
# bootstrap over the 128 positions, paired Wilcoxon tests, and the live Neuronpedia
# checks), and each model's sidecar (the insertion scale). These are fixed numbers
# from the finished four-model run. The per-position, per-genre and per-band numbers
# are computed from the data files below, not stored here.
DOSSIER = {
    "llama70b": dict(
        injection_scale=30, reads_source="obtained from the publicly hosted Llama model on Neuronpedia",
        depth="53 / 80",
        geom=dict(norm="~90", hbar2=0.114, D=0.886, pw_raw=0.107, pw_cen=-0.007, pw_std=-0.007),
        rogue=dict(desc="dimension 5114 (44% of the average magnitude)",
                   topval="-6", sigma="5", share=0.44),
        boot=dict(dfve_ci=[0.10, 0.21], mse_ci=[1.30, 1.60], wilcoxon="3e-10"),
        np=dict(hosted=True, kind="self-consistency",
                cos_np=None, cos_ours=None, norm_field=0.9169, our_D=0.886,
                note=("The Llama descriptions used here were obtained from the hosted "
                      "service on Neuronpedia. Consequently, the very close agreement "
                      "between the present scores and the hosted scores, a correlation of "
                      "0.9995, confirms primarily that the scoring procedure matches the "
                      "hosted one; it does not establish an independent regeneration of "
                      "the descriptions. When the descriptions were instead regenerated "
                      "live and scored, the agreement was lower, a correlation of 0.98, "
                      "with the additional discrepancy attributable to an off-by-one "
                      "position issue on some chat-formatted documents. Separately, the "
                      "dispersion figure Neuronpedia reports for this model, 0.9169, is "
                      "close to the value computed independently here, 0.886.")),
    ),
    "qwen7b": dict(
        injection_scale=150, reads_source="generated here by inserting the vector at the first layer",
        depth="20 / 28",
        geom=dict(norm="~227", hbar2=0.275, D=0.725, pw_raw=0.268, pw_cen=0.526, pw_std=-0.007),
        rogue=dict(desc="dimensions 458 and 2570 (91% combined)",
                   topval="-112", sigma="961", share=0.91),
        boot=dict(dfve_ci=[0.37, 0.48], mse_ci=[2.08, 2.49], wilcoxon="1e-22"),
        np=dict(hosted=False, kind="unhosted",
                note=("Qwen-7B is not hosted publicly, so its generated descriptions "
                      "cannot be compared against an external reference. Confidence in "
                      "them rests instead on indirect indicators: the insertion check was "
                      "clean, the positions that were harder for Llama were also harder "
                      "here, and the overall reconstruction quality fell within the 0.6 "
                      "to 0.8 range reported in the paper. It should be noted that Qwen "
                      "exhibits the largest effect of the four models yet has the least "
                      "external confirmation. One further property is worth stating. "
                      "After standardization, Qwen's vectors continue to vary considerably "
                      "in overall length. Cosine depends only on direction and not on "
                      "length, and by that measure Qwen's direction is recovered the most "
                      "accurately of the four models. The coefficient of determination, R "
                      "squared, penalizes errors in length, which is why Qwen's R squared "
                      "is comparatively low at 0.18 despite the strong agreement in "
                      "direction.")),
    ),
    "gemma12b": dict(
        injection_scale=80000, reads_source="generated here by inserting the vector at the first layer",
        depth="32 / 48",
        geom=dict(norm="~62k", hbar2=0.954, D=0.046, pw_raw=0.953, pw_cen=-0.007, pw_std=-0.008),
        rogue=dict(desc="dimension 2339 alone (99.7%)",
                   topval="69,528", sigma="13,389", share=0.997),
        boot=dict(dfve_ci=[0.21, 0.40], mse_ci=[1.73, 2.04], wilcoxon="1e-18"),
        np=dict(hosted=False, kind="unhosted",
                note=("Gemma-3-12B is not hosted publicly and therefore relies on the "
                      "same indirect indicators as Qwen: a clean insertion check, "
                      "agreement between the positions that are difficult for Llama and "
                      "those that are difficult here, and reconstruction quality within "
                      "the range reported in the paper. Its vectors have the most extreme "
                      "distribution of the four models. A single dimension accounts for "
                      "99.7 percent of the average vector magnitude, which is precisely "
                      "why an uncorrected cosine is nearly identical for every description "
                      "here.")),
    ),
    "gemma27b": dict(
        injection_scale=60000, reads_source="generated here by inserting the vector at the first layer",
        depth="41 / 62",
        geom=dict(norm="~56k", hbar2=0.926, D=0.074, pw_raw=0.925, pw_cen=-0.008, pw_std=-0.008),
        rogue=dict(desc="dimensions 104 and 2733 (99.7% combined)",
                   topval="43,394 / 33,036", sigma="", share=0.997),
        boot=dict(dfve_ci=[0.25, 0.41], mse_ci=[1.92, 2.28], wilcoxon="3e-22"),
        np=dict(hosted=True, kind="external",
                cos_np=0.991, cos_ours=0.992, l2_norm="40,000 to 72,000", norm_field=0.0579, our_D=0.0745,
                np_dist=dict(n=286, genres=24, median=0.991, mean=0.977, frac95=0.955),
                note=("Gemma-3-27B is hosted publicly on Neuronpedia (model "
                      "gemma-3-27b-it, source kitft-l41), which provides the strongest "
                      "external check of the four. The present cosine of 0.992 matches the "
                      "hosted value of 0.991, and the two agree position by position to "
                      "within 0.003. The vector lengths Neuronpedia reports, between "
                      "40,000 and 72,000, match the large lengths measured here, which is "
                      "the signature of the few dominant dimensions. The description text "
                      "from Neuronpedia conveys the same content as the description "
                      "generated here, with several nearly identical, which confirms that "
                      "the insertion procedure operates correctly from end to end. Across "
                      "286 described positions in 24 categories, Neuronpedia returns a "
                      "median cosine of 0.991, a mean of 0.977, and values above 0.95 for "
                      "95.5 percent of positions, the same pattern of near-identical "
                      "cosines observed here. Finally, the dispersion figure Neuronpedia "
                      "reports, 0.0579, is close to the value computed independently here, "
                      "0.0745.")),
    ),
}

# Shared explanation, provenance and honest limits (the same for all four models).
# Taken from VETTING_REPORT.md (the independent review of 18 June 2026, which found the
# result sound).
SUITE = dict(
    abstract=("This study tests whether the internal vector a model reads carries "
              "information of its own, or whether the same content could be inferred from "
              "the surrounding text. Across all four models examined, the vector carries "
              "information that the text alone does not provide. The measure used is the "
              "standardized cosine, which first removes the one or two coordinates that "
              "take very large and nearly constant values across positions and would "
              "otherwise dominate the comparison. After this correction, a deliberately "
              "mismatched description scores approximately 0.045, no better than chance, "
              "whereas the genuine description scores substantially higher and exceeds a "
              "strong text-only description. An independent review that recomputed the "
              "statistics, using a 20,000-sample bootstrap, paired significance tests, and "
              "direct comparison against the publicly hosted copies of these models, found "
              "the result sound and identified no calculation error."),
    bootstrap=("To confirm that the result does not depend on the particular 128 "
               "positions sampled, the 128 positions were resampled with replacement "
               "20,000 times and the statistics recomputed for each resample. In every "
               "model, the resulting confidence interval for the vector's contribution lay "
               "entirely above zero, and the interval for the ratio of reconstruction "
               "errors lay entirely above one, indicating that the text-only description "
               "was consistently the less accurate of the two. Removing any single "
               "position changes the result by at most 0.07, and the five positions with "
               "the smallest variance together account for less than 3.3 percent of the "
               "total weight. The conclusion therefore does not rest on a small number of "
               "unusual positions."),
    mse_identity=("Two quantities are commonly reported for reconstruction of this kind: "
                  "cosine similarity and mean squared error. Here they are not "
                  "independent. Before scoring, both vectors are rescaled to a common "
                  "length, namely the square root of the number of dimensions. For vectors "
                  "of equal length, the mean squared error equals exactly two times one "
                  "minus the cosine similarity. This identity was verified analytically, to "
                  "fifteen decimal places, and against the hosted models, where the mean "
                  "discrepancy was 0.0000007 across 44 positions. Because the two "
                  "quantities convey the same information, cosine is reported as the "
                  "primary measure and mean squared error is included only as a familiar "
                  "reference."),
    mimic=dict(scored="v2_3", disc=0.297, alt="v1_1", alt_disc=0.472,
               note=("The text-only comparison was produced by a separate model (Claude "
                     "Opus 4.8) that was given the same passage but not the internal vector "
                     "and was asked to write a description in the same style. The version "
                     "used here is designated v2_3. The persuasiveness of these "
                     "descriptions was assessed by asking a separate classifier to "
                     "distinguish them from genuine descriptions; the classifier was "
                     "correct only 29.7 percent of the time, below the chance rate of 50 "
                     "percent. The text-only descriptions therefore resemble genuine "
                     "descriptions at least as closely as the genuine descriptions "
                     "themselves, which makes them a demanding comparison. That the "
                     "genuine description nonetheless prevails is consequently a "
                     "conservative test rather than a lenient one. The text-only "
                     "descriptions also reconstruct the vector well on their own terms: "
                     "they exceed the mismatched control at 125 to 127 of the 128 "
                     "positions and recover between 69 and 90 percent of the difference "
                     "between the genuine description and the control. The comparison is "
                     "therefore both strong and fair. A separate version, v1_1, was tuned "
                     "so that the classifier was correct almost exactly half the time; it "
                     "was not the version scored here. An earlier write-up referred to "
                     "v1_1 in error, as the review noted.")),
    smell=("When the vector is inserted incorrectly, the model that writes the description "
           "fails in a characteristic way: it begins to emit Chinese characters at random. "
           "This provides a simple diagnostic for a faulty setup. Across all 384 "
           "descriptions generated, it occurred exactly once. A substantial scaling error "
           "would have produced pervasive corruption rather than a single instance."),
    injection=("For Qwen and Gemma, ready-made descriptions were not available, so the "
               "genuine descriptions were generated directly by inserting the internal "
               "vector into the model. The insertion replaces the hidden state of a single "
               "marker position, at the first layer, with the vector rescaled to a fixed "
               "length specified by the model. The official serving code performs the same "
               "operation. One concern was specific to Gemma, which multiplies its token "
               "embeddings by a large factor, the square root of the hidden size, and "
               "could in principle rescale the inserted vector by approximately a factor "
               "of 62. Simulating both the procedure used here and the official procedure "
               "yielded an identical length for the inserted vector, so this error does "
               "not occur; the large factor applies only to the remaining positions, not "
               "to the overwritten marker. The layer read from was also confirmed to match "
               "the layer the models were trained to use."),
    ceilings=[
        "The original internal vectors reside only on the compute cluster, so checks "
        "performed off-cluster rely on the per-position scores saved earlier. An error "
        "affecting the genuine and the text-only descriptions equally would cancel and "
        "would not alter the comparison. An error that misaligned vectors with positions "
        "would alter it, and such an error cannot be excluded by the off-cluster checks "
        "alone.",
        "The reconstruction scoring was verified analytically and against the hosted "
        "models, but it was not re-executed on a GPU for the descriptions generated here.",
        "Qwen-7B and Gemma-12B are not hosted publicly, so their generated descriptions "
        "have no external reference for comparison. This is notable because Qwen exhibits "
        "the largest effect of the four models yet has the least external confirmation.",
        "The comparison against the hosted Llama model is partly circular, because the "
        "Llama descriptions were obtained from that same service. The very high agreement "
        "of 0.9995 therefore demonstrates primarily that the present scoring matches the "
        "hosted scoring. An independent regeneration of the descriptions agrees less "
        "closely, at 0.98.",
    ],
    citations=[
        "Timkey and van Schijndel, EMNLP 2021, “All Bark and No Bite: Rogue Dimensions in "
        "Transformer Language Models Obscure Representational Quality.” This paper shows "
        "that a small number of dimensions can dominate cosine similarity and render it "
        "misleading, and it recommends correcting this by standardizing each dimension.",
        "Sun et al. 2024 (arXiv 2402.17762), “Massive Activations in Large Language "
        "Models.” This paper documents that a few activations sit roughly 1,000 to 10,000 "
        "times above the median and act as nearly constant offsets in fixed dimensions.",
        "Gemma multiplies its token embeddings by the square root of the hidden size, "
        "which inflates the magnitude of the internal vectors and makes the effect of the "
        "dominant coordinates especially strong. This is confirmed in the HuggingFace "
        "Gemma-3 model code.",
    ],
    review=("An independent review on 18 June 2026 re-derived the analysis from first "
            "principles, recomputed every headline value from the saved per-position data, "
            "reproduced the numbers against the hosted models, and confirmed the "
            "supporting literature. It concluded that the result is sound. The principal "
            "finding holds in all four models. The unusual cosine values for Gemma are "
            "correct rather than erroneous; they are simply uninformative in isolation. No "
            "calculation error was found. The full report is in athena/VETTING_REPORT.md."),
)

BANDS = [("under 0.60", -1, 0.60), ("0.60 to 0.80", 0.60, 0.80), ("0.80 to 0.90", 0.80, 0.90),
         ("0.90 to 0.95", 0.90, 0.95), ("0.95 and up", 0.95, 2.0)]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def mse(cos):
    return 2.0 * (1.0 - cos)


def agg(rs):
    """Aggregate a list of rows into the block the viewer renders."""
    n = len(rs)
    cr, cm, cc = mean(r["cos_real"] for r in rs), mean(r["cos_mimic"] for r in rs), mean(r["cos_control"] for r in rs)
    dl = mean(r["delta"] for r in rs)
    win = sum(1 for r in rs if r["delta"] > 0)
    return dict(n=n, real=round(cr, 4), mimic=round(cm, 4), control=round(cc, 4),
                mse_real=round(mse(cr), 4), mse_mimic=round(mse(cm), 4), mse_control=round(mse(cc), 4),
                delta=round(dl, 4), win=win)


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "llama70b"
    default_out = "out" if key == "llama70b" else f"out_{key}"
    res_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "athena" / default_out / "results.json"
    inp_path = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "athena" / "run_input.json"
    assert key in REGISTRY, f"unknown model key {key!r}; add it to REGISTRY"
    reg = REGISTRY[key]

    # FVE (Fraction of Variance Explained) — the paper's mean-centered metric;
    # the HONEST fidelity once the shared-mean component is removed. Raw cosine is
    # saturated for high-anisotropy models (gemma: ~95% of unit energy is the mean).
    fve_path = HERE / "athena" / "fve" / f"fve_{key}.json"
    fve = json.loads(fve_path.read_text()) if fve_path.exists() else None
    fve_by_idx = {r["idx"]: r for r in (fve["rows"] if fve else [])}

    # Standardized (de-biased) scoring — the ROBUST primary metric: per-dim z-score
    # removes the rogue/massive-activation dims (Timkey & van Schijndel 2021), so a
    # mismatched control collapses to ~0 and cosine measures content, not 1-2 dims.
    std_path = HERE / "athena" / "fve" / f"std_full_{key}.json"
    std = json.loads(std_path.read_text()) if std_path.exists() else None
    std_by_idx = {r["idx"]: r for r in (std["rows"] if std else [])}

    res = json.loads(res_path.read_text())
    rows = res["rows"]
    val = res.get("validation")           # null for the locally-generated (non-llama) runs
    summ = res["summary"]

    texts = {it["idx"]: it for it in json.loads(inp_path.read_text(encoding="utf-8"))}
    n_total = len(texts)

    # conversation context per position (ends at the read token) — same source the
    # calibration tabs use; shared across models since docs+location are held fixed.
    prefixes = {}
    samp = CORPUS / "calib_sample_v2_3.json"
    if samp.exists():
        for tk in json.loads(samp.read_text())["tasks"]:
            prefixes[int(tk["idx"])] = tk.get("prefix", "")

    # per-row enrichment: prefer text embedded in results (generated AV read for the
    # replication models); fall back to run_input (llama reused the corpus reads).
    out_rows = []
    for r in rows:
        t = texts.get(r["idx"], {})
        fr = fve_by_idx.get(r["idx"], {})
        sr = std_by_idx.get(r["idx"], {})
        out_rows.append(dict(
            idx=r["idx"], theme=r["theme"], position=r["position"],
            cosine_stored=r["cosine_stored"],
            cos_real=r["cos_real"], cos_mimic=r["cos_mimic"],
            cos_control=r["cos_control"], delta=r["delta"],
            std_real=sr.get("std_real"), std_mimic=sr.get("std_mimic"),
            std_control=sr.get("std_control"),
            std_delta=(round(sr["std_real"] - sr["std_mimic"], 4)
                       if sr.get("std_real") is not None else None),
            fve_real=fr.get("fve_real"), fve_mimic=fr.get("fve_mimic"),
            fve_control=fr.get("fve_control"),
            fve_delta=(round(fr["fve_real"] - fr["fve_mimic"], 4)
                       if fr.get("fve_real") is not None else None),
            token=r.get("token", t.get("token", "")), is_chat=bool(t.get("is_chat", False)),
            llama_token=t.get("token", ""),     # matches the shared (llama) prefix for highlighting
            real_text=r.get("real_text", t.get("real_text", "")),
            mimic_text=r.get("mimic_text", t.get("mimic_text", "")),
            prefix=prefixes.get(r["idx"], ""),
        ))

    # alignment gate: llama filtered to in_range & token_match upstream (count == rows);
    # replication runs report anchor success in results["alignment"].
    alignment = res.get("alignment") or dict(match=len(rows), n=n_total)

    # by cosine band (on the stored cosine = difficulty of the position)
    bands = []
    for name, lo, hi in BANDS:
        sub = [r for r in rows if lo <= r["cosine_stored"] < hi]
        if sub:
            b = agg(sub); b["name"] = name; bands.append(b)

    # by genre, sorted by Δ descending (activation matters most at the top)
    genres = []
    for th in sorted({r["theme"] for r in rows}):
        g = agg([r for r in rows if r["theme"] == th]); g["theme"] = th; genres.append(g)
    genres.sort(key=lambda g: -g["delta"])

    # join FVE aggregates onto bands (same thresholds/order) and genres (by theme)
    if fve:
        fg = {x["theme"]: x for x in fve["genres"]}
        for g in genres:
            x = fg.get(g["theme"])
            if x:
                g["fve_real"], g["fve_mimic"], g["fve_dfve"] = x["fve_real"], x["fve_mimic"], x["dfve"]
        if len(fve["bands"]) == len(bands):
            for b, x in zip(bands, fve["bands"]):
                b["fve_real"], b["fve_mimic"], b["fve_dfve"] = x["fve_real"], x["fve_mimic"], x["dfve"]
    # join standardized (de-biased) aggregates the same way
    if std:
        sg = {x["theme"]: x for x in std["genres"]}
        for g in genres:
            x = sg.get(g["theme"])
            if x:
                g["std_real"], g["std_mimic"], g["std_delta"] = x["std_real"], x["std_mimic"], x["std_delta"]
        if len(std["bands"]) == len(bands):
            for b, x in zip(bands, std["bands"]):
                b["std_real"], b["std_mimic"], b["std_delta"] = x["std_real"], x["std_mimic"], x["std_delta"]

    # Δ distribution
    ds = sorted(r["delta"] for r in rows)
    nrow = len(ds)
    dist = dict(
        min=round(ds[0], 4), p10=round(ds[max(0, nrow // 10)], 4),
        median=round(ds[nrow // 2], 4), p90=round(ds[min(nrow - 1, 9 * nrow // 10)], 4),
        max=round(ds[-1], 4),
        neg=sum(1 for x in ds if x < 0), big=sum(1 for x in ds if abs(x) > 0.05),
    )

    entry = dict(
        key=key, label=reg["label"],
        meta=dict(base=reg["base"], ar=reg["ar"], layer=reg["layer"],
                  d_model=reg["d_model"], mse_scale=reg["mse_scale"], date=reg["date"],
                  goal=GOAL),
        gates=dict(alignment=alignment, validation=val,
                   injection=res.get("injection"), difficulty_corr=res.get("difficulty_corr")),
        summary=dict(
            real=summ["mean_cos_real"], mimic=summ["mean_cos_mimic"], control=summ["mean_cos_control"],
            mse_real=round(mse(summ["mean_cos_real"]), 4), mse_mimic=round(mse(summ["mean_cos_mimic"]), 4),
            mse_control=round(mse(summ["mean_cos_control"]), 4),
            mean_delta=summ["mean_delta"], median_delta=summ["median_delta"],
            real_beats_mimic=summ["real_beats_mimic"], real_beats_control=summ["real_beats_control"],
            n=len(rows),
        ),
        dist=dist, bands=bands, genres=genres, rows=out_rows,
        fve=(dict(
            real=fve["summary"]["FVE_real"], mimic=fve["summary"]["FVE_mimic"],
            control=fve["summary"]["FVE_control"], dfve=fve["summary"]["dFVE"],
            act_frac=fve["summary"]["activation_specific_frac"],
            mse_ratio=fve["summary"]["mse_ratio_mimic_over_real"],
            norm_share=fve["norm_share_in_mean"], D=fve["D_variance"],
            real_beats_mimic=fve["summary"]["real_beats_mimic_FVE"],
        ) if fve else None),
        std=(dict(
            real=std["summary"]["std_cos"]["real"], mimic=std["summary"]["std_cos"]["mimic"],
            control=std["summary"]["std_cos"]["control"], delta=std["summary"]["std_delta"],
            win=std["summary"]["std_win"],
            r2_real=std["summary"]["std_R2"]["real"], r2_mimic=std["summary"]["std_R2"]["mimic"],
            r2_control=std["summary"]["std_R2"]["control"],
        ) if std else None),
        dossier=DOSSIER.get(key),
        suite=SUITE,
    )

    js = ("// auto-generated by build_act_audit_view.py, do not edit by hand\n"
          "(window.ACT_AUDIT = window.ACT_AUDIT || {models: []}).models.push(\n"
          + json.dumps(entry, ensure_ascii=False) + "\n);\n")
    out = CORPUS / f"act_audit_{key}.js"
    out.write_text(js, encoding="utf-8")
    print(f"wrote {out.name}: {len(rows)} rows, {len(genres)} genres, {len(bands)} bands, "
          f"{out.stat().st_size:,} bytes")
    print(f"  headline: real={entry['summary']['real']} mimic={entry['summary']['mimic']} "
          f"control={entry['summary']['control']} Δ={entry['summary']['mean_delta']} "
          f"win={entry['summary']['real_beats_mimic']}")
    if val:
        print(f"  gates: alignment {alignment['match']}/{alignment['n']}  "
              f"validation pearson={val['pearson']} mean|err|={val['mean_abs_err']}")
    else:
        inj = res.get("injection") or {}
        print(f"  gates: anchor {alignment['match']}/{alignment['n']}  "
              f"injection CJK={inj.get('cjk')}/{inj.get('n')} "
              f"difficulty_corr={res.get('difficulty_corr')}")


if __name__ == "__main__":
    main()
