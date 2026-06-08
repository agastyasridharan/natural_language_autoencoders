"""CLI for the white-box harness. argparse (matches miles/nla convention).

Subcommands:
  gate    reproduce a logged decode dump end-to-end (the Phase-0 validation gate)
  e1      dump per-layer marker-propagation curves for a conversation
  decode  verbalize + score a single position (quick smoke test)

Examples
--------
# Validation gate on the released Qwen NLA (auto-downloads from HF):
python -m nla.whitebox.cli gate --released qwen2.5-7b-L20 \
    --log examples/qwen7b_layer20_step4200.txt --level L1

# Same, from local checkpoint dirs, end-to-end:
python -m nla.whitebox.cli gate \
    --av ./qwen-L20-av --ar ./qwen-L20-ar \
    --extraction-model Qwen/Qwen2.5-7B-Instruct --layer 20 \
    --log examples/qwen7b_layer20_step4200.txt --level L2

# E1 propagation curves over the reply region:
python -m nla.whitebox.cli e1 --released qwen2.5-7b-L20 \
    --log examples/qwen7b_layer20_step4200.txt --min-position 34 --out e1_qwen.npz
"""

from __future__ import annotations

import argparse
import os
import sys

# Reduce CUDA fragmentation on small/shared GPUs (set before any torch import).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from nla.whitebox.constants import RELEASED


def _add_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--released", choices=sorted(RELEASED), default=None,
                    help="fill av/ar/extraction/layer/baseline from the registry")
    ap.add_argument("--av", default=None, help="AV checkpoint dir or HF repo id")
    ap.add_argument("--ar", default=None, help="AR checkpoint dir or HF repo id")
    ap.add_argument("--extraction-model", default=None, help="target model (HF id) whose activations are explained")
    ap.add_argument("--layer", type=int, default=None, help="extraction layer K (block-K output = hidden_states[K+1])")
    ap.add_argument("--device", default="cuda", help="'cuda', 'cpu', 'cuda:0', or 'auto' (device_map)")
    ap.add_argument("--injection-scale", type=float, default=None, help="override sidecar injection_scale (OOD)")


def _build(args):
    from nla.whitebox.harness import WhiteBoxNLA

    if args.released:
        m = RELEASED[args.released]
        av = args.av or m.av_repo
        ar = args.ar or m.ar_repo
        extraction = args.extraction_model or m.extraction_model
        layer = args.layer if args.layer is not None else m.layer
        assert av and ar, (
            f"{m.name} has no registry repo ids — pass --av/--ar local dirs."
        )
    else:
        av, ar, extraction, layer = args.av, args.ar, args.extraction_model, args.layer
        assert av and ar and extraction and layer is not None, (
            "need --av, --ar, --extraction-model, --layer (or --released)."
        )
    return WhiteBoxNLA(
        av, ar, extraction, layer=layer,
        device=args.device, injection_scale_override=args.injection_scale,
    )


def _baseline(args) -> float:
    if getattr(args, "baseline", None) is not None:
        return args.baseline
    if args.released and RELEASED[args.released].fve_nrm_baseline is not None:
        return RELEASED[args.released].fve_nrm_baseline
    return 0.7335  # Qwen-2.5-7B L20 default


def cmd_gate(args) -> int:
    from nla.whitebox.gate import format_report, run_gate

    nla = _build(args)
    report = run_gate(
        nla, args.log, level=args.level, baseline=_baseline(args),
        min_position=args.min_position, max_new_tokens=args.max_new_tokens, verbose=True,
    )
    print(format_report(report))
    return 0 if report.passed else 1


def cmd_e1(args) -> int:
    from nla.whitebox.experiments.e1_propagation import run_e1, summarize_e1

    nla = _build(args)
    if args.log:
        from nla.whitebox import example_log as el

        conv = el.parse_conversation(args.log)
        messages = [{"role": "user", "content": conv.user_message if conv else args.prompt}]
    else:
        messages = [{"role": "user", "content": args.prompt}]
    acts = nla.extract_conversation(messages, generate_reply=True, max_new_tokens=args.max_new_tokens)
    # restrict to the in-distribution region by default
    idxs = [i for i in range(len(acts)) if i >= args.min_position]
    vecs = acts.residuals[idxs]
    labels = [acts.token_strs[i] for i in idxs]
    nla.unload_extractor()  # free the extractor before the AV loads (small-GPU safe)
    res = run_e1(nla, vecs, labels=labels, out_path=args.out)
    print(summarize_e1(res, cos_threshold=args.cos_threshold))
    if args.out:
        print(f"[e1] saved curves to {args.out}")
    return 0


def cmd_decode(args) -> int:
    from nla.whitebox import example_log as el

    nla = _build(args)
    conv = el.parse_conversation(args.log) if args.log else None
    messages = [{"role": "user", "content": conv.user_message if conv else args.prompt}]
    acts = nla.extract_conversation(messages, generate_reply=True, max_new_tokens=args.max_new_tokens)
    p = args.position
    activation = acts.residuals[p]
    token, norm = acts.token_strs[p], float(activation.norm())
    nla.unload_extractor()  # stage: free extractor before AV
    expl = nla.verbalize(activation, greedy=True, max_new_tokens=args.max_new_tokens)
    nla.unload_av()         # stage: free AV before AR
    mse, cos = nla.score(expl, activation)
    fve = nla.fve_nrm(mse, _baseline(args))
    print(f"position {p} token={token!r}  ||v||={norm:.1f}")
    print(f"  mse_nrm={mse:.3f}  cos={cos:.3f}  fve_nrm={fve:.3f}")
    print(f"  explanation: {expl}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nla.whitebox.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="reproduce a logged decode dump (validation gate)")
    _add_model_args(g)
    g.add_argument("--log", required=True, help="path to a examples/*.txt decode dump")
    g.add_argument("--level", choices=["extract", "L1", "L2"], default="L1")
    g.add_argument("--baseline", type=float, default=None, help="FVE denominator Var(v_nrm)")
    g.add_argument("--min-position", type=int, default=24)
    g.add_argument("--max-new-tokens", type=int, default=200)
    g.set_defaults(func=cmd_gate)

    e = sub.add_parser("e1", help="dump marker-propagation curves")
    _add_model_args(e)
    e.add_argument("--log", default=None, help="dump to take the conversation from")
    e.add_argument("--prompt", default="What are you hiding?", help="user message (if --log absent)")
    e.add_argument("--min-position", type=int, default=34, help="skip OOD prompt region")
    e.add_argument("--cos-threshold", type=float, default=0.5)
    e.add_argument("--max-new-tokens", type=int, default=200)
    e.add_argument("--baseline", type=float, default=None)
    e.add_argument("--out", default=None, help="save curves to this .npz")
    e.set_defaults(func=cmd_e1)

    d = sub.add_parser("decode", help="verbalize + score one position")
    _add_model_args(d)
    d.add_argument("--log", default=None)
    d.add_argument("--prompt", default="What are you hiding?")
    d.add_argument("--position", type=int, default=50)
    d.add_argument("--baseline", type=float, default=None)
    d.add_argument("--max-new-tokens", type=int, default=200)
    d.set_defaults(func=cmd_decode)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
