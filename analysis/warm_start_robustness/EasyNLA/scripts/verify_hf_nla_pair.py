"""End-to-end smoke test for a full-HF NLA AV/AR checkpoint pair.

Loads one raw activation from parquet, generates an explanation with the AV,
reconstructs it with the AR, and writes a durable JSON record. All NLA-specific
constants are loaded from the two checkpoint sidecars.
"""

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.schema import EXPLANATION_RE, normalize_activation
from nla.utils import register_karvonen_hook
from nla.utils.critic import critic_predict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--av-ckpt", required=True)
    p.add_argument("--ar-ckpt", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    table = pq.read_table(
        args.parquet,
        columns=["activation_vector", "detokenized_text_truncated", "doc_id"],
    ).slice(args.row, 1)
    assert table.num_rows == 1, f"row {args.row} is outside {args.parquet}"
    activation = torch.tensor(
        table.column("activation_vector")[0].as_py(), dtype=torch.float32
    )
    source = table.column("detokenized_text_truncated")[0].as_py()
    doc_id = table.column("doc_id")[0].as_py()

    tokenizer = AutoTokenizer.from_pretrained(args.av_ckpt, trust_remote_code=True)
    av_cfg = load_nla_config(args.av_ckpt, tokenizer)
    assert activation.numel() == av_cfg.d_model

    content = av_cfg.actor_prompt_template.format(
        injection_char=av_cfg.injection_char
    )
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded

    actor = AutoModelForCausalLM.from_pretrained(
        args.av_ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="cuda:0",
    ).eval()
    input_ids = input_ids.to("cuda:0")
    # Match the AV SFT/RL mechanism exactly: inject the RAW activation with the
    # Karvonen norm-matched residual hook at layer 1.  `injection_scale` belongs
    # to the older embedding-overwrite path and is intentionally irrelevant here.
    vectors_ref = [activation.unsqueeze(0)]
    hook_state = register_karvonen_hook(
        actor,
        vectors_ref,
        av_cfg.injection_token_id,
        av_cfg.injection_left_neighbor_id,
        av_cfg.injection_right_neighbor_id,
        layer_idx=1,
    )

    try:
        with torch.inference_mode():
            generated = actor.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        vectors_ref[0] = None
    # `generate()` returns the prompt followed by the completion when called
    # with input_ids.  Decode only newly generated tokens: the prompt itself
    # contains both the literal injection marker and instructional
    # <explanation> text, so searching the full conversation creates a false
    # explanation match and a false CJK-leak failure.
    raw_response = tokenizer.decode(
        generated[0, input_ids.shape[1] :], skip_special_tokens=True
    )
    print(f"[smoke] Karvonen apply_count={hook_state['apply_count']}")
    print(f"[smoke] raw_response={raw_response!r}")
    assert hook_state["apply_count"] == 1, (
        f"Karvonen injection applied {hook_state['apply_count']} times; expected one prefill"
    )
    match = EXPLANATION_RE.search(raw_response)
    assert match is not None, f"AV output lacks complete explanation tags: {raw_response!r}"
    explanation = match.group(1).strip()
    assert not re.search(r"[\u3200-\u32ff]", explanation), (
        f"CJK marker-like text in AV explanation despite apply_count=1: {explanation!r}"
    )

    del actor, generated
    torch.cuda.empty_cache()

    ar_cfg = load_nla_config(args.ar_ckpt, tokenizer)
    assert ar_cfg.critic_prompt_template is not None
    critic = NLACriticModel.from_pretrained(
        args.ar_ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    critic_prompt = ar_cfg.critic_prompt_template.format(explanation=explanation)
    critic_ids = tokenizer(
        critic_prompt, return_tensors="pt", add_special_tokens=True
    )["input_ids"].to("cuda:0")
    with torch.inference_mode():
        prediction = critic_predict(
            critic, critic_ids, None, ar_cfg.mse_scale
        )[0]
    pred_n = normalize_activation(prediction.unsqueeze(0), ar_cfg.mse_scale)[0]
    gold_n = normalize_activation(activation.unsqueeze(0), ar_cfg.mse_scale)[0]
    mse = F.mse_loss(pred_n.cpu(), gold_n.cpu()).item()
    cosine = F.cosine_similarity(pred_n.cpu(), gold_n.cpu(), dim=0).item()

    result = {
        "av_checkpoint": str(Path(args.av_ckpt).resolve()),
        "ar_checkpoint": str(Path(args.ar_ckpt).resolve()),
        "parquet": str(Path(args.parquet).resolve()),
        "row": args.row,
        "doc_id": doc_id,
        "activation_norm": activation.norm().item(),
        "source_text": source,
        "raw_response": raw_response,
        "explanation": explanation,
        "normalized_mse": mse,
        "cosine_similarity": cosine,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
