"""Measure batched greedy generation without modifying benchmark evidence."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from calix_telco_bench.contracts import load_manifest
from calix_telco_bench.dataset import iter_records
from calix_telco_bench.generation import load_prompt_protocol, system_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/MANIFEST.json"))
    parser.add_argument("--prompts", type=Path, default=Path("prompts/generation-route-expert-v2.yaml"))
    parser.add_argument("--route", default="bgp")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("/tmp/calix-telco-bench-batch-smoke.json"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch size and max-new-tokens must be positive")

    protocol = load_prompt_protocol(args.prompts)
    records = [record for record in iter_records(args.manifest, args.batch_size, [args.route])]
    if len(records) != args.batch_size:
        raise RuntimeError(f"route {args.route!r} has fewer than {args.batch_size} records")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, fix_mistral_regex=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rendered = [tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt(protocol, record["route"])}, {"role": "user", "content": record["prompt"]}],
        tokenize=False, add_generation_prompt=True,
    ) for record in records]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(args.device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(args.device).eval()
    compiled = False
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        compiled = True
    except Exception:
        pass
    torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, cache_implementation="static",
        )
    torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started
    completions = outputs[:, inputs["input_ids"].shape[1]:]
    result = {
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "attention_backend": "sdpa", "cache_implementation": "static",
        "torch_compile": compiled, "elapsed_seconds": elapsed,
        "output_tokens": [int(row.shape[0]) for row in completions],
        "tokens_per_second": int(completions.numel()) / elapsed if elapsed else None,
        "answers": [tokenizer.decode(row, skip_special_tokens=True) for row in completions],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "answers"}, indent=2))


if __name__ == "__main__":
    main()
