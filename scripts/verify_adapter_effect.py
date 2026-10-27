#!/usr/bin/env python3
"""Verify that a configured LoRA target changes CPT logits on one held-out prompt.

This is a read-only, single-prompt inference check.  It loads the CPT backbone,
records its next-token logits, attaches the configured student LoRA to that same
backbone, and records the resulting logits.  It is not a quality benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--base-target", default="cpt_base")
    parser.add_argument("--adapter-target", default="mopd_student_step_2000")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--route", default="bgp")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path, help="optional JSON evidence path")
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index must be non-negative")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from calix_telco_bench.generation import _render, load_prompt_protocol, system_prompt
    from calix_telco_bench.plan import load_run_config, resolve_target

    config_path = args.config.resolve()
    config = load_run_config(config_path)
    base = resolve_target(config, config_path, args.base_target)
    student = resolve_target(config, config_path, args.adapter_target)
    if base["adapter"] is not None:
        raise ValueError(f"{args.base_target} is not a bare CPT target")
    if not student["adapter"]:
        raise ValueError(f"{args.adapter_target} has no adapter")
    if base["model_path"] != student["model_path"] or base["model_revision"] != student["model_revision"]:
        raise ValueError("base and adapter targets do not share the configured immutable CPT backbone")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    manifest = json.loads(Path(base["manifest"]).read_text(encoding="utf-8"))
    entry = next((value for value in manifest["routes"] if value["route"] == args.route), None)
    if entry is None:
        raise ValueError(f"route {args.route!r} is absent from the manifest")
    records = [json.loads(line) for line in (Path(base["manifest"]).parent / entry["path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    record = records[args.sample_index]
    route = args.route
    protocol = load_prompt_protocol(base["prompts"])
    tokenizer = AutoTokenizer.from_pretrained(base["model_path"], local_files_only=True, fix_mistral_regex=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base["model_path"], local_files_only=True, dtype=torch.bfloat16).to(args.device).eval()
    input_ids = _render(tokenizer, system_prompt(protocol, route), record["prompt"]).to(args.device)
    with torch.inference_mode():
        base_logits = model(input_ids).logits[0, -1].float().cpu()
    # This is the original MOPD student-evaluator load contract.
    student_model = PeftModel.from_pretrained(model, student["adapter"], adapter_name="student", is_trainable=False)
    student_model.set_adapter("student")
    student_model.eval()
    with torch.inference_mode():
        attached_logits = student_model(input_ids).logits[0, -1].float().cpu()
    # PEFT exposes an explicit enable hook.  A correct evaluator must never
    # silently score a loaded-but-disabled LoRA adapter.
    student_model.enable_adapter_layers()
    with torch.inference_mode():
        enabled_logits = student_model(input_ids).logits[0, -1].float().cpu()
    adapter_layers = []
    for name, module in student_model.named_modules():
        if not hasattr(module, "lora_A") or "student" not in getattr(module, "lora_A", {}):
            continue
        a = module.lora_A["student"].weight.detach().float()
        b = module.lora_B["student"].weight.detach().float()
        adapter_layers.append({
            "name": name,
            "disabled": bool(getattr(module, "disable_adapters", False)),
            "merged": bool(getattr(module, "merged", False)),
            "a_nonzero": int(a.count_nonzero()), "b_nonzero": int(b.count_nonzero()),
            "a_max_abs": float(a.abs().max()), "b_max_abs": float(b.abs().max()),
        })
        if len(adapter_layers) == 3:
            break
    merged_model = student_model.merge_and_unload().eval()
    with torch.inference_mode():
        merged_logits = merged_model(input_ids).logits[0, -1].float().cpu()
    attached_delta = (attached_logits - base_logits).abs()
    enabled_delta = (enabled_logits - base_logits).abs()
    merged_delta = (merged_logits - base_logits).abs()
    adapter_file = Path(student["adapter"]) / "adapter_model.safetensors"
    result: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "base_vs_lora_next_token_logit_check",
        "device": args.device,
        "sample": {"route": route, "sample_id": record["sample_id"], "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest()},
        "base": {"target": args.base_target, "model_path": base["model_path"], "model_revision": base["model_revision"]},
        "student": {"target": args.adapter_target, "adapter": student["adapter"], "adapter_sha256": _sha256(adapter_file), "active_adapters": list(student_model.active_adapters)},
        "adapter_layer_samples": adapter_layers,
        "logit_delta": {
            "attached": {"max_abs": float(attached_delta.max()), "mean_abs": float(attached_delta.mean()), "nonzero_count": int((attached_delta > 0).sum())},
            "explicitly_enabled": {"max_abs": float(enabled_delta.max()), "mean_abs": float(enabled_delta.mean()), "nonzero_count": int((enabled_delta > 0).sum())},
            "merged": {"max_abs": float(merged_delta.max()), "mean_abs": float(merged_delta.mean()), "nonzero_count": int((merged_delta > 0).sum())},
            "vocabulary_size": int(enabled_delta.numel()),
            "base_top1_token_id": int(base_logits.argmax()), "student_top1_token_id": int(enabled_logits.argmax()),
        },
    }
    result["adapter_effect_detected"] = result["logit_delta"]["explicitly_enabled"]["nonzero_count"] > 0
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["adapter_effect_detected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
