"""MOPD-compatible Hugging Face and PEFT benchmark generation."""
from __future__ import annotations
import json, os, platform, sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
import ssl
from time import sleep
from typing import Any, Callable
from urllib import error, request
import yaml
from .contracts import identity, load_manifest, sha256, sha256_text, validate_manifest
from .dataset import iter_records

_API_MAX_RETRIES = 5
_API_RETRY_WAIT_SECONDS = 300


def _retry_api_call(operation: Callable[[], Any]) -> Any:
    for retry in range(_API_MAX_RETRIES + 1):
        try:
            return operation()
        except Exception as exc:
            if retry == _API_MAX_RETRIES:
                raise
            print(
                f"OpenAI-compatible API call failed ({type(exc).__name__}); "
                f"retrying in 5 minutes ({retry + 1}/{_API_MAX_RETRIES})",
                flush=True,
            )
            sleep(_API_RETRY_WAIT_SECONDS)


def _urlopen(req: request.Request, timeout: int):
    # NODE_TLS_REJECT_UNAUTHORIZED is Node-specific; apply its explicit opt-out
    # to this Python HTTPS client only when the caller sets it to "0".
    if req.type == "https" and os.getenv("NODE_TLS_REJECT_UNAUTHORIZED") == "0":
        return request.urlopen(req, timeout=timeout, context=ssl._create_unverified_context())
    return request.urlopen(req, timeout=timeout)

def load_prompt_protocol(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("invalid generation-prompt protocol")
    has_shared_system = isinstance(value.get("system"), str) and bool(value["system"].strip())
    has_route_systems = isinstance(value.get("routes"), dict) and all(
        isinstance(route, str) and isinstance(prompt, str) and prompt.strip()
        for route, prompt in value["routes"].items()
    )
    if has_shared_system == has_route_systems:
        raise ValueError("generation-prompt protocol must define exactly one of system or routes")
    if not isinstance(value.get("protocol_id"), str) or not value["protocol_id"].strip():
        raise ValueError("generation-prompt protocol requires protocol_id")
    if not isinstance(value.get("decoding"), dict):
        raise ValueError("generation-prompt protocol requires decoding")
    return value


def system_prompt(protocol: dict[str, Any], route: str) -> str:
    """Resolve a shared deployment prompt or a route-labelled diagnostic prompt."""
    shared = protocol.get("system")
    if isinstance(shared, str):
        return shared
    routes = protocol["routes"]
    if route not in routes:
        raise ValueError(f"prompt protocol has no system prompt for route {route!r}")
    return str(routes[route])

def _render(tokenizer: Any, system: str, prompt: str) -> Any:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    # Transformers releases have returned either a tensor or a BatchEncoding
    # here. Generation below requires the actual token tensor in both cases.
    return rendered["input_ids"] if isinstance(rendered, Mapping) else rendered


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def run(*, manifest: Path, prompts: Path, model_path: str, adapter: str | None,
        model_revision: str | None, adapter_revision: str | None, device: str,
        output: Path, max_new_tokens: int | None, limit_per_route: int | None,
        resume: bool, local_files_only: bool, trust_remote_code: bool,
        routes: list[str] | None = None) -> dict[str, int]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc: raise RuntimeError("install telcobench[huggingface]") from exc
    if not device.startswith("cuda") or not torch.cuda.is_available(): raise RuntimeError("MOPD-compatible evaluation requires --device cuda[:N]")
    validate_manifest(manifest); protocol = load_prompt_protocol(prompts)
    available_routes = [entry["route"] for entry in load_manifest(manifest)["routes"]]
    if routes is not None:
        unknown = sorted(set(routes) - set(available_routes))
        if unknown:
            raise ValueError(f"selected routes are absent from manifest: {', '.join(unknown)}")
        selected_routes = [route for route in available_routes if route in set(routes)]
    else:
        selected_routes = available_routes
    output.mkdir(parents=True, exist_ok=True); generations = output / "generations.jsonl"
    if generations.exists() and not resume: raise FileExistsError(f"{generations} exists; use --resume")
    prior_lines = generations.read_text(encoding="utf-8").splitlines() if generations.exists() else []
    prior_rows = [json.loads(line) for line in prior_lines if line.strip()]
    done = {identity(x["route"], x["sample_id"]) for x in prior_rows if x.get("status") in {"ok", "error"}}
    expected_by_route: dict[str, int] = {route: 0 for route in selected_routes}
    for record in iter_records(manifest, limit_per_route, selected_routes): expected_by_route[record["route"]] += 1
    expected_total = sum(expected_by_route.values())
    # Publish loading state before any heavyweight Hugging Face / PEFT load so
    # the TUI never mistakes a legitimate model initialization for a stalled
    # or absent worker.
    progress_path = output / "generation-progress.json"
    route_progress = {route: {"expected": expected, "completed": 0, "ok": 0, "errors": 0, "status": "QUEUED"} for route, expected in expected_by_route.items()}
    for row in prior_rows:
        route = row.get("route")
        if route in route_progress and row.get("status") in {"ok", "error"}:
            state = route_progress[route]; state["completed"] += 1; state["ok"] += int(row["status"] == "ok"); state["errors"] += int(row["status"] == "error")
    for state in route_progress.values():
        if state["completed"] == state["expected"]: state["status"] = "COMPLETED" if not state["errors"] else "FAILED"
    progress = {
        "schema_version": 1, "status": "LOADING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), "expected": expected_total,
        "completed": len(done), "ok": sum(row.get("status") == "ok" for row in prior_rows),
        "errors": sum(row.get("status") == "error" for row in prior_rows),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in prior_rows if row.get("status") == "ok"),
        "selected_routes": selected_routes, "device": device, "routes": route_progress,
        "completed_routes": [route for route, state in route_progress.items() if state["status"] == "COMPLETED"],
    }
    _write_json(progress_path, progress)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only, fix_mistral_regex=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=local_files_only, dtype=torch.bfloat16, trust_remote_code=trust_remote_code).to(device)
    adapter_runtime: dict[str, Any] | None = None
    if adapter:
        # Match the original MOPD student evaluator exactly: use an explicit
        # named adapter and select it before generation.  Do not rely on PEFT's
        # version-dependent implicit ``default`` selection.
        model = PeftModel.from_pretrained(model, adapter, adapter_name="student", is_trainable=False)
        model.set_adapter("student")
        # Loading an adapter must not be conflated with applying it.  Explicitly
        # enable the LoRA layers and fail closed if PEFT reports no live target.
        model.enable_adapter_layers()
        lora_layers = [
            module for module in model.modules()
            if hasattr(module, "lora_A") and "student" in getattr(module, "lora_A", {})
        ]
        if not lora_layers:
            raise RuntimeError("PEFT loaded no student LoRA target layers; refusing a student evaluation")
        disabled = sum(bool(getattr(module, "disable_adapters", False)) for module in lora_layers)
        if disabled:
            raise RuntimeError(f"{disabled}/{len(lora_layers)} LoRA target layers remain disabled; refusing a student evaluation")
        adapter_runtime = {"adapter_name": "student", "adapter_enabled": True, "student_lora_target_layers": len(lora_layers), "disabled_lora_target_layers": disabled}
    model.eval(); decoding = dict(protocol["decoding"]); decoding["max_new_tokens"] = max_new_tokens or decoding["max_new_tokens"]
    # Frozen neutral evaluation is greedy.  Recent Transformers warns (and can
    # change behavior across releases) when sampling-only knobs accompany
    # ``do_sample: false``; omit them from the actual generation call.
    if not decoding.get("do_sample", False):
        for key in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
            decoding.pop(key, None)
    prompt_map = {route: system_prompt(protocol, route) for route in selected_routes}
    metadata = {
        "schema_version": 2,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_manifest_sha256": sha256(manifest),
        "selected_routes": selected_routes,
        "prompt_protocol": protocol,
        "prompt_protocol_sha256": sha256(prompts),
        "evaluation_condition": protocol.get("condition", "route_conditioned_diagnostic"),
        "system_prompts": {route: {"text": text, "sha256": sha256_text(text)} for route, text in prompt_map.items()},
        "model": {"path": model_path, "revision": model_revision, "config_name_or_path": getattr(model.config, "_name_or_path", None)},
        "adapter": {"path": adapter, "revision": adapter_revision, "runtime": adapter_runtime},
        "device": device,
        "tokenizer": {"name_or_path": getattr(tokenizer, "name_or_path", None), "chat_template": getattr(tokenizer, "chat_template", None), "chat_template_sha256": sha256_text(getattr(tokenizer, "chat_template", "") or "")},
        "tokenizer_fix_mistral_regex": True,
        "rendering": {"implementation": "tokenizer.apply_chat_template", "add_generation_prompt": True, "roles": ["system", "user"]},
        "decoding": decoding,
        "python": sys.version,
        "platform": platform.platform(),
    }
    run_record = output / "run.json"
    if resume and run_record.exists():
        prior = json.loads(run_record.read_text(encoding="utf-8"))
        for key in ("benchmark_manifest_sha256", "selected_routes", "prompt_protocol_sha256", "model", "adapter", "decoding"):
            if prior.get(key) != metadata.get(key):
                raise ValueError(f"cannot resume: {key} differs from existing run metadata")
        metadata["started_at_utc"] = prior.get("started_at_utc")
        metadata["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(run_record, metadata)
    progress["status"] = "GENERATING"; progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat(); _write_json(progress_path, progress)
    with generations.open("a", encoding="utf-8") as stream:
        for record in iter_records(manifest, limit_per_route, selected_routes):
            key = identity(record["route"], record["sample_id"])
            if key in done: continue
            try:
                rendered_system = prompt_map[record["route"]]
                input_ids = _render(tokenizer, rendered_system, record["prompt"]).to(device)
                with torch.inference_mode(): output_ids = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, **decoding)
                completion = output_ids[0, input_ids.shape[-1]:]
                result = {"route": record["route"], "sample_id": record["sample_id"], "prompt": record["prompt"], "system_prompt_sha256": sha256_text(rendered_system), "answer": tokenizer.decode(completion, skip_special_tokens=True), "input_tokens": int(input_ids.shape[-1]), "output_tokens": int(completion.shape[-1]), "status": "ok"}
            except Exception as exc:
                message = str(exc) or repr(exc)
                result = {"route": record["route"], "sample_id": record["sample_id"], "prompt": record["prompt"], "status": "error", "error_type": type(exc).__name__, "error": message}
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True)+"\n"); stream.flush()
            progress["completed"] = int(progress["completed"]) + 1
            progress["ok"] = int(progress["ok"]) + int(result["status"] == "ok")
            progress["errors"] = int(progress["errors"]) + int(result["status"] == "error")
            progress["output_tokens"] = int(progress["output_tokens"]) + int(result.get("output_tokens", 0))
            route_state = route_progress[record["route"]]
            route_state["completed"] += 1; route_state["ok"] += int(result["status"] == "ok"); route_state["errors"] += int(result["status"] == "error")
            route_state["status"] = ("COMPLETED" if not route_state["errors"] else "FAILED") if route_state["completed"] == route_state["expected"] else "GENERATING"
            progress["completed_routes"] = [route for route, state in route_progress.items() if state["status"] == "COMPLETED"]
            progress["current_route"] = record["route"]
            progress["current_sample_id"] = record["sample_id"]
            progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(progress_path, progress)
    result = coverage(manifest, generations, output / "coverage.json", limit_per_route, selected_routes)
    progress.update({"status": "COMPLETED", "completed": result["unique_records"], "ok": result["ok"], "errors": result["errors"], "missing": result["missing"], "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    progress["updated_at_utc"] = progress["finished_at_utc"]
    _write_json(progress_path, progress)
    return result

def coverage(
    manifest: Path,
    generations: Path,
    output: Path,
    limit_per_route: int | None,
    routes: list[str] | None = None,
) -> dict[str, int]:
    expected = {identity(r["route"], r["sample_id"]) for r in iter_records(manifest, limit_per_route, routes)}
    rows = [json.loads(line) for line in generations.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = {identity(r["route"], r["sample_id"]) for r in rows}
    result = {"expected": len(expected), "records": len(rows), "unique_records": len(seen), "missing": len(expected-seen), "ok": sum(r.get("status")=="ok" for r in rows), "errors": sum(r.get("status")=="error" for r in rows)}
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8"); return result


def _openai_models(*, base_url: str, endpoint: str, api_key: str | None,
                   timeout: int) -> set[str]:
    """Read the configured model catalog before generation starts."""
    headers = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}", headers=headers, method="GET")
    try:
        with _urlopen(req, timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible models endpoint returned HTTP {exc.code}: {body[:500]}") from exc
    entries = value.get("data", value.get("models")) if isinstance(value, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError("OpenAI-compatible models endpoint response has no data/models list")
    return {
        entry if isinstance(entry, str) else entry.get("id")
        for entry in entries
        if isinstance(entry, str) or (isinstance(entry, dict) and isinstance(entry.get("id"), str))
    }


def _openai_completion(*, base_url: str, endpoint: str, api_key: str | None, model: str,
                       system: str, prompt: str, max_new_tokens: int,
                       temperature: float, top_p: float, seed: object,
                       timeout: int,
                       chat_template_kwargs: dict[str, Any] | None = None) -> tuple[str | None, int, int, str | None]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p,
        "stream": False,
    }
    if isinstance(seed, int): payload["seed"] = seed
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    headers = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with _urlopen(req, timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible endpoint returned HTTP {exc.code}: {body[:500]}") from exc
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("OpenAI-compatible response has no choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("OpenAI-compatible response has no choices[0].message")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise RuntimeError("OpenAI-compatible response choices[0].message.content is not a string")
    usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
    finish_reason = choices[0].get("finish_reason")
    return (
        content, int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
        finish_reason if isinstance(finish_reason, str) else None,
    )


def run_openai_compatible(*, backend: str, manifest: Path, prompts: Path, output: Path,
                          model: str, base_url: str, models_endpoint: str,
                          chat_completions_endpoint: str, api_key_env: object,
                          allow_missing_api_key: bool, request_timeout_seconds: int,
                          temperature: float, top_p: float, seed: object,
                          max_new_tokens: int, routes: list[str] | None, device: str,
                          empty_content_fallback: dict[str, Any] | None = None,
                          limit_per_route: int | None, resume: bool) -> dict[str, int]:
    """Generate against a local OpenAI-compatible endpoint with the HF evidence contract."""
    if backend != "openai_compatible": raise ValueError(f"unsupported generation backend: {backend}")
    api_key = os.getenv(str(api_key_env)) if isinstance(api_key_env, str) and api_key_env else None
    if not api_key and not allow_missing_api_key:
        raise RuntimeError(f"{api_key_env} is required for this endpoint")
    available_models = _retry_api_call(lambda: _openai_models(
        base_url=base_url, endpoint=models_endpoint, api_key=api_key, timeout=request_timeout_seconds,
    ))
    if model not in available_models:
        raise RuntimeError(f"model {model!r} is not listed by {models_endpoint}; available models: {', '.join(sorted(available_models))}")
    validate_manifest(manifest); protocol = load_prompt_protocol(prompts)
    available = [entry["route"] for entry in load_manifest(manifest)["routes"]]
    if routes is not None:
        unknown = sorted(set(routes) - set(available))
        if unknown: raise ValueError(f"selected routes are absent from manifest: {', '.join(unknown)}")
        selected = [route for route in available if route in set(routes)]
    else: selected = available
    output.mkdir(parents=True, exist_ok=True); generations = output / "generations.jsonl"
    if generations.exists() and not resume: raise FileExistsError(f"{generations} exists; use --resume")
    rows = [json.loads(line) for line in generations.read_text(encoding="utf-8").splitlines() if line.strip()] if generations.exists() else []
    done = {identity(row["route"], row["sample_id"]) for row in rows if row.get("status") in {"ok", "error"}}
    fallback_retry_ids: set[tuple[str, str]] = set()
    fallback_policy = empty_content_fallback or {}
    error_backup_name = fallback_policy.get("prior_error_backup")
    if isinstance(error_backup_name, str):
        error_backup = output / error_backup_name
        if error_backup.is_file():
            for line in error_backup.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                prior_row = json.loads(line)
                if prior_row.get("status") == "error":
                    fallback_retry_ids.add(identity(prior_row["route"], prior_row["sample_id"]))
    expected_by_route = {route: 0 for route in selected}
    for record in iter_records(manifest, limit_per_route, selected): expected_by_route[record["route"]] += 1
    route_progress = {route: {"expected": expected, "completed": 0, "ok": 0, "errors": 0, "status": "QUEUED"} for route, expected in expected_by_route.items()}
    for row in rows:
        state = route_progress.get(row.get("route"))
        if state is not None and row.get("status") in {"ok", "error"}:
            state["completed"] += 1; state["ok"] += int(row["status"] == "ok"); state["errors"] += int(row["status"] == "error")
    prompt_map = {route: system_prompt(protocol, route) for route in selected}
    now = datetime.now(timezone.utc).isoformat()
    progress: dict[str, Any] = {"schema_version": 1, "status": "GENERATING", "started_at_utc": now, "updated_at_utc": now,
        "expected": sum(expected_by_route.values()), "completed": len(done), "ok": sum(row.get("status") == "ok" for row in rows),
        "errors": sum(row.get("status") == "error" for row in rows), "output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows if row.get("status") == "ok"),
        "selected_routes": selected, "device": device, "routes": route_progress, "completed_routes": []}
    progress_path = output / "generation-progress.json"; _write_json(progress_path, progress)
    metadata = {"schema_version": 2, "started_at_utc": now, "benchmark_manifest_sha256": sha256(manifest), "selected_routes": selected,
        "prompt_protocol": protocol, "prompt_protocol_sha256": sha256(prompts), "evaluation_condition": protocol.get("condition", "route_conditioned_diagnostic"),
        "system_prompts": {route: {"text": text, "sha256": sha256_text(text)} for route, text in prompt_map.items()},
        "model": {"provider": "openai_compatible", "base_url": base_url, "id": model}, "adapter": {"path": None, "revision": None, "runtime": None},
        "device": device, "rendering": {"implementation": "OpenAI chat.completions", "roles": ["system", "user"]},
        "empty_content_fallback": empty_content_fallback,
        "decoding": {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "seed": seed}, "python": sys.version, "platform": platform.platform()}
    run_record = output / "run.json"
    if resume and run_record.exists():
        prior = json.loads(run_record.read_text(encoding="utf-8"))
        for key in ("benchmark_manifest_sha256", "selected_routes", "prompt_protocol_sha256", "model", "decoding"):
            if prior.get(key) != metadata.get(key): raise ValueError(f"cannot resume: {key} differs from existing run metadata")
        metadata["started_at_utc"] = prior.get("started_at_utc"); metadata["resumed_at_utc"] = now
    _write_json(run_record, metadata)
    with generations.open("a", encoding="utf-8") as stream:
        for record in iter_records(manifest, limit_per_route, selected):
            key = identity(record["route"], record["sample_id"])
            if key in done: continue
            system = prompt_map[record["route"]]
            try:
                request_kwargs = dict(
                    base_url=base_url, endpoint=chat_completions_endpoint, api_key=api_key,
                    model=model, system=system, prompt=record["prompt"],
                    max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                    seed=seed, timeout=request_timeout_seconds,
                )
                fallback_kwargs = (empty_content_fallback or {}).get("chat_template_kwargs")
                used_fallback = key in fallback_retry_ids and bool(fallback_kwargs)
                if used_fallback:
                    answer, input_tokens, output_tokens, finish_reason = _retry_api_call(
                        lambda: _openai_completion(**request_kwargs, chat_template_kwargs=fallback_kwargs)
                    )
                else:
                    answer, input_tokens, output_tokens, finish_reason = _retry_api_call(
                        lambda: _openai_completion(**request_kwargs)
                    )
                    if (not answer or not answer.strip()) and finish_reason == "length" and fallback_kwargs:
                        answer, input_tokens, output_tokens, finish_reason = _retry_api_call(
                            lambda: _openai_completion(**request_kwargs, chat_template_kwargs=fallback_kwargs)
                        )
                        used_fallback = True
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError(
                        f"OpenAI-compatible response contained no answer content (finish_reason={finish_reason!r})"
                    )
                result = {"route": record["route"], "sample_id": record["sample_id"], "prompt": record["prompt"], "system_prompt_sha256": sha256_text(system), "answer": answer, "input_tokens": input_tokens, "output_tokens": output_tokens, "finish_reason": finish_reason, "status": "ok"}
                if used_fallback:
                    result["generation_overrides"] = {"chat_template_kwargs": fallback_kwargs}
            except Exception as exc:
                result = {"route": record["route"], "sample_id": record["sample_id"], "prompt": record["prompt"], "status": "error", "error_type": type(exc).__name__, "error": str(exc) or repr(exc)}
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"); stream.flush()
            progress["completed"] += 1; progress["ok"] += int(result["status"] == "ok"); progress["errors"] += int(result["status"] == "error"); progress["output_tokens"] += int(result.get("output_tokens", 0))
            state = route_progress[record["route"]]; state["completed"] += 1; state["ok"] += int(result["status"] == "ok"); state["errors"] += int(result["status"] == "error")
            state["status"] = "COMPLETED" if state["completed"] == state["expected"] and not state["errors"] else "FAILED" if state["completed"] == state["expected"] else "GENERATING"
            progress["completed_routes"] = [route for route, state in route_progress.items() if state["status"] == "COMPLETED"]; progress["current_route"] = record["route"]; progress["current_sample_id"] = record["sample_id"]; progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat(); _write_json(progress_path, progress)
    result = coverage(manifest, generations, output / "coverage.json", limit_per_route, selected)
    progress.update({"status": "COMPLETED", "completed": result["unique_records"], "ok": result["ok"], "errors": result["errors"], "missing": result["missing"], "finished_at_utc": datetime.now(timezone.utc).isoformat()}); progress["updated_at_utc"] = progress["finished_at_utc"]; _write_json(progress_path, progress)
    return result
