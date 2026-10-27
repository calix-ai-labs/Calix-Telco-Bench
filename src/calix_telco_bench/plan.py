"""Config-driven execution of the existing generation contract."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .env import load_runtime_env


def load_run_config(path: Path) -> dict[str, Any]:
    load_runtime_env(path.parent / ".env")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("evaluation config requires schema_version: 1")
    for key in ("benchmark", "targets"):
        if not isinstance(value.get(key), dict):
            raise ValueError(f"evaluation config requires mapping: {key}")
    provider = value.get("provider")
    if not isinstance(value.get("release"), dict) and not (isinstance(provider, dict) and provider.get("kind") == "openai_compatible"):
        raise ValueError("evaluation config requires release, or provider.kind: openai_compatible")
    def expand(item: Any) -> Any:
        if isinstance(item, str):
            expanded = os.path.expandvars(item)
            if "${" in expanded or "$" in expanded:
                raise ValueError(f"unresolved environment variable in config: {item}")
            return expanded
        if isinstance(item, list):
            return [expand(child) for child in item]
        if isinstance(item, dict):
            return {key: expand(child) for key, child in item.items()}
        return item
    return expand(value)


def _path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def resolve_target(config: dict[str, Any], config_path: Path, target_name: str) -> dict[str, Any]:
    benchmark = config["benchmark"]
    target = config["targets"].get(target_name)
    if not isinstance(target, dict):
        raise ValueError(f"unknown evaluation target: {target_name}")
    manifest = _path(config_path, str(benchmark["manifest"]))
    prompts = _path(config_path, str(benchmark["prompts"]))
    output_root = _path(config_path, str(benchmark["output_root"]))
    output = output_root / str(target["output"])
    routes = target.get("routes")
    if routes is not None and (not isinstance(routes, list) or not all(isinstance(route, str) for route in routes)):
        raise ValueError(f"target {target_name!r} routes must be a list of strings or null")
    for label, path in (("manifest", manifest), ("prompts", prompts)):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    provider = config.get("provider")
    if isinstance(provider, dict) and provider.get("kind") == "openai_compatible":
        base_url, model = provider.get("base_url"), target.get("model")
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("provider.base_url must be an HTTP(S) URL")
        if not isinstance(model, str) or not model:
            raise ValueError(f"remote target {target_name!r} requires model")
        models_endpoint = provider.get("models_endpoint", "/models")
        chat_completions_endpoint = provider.get("chat_completions_endpoint", "/chat/completions")
        if not isinstance(models_endpoint, str) or not models_endpoint.startswith("/"):
            raise ValueError("provider.models_endpoint must be an absolute path beginning with '/'")
        if not isinstance(chat_completions_endpoint, str) or not chat_completions_endpoint.startswith("/"):
            raise ValueError("provider.chat_completions_endpoint must be an absolute path beginning with '/'")
        generation = config.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("generation must be a mapping")
        empty_content_fallback = generation.get("empty_content_fallback")
        if empty_content_fallback is not None:
            if not isinstance(empty_content_fallback, dict) or not isinstance(empty_content_fallback.get("chat_template_kwargs"), dict):
                raise ValueError("generation.empty_content_fallback must contain a chat_template_kwargs mapping")
            backup_name = empty_content_fallback.get("prior_error_backup")
            if backup_name is not None and (not isinstance(backup_name, str) or not backup_name):
                raise ValueError("generation.empty_content_fallback.prior_error_backup must be a non-empty filename")
        return {
            "backend": "openai_compatible", "manifest": manifest, "prompts": prompts,
            "output": output, "routes": routes, "model": model,
            "device": "remote",
            "base_url": base_url.rstrip("/"),
            "models_endpoint": models_endpoint,
            "chat_completions_endpoint": chat_completions_endpoint,
            "api_key_env": target.get("api_key_env", provider.get("api_key_env")),
            "allow_missing_api_key": bool(provider.get("allow_missing_api_key", False)),
            "request_timeout_seconds": int(generation.get("request_timeout_seconds", 180)),
            "temperature": float(generation.get("temperature", 0.0)),
            "top_p": float(generation.get("top_p", 1.0)), "seed": generation.get("seed"),
            "max_new_tokens": int(benchmark["max_new_tokens"]),
            "empty_content_fallback": empty_content_fallback,
        }
    release = config["release"]
    base_name = target.get("base_model", "cpt_base")
    if not isinstance(base_name, str) or not base_name:
        raise ValueError(f"target {target_name!r} base_model must be a non-empty release model name")
    base = release.get(base_name)
    if not isinstance(base, dict) or not isinstance(base.get("path"), str):
        raise ValueError(f"release.{base_name}.path is required for target {target_name!r}")
    if not Path(base["path"]).exists():
        raise FileNotFoundError(f"model does not exist: {base['path']}")
    adapter = target.get("adapter")
    if adapter is not None and not Path(adapter).is_dir():
        raise FileNotFoundError(f"adapter does not exist: {adapter}")
    return {
        "manifest": manifest,
        "prompts": prompts,
        "model_path": str(base["path"]),
        "adapter": adapter,
        "model_revision": base.get("model_safetensors_sha256"),
        "adapter_revision": target.get("adapter_revision"),
        "device": str(target.get("device", "cuda:0")),
        "output": output,
        "max_new_tokens": int(benchmark["max_new_tokens"]),
        "local_files_only": bool(benchmark.get("local_files_only", True)),
        "trust_remote_code": bool(benchmark.get("trust_remote_code", False)),
        "routes": routes,
    }
