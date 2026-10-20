from pathlib import Path
import hashlib
import yaml
import pytest
from network_llm_eval import artifacts
from network_llm_eval.artifacts import verify_target_artifacts
from network_llm_eval.contracts import load_manifest, validate_manifest
from network_llm_eval.env import load_judge_env
from network_llm_eval.generation import _render, load_prompt_protocol, system_prompt
from network_llm_eval.plan import load_run_config, resolve_target

def test_release_manifest_is_valid() -> None:
    root = Path(__file__).parents[1]
    assert validate_manifest(root / "datasets" / "MANIFEST.json") == {"routes": 14, "prompts": 2850}


def test_generation_manifest_contains_routes() -> None:
    root = Path(__file__).parents[1]
    assert len(load_manifest(root / "datasets" / "MANIFEST.json")["routes"]) == 14


def test_prompt_protocols_express_distinct_conditions() -> None:
    root = Path(__file__).parents[1]
    neutral = load_prompt_protocol(root / "prompts" / "generation-neutral-v1.yaml")
    diagnostic = load_prompt_protocol(root / "prompts" / "generation-v1.yaml")
    route_expert = load_prompt_protocol(root / "prompts" / "generation-route-expert-v2.yaml")
    assert neutral["condition"] == "neutral_deployment_like"
    assert diagnostic["condition"] == "route_conditioned_diagnostic"
    assert system_prompt(neutral, "bgp") == system_prompt(neutral, "ospf")
    assert system_prompt(diagnostic, "bgp") != system_prompt(diagnostic, "ospf")
    assert route_expert["condition"] == "route_conditioned_expert_diagnostic"
    assert set(route_expert["routes"]) == {route["route"] for route in load_manifest(root / "datasets" / "MANIFEST.json")["routes"]}
    assert "IP-multicast" in system_prompt(route_expert, "multicast")


def test_renderer_extracts_input_ids_from_batch_encoding() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            assert kwargs["return_tensors"] == "pt"
            return {"input_ids": "token-tensor"}
    assert _render(FakeTokenizer(), "system", "prompt") == "token-tensor"


def test_run_config_resolves_a_route_scoped_target(tmp_path: Path) -> None:
    model = tmp_path / "model"
    adapter = tmp_path / "adapter"
    model.mkdir(); adapter.mkdir()
    manifest = tmp_path / "MANIFEST.json"
    prompts = tmp_path / "prompts.yaml"
    manifest.write_text('{"schema_version": 1, "routes": []}\n', encoding="utf-8")
    prompts.write_text("schema_version: 1\nprotocol_id: test\nsystem: test\ndecoding: {}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "release": {"cpt_base": {"path": str(model), "model_safetensors_sha256": "model-sha"}},
        "benchmark": {"manifest": "MANIFEST.json", "prompts": "prompts.yaml", "output_root": str(tmp_path / "outputs"), "max_new_tokens": 12},
        "targets": {"teacher": {"adapter": str(adapter), "adapter_revision": "adapter-sha", "device": "cuda:2", "output": "teacher", "routes": ["bgp"]}},
    }), encoding="utf-8")
    plan = resolve_target(load_run_config(config_path), config_path, "teacher")
    assert plan["model_path"] == str(model)
    assert plan["routes"] == ["bgp"]
    assert plan["output"] == tmp_path / "outputs" / "teacher"


def test_run_config_supports_a_target_specific_release_base(tmp_path: Path) -> None:
    raw = tmp_path / "raw"; cpt = tmp_path / "cpt"
    raw.mkdir(); cpt.mkdir()
    manifest = tmp_path / "MANIFEST.json"; manifest.write_text('{"schema_version": 1, "routes": []}\n', encoding="utf-8")
    prompts = tmp_path / "prompts.yaml"; prompts.write_text("schema_version: 1\nprotocol_id: test\nsystem: test\ndecoding: {}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "release": {"cpt_base": {"path": str(cpt)}, "raw_base": {"path": str(raw)}},
        "benchmark": {"manifest": "MANIFEST.json", "prompts": "prompts.yaml", "output_root": str(tmp_path / "outputs"), "max_new_tokens": 12},
        "targets": {"raw": {"base_model": "raw_base", "output": "raw"}},
    }), encoding="utf-8")
    assert resolve_target(load_run_config(config_path), config_path, "raw")["model_path"] == str(raw)


def test_artifact_digest_preflight_uses_and_invalidates_its_cache(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model"; adapter = tmp_path / "adapter"
    model.mkdir(); adapter.mkdir()
    weights = model / "model.safetensors"; weights.write_bytes(b"base-v1")
    adapter_weights = adapter / "adapter_model.safetensors"; adapter_weights.write_bytes(b"adapter-v1")
    config_path = tmp_path / "config.yaml"
    config = {
        "release": {"cpt_base": {"path": str(model), "model_safetensors_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}},
        "targets": {"student": {"adapter": str(adapter), "adapter_revision": hashlib.sha256(adapter_weights.read_bytes()).hexdigest()}},
    }
    calls: list[Path] = []
    real_sha256 = artifacts.sha256
    monkeypatch.setattr(artifacts, "sha256", lambda path: (calls.append(path), real_sha256(path))[1])

    verify_target_artifacts(config, config_path, "student")
    verify_target_artifacts(config, config_path, "student")
    assert calls == [weights, adapter_weights]

    adapter_weights.write_bytes(b"adapter-v2")
    config["targets"]["student"]["adapter_revision"] = real_sha256(adapter_weights)
    verify_target_artifacts(config, config_path, "student")
    assert calls == [weights, adapter_weights, adapter_weights]


def test_artifact_digest_preflight_rejects_mismatch(tmp_path: Path) -> None:
    model = tmp_path / "model"; model.mkdir()
    (model / "model.safetensors").write_bytes(b"not-the-expected-model")
    config = {
        "release": {"cpt_base": {"path": str(model), "model_safetensors_sha256": "0" * 64}},
        "targets": {"base": {}},
    }
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_target_artifacts(config, tmp_path / "config.yaml", "base")


def test_judge_environment_loads_identity_without_exposing_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NETWORK_LLM_EVAL_JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("NETWORK_LLM_EVAL_JUDGE_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NETWORK_LLM_EVAL_JUDGE_PROVIDER=test-provider\n"
        "NETWORK_LLM_EVAL_JUDGE_MODEL=test-model\n"
        "ANTHROPIC_API_KEY=must-not-be-returned\n",
        encoding="utf-8",
    )
    judge = load_judge_env(env_file)
    assert judge["provider"] == "test-provider"
    assert judge["model"] == "test-model"
    assert "ANTHROPIC_API_KEY" not in judge
