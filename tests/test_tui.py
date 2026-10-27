from pathlib import Path
import hashlib
import json

import yaml

from calix_telco_bench.tui import target_status


def test_target_status_reads_persisted_generation_evidence(tmp_path: Path) -> None:
    model = tmp_path / "model"; adapter = tmp_path / "adapter"
    model.mkdir(); adapter.mkdir()
    heldout = tmp_path / "heldout"; heldout.mkdir()
    route_file = heldout / "bgp.jsonl"
    route_file.write_text('{"prompt":"test","sample_id":"BGP-1","teacher_routing_key":"bgp"}\n', encoding="utf-8")
    digest = hashlib.sha256(route_file.read_bytes()).hexdigest()
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"schema_version": 1, "routes": [{"route": "bgp", "path": "heldout/bgp.jsonl", "records": 1, "sha256": digest}]}), encoding="utf-8")
    (tmp_path / "prompts.yaml").write_text("schema_version: 1\nprotocol_id: test\nsystem: test\ndecoding: {}\n", encoding="utf-8")
    output = tmp_path / "output"; output.mkdir()
    (output / "generations.jsonl").write_text('{"route":"bgp","sample_id":"BGP-1","status":"ok"}\n', encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "release": {"cpt_base": {"path": str(model)}},
        "benchmark": {"manifest": "MANIFEST.json", "prompts": "prompts.yaml", "output_root": str(tmp_path), "max_new_tokens": 12},
        "targets": {"student": {"adapter": str(adapter), "output": "output", "routes": ["bgp"]}},
    }), encoding="utf-8")
    status = target_status(yaml.safe_load(config_path.read_text(encoding="utf-8")), config_path, "student")
    assert status.phase == "AWAITING JUDGE"
    assert status.succeeded == 1
    assert status.missing == 0


def test_target_status_uses_limited_worker_plan_for_expected_count(tmp_path: Path) -> None:
    model = tmp_path / "model"; adapter = tmp_path / "adapter"
    model.mkdir(); adapter.mkdir()
    heldout = tmp_path / "heldout"; heldout.mkdir()
    route_file = heldout / "bgp.jsonl"
    route_file.write_text(''.join(f'{{"prompt":"test","sample_id":"BGP-{i}","teacher_routing_key":"bgp"}}\\n' for i in range(3)), encoding="utf-8")
    digest = hashlib.sha256(route_file.read_bytes()).hexdigest()
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"schema_version": 1, "routes": [{"route": "bgp", "path": "heldout/bgp.jsonl", "records": 3, "sha256": digest}]}), encoding="utf-8")
    (tmp_path / "prompts.yaml").write_text("schema_version: 1\\nprotocol_id: test\\nsystem: test\\ndecoding: {}\\n", encoding="utf-8")
    output = tmp_path / "output"; output.mkdir()
    (output / "generation-progress.json").write_text(json.dumps({"status": "GENERATING", "expected": 1, "completed": 0}), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"schema_version": 1, "release": {"cpt_base": {"path": str(model)}}, "benchmark": {"manifest": "MANIFEST.json", "prompts": "prompts.yaml", "output_root": str(tmp_path), "max_new_tokens": 12}, "targets": {"student": {"adapter": str(adapter), "output": "output", "routes": ["bgp"]}}}), encoding="utf-8")
    status = target_status(yaml.safe_load(config_path.read_text(encoding="utf-8")), config_path, "student")
    assert status.expected == 1
