from pathlib import Path

from network_llm_eval import control


def test_stop_pipeline_still_stops_children_when_coordinator_is_not_running(
    tmp_path: Path, monkeypatch,
) -> None:
    config = {"targets": {"student": {}}}
    current = {
        "output": str(tmp_path), "log": str(tmp_path / "pipeline-all.log"),
        "pid": 123, "running": False, "reason": "cannot verify managed process identity",
        "control": {"pid": 123},
    }
    monkeypatch.setattr(control, "pipeline_status", lambda *_: current)
    monkeypatch.setattr(control, "_judge_all_root", lambda *_: tmp_path)
    monkeypatch.setattr(
        control, "stop", lambda *_: {"target": "student", "stopped": True, "signal": "SIGTERM"},
    )
    monkeypatch.setattr(control, "stop_judge_all", lambda *_: {"stopped": False})

    result = control.stop_pipeline(config, tmp_path / "config.yaml")

    assert result["stopped"] is True
    assert result["workers"]["student"]["stopped"] is True
