import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "compare_eval_runs.py"
    spec = importlib.util.spec_from_file_location("compare_eval_runs", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_eval_extracts_summary_fields(tmp_path: Path) -> None:
    module = _load_module()
    payload = {
        "policy": "random",
        "checkpoint": "",
        "summary": {
            "mean_reward": 1.25,
            "win_rate": 0.1,
            "illegal_action_rate": 0.02,
            "mean_episode_len": 90.0,
        },
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    out = module.load_eval(p)
    assert out["policy"] == "random"
    assert out["mean_reward"] == 1.25
    assert out["illegal_action_rate"] == 0.02
