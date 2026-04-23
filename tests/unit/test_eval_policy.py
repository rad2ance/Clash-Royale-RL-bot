import importlib.util
import sys
from pathlib import Path


def _load_eval_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "eval_policy.py"
    spec = importlib.util.spec_from_file_location("eval_policy", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_outputs_expected_rates() -> None:
    module = _load_eval_module()
    rows = [
        module.EpisodeEval(episode=0, total_reward=1.0, steps=10, illegal_actions=1, win=True, loss=False),
        module.EpisodeEval(episode=1, total_reward=-2.0, steps=20, illegal_actions=3, win=False, loss=True),
        module.EpisodeEval(episode=2, total_reward=0.0, steps=30, illegal_actions=0, win=False, loss=False),
    ]
    out = module.summarize(rows)
    assert out["episodes"] == 3.0
    assert abs(out["mean_reward"] - (-1.0 / 3.0)) < 1e-6
    assert abs(out["mean_episode_len"] - 20.0) < 1e-6
    assert abs(out["illegal_action_rate"] - (4.0 / 60.0)) < 1e-6
    assert abs(out["win_rate"] - (1.0 / 3.0)) < 1e-6
    assert abs(out["loss_rate"] - (1.0 / 3.0)) < 1e-6
    assert abs(out["draw_rate"] - (1.0 / 3.0)) < 1e-6
