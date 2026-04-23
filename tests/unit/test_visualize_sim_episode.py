import csv
import importlib.util
from pathlib import Path

from crbot.sim import CrLikeSimEnv


def _load_visualize_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "visualize_sim_episode.py"
    spec = importlib.util.spec_from_file_location("visualize_sim_episode", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_row_and_csv_roundtrip(tmp_path: Path) -> None:
    module = _load_visualize_module()
    env = CrLikeSimEnv()
    env.reset(seed=0)
    row = module.metric_row(
        step_idx=0,
        action=None,
        reward=None,
        terminated=False,
        truncated=False,
        info=None,
        env=env,
    )
    out = tmp_path / "metrics.csv"
    module.save_metrics_csv(out, [row])

    with out.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["step"] == "0"
    assert rows[0]["action"] == "-1"
