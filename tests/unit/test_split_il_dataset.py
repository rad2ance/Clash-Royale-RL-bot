import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "split_il_dataset.py"
    spec = importlib.util.spec_from_file_location("split_il_dataset", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_assign_split_is_deterministic() -> None:
    module = _load_module()
    keys = [f"k{i}" for i in range(20)]
    a = module.assign_split(keys, train_ratio=0.7, val_ratio=0.2, seed=123)
    b = module.assign_split(keys, train_ratio=0.7, val_ratio=0.2, seed=123)
    assert a == b


def test_assign_split_all_keys_assigned_once() -> None:
    module = _load_module()
    keys = [f"k{i}" for i in range(13)]
    out = module.assign_split(keys, train_ratio=0.6, val_ratio=0.2, seed=1)
    assert set(out.keys()) == set(keys)
    splits = list(out.values())
    assert all(s in {"train", "val", "test"} for s in splits)
