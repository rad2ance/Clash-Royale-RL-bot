import importlib.util
import sys
from pathlib import Path

import torch


def _load_train_bc_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "train_bc.py"
    spec = importlib.util.spec_from_file_location("train_bc", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_apply_action_mask_blocks_illegal_logits() -> None:
    module = _load_train_bc_module()
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    masks = torch.tensor([[True, False, True]], dtype=torch.bool)
    out = module.apply_action_mask(logits, masks)
    assert out[0, 0].item() == 1.0
    assert out[0, 2].item() == 3.0
    assert out[0, 1].item() < -1e8
