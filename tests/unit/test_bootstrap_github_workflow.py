import importlib.util
from pathlib import Path


def _load_bootstrap_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_github_workflow.py"
    spec = importlib.util.spec_from_file_location("bootstrap_github_workflow", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_issue_title_collapses_spaces_and_case() -> None:
    module = _load_bootstrap_module()
    a = module.normalize_issue_title("  Simulator: add river constraints  ")
    b = module.normalize_issue_title("simulator: ADD   river constraints")
    assert a == b
