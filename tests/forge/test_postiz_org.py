import pytest
from pathlib import Path
from core.forge import postiz_client as pz


@pytest.fixture(autouse=True)
def _no_env_file(monkeypatch, tmp_path):
    # postiz_client._env reads config/.env first, then os.environ. Point the
    # env file at a nonexistent path so these tests exercise the per-org env-var
    # selection deterministically (config/.env already carries the real keys).
    monkeypatch.setattr(pz, "_ENV_FILE", tmp_path / "nonexistent.env")


def test_key_for_org_selects_per_org_env(monkeypatch):
    monkeypatch.setenv("POSTIZ_MAINSTAY_API_KEY", "MAIN-KEY")
    monkeypatch.setenv("POSTIZ_RUCKTALK_API_KEY", "RUCK-KEY")
    assert pz.key_for_org("mainstay") == "MAIN-KEY"
    assert pz.key_for_org("rucktalk") == "RUCK-KEY"


def test_unknown_org_has_no_key(monkeypatch):
    monkeypatch.delenv("POSTIZ_GROUNDRUSH_API_KEY", raising=False)
    assert pz.key_for_org("groundrush") is None
