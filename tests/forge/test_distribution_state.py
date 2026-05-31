from core.forge import distribution as dist


def test_roster_defaults_then_set_get(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_ACCOUNTS_PATH", str(tmp_path / "acc.json"))
    accts = dist.get_accounts()
    assert isinstance(accts, list) and len(accts) >= 1
    dist.set_accounts([{"handle": "@rod.daily", "platform": "TikTok", "tier": "burner"}])
    got = dist.get_accounts()
    assert got[0]["handle"] == "@rod.daily" and got[0]["platform"] == "TikTok"


def test_mark_posted_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "d.db"))
    dist.mark_posted("job9:0", True)
    assert dist.posted_map(["job9:0", "job9:1"]) == {"job9:0": True}
