import importlib


def test_priority_from_seed(tmp_path, monkeypatch):
    f = tmp_path / "waitlist_emails.txt"
    f.write_text("fan1@x.com\nVIP@x.com\n")
    monkeypatch.setenv("FAIRGAME_WAITLIST_FILE", str(f))
    from core.fairgame import waitlist
    importlib.reload(waitlist)
    assert waitlist.is_priority("fan1@x.com") is True
    assert waitlist.is_priority("VIP@X.com") is True  # case-insensitive
    assert waitlist.is_priority("nobody@x.com") is False


def test_priority_no_file(monkeypatch):
    monkeypatch.setenv("FAIRGAME_WAITLIST_FILE", "/nonexistent/path.txt")
    from core.fairgame import waitlist
    importlib.reload(waitlist)
    assert waitlist.is_priority("anyone@x.com") is False
