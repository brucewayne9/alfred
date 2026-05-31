from core.forge import clips


def test_url_spec_builds_direct_target():
    target, kind = clips.resolve_source("https://www.youtube.com/watch?v=abc")
    assert kind == "url"
    assert target == "https://www.youtube.com/watch?v=abc"


def test_search_prefix_builds_ytsearch():
    target, kind = clips.resolve_source("search:lonely night rain")
    assert kind == "search"
    assert target == "ytsearch3:lonely night rain"


def test_bare_phrase_treated_as_search():
    target, kind = clips.resolve_source("cinematic city dusk")
    assert kind == "search"
    assert target.startswith("ytsearch3:")


def test_ytdlp_cmd_requests_mp4_into_outdir(tmp_path):
    cmd = clips.ytdlp_cmd("ytsearch3:rain", tmp_path)
    assert cmd[0] == "yt-dlp"
    assert "ytsearch3:rain" in cmd
    assert any("-o" == c for c in cmd)
    assert any(str(tmp_path) in c for c in cmd)
    assert "mp4" in " ".join(cmd)
