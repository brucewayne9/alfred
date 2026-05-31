from unittest import mock

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


def test_two_fetches_same_outdir_dont_collide(tmp_path):
    """Regression: each fetch must land in an isolated dir so a second source's
    file can't be skipped as 'already downloaded' (the autonumber-collision bug
    that broke every multi-source montage on the 2nd source onward)."""
    # Simulate yt-dlp always writing the SAME filename into its -o directory.
    def fake_run(cmd, **kw):
        out_idx = cmd.index("-o") + 1
        out_dir = __import__("pathlib").Path(cmd[out_idx]).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "src_00001.mp4").write_bytes(b"fake")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch.object(clips.subprocess, "run", side_effect=fake_run):
        first = clips.fetch_source("https://youtube.com/shorts/AAA", tmp_path)
        second = clips.fetch_source("https://youtube.com/shorts/BBB", tmp_path)

    assert first and second, "both fetches must return a file"
    assert set(first).isdisjoint(set(second)), "fetched paths must be distinct"
