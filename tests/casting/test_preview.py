# tests/casting/test_preview.py
import importlib
import core.casting.preview as pv

def test_preview_calls_qwen_with_neutral_voice(monkeypatch, tmp_path):
    importlib.reload(pv)
    calls = {}
    def fake_get(url, params=None, timeout=None):
        calls["url"] = url; calls["params"] = params
        class R:
            status_code = 200
            content = b"RIFFfakewav"
            def raise_for_status(self): pass
        return R()
    monkeypatch.setattr(pv.requests, "get", fake_get)
    out_path = str(tmp_path / "preview.wav")
    result = pv.render_preview(voice_name="cc7_neutral",
                               line="Good morning, this is a test.", out_path=out_path)
    assert result == out_path
    assert "synthesize_speech" in calls["url"]
    assert calls["params"]["text"] == "Good morning, this is a test."
    assert calls["params"]["voice"] == "cc7_neutral"
    with open(out_path, "rb") as fh:
        assert fh.read().startswith(b"RIFF")
