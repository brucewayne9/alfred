"""Tests for the _topic_clip_handler registered in core.forge.handlers.

All external I/O is monkeypatched so tests run without ffmpeg, ComfyUI, or
Nextcloud. Confirms:
  - result shape: format, variant_count, variations_each, delivered_dirs
  - allow_flip=False used (via multiply call signature capture)
  - variant_count=1 -> exactly one variant rendered
  - "topic_clip" is in the handler registry after register_default_handlers()
"""
import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def stub_chromadb(monkeypatch):
    """Keep chromadb out of the import chain (same pattern as other forge tests)."""
    mod = types.ModuleType("chromadb")
    mod.Client = object
    mod.PersistentClient = object
    sys.modules.setdefault("chromadb", mod)


@pytest.fixture()
def tmp_forge_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import jobs
    jobs._HANDLERS.clear()
    return tmp_path


@pytest.fixture()
def three_segments():
    """Three segments totalling 30 s — pass enforce_duration (min 10 s)."""
    return [
        {"start_s": 0.0,  "end_s": 10.0, "text": "First sentence here.", "speaker": "A", "score": 0.80},
        {"start_s": 10.0, "end_s": 20.0, "text": "Second sentence.", "speaker": "A", "score": 0.75},
        {"start_s": 20.0, "end_s": 30.0, "text": "Third sentence.", "speaker": "B", "score": 0.72},
    ]


# ---------------------------------------------------------------------------
# Helpers to track calls without touching real IO
# ---------------------------------------------------------------------------


def _make_stub_render(tmp_path):
    """Return a render() stub that writes a tiny placeholder .mp4 file."""
    def _render(params, out_path):
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"STUB_MASTER")
        return p
    return _render


def _noop_deliver(local_path, subfolder, filename=None):
    return "Content/Mainstay-RodWave/stub"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTopicClipHandlerResult:
    """Handler returns the correct result dict shape."""

    def test_result_shape_variant_count_3(self, tmp_forge_db, three_segments, monkeypatch):
        """variant_count=3 => three variants, result has all required keys."""
        # Monkeypatch render, multiply, delivery.deliver
        import core.forge.renderers.topic_clip as tc_mod
        import core.forge.multiply as multiply_mod
        import core.forge.delivery as delivery_mod

        stub_render = _make_stub_render(tmp_forge_db)
        monkeypatch.setattr(tc_mod, "render", stub_render)

        multiply_calls = []
        def stub_multiply(master, count, out_dir, *, base_name="variant", allow_flip=True):
            multiply_calls.append({"master": master, "count": count, "allow_flip": allow_flip})
            return []  # no stealth copies for simplicity
        monkeypatch.setattr(multiply_mod, "multiply", stub_multiply)

        deliver_calls = []
        def stub_deliver(local_path, subfolder, filename=None):
            deliver_calls.append({"path": local_path, "subfolder": subfolder})
            return "Content/Mainstay-RodWave/" + subfolder
        monkeypatch.setattr(delivery_mod, "deliver", stub_deliver)

        from core.forge.handlers import _topic_clip_handler
        result = _topic_clip_handler({
            "source_id": "test-src",
            "segments": three_segments,
            "variant_count": 3,
            "variations": 0,  # no stealth so delivered=3 (one master per variant)
        })

        assert result["format"] == "topic_clip"
        assert result["variant_count"] == 3
        assert "variations_each" in result
        assert "delivered_dirs" in result
        assert isinstance(result["delivered_dirs"], list)
        # 3 master delivers (no stealth copies)
        assert result["delivered"] == 3
        assert len(result["delivered_dirs"]) == 3

    def test_allow_flip_is_false(self, tmp_forge_db, three_segments, monkeypatch):
        """multiply() is always called with allow_flip=False."""
        import core.forge.renderers.topic_clip as tc_mod
        import core.forge.multiply as multiply_mod
        import core.forge.delivery as delivery_mod

        monkeypatch.setattr(tc_mod, "render", _make_stub_render(tmp_forge_db))

        flip_values = []
        def stub_multiply(master, count, out_dir, *, base_name="variant", allow_flip=True):
            flip_values.append(allow_flip)
            return []
        monkeypatch.setattr(multiply_mod, "multiply", stub_multiply)

        monkeypatch.setattr(delivery_mod, "deliver", _noop_deliver)

        from core.forge.handlers import _topic_clip_handler
        _topic_clip_handler({
            "source_id": "test-src",
            "segments": three_segments,
            "variant_count": 2,
            "variations": 2,
        })

        assert flip_values, "multiply was never called"
        assert all(v is False for v in flip_values), f"allow_flip must always be False; got {flip_values}"

    def test_variant_count_1(self, tmp_forge_db, three_segments, monkeypatch):
        """With variant_count=1, exactly one render and one deliver call."""
        import core.forge.renderers.topic_clip as tc_mod
        import core.forge.multiply as multiply_mod
        import core.forge.delivery as delivery_mod

        render_calls = []
        def counting_render(params, out_path):
            render_calls.append(params.get("variant_index", 0))
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"STUB")
            return p
        monkeypatch.setattr(tc_mod, "render", counting_render)
        monkeypatch.setattr(multiply_mod, "multiply", lambda *a, **kw: [])
        monkeypatch.setattr(delivery_mod, "deliver", _noop_deliver)

        from core.forge.handlers import _topic_clip_handler
        result = _topic_clip_handler({
            "source_id": "test-src",
            "segments": three_segments,
            "variant_count": 1,
            "variations": 0,
        })

        assert result["variant_count"] == 1
        assert len(render_calls) == 1
        assert render_calls[0] == 0  # variant_index=0

    def test_variant_count_clamped_to_10(self, tmp_forge_db, three_segments, monkeypatch):
        """variant_count is clamped to max 10."""
        import core.forge.renderers.topic_clip as tc_mod
        import core.forge.multiply as multiply_mod
        import core.forge.delivery as delivery_mod

        monkeypatch.setattr(tc_mod, "render", _make_stub_render(tmp_forge_db))
        monkeypatch.setattr(multiply_mod, "multiply", lambda *a, **kw: [])
        monkeypatch.setattr(delivery_mod, "deliver", _noop_deliver)

        from core.forge.handlers import _topic_clip_handler
        result = _topic_clip_handler({
            "source_id": "test-src",
            "segments": three_segments,
            "variant_count": 999,
            "variations": 0,
        })
        assert result["variant_count"] <= 10


class TestHandlerRegistry:
    """topic_clip is present in the handler registry after register_default_handlers()."""

    def test_topic_clip_registered(self, tmp_forge_db, monkeypatch):
        """After register_default_handlers(), 'topic_clip' is in the registry."""
        # Stub heavy imports so register_default_handlers doesn't trigger GPU init
        import core.forge.jobs as jobs_mod
        jobs_mod._HANDLERS.clear()

        # Lazy imports inside handlers are fine; we just confirm the key is present
        from core.forge import handlers
        handlers.register_default_handlers()

        from core.forge import jobs
        assert "topic_clip" in jobs._HANDLERS, (
            "topic_clip handler not registered — add "
            'forge_jobs.register_handler("topic_clip", _topic_clip_handler) '
            "to register_default_handlers()"
        )
