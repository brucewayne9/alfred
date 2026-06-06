# tests/casting/test_persona.py
import importlib
import core.casting.persona as p

def test_build_prompt_includes_brief_and_archetype():
    msg = p._build_prompt(name="Sloan", brief="warm realist, sales-driven", archetype_summary="Art-of-War strategist")
    assert "Sloan" in msg and "sales-driven" in msg and "Art-of-War" in msg

def test_draft_persona_uses_llm(monkeypatch):
    importlib.reload(p)
    captured = {}
    def fake_chat(url, json=None, timeout=None):
        captured["url"] = url; captured["body"] = json
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"message": {"content": "PERSONA: Sloan is a warm-realist host."}}
        return R()
    monkeypatch.setattr(p.requests, "post", fake_chat)
    out = p.draft_persona(name="Sloan", brief="warm realist", archetype_id="strategist")
    assert "warm-realist" in out.persona_prompt
    assert captured["body"]["model"]  # model set
    assert captured["body"]["think"] is False
    assert captured["body"]["stream"] is False
