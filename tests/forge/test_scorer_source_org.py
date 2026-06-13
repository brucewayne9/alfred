"""Verify that score_source stamps clip_candidates with the source's org_id.

Seam: score_source accepts a ``chat_fn`` injectable — we supply a fake that
returns one valid JSON candidate, bypassing Ollama entirely. We also seed the
source via ``ingest.create_source`` (so org_id is set in the DB) and insert a
transcript segment so score_source has something to score.
"""
import json

import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    return _db


def test_scored_candidates_inherit_source_org(db, monkeypatch):
    from core.forge import scorer, ingest

    # Create a source belonging to rucktalk.
    sid = ingest.create_source("url", "ruck talk ep", None, org="rucktalk")

    # Insert one transcript segment so score_source has something to process.
    ingest.save_segments(sid, [
        {"seq": 0, "start_s": 0.0, "end_s": 8.0, "text": "I almost quit music",
         "speaker": "A", "words": [
             {"word": "I", "start": 0.0, "end": 0.3},
             {"word": "almost", "start": 0.4, "end": 0.9},
             {"word": "quit", "start": 1.0, "end": 1.5},
             {"word": "music", "start": 1.6, "end": 2.2},
         ]},
    ])

    # Fake judge: returns one valid candidate — no network call.
    def fake_judge(messages, model):
        return json.dumps([
            {"start_s": 0.4, "end_s": 2.0, "score": 85,
             "hook": "I almost quit", "emotion": "vulnerable",
             "reason": "raw admission", "caption": "He almost walked away."},
        ])

    scorer.score_source(sid, chat_fn=fake_judge)

    rows = scorer.get_candidates(sid)
    assert rows, "expected at least one candidate"
    assert all(r["org_id"] == "rucktalk" for r in rows), (
        f"expected org_id='rucktalk' on all rows, got {[r['org_id'] for r in rows]}"
    )
