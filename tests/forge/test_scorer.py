import json

import pytest

from core.forge import scorer


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    return _db


# ── parse_candidates ──────────────────────────────────────────────────────
# Kimi is shaky on strict JSON, so the parser is the robustness heart of the
# scorer: tolerate wrapping, alternate keys, clamp scores, drop garbage.

def test_parse_candidates_reads_a_clean_json_list():
    raw = json.dumps([
        {"start_s": 12.0, "end_s": 38.5, "score": 87, "hook": "I almost quit",
         "emotion": "vulnerable", "reason": "raw admission", "caption": "He almost walked away."},
    ])
    out = scorer.parse_candidates(raw)
    assert len(out) == 1
    c = out[0]
    assert c["start_s"] == 12.0
    assert c["end_s"] == 38.5
    assert c["score"] == 87
    assert c["hook"] == "I almost quit"
    assert c["emotion"] == "vulnerable"


def test_parse_candidates_unwraps_markdown_fenced_json():
    raw = "Here are the clips:\n```json\n[{\"start_s\": 1, \"end_s\": 9, \"score\": 50}]\n```\nHope that helps!"
    out = scorer.parse_candidates(raw)
    assert len(out) == 1
    assert out[0]["start_s"] == 1.0
    assert out[0]["end_s"] == 9.0


def test_parse_candidates_accepts_a_moments_container_object():
    raw = json.dumps({"moments": [{"start_s": 5, "end_s": 20, "score": 70}]})
    out = scorer.parse_candidates(raw)
    assert len(out) == 1
    assert out[0]["score"] == 70


def test_parse_candidates_clamps_score_into_0_100():
    raw = json.dumps([
        {"start_s": 0, "end_s": 5, "score": 140},
        {"start_s": 6, "end_s": 9, "score": -10},
    ])
    out = scorer.parse_candidates(raw)
    assert out[0]["score"] == 100
    assert out[1]["score"] == 0


def test_parse_candidates_drops_entries_missing_required_timestamps():
    raw = json.dumps([
        {"start_s": 0, "end_s": 5, "score": 80},      # good
        {"score": 90, "hook": "no timestamps"},        # drop
        {"start_s": 10, "end_s": 10, "score": 60},     # drop: zero-length
        {"start_s": 30, "end_s": 12, "score": 60},     # drop: end before start
    ])
    out = scorer.parse_candidates(raw)
    assert len(out) == 1
    assert out[0]["start_s"] == 0.0


def test_parse_candidates_defaults_optional_text_fields_to_empty():
    raw = json.dumps([{"start_s": 0, "end_s": 5, "score": 80}])
    out = scorer.parse_candidates(raw)
    assert out[0]["hook"] == ""
    assert out[0]["emotion"] == ""
    assert out[0]["reason"] == ""
    assert out[0]["caption"] == ""


def test_parse_candidates_returns_empty_on_unparseable_garbage():
    assert scorer.parse_candidates("the model refused to answer") == []
    assert scorer.parse_candidates("") == []


# ── build_transcript_text ─────────────────────────────────────────────────

def test_build_transcript_text_emits_one_timestamped_line_per_segment():
    segs = [
        {"start_s": 0.0, "end_s": 4.2, "text": "Welcome back", "speaker": "A"},
        {"start_s": 4.2, "end_s": 9.9, "text": "I almost quit music", "speaker": "B"},
    ]
    text = scorer.build_transcript_text(segs)
    lines = text.splitlines()
    assert len(lines) == 2
    assert lines[0] == "[0.0] (A) Welcome back"
    assert lines[1] == "[4.2] (B) I almost quit music"


def test_build_transcript_text_omits_speaker_tag_when_absent():
    segs = [{"start_s": 1.0, "end_s": 2.0, "text": "hi", "speaker": None}]
    assert scorer.build_transcript_text(segs) == "[1.0] hi"


# ── snap_to_words ─────────────────────────────────────────────────────────

def _segs_with_words():
    return [{
        "start_s": 0.0, "end_s": 6.0, "text": "I almost quit music last year",
        "words": [
            {"word": "I", "start": 0.0, "end": 0.4},
            {"word": "almost", "start": 0.5, "end": 1.0},
            {"word": "quit", "start": 1.1, "end": 1.6},
            {"word": "music", "start": 1.7, "end": 2.3},
            {"word": "last", "start": 2.4, "end": 2.8},
            {"word": "year", "start": 2.9, "end": 3.4},
        ],
    }]


def test_snap_to_words_moves_bounds_to_nearest_word_edges():
    segs = _segs_with_words()
    # start 0.45 is closest to "almost".start (0.5); end 2.35 closest to "music".end (2.3)
    snapped = scorer.snap_to_words({"start_s": 0.45, "end_s": 2.35, "score": 80}, segs)
    assert snapped["start_s"] == 0.5
    assert snapped["end_s"] == 2.3
    assert snapped["score"] == 80  # other fields preserved


def test_snap_to_words_is_a_noop_without_word_timings():
    segs = [{"start_s": 0.0, "end_s": 6.0, "text": "no words here", "words": []}]
    c = {"start_s": 1.23, "end_s": 4.56, "score": 50}
    assert scorer.snap_to_words(c, segs) == c


# ── save_candidates / get_candidates ──────────────────────────────────────

def test_save_and_get_candidates_round_trips_ordered_by_score_desc(db):
    cands = [
        {"start_s": 0, "end_s": 5, "score": 60, "hook": "b", "emotion": "funny",
         "reason": "r2", "caption": "c2"},
        {"start_s": 10, "end_s": 20, "score": 91, "hook": "a", "emotion": "vulnerable",
         "reason": "r1", "caption": "c1"},
    ]
    n = scorer.save_candidates("src1", cands, judge_model="kimi-k2.6:cloud", now=1000)
    assert n == 2
    got = scorer.get_candidates("src1")
    assert [c["score"] for c in got] == [91, 60]   # highest first
    assert got[0]["hook"] == "a"
    assert got[0]["judge_model"] == "kimi-k2.6:cloud"
    assert got[0]["rendered"] == 0
    assert "id" in got[0]


def test_save_candidates_replaces_prior_scores_for_the_source(db):
    scorer.save_candidates("src1", [{"start_s": 0, "end_s": 5, "score": 50}], now=1)
    scorer.save_candidates("src1", [{"start_s": 0, "end_s": 5, "score": 80}], now=2)
    got = scorer.get_candidates("src1")
    assert len(got) == 1
    assert got[0]["score"] == 80


def test_get_candidates_returns_empty_for_unknown_source(db):
    assert scorer.get_candidates("nope") == []


# ── mark_rendered / mark_posted / get_candidate (Phase 2 instrumentation) ──

def test_mark_rendered_flags_a_candidate(db):
    scorer.save_candidates("src1", [{"start_s": 0, "end_s": 5, "score": 88}], now=1)
    cid = scorer.get_candidates("src1")[0]["id"]
    assert scorer.get_candidate(cid)["rendered"] == 0
    scorer.mark_rendered(cid)
    assert scorer.get_candidate(cid)["rendered"] == 1
    assert scorer.get_candidate(cid)["posted"] == 0


def test_mark_posted_implies_rendered(db):
    scorer.save_candidates("src1", [{"start_s": 0, "end_s": 5, "score": 88}], now=1)
    cid = scorer.get_candidates("src1")[0]["id"]
    scorer.mark_posted(cid)
    c = scorer.get_candidate(cid)
    assert c["posted"] == 1
    assert c["rendered"] == 1   # posting something means it was rendered


def test_get_candidate_returns_none_for_unknown_id(db):
    assert scorer.get_candidate(999999) is None


def test_mark_rendered_is_a_noop_for_unknown_id(db):
    scorer.mark_rendered(999999)  # must not raise


# ── score_source (orchestration, judge injected) ──────────────────────────

def _seed_source(db, source_id="src1"):
    with db._conn() as c:
        c.execute(
            "INSERT INTO sources (id, kind, spec, status, created_at, updated_at) "
            "VALUES (?, 'url', 'x', 'done', 1, 1)", (source_id,))
        c.execute(
            "INSERT INTO transcript_segments (source_id, seq, start_s, end_s, text, speaker, words) "
            "VALUES (?, 0, 0.0, 6.0, 'I almost quit music last year', 'B', ?)",
            (source_id, json.dumps([
                {"word": "I", "start": 0.0, "end": 0.4},
                {"word": "almost", "start": 0.5, "end": 1.0},
                {"word": "quit", "start": 1.1, "end": 1.6},
                {"word": "music", "start": 1.7, "end": 2.3},
            ])))


def test_score_source_runs_judge_parses_snaps_and_stores(db):
    _seed_source(db)
    seen = {}

    def fake_judge(messages, model):
        seen["model"] = model
        seen["prompt"] = messages[-1]["content"]
        return json.dumps([
            {"start_s": 0.45, "end_s": 2.35, "score": 88, "hook": "I almost quit",
             "emotion": "vulnerable", "reason": "raw", "caption": "He almost walked away."}
        ])

    out = scorer.score_source("src1", chat_fn=fake_judge, now=1000)
    assert len(out) == 1
    assert out[0]["score"] == 88
    assert out[0]["start_s"] == 0.5   # snapped to "almost".start
    assert out[0]["end_s"] == 2.3     # snapped to "music".end
    # judge saw the real transcript text
    assert "I almost quit music last year" in seen["prompt"]
    # persisted
    assert scorer.get_candidates("src1")[0]["score"] == 88


def test_score_source_skips_judge_when_no_segments(db):
    _seed_source(db, "empty")
    with db._conn() as c:
        c.execute("DELETE FROM transcript_segments WHERE source_id = 'empty'")
    called = {"n": 0}

    def fake_judge(messages, model):
        called["n"] += 1
        return "[]"

    out = scorer.score_source("empty", chat_fn=fake_judge)
    assert out == []
    assert called["n"] == 0  # never bothered the judge


def test_score_source_respects_judge_model_env(db, monkeypatch):
    _seed_source(db)
    monkeypatch.setenv("FORGE_JUDGE_MODEL", "glm-5.1:cloud")
    seen = {}

    def fake_judge(messages, model):
        seen["model"] = model
        return "[]"

    scorer.score_source("src1", chat_fn=fake_judge)
    assert seen["model"] == "glm-5.1:cloud"
