from core.forge.renderers.kinetic_lyric import build_karaoke_lines

WORDS = [
    {"word": "21,", "start": 0.0, "end": 0.9}, {"word": "when", "start": 1.16, "end": 1.3},
    {"word": "my", "start": 1.3, "end": 1.48}, {"word": "life", "start": 1.48, "end": 1.78},
    {"word": "22,", "start": 2.74, "end": 3.18}, {"word": "many", "start": 3.48, "end": 3.76},
    {"word": "things", "start": 3.76, "end": 4.14},
]


def test_groups_break_on_age_markers_and_upper_strips_punct():
    lines = build_karaoke_lines(WORDS, fps=30)
    assert lines[0][0] == {"text": "21", "startFrame": 0, "endFrame": 27}
    assert any(line[0]["text"] == "22" for line in lines)
    assert all(w["text"] == w["text"].upper() for line in lines for w in line)
    assert all(not w["text"].endswith((",", ".", "?", "!")) for line in lines for w in line)


def test_frames_are_seconds_times_fps():
    lines = build_karaoke_lines([{"word": "hi", "start": 2.0, "end": 2.5}], fps=30)
    assert lines[0][0]["startFrame"] == 60 and lines[0][0]["endFrame"] == 75


def test_caps_line_length():
    words = [{"word": f"w{i}", "start": i * 0.2, "end": i * 0.2 + 0.1} for i in range(10)]
    lines = build_karaoke_lines(words, fps=30, max_words=7)
    assert all(len(line) <= 7 for line in lines)
