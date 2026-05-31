from core.forge.remix import build_remixes


def test_n1_returns_single_unchanged_copy():
    p = {"caption": "lonely", "variations": 6}
    out = build_remixes(p, 1)
    assert len(out) == 1
    assert out[0]["caption"] == "lonely"
    assert out[0] is not p


def test_n_gives_distinct_vessel_prompts():
    out = build_remixes({"caption": "lonely night"}, 4)
    assert len(out) == 4
    vps = [o["vessel_prompt"] for o in out]
    assert len(set(vps)) == 4
    assert all("lonely night" in v for v in vps)
    assert all(o["remix_index"] == i for i, o in enumerate(out))


def test_caption_text_is_never_mutated():
    out = build_remixes({"caption": "I ain't mad"}, 3)
    assert all(o["caption"] == "I ain't mad" for o in out)


def test_clamps_to_at_least_one():
    assert len(build_remixes({"caption": "x"}, 0)) == 1
