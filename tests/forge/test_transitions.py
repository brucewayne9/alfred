"""M1 transitions — catalog resolution and Auto selection (pure logic)."""
from core.forge import transitions


def test_resolve_dissolve_maps_to_xfade_fade():
    spec = transitions.resolve("dissolve")
    assert spec["xfade"] == "fade"


def test_resolve_cut_has_no_xfade():
    # Cut routes to the hard-cut concat path, so it carries no xfade transition.
    spec = transitions.resolve("cut")
    assert spec["xfade"] is None


def test_resolve_unknown_key_falls_back_to_dissolve():
    spec = transitions.resolve("does-not-exist")
    assert spec["key"] == "dissolve"
    assert spec["xfade"] == "fade"


def test_directional_transition_flagged_directional():
    assert transitions.resolve("whip")["directional"] is True
    assert transitions.resolve("dissolve")["directional"] is False


def test_pick_auto_defaults_to_dissolve():
    assert transitions.pick_auto({}) == "dissolve"


def test_pick_auto_high_energy_picks_whip():
    assert transitions.pick_auto({"energy": "high"}) == "whip"


def test_pick_auto_never_returns_flashy_types():
    # Auto must stay tasteful — flashy transitions are deliberate opt-ins only.
    for params in ({}, {"energy": "high"}, {"energy": "low"}, {"bpm": 140}):
        assert transitions.pick_auto(params) not in ("flash", "glitch")


def test_menu_lists_auto_first():
    keys = [entry["key"] for entry in transitions.menu()]
    assert keys[0] == "auto"
    assert "dissolve" in keys
    assert "cut" in keys


def test_render_spec_cut_is_hard_cut():
    spec = transitions.render_spec({"transition": "cut"})
    assert spec["hard_cut"] is True
    assert spec["xfade"] is None


def test_render_spec_concrete_transition():
    spec = transitions.render_spec({"transition": "whip"})
    assert spec["xfade"] == "slideleft"
    assert spec["directional"] is True
    assert spec["hard_cut"] is False


def test_render_spec_defaults_to_dissolve():
    assert transitions.render_spec({})["key"] == "dissolve"


def test_render_spec_auto_resolves_via_pick_auto():
    assert transitions.render_spec({"transition": "auto"})["key"] == "dissolve"
    assert transitions.render_spec({"transition": "auto", "energy": "high"})["key"] == "whip"
