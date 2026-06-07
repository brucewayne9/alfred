import importlib

cb = importlib.import_module("core.api.rucktalk_content_board")


def test_new_card_defaults():
    c = cb.new_card("the walk is cheaper than therapy, reset before the day")
    assert c["status"] == "to_shoot"
    assert c["polished"] is False
    assert c["raw"].startswith("the walk")
    assert c["title"]  # non-empty short label
    assert c["reel_url"] == ""
    assert c["stats"] is None
    assert "id" in c


def test_shortcode_from_reel_url():
    assert cb.shortcode_from_url("https://www.instagram.com/reel/Cabc123XYZ/") == "Cabc123XYZ"
    assert cb.shortcode_from_url("https://instagram.com/p/DEF456/?igsh=x") == "DEF456"
    assert cb.shortcode_from_url("not a url") is None


def test_lane_extraction():
    bot = importlib.import_module("interfaces.telegram.bot")
    f = bot._extract_brain_dump_lane
    assert f("brain dump radio talk about emotion") == ("radio", "talk about emotion")
    assert f("brain dump social the walk is cheaper") == ("social", "the walk is cheaper")
    assert f("brain dump instagram studio reset") == ("social", "studio reset")
    assert f("brain dump just an idea") == ("none", "just an idea")
    assert f("/dump quick note") == ("none", "quick note")
    assert f("what is on my calendar") is None
