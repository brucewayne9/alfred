from core.fairgame import holdings

def test_classify_tier_floor_lower_upper():
    assert holdings.classify_tier("C4W", False) == "floor"   # court/letter code
    assert holdings.classify_tier("FLOOR", True) == "floor"   # GA flag
    assert holdings.classify_tier("105", False) == "lower"    # 100s
    assert holdings.classify_tier("224", False) == "upper"    # 200s+

def test_section_status_upper_is_not_ours():
    # upper bowl is never held -> grey
    assert holdings.section_status("224", False) == "not_ours"

def test_section_status_held_is_available_or_sold_deterministic():
    # floor + lower are held; status is available or sold, and STABLE across calls
    s1 = holdings.section_status("105", False)
    s2 = holdings.section_status("105", False)
    assert s1 == s2
    assert s1 in ("available", "sold")
    assert holdings.section_status("C1", False) in ("available", "sold")

def test_overlay_none_when_unmapped(monkeypatch):
    monkeypatch.setattr(holdings.seatmap, "overview", lambda sid: None)
    assert holdings.overlay("show_999") is None

def test_overlay_counts_and_keys(monkeypatch):
    fake = {"sections": [
        {"id":"a","name":"101","ga":False,"bbox":[0,0,1,1],"c":[0,0],"n":10},
        {"id":"b","name":"224","ga":False,"bbox":[0,0,1,1],"c":[0,0],"n":10},
        {"id":"c","name":"C1","ga":False,"bbox":[0,0,1,1],"c":[0,0],"n":10},
    ]}
    monkeypatch.setattr(holdings.seatmap, "overview", lambda sid: fake)
    ov = holdings.overlay("show_1")
    assert set(ov["sections"].keys()) == {"101","224","C1"}
    assert ov["sections"]["224"] == "not_ours"          # upper = grey
    assert ov["sections"]["101"] in ("available","sold")
    assert ov["held"] == ov["available"] + ov["sold"]   # held = available + sold
    assert ov["held"] == 2                               # 101 + C1 held; 224 not
