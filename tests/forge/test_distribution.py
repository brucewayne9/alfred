from core.forge.distribution import build_caption, assign_posts, PLATFORMS


def test_caption_includes_hook_and_platform_tag():
    cap = build_caption("still waiting on the album", "TikTok")
    assert "still waiting on the album" in cap
    assert "#fyp" in cap.lower() or "#foryou" in cap.lower()
    assert "rod wave" in cap.lower() or "#rodwave" in cap.lower()


def test_caption_platform_specific_tag():
    assert "#shorts" in build_caption("x", "YouTube Shorts").lower()
    assert "#reels" in build_caption("x", "Instagram Reels").lower()


def test_assign_round_robins_files_across_accounts_with_stagger():
    files = [{"name": f"v{i}.mp4", "path": f"p/v{i}.mp4"} for i in range(4)]
    accounts = [
        {"handle": "@a", "platform": "TikTok"},
        {"handle": "@b", "platform": "Instagram Reels"},
    ]
    posts = assign_posts("job1", files, accounts, caption="hook", stagger_minutes=20)
    assert len(posts) == 4
    assert [p["account"] for p in posts] == ["@a", "@b", "@a", "@b"]
    assert [p["platform"] for p in posts] == ["TikTok", "Instagram Reels", "TikTok", "Instagram Reels"]
    assert [p["stagger_minutes"] for p in posts] == [0, 20, 40, 60]
    assert all(p["file_path"].startswith("p/") for p in posts)
    assert all(p["caption"] and p["post_id"] for p in posts)
    assert len({p["post_id"] for p in posts}) == 4


def test_assign_empty_accounts_returns_empty():
    assert assign_posts("j", [{"name": "v.mp4", "path": "p"}], [], caption="h") == []
