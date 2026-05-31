from core.forge.multiply import _make_structural_pool


def test_pool_excludes_flip_when_disabled():
    pool = _make_structural_pool(1080, 1920, allow_flip=False)
    assert pool and all("hflip" not in vf for vf, _ in pool)


def test_pool_includes_flip_by_default():
    pool = _make_structural_pool(1080, 1920)
    assert any("hflip" in vf for vf, _ in pool)
