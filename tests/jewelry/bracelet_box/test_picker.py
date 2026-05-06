"""Pick algorithm tests."""
from __future__ import annotations
import random
import pytest
from collections import Counter

from core.jewelry.bracelet_box import picker
from core.jewelry.bracelet_box.tags import InsufficientStock, BUNDLE_SIZE


def fake_product(pid, color, material, style, days_in_stock=7):
    return {
        'id': pid,
        'name': f'Bracelet {pid}',
        'color_family': color,
        'material_class': material,
        'style_class': style,
        'days_in_stock': days_in_stock,
    }


@pytest.fixture
def fixture_30():
    """30 fake bracelets across known tag combinations.

    5 color families × 3 materials × 2 styles = 30 unique tag combos.
    """
    products = []
    pid = 1
    for color in ('warm', 'cool', 'neutral', 'mixed', 'statement'):
        for material in ('beaded', 'metal-chain', 'leather'):
            for style in ('minimal', 'classic'):
                products.append(fake_product(pid, color, material, style))
                pid += 1
    return products


def test_picker_returns_five(fixture_30):
    rng = random.Random(42)
    picks = picker.pick_five(fixture_30, history=[], rng=rng)
    assert len(picks) == BUNDLE_SIZE
    assert len({p['id'] for p in picks}) == BUNDLE_SIZE  # all unique


def test_picker_variety_within_five(fixture_30):
    """No two picks share BOTH color_family AND material_class."""
    rng = random.Random(42)
    picks = picker.pick_five(fixture_30, history=[], rng=rng)
    pairs = Counter((p['color_family'], p['material_class']) for p in picks)
    assert max(pairs.values()) == 1, f"duplicate color+material pair: {pairs}"


def test_picker_dedup_against_history(fixture_30):
    """Heavy history skews picks away from over-represented color families.

    Statistical test: average over many seeds to smooth out individual noise.
    """
    history = [
        {'color_tags': ['warm'], 'style_tags': ['minimal']},
        {'color_tags': ['warm'], 'style_tags': ['minimal']},
        {'color_tags': ['warm'], 'style_tags': ['classic']},
    ]

    warm_with_history = 0
    warm_without = 0
    for seed in range(50):
        rng_a = random.Random(seed)
        picks_a = picker.pick_five(fixture_30, history=history, rng=rng_a)
        warm_with_history += sum(1 for p in picks_a if p['color_family'] == 'warm')

        rng_b = random.Random(seed)
        picks_b = picker.pick_five(fixture_30, history=[], rng=rng_b)
        warm_without += sum(1 for p in picks_b if p['color_family'] == 'warm')

    # With heavy 'warm' history, the picker should select fewer warm pieces on average.
    assert warm_with_history < warm_without, (
        f"dedup not biting: warm_with={warm_with_history} warm_without={warm_without}"
    )


def test_picker_exact_five(fixture_30):
    """Candidate set == BUNDLE_SIZE returns all 5 with no scoring."""
    five = fixture_30[:5]
    picks = picker.pick_five(five, history=[], rng=random.Random(42))
    assert {p['id'] for p in picks} == {p['id'] for p in five}


def test_picker_insufficient_stock(fixture_30):
    """Candidate set < BUNDLE_SIZE raises InsufficientStock."""
    with pytest.raises(InsufficientStock):
        picker.pick_five(fixture_30[:4], history=[], rng=random.Random(42))


def test_picker_zero_candidates():
    """Empty candidate list raises."""
    with pytest.raises(InsufficientStock):
        picker.pick_five([], history=[], rng=random.Random(42))


def test_picker_deterministic_with_seed(fixture_30):
    """Same seed → same picks."""
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    picks_a = picker.pick_five(fixture_30, history=[], rng=rng_a)
    picks_b = picker.pick_five(fixture_30, history=[], rng=rng_b)
    assert [p['id'] for p in picks_a] == [p['id'] for p in picks_b]


def test_picker_different_seeds_likely_differ(fixture_30):
    """Different seeds produce different picks (strong probability)."""
    rng_a = random.Random(1)
    rng_b = random.Random(99)
    picks_a = picker.pick_five(fixture_30, history=[], rng=rng_a)
    picks_b = picker.pick_five(fixture_30, history=[], rng=rng_b)
    # Astronomically unlikely they match exactly with 30 candidates.
    assert {p['id'] for p in picks_a} != {p['id'] for p in picks_b}


def test_picker_freshness_bias():
    """Items in stock longer have a slight upward weight bias."""
    # Two color/material combos available; older one should be picked more often.
    candidates = [
        fake_product(1, 'warm', 'beaded',       'minimal', days_in_stock=30),
        fake_product(2, 'cool', 'metal-chain',  'classic', days_in_stock=2),
        fake_product(3, 'warm', 'metal-chain',  'minimal', days_in_stock=30),
        fake_product(4, 'cool', 'beaded',       'classic', days_in_stock=2),
        fake_product(5, 'neutral', 'leather',   'minimal', days_in_stock=30),
        fake_product(6, 'mixed', 'leather',     'classic', days_in_stock=2),
        fake_product(7, 'statement', 'gemstone','minimal', days_in_stock=30),
    ]
    older_picked = 0
    newer_picked = 0
    for seed in range(100):
        picks = picker.pick_five(candidates, history=[], rng=random.Random(seed))
        for p in picks:
            if p['days_in_stock'] >= 30:
                older_picked += 1
            elif p['days_in_stock'] <= 2:
                newer_picked += 1
    assert older_picked > newer_picked, (
        f"freshness not biting: older={older_picked} newer={newer_picked}"
    )


def test_picker_homogeneous_catalog_best_effort():
    """If the catalog is too homogeneous to satisfy variety, picker still
    returns 5 picks (best effort, doesn't crash)."""
    # All same color+material pair → impossible to have variety.
    homogeneous = [
        fake_product(i, 'warm', 'beaded', 'minimal') for i in range(1, 11)
    ]
    picks = picker.pick_five(homogeneous, history=[], rng=random.Random(42))
    assert len(picks) == BUNDLE_SIZE
