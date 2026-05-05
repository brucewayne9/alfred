"""Pick session CRUD."""
from __future__ import annotations
import json
import time

from core.jewelry.bracelet_box import db as box_db


def test_create_and_fetch_pick(temp_db):
    pick_id = box_db.create_pick(
        order_id=101, line_item_id=201, bundle_index=1,
        customer_email="ada@example.com", customer_first_name="Ada",
        picked_skus=[1, 2, 3, 4, 5],
        color_tags=["warm", "neutral"], style_tags=["minimal"],
        note_text="Roen chose...",
    )
    assert pick_id > 0

    row = box_db.get_pick(pick_id)
    assert row is not None
    assert row['customer_email'] == 'ada@example.com'
    assert row['status'] == 'suggested'
    assert json.loads(row['picked_skus']) == [1, 2, 3, 4, 5]
    assert json.loads(row['color_tags']) == ['warm', 'neutral']


def test_email_normalization(temp_db):
    """Emails are lowercased and stripped before storage."""
    pid = box_db.create_pick(
        order_id=999, line_item_id=999, bundle_index=1,
        customer_email="  ADA@Example.COM  ", customer_first_name=None,
        picked_skus=[1, 2, 3, 4, 5],
        color_tags=[], style_tags=[],
        note_text="x",
    )
    row = box_db.get_pick(pid)
    assert row['customer_email'] == 'ada@example.com'


def test_status_transitions(temp_db):
    pick_id = box_db.create_pick(
        order_id=102, line_item_id=202, bundle_index=1,
        customer_email="b@c.com", customer_first_name="B",
        picked_skus=[10, 11, 12, 13, 14],
        color_tags=["cool"], style_tags=["classic"],
        note_text="...",
    )
    box_db.set_status(pick_id, 'awaiting_sarah')
    assert box_db.get_pick(pick_id)['status'] == 'awaiting_sarah'

    approved_at = int(time.time())
    box_db.set_status(pick_id, 'approved', approved_at=approved_at)
    row = box_db.get_pick(pick_id)
    assert row['status'] == 'approved'
    assert row['approved_at'] == approved_at


def test_update_picks(temp_db):
    pid = box_db.create_pick(
        order_id=103, line_item_id=203, bundle_index=1,
        customer_email="x@y.com", customer_first_name="X",
        picked_skus=[1, 2, 3, 4, 5],
        color_tags=["warm"], style_tags=["minimal"],
        note_text="orig",
    )
    box_db.update_picks(pid, picked_skus=[9, 8, 7, 6, 5],
                        color_tags=["cool"], style_tags=["classic"])
    row = box_db.get_pick(pid)
    assert json.loads(row['picked_skus']) == [9, 8, 7, 6, 5]
    assert json.loads(row['color_tags']) == ['cool']
    assert json.loads(row['style_tags']) == ['classic']


def test_update_note(temp_db):
    pid = box_db.create_pick(
        order_id=104, line_item_id=204, bundle_index=1,
        customer_email="x@y.com", customer_first_name="X",
        picked_skus=[1, 2, 3, 4, 5],
        color_tags=[], style_tags=[],
        note_text="orig",
    )
    box_db.update_note(pid, "rewritten note")
    assert box_db.get_pick(pid)['note_text'] == "rewritten note"


def test_history_for_email(temp_db):
    """Past shipped/approved picks for the same email come back recency-sorted."""
    for i in range(3):
        pid = box_db.create_pick(
            order_id=200 + i, line_item_id=300 + i, bundle_index=1,
            customer_email="repeat@example.com", customer_first_name="R",
            picked_skus=[i, i+1, i+2, i+3, i+4],
            color_tags=["warm"] if i % 2 == 0 else ["cool"],
            style_tags=["minimal"],
            note_text=f"note {i}",
        )
        box_db.set_status(pid, 'shipped', shipped_at=int(time.time()) + i)

    history = box_db.history_for_email("repeat@example.com")
    assert len(history) == 3
    # Most recent first — note ids ascend with i, but created_at also ascends
    # (one INSERT after another) so newest is "note 2".
    assert history[0]['note_text'] == "note 2"
    assert history[2]['note_text'] == "note 0"


def test_history_excludes_unshipped(temp_db):
    """Only approved/shipped picks count as history — pending ones are skipped."""
    box_db.create_pick(
        order_id=300, line_item_id=400, bundle_index=1,
        customer_email="pending@x.com", customer_first_name="P",
        picked_skus=[1, 2, 3, 4, 5],
        color_tags=[], style_tags=[],
        note_text="still pending",
    )
    history = box_db.history_for_email("pending@x.com")
    assert history == []


def test_list_pending(temp_db):
    """Pending picks (suggested + awaiting_sarah) returned, shipped excluded."""
    pid_a = box_db.create_pick(
        order_id=500, line_item_id=600, bundle_index=1,
        customer_email="a@a.com", customer_first_name="A",
        picked_skus=[1, 2, 3, 4, 5],
        color_tags=[], style_tags=[],
        note_text="a",
    )
    pid_b = box_db.create_pick(
        order_id=501, line_item_id=601, bundle_index=1,
        customer_email="b@b.com", customer_first_name="B",
        picked_skus=[1, 2, 3, 4, 5],
        color_tags=[], style_tags=[],
        note_text="b",
    )
    box_db.set_status(pid_b, 'awaiting_sarah')

    pid_c = box_db.create_pick(
        order_id=502, line_item_id=602, bundle_index=1,
        customer_email="c@c.com", customer_first_name="C",
        picked_skus=[1, 2, 3, 4, 5],
        color_tags=[], style_tags=[],
        note_text="c",
    )
    box_db.set_status(pid_c, 'shipped', shipped_at=int(time.time()))

    pending = box_db.list_pending(older_than_seconds=0)
    pending_ids = {p['id'] for p in pending}
    assert pid_a in pending_ids
    assert pid_b in pending_ids
    assert pid_c not in pending_ids


def test_uniqueness_constraint(temp_db):
    """Same (order_id, line_item_id, bundle_index) cannot be inserted twice."""
    import sqlite3 as _sqlite3
    box_db.create_pick(
        order_id=700, line_item_id=800, bundle_index=1,
        customer_email="x@y.com", customer_first_name=None,
        picked_skus=[1, 2, 3, 4, 5],
        color_tags=[], style_tags=[],
        note_text="x",
    )
    import pytest
    with pytest.raises(_sqlite3.IntegrityError):
        box_db.create_pick(
            order_id=700, line_item_id=800, bundle_index=1,
            customer_email="x@y.com", customer_first_name=None,
            picked_skus=[6, 7, 8, 9, 10],
            color_tags=[], style_tags=[],
            note_text="dup",
        )
