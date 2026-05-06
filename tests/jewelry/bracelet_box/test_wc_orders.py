"""WC order poller tests — all HTTP is mocked."""
from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.jewelry.bracelet_box import wc_orders


def make_order(order_id, line_items, billing_email="ada@example.com",
               billing_first="Ada"):
    return {
        'id': order_id,
        'status': 'processing',
        'billing': {
            'first_name': billing_first,
            'last_name': 'Lovelace',
            'email': billing_email,
        },
        'line_items': line_items,
    }


@patch('core.jewelry.bracelet_box.wc_orders._fetch_orders_after')
def test_iter_yields_box_items_only(mock_fetch):
    mock_fetch.return_value = [
        make_order(1001, [{'id': 5001, 'sku': 'bracelet-aa', 'quantity': 1}]),
        make_order(1002, [{'id': 5002, 'sku': 'bracelet-box', 'quantity': 2}]),
        make_order(1003, [
            {'id': 5003, 'sku': 'bracelet-box', 'quantity': 1},
            {'id': 5004, 'sku': 'bracelet-bb', 'quantity': 1},
        ]),
    ]
    items = list(wc_orders.iter_new_box_line_items(after_id=1000))
    assert len(items) == 2
    assert items[0]['order_id'] == 1002
    assert items[0]['quantity'] == 2
    assert items[1]['order_id'] == 1003
    assert items[1]['quantity'] == 1


@patch('core.jewelry.bracelet_box.wc_orders._fetch_orders_after')
def test_iter_normalizes_email(mock_fetch):
    mock_fetch.return_value = [
        make_order(2000, [{'id': 1, 'sku': 'bracelet-box', 'quantity': 1}],
                   billing_email="  ADA@Example.COM  "),
    ]
    items = list(wc_orders.iter_new_box_line_items(after_id=0))
    assert items[0]['customer_email'] == 'ada@example.com'


@patch('core.jewelry.bracelet_box.wc_orders._fetch_orders_after')
def test_iter_first_name_blank_becomes_none(mock_fetch):
    mock_fetch.return_value = [
        make_order(2001, [{'id': 1, 'sku': 'bracelet-box', 'quantity': 1}],
                   billing_first=""),
    ]
    items = list(wc_orders.iter_new_box_line_items(after_id=0))
    assert items[0]['customer_first_name'] is None


def test_cursor_roundtrip(tmp_path):
    cursor = tmp_path / "cursor.txt"
    wc_orders.save_cursor(cursor, 999)
    assert wc_orders.load_cursor(cursor) == 999


def test_cursor_missing_returns_zero(tmp_path):
    assert wc_orders.load_cursor(tmp_path / "missing.txt") == 0


def test_cursor_corrupted_returns_zero(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("not-an-integer")
    assert wc_orders.load_cursor(p) == 0


@patch('core.jewelry.bracelet_box.wc_orders.requests.put')
@patch('core.jewelry.bracelet_box.wc_orders.requests.get')
def test_reserve_skus_succeeds(mock_get, mock_put):
    """All SKUs in stock → reserve_skus returns True and writes outofstock."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={'stock_status': 'instock', 'stock_quantity': 1}),
        raise_for_status=MagicMock(),
    )
    mock_put.return_value = MagicMock(raise_for_status=MagicMock())
    ok = wc_orders.reserve_skus([1, 2, 3])
    assert ok is True
    assert mock_put.call_count == 3
    # Confirm the outofstock payload was used
    payload = mock_put.call_args_list[0].kwargs['json']
    assert payload == {'stock_quantity': 0, 'stock_status': 'outofstock'}


@patch('core.jewelry.bracelet_box.wc_orders.requests.put')
@patch('core.jewelry.bracelet_box.wc_orders.requests.get')
def test_reserve_skus_bails_if_any_out_of_stock(mock_get, mock_put):
    """If any SKU is already out of stock, reserve_skus returns False
    without writing anything."""
    def side_effect(url, **kwargs):
        # Second SKU is already gone
        m = MagicMock(raise_for_status=MagicMock())
        if "2" in url:
            m.json.return_value = {'stock_status': 'outofstock', 'stock_quantity': 0}
        else:
            m.json.return_value = {'stock_status': 'instock', 'stock_quantity': 1}
        return m
    mock_get.side_effect = side_effect

    ok = wc_orders.reserve_skus([1, 2, 3])
    assert ok is False
    mock_put.assert_not_called()  # no commit pass when pre-check fails


@patch('core.jewelry.bracelet_box.wc_orders.requests.put')
def test_release_skus(mock_put):
    mock_put.return_value = MagicMock(raise_for_status=MagicMock())
    wc_orders.release_skus([10, 11])
    assert mock_put.call_count == 2
    assert mock_put.call_args_list[0].kwargs['json'] == {'stock_quantity': 1, 'stock_status': 'instock'}


@patch('core.jewelry.bracelet_box.wc_orders.requests.get')
def test_fetch_in_stock_bracelets_paginates(mock_get):
    """Paginated response: page 1 has 2 products, page 2 empty → loop ends."""
    page1 = [
        {
            'id': 100, 'name': 'Bracelet A',
            'short_description': 'short',
            'images': [{'src': 'https://x/a.jpg'}],
            'meta_data': [
                {'key': '_roen_color_family', 'value': 'warm'},
                {'key': '_roen_material_class', 'value': 'beaded'},
                {'key': '_roen_style_class', 'value': 'minimal'},
            ],
        },
        {
            'id': 101, 'name': 'Bracelet B',
            'short_description': '',
            'images': [],
            'meta_data': [],  # no tags → defaults
        },
    ]

    # First call: cat lookup → returns one row
    cat_resp = MagicMock(raise_for_status=MagicMock())
    cat_resp.json.return_value = [{'id': 17, 'slug': 'bracelets'}]

    page1_resp = MagicMock(raise_for_status=MagicMock())
    page1_resp.json.return_value = page1

    page2_resp = MagicMock(raise_for_status=MagicMock())
    page2_resp.json.return_value = []

    mock_get.side_effect = [cat_resp, page1_resp, page2_resp]

    # Reset module-cached cat id
    import core.jewelry.bracelet_box.wc_orders as wo
    wo._BRACELETS_CAT_ID = None

    rows = wc_orders.fetch_in_stock_bracelets()
    assert len(rows) == 2
    assert rows[0]['id'] == 100
    assert rows[0]['color_family'] == 'warm'
    assert rows[1]['color_family'] == 'mixed'  # default when missing
    assert rows[1]['image_url'] == ''


@patch('core.jewelry.bracelet_box.wc_orders.requests.get')
def test_fetch_returns_empty_when_no_bracelets_category(mock_get):
    """If the bracelets product_cat doesn't exist, return empty without error."""
    cat_resp = MagicMock(raise_for_status=MagicMock())
    cat_resp.json.return_value = []  # no category
    mock_get.return_value = cat_resp

    import core.jewelry.bracelet_box.wc_orders as wo
    wo._BRACELETS_CAT_ID = None

    assert wc_orders.fetch_in_stock_bracelets() == []
