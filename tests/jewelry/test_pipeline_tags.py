"""Pipeline writes vision tags as WC product meta after creating draft."""
from __future__ import annotations
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest


@pytest.fixture
def mock_intake():
    """A db row that pipeline.process_intake will fetch."""
    return {
        "id": 1,
        "chat_id": 7582976864,
        "status": "received",
        "photos_json": json.dumps([{"path": "/tmp/fake.jpg", "telegram_file_id": "x"}]),
        "price_cents": 5000,
        "description": None,  # forces a fresh vision pass
    }


def test_pipeline_writes_tag_meta_on_fresh_intake(mock_intake):
    with patch('core.jewelry.pipeline.db') as mock_db, \
         patch('core.jewelry.pipeline.vision.describe_piece') as mock_vision, \
         patch('core.jewelry.pipeline.copywriter.write_copy') as mock_copy, \
         patch('core.jewelry.pipeline.woocommerce.upload_image') as mock_upload, \
         patch('core.jewelry.pipeline.woocommerce.create_draft_product') as mock_create, \
         patch('core.jewelry.pipeline.woocommerce.update_product_meta') as mock_meta:

        from core.jewelry import pipeline

        mock_db.get_intake.return_value = mock_intake
        mock_vision.return_value = {
            "description": "Beaded warm bracelet",
            "color_family": "warm",
            "dominant_hex": "#C8794E",
            "material_class": "beaded",
            "style_class": "minimal",
        }
        mock_copy.return_value = {
            "name": "X",
            "sku": "x-sku",
            "short_description": "y",
            "long_description": "z",
            "category": "bracelets",
            "tags": [],
        }
        mock_upload.return_value = 999
        mock_create.return_value = 1234

        pipeline.process_intake(intake_id=1)

        # Confirm all four meta writes happened with expected key/value pairs.
        calls = {c.args[1]: c.args[2] for c in mock_meta.call_args_list}
        assert calls['_roen_color_family'] == 'warm'
        assert calls['_roen_material_class'] == 'beaded'
        assert calls['_roen_style_class'] == 'minimal'
        assert calls['_roen_dominant_hex'] == '#C8794E'
        # All four meta writes targeted post_id 1234
        for c in mock_meta.call_args_list:
            assert c.args[0] == 1234


def test_pipeline_skips_meta_when_description_resumed():
    """When intake already has a description (resume), no fresh vision pass,
    so we have no fresh tags. Pipeline must NOT call update_product_meta."""
    intake = {
        "id": 2,
        "status": "received",
        "photos_json": json.dumps([{"path": "/tmp/fake.jpg", "telegram_file_id": "x"}]),
        "price_cents": 5000,
        "description": "Existing description from a previous attempt.",
    }
    with patch('core.jewelry.pipeline.db') as mock_db, \
         patch('core.jewelry.pipeline.vision.describe_piece') as mock_vision, \
         patch('core.jewelry.pipeline.copywriter.write_copy') as mock_copy, \
         patch('core.jewelry.pipeline.woocommerce.upload_image') as mock_upload, \
         patch('core.jewelry.pipeline.woocommerce.create_draft_product') as mock_create, \
         patch('core.jewelry.pipeline.woocommerce.update_product_meta') as mock_meta:

        from core.jewelry import pipeline

        mock_db.get_intake.return_value = intake
        mock_copy.return_value = {
            "name": "X", "sku": "x-sku", "short_description": "y",
            "long_description": "z", "category": "bracelets", "tags": [],
        }
        mock_upload.return_value = 999
        mock_create.return_value = 5678

        pipeline.process_intake(intake_id=2)

        # No fresh vision pass → no tag writes.
        mock_vision.assert_not_called()
        mock_meta.assert_not_called()
