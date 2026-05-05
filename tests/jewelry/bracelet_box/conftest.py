"""Shared pytest fixtures for bracelet_box tests."""
from __future__ import annotations
import pytest
from unittest.mock import patch


@pytest.fixture
def temp_db(tmp_path):
    """Run tests against a temp SQLite db so we don't touch the real one.

    Patches BOTH core.jewelry.db.DB_PATH (used by init() to create the
    schema) AND core.jewelry.bracelet_box.db.DB_PATH (used by CRUD).
    """
    db_path = tmp_path / "test_jewelry.db"
    with patch('core.jewelry.db.DB_PATH', db_path), \
         patch('core.jewelry.bracelet_box.db.DB_PATH', db_path):
        from core.jewelry import db as core_db
        core_db.init()
        yield db_path
