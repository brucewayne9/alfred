import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    return _db


def test_orgs_table_seeded_with_three_orgs(db):
    with db._conn() as c:
        ids = {r["id"] for r in c.execute("SELECT id FROM orgs")}
    assert {"mainstay", "rucktalk", "groundrush"} <= ids


def test_org_id_column_on_scoped_tables_defaults_to_mainstay(db):
    with db._conn() as c:
        c.execute(
            "INSERT INTO sources (id, kind, spec, status, created_at, updated_at) "
            "VALUES ('s1','url','x','done',0,0)"
        )
        row = c.execute("SELECT org_id FROM sources WHERE id='s1'").fetchone()
    assert row["org_id"] == "mainstay"


def test_all_four_scoped_tables_have_org_id(db):
    with db._conn() as c:
        for table in ("sources", "jobs", "clip_candidates", "dist_posts"):
            cols = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
            assert "org_id" in cols, f"{table} missing org_id"
