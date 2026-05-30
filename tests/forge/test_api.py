import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.security.auth import require_auth


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.api import forge
    app = FastAPI()
    forge.register(app)
    app.dependency_overrides[require_auth] = lambda: {"username": "tester", "role": "admin"}
    return TestClient(app)


def test_health_is_public(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.api import forge
    app = FastAPI()
    forge.register(app)
    c = TestClient(app)
    r = c.get("/forge/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_job_returns_pending(client):
    r = client.post("/forge/jobs", json={"job_type": "echo", "params": {"a": 1}})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert body["job_type"] == "echo"
    assert body["params"] == {"a": 1}


def test_create_job_requires_job_type(client):
    r = client.post("/forge/jobs", json={"params": {}})
    assert r.status_code == 400


def test_get_job_roundtrip(client):
    created = client.post("/forge/jobs", json={"job_type": "echo"}).json()
    r = client.get(f"/forge/jobs/{created['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == created["id"]


def test_get_unknown_job_404(client):
    assert client.get("/forge/jobs/does-not-exist").status_code == 404


def test_list_jobs_and_status_filter(client):
    client.post("/forge/jobs", json={"job_type": "echo"})
    client.post("/forge/jobs", json={"job_type": "echo"})
    r = client.get("/forge/jobs?status=pending")
    assert r.status_code == 200
    assert len(r.json()["jobs"]) == 2
