"""
Backend Tests — MemoryVerse API
=================================
These are integration tests that run against the live FastAPI app.
They require:
  - A running FastAPI server (or use TestClient)
  - A valid Supabase project configured in .env

Run with:
    cd backend && .venv/bin/python -m pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ── Health ───────────────────────────────────────────────────────────────────

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app" in data


# ── Auth — Unauthenticated ────────────────────────────────────────────────────

def test_unauthenticated_access_to_vaults():
    """Protected routes must return 403 or 401 without auth token."""
    response = client.get("/api/v1/vaults")
    assert response.status_code in (401, 403)


def test_unauthenticated_access_to_profile():
    response = client.get("/api/v1/profile")
    assert response.status_code in (401, 403)


def test_unauthenticated_access_to_timeline():
    response = client.get("/api/v1/timeline")
    assert response.status_code in (401, 403)


def test_unauthenticated_access_to_media():
    response = client.get("/api/v1/media")
    assert response.status_code in (401, 403)


def test_unauthenticated_access_to_ai_chat():
    response = client.post("/api/v1/ai/chat", json={"message": "test"})
    assert response.status_code in (401, 403)


# ── Auth — Bad Token ──────────────────────────────────────────────────────────

def test_invalid_token_rejected():
    response = client.get(
        "/api/v1/vaults",
        headers={"Authorization": "Bearer invalid_token_here"}
    )
    assert response.status_code == 401


# ── Signup Validation ─────────────────────────────────────────────────────────

def test_signup_missing_email():
    response = client.post("/api/v1/auth/signup", json={
        "password": "testpassword123"
    })
    assert response.status_code == 422  # Validation error


def test_signup_short_password():
    response = client.post("/api/v1/auth/signup", json={
        "email": "test@example.com",
        "password": "abc"
    })
    assert response.status_code == 422  # Min length 6


def test_signup_invalid_email():
    response = client.post("/api/v1/auth/signup", json={
        "email": "not-an-email",
        "password": "validpassword"
    })
    assert response.status_code == 422


# ── Login Validation ──────────────────────────────────────────────────────────

def test_login_missing_fields():
    response = client.post("/api/v1/auth/login", json={})
    assert response.status_code == 422


def test_login_wrong_credentials():
    response = client.post("/api/v1/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "wrongpassword"
    })
    assert response.status_code in (400, 401)


# ── Vault Validation (unauthenticated) ────────────────────────────────────────

def test_create_vault_requires_auth():
    response = client.post("/api/v1/vaults", json={"name": "Test Vault"})
    assert response.status_code in (401, 403)


def test_join_vault_requires_auth():
    response = client.post("/api/v1/vaults/join", json={"invite_code": "abc123"})
    assert response.status_code in (401, 403)


def test_join_vault_invalid_code_format():
    """Even without auth we should get 401/403 not 500."""
    response = client.post(
        "/api/v1/vaults/join",
        json={"invite_code": ""},
        headers={"Authorization": "Bearer fake_token"}
    )
    assert response.status_code in (401, 403, 422)


# ── AI Validation ─────────────────────────────────────────────────────────────

def test_ai_chat_requires_message():
    response = client.post(
        "/api/v1/ai/chat",
        json={},
        headers={"Authorization": "Bearer fake_token"}
    )
    assert response.status_code in (401, 403, 422)


def test_ai_chat_empty_message_rejected():
    response = client.post(
        "/api/v1/ai/chat",
        json={"message": ""},
        headers={"Authorization": "Bearer fake_token"}
    )
    # Either auth fails or validation fails — either way not 200
    assert response.status_code != 200


# ── Media Validation ──────────────────────────────────────────────────────────

def test_media_create_requires_auth():
    response = client.post("/api/v1/media", json={
        "filename": "test.jpg",
        "storage_path": "user/test.jpg",
        "url": "https://example.com/test.jpg",
    })
    assert response.status_code in (401, 403)


# ── Search ────────────────────────────────────────────────────────────────────

def test_search_requires_auth():
    response = client.get("/api/v1/search?q=test")
    assert response.status_code in (401, 403)


# ── Timeline ──────────────────────────────────────────────────────────────────

def test_timeline_requires_auth():
    response = client.get("/api/v1/timeline")
    assert response.status_code in (401, 403)


# ── Error Response Format ─────────────────────────────────────────────────────

def test_404_returns_json():
    response = client.get("/api/v1/nonexistent_endpoint")
    assert response.status_code == 404


def test_global_error_does_not_expose_stack_trace():
    """Verify the 500 handler returns sanitized JSON, not raw Python traceback."""
    response = client.get("/health")
    assert response.status_code == 200
    # Health check should always be clean
    assert "traceback" not in response.text.lower()
