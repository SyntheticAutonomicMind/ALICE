# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 The ALICE Authors

"""
ALICE API Endpoint Tests

Comprehensive tests for the FastAPI endpoints.
Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def client():
    """Create test client with app lifespan."""
    from src.main import app
    # Use TestClient as context manager to properly handle lifespan
    with TestClient(app) as client:
        yield client


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================

def test_health_endpoint(client):
    """Test health endpoint returns expected format."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert data["status"] in ["ok", "starting"]
    assert "gpuAvailable" in data
    assert "modelsLoaded" in data
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_health_response_types(client):
    """Test health endpoint returns correct types."""
    response = client.get("/health")
    data = response.json()
    
    assert isinstance(data["status"], str)
    assert isinstance(data["gpuAvailable"], bool)
    assert isinstance(data["modelsLoaded"], int)
    assert isinstance(data["version"], str)


# =============================================================================
# MODELS ENDPOINT TESTS
# =============================================================================

def test_models_endpoint(client):
    """Test models endpoint returns expected format."""
    response = client.get("/v1/models")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)


def test_models_openai_compatible(client):
    """Test models endpoint is OpenAI-compatible."""
    response = client.get("/v1/models")
    data = response.json()
    
    # OpenAI models endpoint returns "list" object
    assert data["object"] == "list"
    
    # Each model should have required fields
    for model in data["data"]:
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model


def test_models_refresh(client):
    """Test models refresh endpoint."""
    response = client.post("/v1/models/refresh")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert data["status"] == "ok"
    assert "models_found" in data


# =============================================================================
# METRICS ENDPOINT TESTS
# =============================================================================

def test_metrics_endpoint(client):
    """Test metrics endpoint returns expected format."""
    response = client.get("/metrics")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "queueDepth" in data
    assert "modelsLoaded" in data
    assert "totalGenerations" in data
    assert "avgGenerationTime" in data


def test_metrics_types(client):
    """Test metrics endpoint returns correct types."""
    response = client.get("/metrics")
    data = response.json()
    
    assert isinstance(data["queueDepth"], int)
    assert isinstance(data["modelsLoaded"], int)
    assert isinstance(data["totalGenerations"], int)
    assert isinstance(data["avgGenerationTime"], (int, float))
    assert isinstance(data["gpuUtilization"], (int, float))


# =============================================================================
# CHAT COMPLETIONS ENDPOINT TESTS
# =============================================================================

def test_chat_completions_missing_model(client):
    """Test chat completions returns error for missing model."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "sd/nonexistent-model",
            "messages": [{"role": "user", "content": "a cat"}]
        }
    )
    
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert "message" in data["error"]


def test_chat_completions_missing_message(client):
    """Test chat completions returns error for missing user message."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "sd/test-model",
            "messages": []
        }
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_chat_completions_invalid_json(client):
    """Test chat completions returns error for invalid JSON."""
    response = client.post(
        "/v1/chat/completions",
        content="not valid json",
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 422


def test_chat_completions_request_format(client):
    """Test chat completions accepts OpenAI-compatible request."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "sd/test-model",
            "messages": [
                {"role": "system", "content": "You are an image generator"},
                {"role": "user", "content": "a beautiful sunset"}
            ],
            "sam_config": {
                "steps": 25,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512
            }
        }
    )
    
    # Should fail with 404 (model not found) not 422 (validation error)
    assert response.status_code == 404


def test_chat_completions_with_sam_config(client):
    """Test chat completions accepts sam_config parameters."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "sd/test-model",
            "messages": [{"role": "user", "content": "a cat"}],
            "sam_config": {
                "negative_prompt": "blurry, ugly",
                "steps": 30,
                "guidance_scale": 8.0,
                "width": 768,
                "height": 768,
                "seed": 42,
                "scheduler": "euler_a",
                "num_images": 1
            }
        }
    )
    
    # Should fail with 404 (model not found) not 422 (validation error)
    assert response.status_code == 404


# =============================================================================
# ERROR RESPONSE TESTS
# =============================================================================

def test_error_response_format(client):
    """Test error responses follow expected format."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "sd/nonexistent",
            "messages": [{"role": "user", "content": "test"}]
        }
    )
    
    data = response.json()
    
    assert "error" in data
    error = data["error"]
    
    assert "message" in error
    assert "type" in error
    assert "code" in error


# =============================================================================
# WEB INTERFACE TESTS
# =============================================================================

def test_web_interface_accessible(client):
    """Test web interface is accessible."""
    response = client.get("/web/index.html")
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_web_interface_css(client):
    """Test CSS is accessible."""
    response = client.get("/web/style.css")
    
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_web_interface_js(client):
    """Test JavaScript is accessible."""
    response = client.get("/web/app.js")
    
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


# =============================================================================
# API DOCUMENTATION TESTS
# =============================================================================

def test_openapi_schema(client):
    """Test OpenAPI schema is available."""
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "openapi" in data
    assert "info" in data
    assert "paths" in data


def test_docs_endpoint(client):
    """Test Swagger UI is available."""
    response = client.get("/docs")
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_redoc_endpoint(client):
    """Test ReDoc is available."""
    response = client.get("/redoc")
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
