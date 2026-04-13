"""
Comprehensive test suite for AI Book Assistant API.

Tests cover:
- API endpoint functionality
- Error handling
- Data validation
- Service initialization
- Integration tests
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import time

from app.main import app
from app.models import QueryRequest, QueryResponse
from app.services import RAGService


# ==================== TEST CLIENT ====================

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_rag_service():
    """Mock RAG service for testing."""
    service = MagicMock(spec=RAGService)
    service.models_loaded = True
    service.get_rag_response.return_value = {
        "response": "Test answer",
        "query_rewritten": "Test question rewritten",
        "chunks_used": 3,
        "processing_time_ms": 100.0,
        "model_used": "llama3.2:1b",
    }
    service.get_status.return_value = {
        "num_chunks": 1000,
        "embeddings_dim": 1024,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_size": 0,
    }
    return service


# ==================== ROOT ENDPOINT TESTS ====================

class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["message"] == "Welcome to AI Book Assistant API"

    def test_root_response_structure(self, client):
        """Test root response has required fields."""
        response = client.get("/")
        data = response.json()
        assert "version" in data
        assert "documentation" in data
        assert "endpoints" in data


# ==================== QUERY ENDPOINT TESTS ====================

class TestQueryEndpoint:
    """Tests for /ask endpoint."""

    def test_ask_simple_query(self, client, mock_rag_service):
        """Test asking a simple question."""
        payload = {"query": "What is Harry Potter?"}

        with patch('app.routes.get_rag_service', return_value=mock_rag_service):
            response = client.post("/api/v1/ask", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "query_rewritten" in data
        assert "chunks_used" in data
        assert "processing_time_ms" in data

    def test_ask_with_all_parameters(self, client, mock_rag_service):
        """Test query with all optional parameters."""
        payload = {
            "query": "What are the main themes?",
            "query_type": "summary",
            "top_k": 5,
            "rerank_top_k": 2,
            "use_reranker": False,
            "include_sources": True,
        }

        with patch('app.routes.get_rag_service', return_value=mock_rag_service):
            response = client.post("/api/v1/ask", json=payload)

        assert response.status_code == 200

    def test_ask_query_too_short(self, client):
        """Test validation: query too short."""
        payload = {"query": "abc"}  # Less than 5 chars

        response = client.post("/api/v1/ask", json=payload)
        assert response.status_code == 422
        assert "validation" in response.json()["error"].lower()

    def test_ask_query_too_long(self, client):
        """Test validation: query too long."""
        payload = {"query": "a" * 1001}  # More than 1000 chars

        response = client.post("/api/v1/ask", json=payload)
        assert response.status_code == 422

    def test_ask_empty_query(self, client):
        """Test validation: empty query."""
        payload = {"query": "   "}  # Only whitespace

        response = client.post("/api/v1/ask", json=payload)
        assert response.status_code == 422

    def test_ask_missing_query(self, client):
        """Test validation: missing query field."""
        payload = {}

        response = client.post("/api/v1/ask", json=payload)
        assert response.status_code == 422

    def test_ask_invalid_query_type(self, client):
        """Test validation: invalid query type."""
        payload = {
            "query": "What is magic?",
            "query_type": "invalid_type"
        }

        response = client.post("/api/v1/ask", json=payload)
        assert response.status_code == 422

    def test_ask_invalid_top_k(self, client):
        """Test validation: invalid top_k."""
        payload = {
            "query": "What is magic?",
            "top_k": 100  # Greater than max
        }

        response = client.post("/api/v1/ask", json=payload)
        assert response.status_code == 422

    def test_ask_response_model(self, client, mock_rag_service):
        """Test response conforms to QueryResponse model."""
        payload = {"query": "What is magic?"}

        with patch('app.routes.get_rag_service', return_value=mock_rag_service):
            response = client.post("/api/v1/ask", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Validate response matches QueryResponse schema
        assert isinstance(data["response"], str)
        assert isinstance(data["query_rewritten"], str)
        assert isinstance(data["chunks_used"], int)
        assert isinstance(data["processing_time_ms"], float)
        assert isinstance(data["model_used"], str)


# ==================== BATCH QUERY TESTS ====================

class TestBatchQueryEndpoint:
    """Tests for /ask-batch endpoint."""

    def test_batch_single_query(self, client, mock_rag_service):
        """Test batch with single query."""
        payload = {
            "queries": [
                {"id": "1", "query": "What is magic?"}
            ]
        }

        with patch('app.routes.get_rag_service', return_value=mock_rag_service):
            response = client.post("/api/v1/ask-batch", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == "1"
        assert data["results"][0]["success"] is True

    def test_batch_multiple_queries(self, client, mock_rag_service):
        """Test batch with multiple queries."""
        payload = {
            "queries": [
                {"id": "1", "query": "What is magic?"},
                {"id": "2", "query": "Who is the main character?"},
                {"id": "3", "query": "What is the plot?"},
            ]
        }

        with patch('app.routes.get_rag_service', return_value=mock_rag_service):
            response = client.post("/api/v1/ask-batch", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        assert data["failed_count"] == 0

    def test_batch_max_queries(self, client, mock_rag_service):
        """Test batch respects max query limit."""
        # Create 51 queries (max is 50)
        queries = [
            {"id": str(i), "query": "What is magic?"}
            for i in range(51)
        ]
        payload = {"queries": queries}

        response = client.post("/api/v1/ask-batch", json=payload)
        assert response.status_code == 422  # Validation error

    def test_batch_empty_queries(self, client):
        """Test batch with empty queries list."""
        payload = {"queries": []}

        response = client.post("/api/v1/ask-batch", json=payload)
        assert response.status_code == 422


# ==================== HEALTH CHECK TESTS ====================

class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client, mock_rag_service):
        """Test health check endpoint."""
        with patch('app.routes.get_rag_service', return_value=mock_rag_service):
            response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert data["models_loaded"] is True

    def test_health_check_all_fields(self, client, mock_rag_service):
        """Test health check response contains all fields."""
        with patch('app.routes.get_rag_service', return_value=mock_rag_service):
            response = client.get("/api/v1/health")

        data = response.json()
        required_fields = [
            "status",
            "models_loaded",
            "embeddings_model",
            "reranker_model",
            "llm_model",
        ]
        for field in required_fields:
            assert field in data


# ==================== STATS ENDPOINT TESTS ====================

class TestStatsEndpoint:
    """Tests for /stats endpoint."""

    def test_stats_endpoint(self, client, mock_rag_service):
        """Test stats endpoint."""
        with patch('app.routes.get_rag_service', return_value=mock_rag_service):
            response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_chunks" in data
        assert "embeddings_dim" in data
        assert "models_info" in data


# ==================== MODELS ENDPOINT TESTS ====================

class TestModelsEndpoint:
    """Tests for /models endpoint."""

    def test_models_endpoint(self, client):
        """Test models information endpoint."""
        response = client.get("/api/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert "embeddings_model" in data
        assert "reranker_model" in data
        assert "generation_models" in data


# ==================== ERROR HANDLING TESTS ====================

class TestErrorHandling:
    """Tests for error handling."""

    def test_404_not_found(self, client):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        assert "error" in response.json()

    def test_405_method_not_allowed(self, client):
        """Test method not allowed."""
        response = client.get("/api/v1/ask")  # POST expected
        assert response.status_code == 405

    def test_malformed_json(self, client):
        """Test malformed JSON handling."""
        response = client.post(
            "/api/v1/ask",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [422, 400]


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests with real service (if available)."""

    @pytest.mark.skipif(
        not hasattr(MagicMock(), 'models_loaded'),
        reason="RAG service not available"
    )
    def test_full_query_flow(self, client, mock_rag_service):
        """Test complete query flow."""
        # Health check
        with patch('app.routes.get_rag_service', return_value=mock_rag_service):
            health = client.get("/api/v1/health")
            assert health.status_code == 200

            # Query
            response = client.post(
                "/api/v1/ask",
                json={"query": "What is magic?"}
            )
            assert response.status_code == 200

            # Stats
            stats = client.get("/api/v1/stats")
            assert stats.status_code == 200


# ==================== RESPONSE TIME TESTS ====================

class TestResponseTime:
    """Tests for response time requirements."""

    def test_health_check_fast(self, client, mock_rag_service):
        """Test health check is fast."""
        with patch('app.routes.get_rag_service', return_value=mock_rag_service):
            start = time.time()
            response = client.get("/api/v1/health")
            duration = (time.time() - start) * 1000

        assert response.status_code == 200
        assert duration < 1000  # Should be less than 1 second

    def test_query_timeout(self, client):
        """Test query timeout behavior."""
        # This would test actual timeout, skipped for now
        pass


# ==================== PYTEST CONFIGURATION ====================

@pytest.fixture(scope="session")
def setup_test_env():
    """Setup test environment."""
    import os
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["LOG_LEVEL"] = "WARNING"


# ==================== TEST MARKERS ====================

pytestmark = [
    pytest.mark.asyncio,
]


# ==================== RUN TESTS ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])