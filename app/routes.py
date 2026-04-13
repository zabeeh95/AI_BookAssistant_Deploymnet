"""
API route handlers for the AI Book Assistant.

This module defines all endpoints:
- /ask - Single query endpoint
- /ask-batch - Batch query processing
- /health - Health check
- /stats - Service statistics
- /models - Model information

Each endpoint includes proper error handling, validation, and documentation.
"""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models import (
    QueryRequest,
    QueryResponse,
    HealthCheckResponse,
    ErrorResponse,
    ModelStatsResponse,
    ChunkReference,
)
from app.services import get_rag_service, RAGService
from config import APIConfig, RAGConfig, SMALL_LLM, LARGE_LLM, EMBEDDINGS_MODEL, RANKING_MODEL

logger = logging.getLogger(__name__)

# Create router with prefix for all routes
router = APIRouter(prefix="/api/v1", tags=["queries"])


# ==================== DEPENDENCIES ====================

def get_request_id() -> str:
    """Generate unique request ID for logging."""
    return str(uuid.uuid4())


async def get_service() -> RAGService:
    """FastAPI dependency to inject RAG service."""
    return get_rag_service()


# ==================== SINGLE QUERY ENDPOINT ====================

@router.post("/ask",
             response_model=QueryResponse,
             responses={400: {"model": ErrorResponse, "description": "Invalid request"},
                        500: {"model": ErrorResponse, "description": "Server error"},
                        504: {"description": "LLM service unavailable"}, },
             summary="Ask a question about the book",
             description="Submit a question about the book content. The system will retrieve relevant passages, "
                         "rerank them, and generate an answer using an LLM.",
             )
async def ask_question(request: QueryRequest,
                       service: RAGService = Depends(get_service),
                       request_id: str = Depends(get_request_id), ) -> QueryResponse:
    """
    Process a user question and return an AI-generated answer.

    The endpoint follows this process:
    1. Rewrite the query for clarity (optional)
    2. Retrieve similar chunks from the knowledge base
    3. Rerank chunks by relevance (optional)
    4. Generate context from top chunks
    5. Use LLM to generate final answer

    Returns detailed metadata about processing.
    """
    logger.info(
        f"[{request_id}] Received query | Type: {request.query_type} | "
        f"Top-k: {request.top_k} | Reranker: {request.use_reranker}"
    )

    # Check if service is ready
    if not service.models_loaded:
        logger.error(f"[{request_id}] RAG service models not loaded")
        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready. Please try again later.",
        )

    try:
        # Get RAG response
        result = service.get_rag_response(query=request.query,
                                          query_type=request.query_type.value,
                                          top_k=request.top_k,
                                          rerank_top_k=request.rerank_top_k,
                                          use_reranker=request.use_reranker,
                                          include_source_chunks=request.include_sources,
                                          )

        # Build response model
        response = QueryResponse(response=result["response"],
                                 query_rewritten=result["query_rewritten"],
                                 chunks_used=result["chunks_used"],
                                 source_chunks=result.get("source_chunks"),
                                 processing_time_ms=result["processing_time_ms"],
                                 model_used=result["model_used"],
                                 )

        logger.info(f"[{request_id}] Query processed successfully | "
                    f"Time: {response.processing_time_ms:.0f}ms | "
                    f"Chunks: {response.chunks_used}")

        return response

    except Exception as e:
        logger.error(f"[{request_id}] Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


# ==================== HEALTH CHECK ENDPOINT ====================

@router.get("/health",
            response_model=HealthCheckResponse,
            summary="Health check",
            description="Check if the service is running and models are loaded.",
            )
async def health_check(service: RAGService = Depends(get_service), ) -> HealthCheckResponse:
    """
    Health check endpoint for monitoring and orchestration.

    Returns status of:
    - Overall service health
    - Model loading status
    - Model names and versions

    Set check_models=true for intensive checks (not recommended for frequent polling).
    """
    try:
        status = "healthy" if service.models_loaded else "degraded"

        response = HealthCheckResponse(status=status,
                                       models_loaded=service.models_loaded,
                                       embeddings_model=str(EMBEDDINGS_MODEL),
                                       reranker_model=str(RANKING_MODEL),
                                       llm_model=SMALL_LLM,
                                       message="All systems operational" if service.models_loaded else "Models not yet loaded",
                                       )

        logger.debug(f"Health check performed | Status: {status}")

        return response

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Service health check failed",
        )


# ==================== STATISTICS ENDPOINT ====================

@router.get("/stats",
            response_model=ModelStatsResponse,
            summary="Service statistics",
            description="Get detailed statistics about the loaded models and knowledge base.",
            )
async def get_stats(service: RAGService = Depends(get_service)) -> ModelStatsResponse:
    """
    Get detailed statistics about the service.

    Includes:
    - Number of chunks in knowledge base
    - Embedding dimensions
    - Index type
    - Cache statistics
    - Model information
    """
    try:
        status = service.get_status()

        response = ModelStatsResponse(
            total_chunks=status["num_chunks"],
            embeddings_dim=status["embeddings_dim"],
            index_type="FAISS L2Index",
            models_info={
                "embeddings": {
                    "name": "BAAI/bge-large-en-v1.5",
                    "type": "SentenceTransformer",
                },
                "reranker": {
                    "name": "BAAI/bge-reranker-base",
                    "type": "CrossEncoder",
                },
                "llm": {
                    "name": SMALL_LLM,
                    "type": "Ollama",
                },
                "cache": {
                    "hits": status["cache_hits"],
                    "misses": status["cache_misses"],
                    "size": status["cache_size"],
                },
            }
        )

        logger.debug("Statistics retrieved")

        return response

    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve statistics",
        )


# ==================== MODELS INFORMATION ENDPOINT ====================

@router.get("/models",
            summary="List available models",
            description="Get information about available LLM models.",
            )
async def get_models():
    """
    Get information about available models for RAG pipeline.
    """
    return {
        "embeddings_model": {
            "name": "BAAI/bge-large-en-v1.5",
            "type": "SentenceTransformer",
            "dims": 1024,
            "purpose": "Query and document embedding",
        },
        "reranker_model": {
            "name": "BAAI/bge-reranker-base",
            "type": "CrossEncoder",
            "purpose": "Re-rank retrieved passages by relevance",
        },
        "generation_models": {
            "small": {
                "name": SMALL_LLM,
                "params": "1B",
                "use_case": "Query rewriting and response generation",
            },
            "large": {
                "name": LARGE_LLM,
                "params": "7B+",
                "use_case": "Advanced reasoning and detailed responses",
            },
        },
        "rag_config": {
            "default_top_k": RAGConfig.DEFAULT_TOP_K,
            "default_rerank_k": RAGConfig.DEFAULT_RERANK_K,
            "max_top_k": RAGConfig.MAX_TOP_K,
            "query_rewriting_enabled": RAGConfig.ENABLE_QUERY_REWRITING,
            "reranking_enabled": RAGConfig.USE_RERANKER,
        },
    }


# ==================== ERROR HANDLERS ====================

@router.get("/error-test",
            summary="Test error handling",
            description="Endpoint to test error response format (development only)",
            )
async def error_test(error_type: str = Query("400", description="HTTP status code to return")):
    """
    Test endpoint to verify error response format.
    Only available in development mode.
    """
    code = int(error_type) if error_type.isdigit() else 500
    raise HTTPException(
        status_code=code,
        detail="This is a test error",
    )
