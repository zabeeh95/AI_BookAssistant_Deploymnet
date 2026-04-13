"""
Pydantic models for request/response validation in the AI Book Assistant API.

These models ensure type safety, automatic documentation generation (via Swagger UI),
and request/response validation. Each model is used in endpoint definitions.
"""

from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List



class QueryTypeEnum(str, Enum):
    """Enum for different query types - enables API consumers to specify query intent."""
    GENERAL = "general"
    SUMMARY = "summary"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"


# ==================== REQUEST MODELS ====================

class QueryRequest(BaseModel):
    """
    Request model for asking a question about the book.

    Attributes:
        query: The question to ask (required, 5-1000 chars)
        query_type: The type of query to optimize retrieval (default: general)
        top_k: Number of relevant chunks to retrieve (default: 10, range: 1-20)
        rerank_top_k: Number of chunks to rerank before context (default: 3, range: 1-10)
        use_reranker: Whether to use cross-encoder reranking (default: True)
        include_sources: Whether to include source chunk indices in response (default: False)
    """

    query: str = Field(..., min_length=5, max_length=1000,
                       description="The question to ask about the book content",
                       example="What are the main themes in Harry Potter?")
    query_type: QueryTypeEnum = Field(default=QueryTypeEnum.GENERAL,
                                      description="Type of query to optimize retrieval strategy")
    top_k: int = Field(default=10, ge=1, le=20,
                       description="Number of chunks to retrieve from vector database")
    rerank_top_k: int = Field(default=3, ge=1, le=10,
                              description="Number of top-ranked chunks to use for context")
    use_reranker: bool = Field(default=True,
                               description="Whether to use cross-encoder reranking for better relevance")
    include_sources: bool = Field(default=False,
                                  description="Include source chunk indices for traceability")

    @field_validator('query')
    @classmethod
    def query_must_be_meaningful(cls, v):
        """Validate that query is not just whitespace."""
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace')
        return v.strip()


class HealthCheckRequest(BaseModel):
    """Request model for health check endpoint (can be extended for future use)."""
    check_models: bool = Field(default=False,
                               description="Whether to perform intensive model load checks")


# ==================== RESPONSE MODELS ====================

class ChunkReference(BaseModel):
    """Reference to a source chunk used in the response."""
    chunk_id: int = Field(..., description="Index of the chunk in the knowledge base")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score from reranker (0-1)")


class QueryResponse(BaseModel):
    """
    Response model for question answering endpoint.

    Attributes:
        response: The LLM-generated answer
        query_rewritten: The rewritten/clarified version of the user's query
        chunks_used: Number of chunks used for context
        source_chunks: References to source chunks (if requested)
        processing_time_ms: Time taken to process the query
        model_used: Which LLM model was used for generation
    """
    response: str = Field(..., description="The generated answer from the AI assistant")
    query_rewritten: str = Field(..., description="Clarified/rewritten version of the input query")
    chunks_used: int = Field(..., ge=0, description="Number of context chunks used in response generation")
    source_chunks: Optional[List[ChunkReference]] = Field(default=None,
                                                          description="Source chunks used (if include_sources=True)")
    processing_time_ms: float = Field(..., ge=0, description="Total processing time in milliseconds")
    model_used: str = Field(..., description="Name of the LLM model used for generation")


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Status of the service (healthy/degraded/unhealthy)")
    models_loaded: bool = Field(..., description="Whether all required models are loaded")
    embeddings_model: str = Field(..., description="Name of the embeddings model")
    reranker_model: str = Field(..., description="Name of the reranker model")
    llm_model: str = Field(..., description="Name of the LLM model")
    message: Optional[str] = Field(default=None, description="Additional status message")


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    error: str = Field(..., description="Error type/message")
    detail: str = Field(..., description="Detailed error description")
    status_code: int = Field(..., description="HTTP status code")
    request_id: Optional[str] = Field(default=None, description="Unique request ID for debugging")


class ModelStatsResponse(BaseModel):
    """Response model for model statistics endpoint."""
    total_chunks: int = Field(..., description="Total chunks in knowledge base")
    embeddings_dim: int = Field(..., description="Dimension of embeddings")
    index_type: str = Field(..., description="Type of FAISS index")
    models_info: dict = Field(..., description="Dictionary of loaded models and their info")


