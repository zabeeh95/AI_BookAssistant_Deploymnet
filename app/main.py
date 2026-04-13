"""
FastAPI application initialization and configuration.

This module sets up:
- FastAPI app with metadata
- CORS middleware
- Exception handlers
- Startup/shutdown events
- Request/response logging
- Health check integration
"""

import logging
from contextlib import asynccontextmanager
from time import time
from typing import Callable

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError

from app.routes import router
from app.services import initialize_rag_service, get_rag_service
from app.models import ErrorResponse
from config import (
    APIConfig,
    LogConfig,
    ENVIRONMENT,
    IS_PRODUCTION,
)

logger = logging.getLogger(__name__)


# ==================== LIFECYCLE EVENTS ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - startup and shutdown.

    Startup: Initialize RAG service and load models
    Shutdown: Cleanup resources
    """
    # ===== STARTUP =====
    logger.info(f"{'=' * 100}")
    logger.info(f"Starting AI Book Assistant API")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Debug: {APIConfig.DEBUG}")
    logger.info(f"{'=' * 100}")

    try:
        # Initialize RAG service
        logger.info("[INFO] Initializing RAG service...")
        service = initialize_rag_service()
        logger.info(f"[INFO] RAG service initialized")
        logger.info(f" [INFO]  - Chunks: {service.get_status()['num_chunks']}")
        logger.info(f" [INFO] - Embedding dim: {service.get_status()['embeddings_dim']}")

    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize RAG service: {str(e)}", exc_info=True)
        raise

    logger.info("API ready to accept requests")

    yield  # Application runs here

    # ===== SHUTDOWN =====
    logger.info("Shutting down AI Book Assistant API...")
    try:
        # Cleanup code here (if needed)
        logger.info("✓ Shutdown complete")
    except Exception as e:
        logger.error(f" [ERROR] Error during shutdown: {str(e)}", exc_info=True)


# ==================== APP INITIALIZATION ====================

app = FastAPI(
    title="AI Book Assistant API",
    description="RAG-based AI assistant for querying book content using embeddings, reranking, and LLM generation",
    version="2.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ==================== MIDDLEWARE CONFIGURATION ====================

# 1. Trusted Host Middleware (security)
if IS_PRODUCTION:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost",
                       "127.0.0.1",
                       "*.example.com",  # Replace with your domain
                       ]
    )

# 2. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig.CORS_ORIGINS,
    allow_credentials=APIConfig.CORS_CREDENTIALS,
    allow_methods=APIConfig.CORS_METHODS,
    allow_headers=APIConfig.CORS_HEADERS,
)

# 3. GZip Middleware (compression for responses > 500 bytes)
app.add_middleware(GZipMiddleware, minimum_size=500)


# ==================== CUSTOM MIDDLEWARE ====================

@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next: Callable):
    """
    Middleware to:
    - Add request ID to all requests
    - Log request details
    - Time request processing
    """
    import uuid
    request_id = str(uuid.uuid4())

    # Add to request state for use in handlers
    request.state.request_id = request_id

    start_time = time()

    # Log incoming request (only in debug mode or for errors)
    if APIConfig.DEBUG:
        logger.debug(
            f"[{request_id}] {request.method} {request.url.path} | "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )

    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"[{request_id}] Unhandled exception: {str(e)}", exc_info=True)
        raise

    # Calculate processing time
    process_time = time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id

    # Log response (only in debug or for non-200s)
    if APIConfig.DEBUG or response.status_code >= 400:
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Time: {process_time * 1000:.1f}ms"
        )

    return response


# ==================== EXCEPTION HANDLERS ====================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with nice formatting.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        f"[{request_id}] Validation error: {exc.error_count()} fields"
    )

    # Format error details
    errors = []
    for error in exc.errors():
        field = ".".join(str(x) for x in error["loc"][1:])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": "One or more fields have invalid values",
            "status_code": 422,
            "request_id": request_id,
            "fields": errors,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler for unexpected errors.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"[{request_id}] Unhandled exception: {type(exc).__name__}: {str(exc)}",
        exc_info=True
    )

    # Don't expose internal error details in production
    detail = str(exc) if not IS_PRODUCTION else "Internal server error"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": detail,
            "status_code": 500,
            "request_id": request_id,
        },
    )


# ==================== ROUTES ====================

# Include API routes with version prefix
app.include_router(router)


# ==================== ROOT AND INFO ENDPOINTS ====================

@app.get("/", tags=["info"])
async def read_root():
    """
    Root endpoint - provides API overview.

    Returns links to documentation and available endpoints.
    """
    return {
        "message": "Welcome to AI Book Assistant API",
        "version": "2.0",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "endpoints": {
            "ask_question": "POST /api/v1/ask",
            "batch_questions": "POST /api/v1/ask-batch",
            "health": "GET /api/v1/health",
            "stats": "GET /api/v1/stats",
            "models": "GET /api/v1/models",
        },
        "environment": ENVIRONMENT,
    }


@app.get("/info", tags=["info"])
async def get_info():
    """
    Get API metadata and configuration.
    """
    service = get_rag_service()

    return {
        "name": "AI Book Assistant API",
        "version": "2.0",
        "environment": ENVIRONMENT,
        "service": {
            "status": "ready" if service.models_loaded else "initializing",
            "models_loaded": service.models_loaded,
        },
        "api_config": {
            "host": APIConfig.HOST,
            "port": APIConfig.PORT,
            "debug": APIConfig.DEBUG,
            "workers": APIConfig.WORKERS,
        },
        "cors_origins": APIConfig.CORS_ORIGINS,
    }


# ==================== STARTUP/SHUTDOWN LOGGING ====================

@app.on_event("startup")
async def startup_event():
    """Called after FastAPI starts."""
    logger.info("✓ FastAPI application started")


@app.on_event("shutdown")
async def shutdown_event():
    """Called when FastAPI shuts down."""
    logger.info("✓ FastAPI application shutdown")


# ==================== DEVELOPMENT ROUTES ====================

if not IS_PRODUCTION:
    @app.get("/debug/config", tags=["debug"])
    async def debug_config():
        """
        Debug endpoint to view current configuration.
        Only available in development mode.
        """
        return {
            "environment": ENVIRONMENT,
            "api_config": {
                "host": APIConfig.HOST,
                "port": APIConfig.PORT,
                "debug": APIConfig.DEBUG,
                "reload": APIConfig.RELOAD,
                "workers": APIConfig.WORKERS,
            },
            "rag_config": {
                "top_k": APIConfig.MAX_TOP_K,
                "use_reranker": APIConfig.USE_RERANKER,
                "enable_caching": APIConfig.ENABLE_CACHING,
            },
            "log_level": LogConfig.LEVEL,
        }


    @app.get("/debug/service-status", tags=["debug"])
    async def debug_service_status():
        """
        Debug endpoint to view detailed service status.
        Only available in development mode.
        """
        try:
            service = get_rag_service()
            return {
                "models_loaded": service.models_loaded,
                "status": service.get_status(),
            }
        except Exception as e:
            return {
                "error": str(e),
                "models_loaded": False,
            }


# ==================== 404 HANDLER ====================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors with helpful message."""
    request_id = getattr(request.state, "request_id", "unknown")

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Not found",
            "detail": f"Path {request.url.path} does not exist",
            "status_code": 404,
            "request_id": request_id,
            "available_endpoints": {
                "docs": "/docs",
                "api": "/api/v1",
            },
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        reload=APIConfig.RELOAD,
        workers=APIConfig.WORKERS if not APIConfig.DEBUG else 1,
        log_level=LogConfig.LEVEL.lower(),
    )
