"""
Configuration management for the AI Book Assistant application.

This module handles:
- Environment variable loading
- Model path configuration
- Performance settings
- Logging configuration
- Deployment-specific settings

Environment variables can override defaults by setting them in .env file or os.environ.
"""

import os
import logging
from pathlib import Path
from typing import Optional

# ==================== DIRECTORY CONFIGURATION ====================

# Get base directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESOURCE_DIR = BASE_DIR / "resource"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
LOG_DIR.mkdir(exist_ok=True)

# ==================== MODEL PATHS ====================

EMBEDDINGS_MODEL_NAME = "models--BAAI--bge-large-en-v1.5"
EMBEDDINGS_MODEL = RESOURCE_DIR / EMBEDDINGS_MODEL_NAME

RANKING_MODEL_NAME = "models--BAAI--bge-reranker-base"
RANKING_MODEL = RESOURCE_DIR / RANKING_MODEL_NAME

# Chunk and index paths
CHUNKS_PKL_PATH = RESOURCE_DIR / "chunks.pkl"
FAISS_INDEX_PATH = RESOURCE_DIR / "faiss_index.bin"

# ==================== LLM CONFIGURATION ====================

# LLM models (for Ollama)
SMALL_LLM = os.getenv("SMALL_LLM", "llama3.2:1b")
LARGE_LLM = os.getenv("LARGE_LLM", "mistral:latest")
DEFAULT_LLM = SMALL_LLM

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds
OLLAMA_RETRY_ATTEMPTS = int(os.getenv("OLLAMA_RETRY_ATTEMPTS", "3"))


# ==================== RAG CONFIGURATION ====================

class RAGConfig:
    """RAG system configuration with tunable parameters."""

    # Retrieval parameters
    DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "10"))
    DEFAULT_RERANK_K = int(os.getenv("RAG_RERANK_K", "3"))
    MAX_TOP_K = int(os.getenv("RAG_MAX_TOP_K", "20"))
    MAX_RERANK_K = int(os.getenv("RAG_MAX_RERANK_K", "10"))

    # Context generation
    CONTEXT_SEPARATOR = "\n\n"
    MAX_CONTEXT_LENGTH = int(os.getenv("RAG_MAX_CONTEXT_LENGTH", "4000"))

    # Query rewriting
    ENABLE_QUERY_REWRITING = os.getenv("RAG_ENABLE_QUERY_REWRITING", "true").lower() == "true"
    QUERY_REWRITE_MODEL = os.getenv("RAG_QUERY_REWRITE_MODEL", SMALL_LLM)

    # Performance
    USE_RERANKER = os.getenv("RAG_USE_RERANKER", "true").lower() == "true"
    ENABLE_CACHING = os.getenv("RAG_ENABLE_CACHING", "true").lower() == "true"
    CACHE_SIZE = int(os.getenv("RAG_CACHE_SIZE", "1000"))  # number of cached queries


# ==================== API CONFIGURATION ====================

class APIConfig:
    """API server configuration."""

    # Server settings
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
    RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
    WORKERS = int(os.getenv("API_WORKERS", "4"))

    # CORS configuration
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8080"
    ).split(",")
    CORS_CREDENTIALS = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
    CORS_METHODS = ["GET", "POST", "OPTIONS"]
    CORS_HEADERS = ["*"]

    # Rate limiting
    ENABLE_RATE_LIMIT = os.getenv("ENABLE_RATE_LIMIT", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "3600"))  # seconds

    # Timeouts
    QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "30"))  # seconds
    HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "10"))  # seconds

    # Request/Response limits
    MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "1000000"))  # 1MB
    MAX_BATCH_QUERIES = int(os.getenv("MAX_BATCH_QUERIES", "50"))


# ==================== LOGGING CONFIGURATION ====================

class LogConfig:
    """Logging configuration."""

    LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Log file
    LOG_FILE = LOG_DIR / "app.log"
    LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO")
    LOG_FILE_MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", "10485760"))  # 10MB
    LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))

    # Format
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def setup_logging():
        """Configure application logging."""
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(LogConfig.LEVEL)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LogConfig.LEVEL)
        console_formatter = logging.Formatter(LogConfig.LOG_FORMAT, LogConfig.LOG_DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler (if in non-debug mode)
        if not APIConfig.DEBUG:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                LogConfig.LOG_FILE,
                maxBytes=LogConfig.LOG_FILE_MAX_BYTES,
                backupCount=LogConfig.LOG_FILE_BACKUP_COUNT
            )
            file_handler.setLevel(LogConfig.LOG_FILE_LEVEL)
            file_formatter = logging.Formatter(LogConfig.LOG_FORMAT, LogConfig.LOG_DATE_FORMAT)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)


# ==================== ENVIRONMENT ====================

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"
IS_DEVELOPMENT = ENVIRONMENT == "development"


# ==================== VALIDATION ====================

def validate_config():
    """Validate that all required paths and configurations exist."""
    errors = []

    # Check model directories
    if not EMBEDDINGS_MODEL.exists():
        errors.append(f"Embeddings model path not found: {EMBEDDINGS_MODEL}")

    if not RANKING_MODEL.exists():
        errors.append(f"Ranking model path not found: {RANKING_MODEL}")

    # Check data files
    if not CHUNKS_PKL_PATH.exists():
        errors.append(f"Chunks pickle file not found: {CHUNKS_PKL_PATH}")

    if not FAISS_INDEX_PATH.exists():
        errors.append(f"FAISS index not found: {FAISS_INDEX_PATH}")

    # Check data directory
    if not DATA_DIR.exists():
        errors.append(f"Data directory not found: {DATA_DIR}")

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(errors)
        if IS_PRODUCTION:
            raise RuntimeError(error_msg)
        else:
            logging.warning(error_msg)

    return len(errors) == 0


# ==================== INITIALIZATION ====================

# Setup logging
LogConfig.setup_logging()
logger = logging.getLogger(__name__)

# Log configuration on startup
if APIConfig.DEBUG:
    logger.info(f"Running in {ENVIRONMENT} mode")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"RAG Config - Top K: {RAGConfig.DEFAULT_TOP_K}, Rerank K: {RAGConfig.DEFAULT_RERANK_K}")
    logger.info(f"API Config - Host: {APIConfig.HOST}:{APIConfig.PORT}")

# Validate configuration
validate_config()