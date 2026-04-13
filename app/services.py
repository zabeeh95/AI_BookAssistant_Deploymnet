"""
Service layer for RAG (Retrieval-Augmented Generation) functionality.

This module handles:
- Model loading and caching
- Query embedding and retrieval
- Result reranking
- Response generation
- Error handling and logging

The RAGService class is designed to be instantiated once and reused across requests.
"""

import faiss
import pickle
import logging
import time
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

from config import (
    EMBEDDINGS_MODEL,
    RANKING_MODEL,
    SMALL_LLM,
    LARGE_LLM,
    DEFAULT_LLM,
    RAGConfig,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    OLLAMA_RETRY_ATTEMPTS,
    CHUNKS_PKL_PATH,
    FAISS_INDEX_PATH,
)

logger = logging.getLogger(__name__)

# ==================== PROMPT TEMPLATES ====================

SYSTEM_PROMPTS = {
    "general": """You are a helpful book assistant. Use ONLY the provided context to answer questions. 
If the answer is not in the context, clearly state "I don't have that information in the book."
Be concise and accurate.""",

    "summary": """You are a summarization expert. Based on the context provided, give a concise summary.
Focus on key points and main ideas.""",

    "explanation": """You are an explanation expert. Explain the concept clearly and thoroughly based on the context.
Use examples from the text when helpful.""",

    "comparison": """You are a comparison expert. Compare the items/concepts mentioned in the context.
Highlight similarities and differences.""",
}


def get_system_prompt(query_type: str = "general") -> str:
    """Get appropriate system prompt based on query type."""
    return SYSTEM_PROMPTS.get(query_type, SYSTEM_PROMPTS["general"])


def create_user_prompt(context: str, query: str, system_prompt: str) -> str:
    """Create the full prompt for the LLM."""
    return f"""{system_prompt}

Context:
{context}

Question: {query}

Answer (2-3 sentences):"""


# ==================== RAG SERVICE ====================

class RAGService:
    """
    Retrieval-Augmented Generation service.

    Handles embedding, retrieval, reranking, and response generation.
    Models are loaded once during initialization and reused.
    """

    def __init__(self):
        """Initialize RAG service by loading all required models and data."""
        logger.info("Initializing RAG Service...")
        self.start_time = time.time()

        try:
            # Load embedding model
            logger.info(f" [INFO] Loading embeddings model from {EMBEDDINGS_MODEL}")
            self.embed_model = SentenceTransformer(str(EMBEDDINGS_MODEL))
            logger.info(" [INFO] Embeddings model loaded")

            # Load reranker model
            logger.info(f" [INFO] Loading reranker model from {RANKING_MODEL}")
            self.reranker = CrossEncoder(str(RANKING_MODEL))
            logger.info(" [INFO] Reranker model loaded")

            # Load chunks
            logger.info(f" [INFO] Loading chunks from {CHUNKS_PKL_PATH}")
            with open(CHUNKS_PKL_PATH, "rb") as f:
                self.chunks = pickle.load(f)
            logger.info(f" [INFO] Loaded {len(self.chunks)} chunks")

            # Load FAISS index
            logger.info(f" [INFO] Loading FAISS index from {FAISS_INDEX_PATH}")
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            logger.info(f" [INFO] FAISS index loaded with {self.index.ntotal} vectors")

            self.models_loaded = True
            init_time = time.time() - self.start_time
            logger.info(f" [INFO] RAG Service initialized successfully in {init_time:.2f}s")

        except Exception as e:
            self.models_loaded = False
            logger.error(f" [ERROR] Failed to initialize RAG Service: {str(e)}", exc_info=True)
            raise

        # Initialize query cache
        self._query_cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            "models_loaded": self.models_loaded,
            "num_chunks": len(self.chunks) if hasattr(self, 'chunks') else 0,
            "embeddings_dim": self.embed_model.get_sentence_embedding_dimension() if hasattr(self,
                                                                                             'embed_model') else 0,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._query_cache),
        }

    def _get_cache_key(self, query: str, top_k: int, use_reranker: bool) -> str:
        """Generate cache key for a query."""
        return f"{query}|{top_k}|{use_reranker}"

    def _update_cache(self, key: str, value: Any) -> None:
        """Update cache with size limit."""
        if not RAGConfig.ENABLE_CACHING:
            return

        if len(self._query_cache) >= RAGConfig.CACHE_SIZE:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self._query_cache))
            del self._query_cache[first_key]

        self._query_cache[key] = value

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query to be clearer and more specific.

        Uses the small LLM to rephrase ambiguous queries.
        """
        logger.debug(f"Rewriting query: {query}")

        prompt = f"""Rewrite this question to be clearer and more specific for searching a book.
Keep it concise (1-2 sentences). Do not add information not in the original question.

Question: {query}

Rewritten Question:"""

        try:
            response = ollama.chat(
                model=RAGConfig.QUERY_REWRITE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            rewritten = response["message"]["content"].strip()
            logger.debug(f"Rewritten to: {rewritten}")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed: {str(e)}. Using original query.")
            return query

    def retrieve_chunks(self, query: str, top_k: int = RAGConfig.DEFAULT_TOP_K) -> Tuple[List[str], List[int]]:
        """
        Retrieve relevant chunks using vector similarity search.

        Args:
            query: The search query
            top_k: Number of chunks to retrieve

        Returns:
            Tuple of (chunks, indices)
        """
        # Prepare query for embedding
        # This special prefix helps the embeddings model understand we're searching
        search_query = "Represent this sentence for searching relevant passages: " + query

        # Embed and search
        query_vec = self.embed_model.encode([search_query], normalize_embeddings=True)

        distances, indices = self.index.search(query_vec, k=top_k)

        retrieved_chunks = [self.chunks[i] for i in indices[0]]

        logger.debug(f"Retrieved {len(retrieved_chunks)} chunks for query")

        return retrieved_chunks, indices[0].tolist()

    def rerank_chunks(self, query: str, chunks: List[str],
                      top_k: int = RAGConfig.DEFAULT_RERANK_K) \
            -> Tuple[List[str], List[float]]:
        """
        Rerank chunks using cross-encoder for better relevance.

        Args:
            query: The original query
            chunks: List of chunks to rerank
            top_k: Number of top-ranked chunks to return

        Returns:
            Tuple of (reranked_chunks, scores)
        """
        if len(chunks) == 0:
            return [], []

        # Create query-chunk pairs for the reranker
        pairs = [[query, chunk] for chunk in chunks]

        # Get relevance scores
        scores = self.reranker.predict(pairs)

        # Sort by score (descending) and select top-k
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        reranked_chunks = [chunks[i] for i in sorted_indices]
        reranked_scores = [scores[i] for i in sorted_indices]

        logger.debug(f"Reranked {len(chunks)} chunks to top {len(reranked_chunks)}")

        return reranked_chunks, reranked_scores

    def build_context(self, chunks: List[str], max_length: int = RAGConfig.MAX_CONTEXT_LENGTH) -> str:
        """
        Build context string from chunks with length limit.

        Args:
            chunks: List of chunks to combine
            max_length: Maximum length of context string

        Returns:
            Context string
        """
        context = ""
        for chunk in chunks:
            potential_context = context + chunk + RAGConfig.CONTEXT_SEPARATOR
            if len(potential_context) <= max_length:
                context = potential_context
            else:
                logger.debug(f"Context length limit reached at {len(context)} chars")
                break

        return context.strip()

    def generate_response(self, context: str, query: str, query_type: str = "general",
                          model: str = DEFAULT_LLM, ) -> str:
        """
        Generate response using LLM with context.

        Args:
            context: The retrieved context
            query: The user's query
            query_type: Type of query (affects system prompt)
            model: Which LLM model to use

        Returns:
            The generated response
        """
        system_prompt = get_system_prompt(query_type)
        full_prompt = create_user_prompt(context, query, system_prompt)

        logger.debug(f"Generating response using {model}")

        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                stream=False,
                # timeout=OLLAMA_TIMEOUT,
            )

            answer = response["message"]["content"].strip()
            return answer

        except Exception as e:
            logger.error(f"LLM response generation failed: {str(e)}", exc_info=True)
            raise

    def get_rag_response(self, query: str, query_type: str = "general",
                         top_k: int = RAGConfig.DEFAULT_TOP_K, rerank_top_k: int = RAGConfig.DEFAULT_RERANK_K,
                         use_reranker: bool = RAGConfig.USE_RERANKER, include_source_chunks: bool = False,
                         ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: rewrite → retrieve → rerank → generate.

        Args:
            query: User's question
            query_type: Type of query
            top_k: Initial number of chunks to retrieve
            rerank_top_k: Number of chunks to use for context
            use_reranker: Whether to use reranking
            include_source_chunks: Whether to include source chunk info

        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()

        try:
            # Step 1: Rewrite query (if enabled)
            if RAGConfig.ENABLE_QUERY_REWRITING:
                rewritten_query = self.rewrite_query(query)
            else:
                rewritten_query = query

            # Step 2: Retrieve chunks
            retrieved_chunks, chunk_indices = self.retrieve_chunks(
                rewritten_query,
                top_k=top_k
            )

            # Step 3: Rerank (if enabled)
            if use_reranker and len(retrieved_chunks) > 0:
                reranked_chunks, scores = self.rerank_chunks(
                    rewritten_query,
                    retrieved_chunks,
                    top_k=rerank_top_k
                )
                final_chunks = reranked_chunks
                chunk_scores = scores if include_source_chunks else []
            else:
                final_chunks = retrieved_chunks[:rerank_top_k]
                chunk_scores = []

            # Step 4: Build context
            context = self.build_context(final_chunks)

            # Step 5: Generate response
            response_text = self.generate_response(
                context,
                rewritten_query,
                query_type=query_type,
            )

            processing_time_ms = (time.time() - start_time) * 1000

            # Build result
            result = {
                "response": response_text,
                "query_rewritten": rewritten_query,
                "chunks_used": len(final_chunks),
                "processing_time_ms": processing_time_ms,
                "model_used": DEFAULT_LLM,
            }

            # Add source chunks if requested
            if include_source_chunks and chunk_scores:
                result["source_chunks"] = [
                    {
                        "chunk_id": idx,
                        "relevance_score": float(score)
                    }
                    for idx, score in zip(chunk_indices[:len(chunk_scores)], chunk_scores)
                ]

            logger.info(
                f"Query processed in {processing_time_ms:.0f}ms | "
                f"Chunks: {len(final_chunks)} | "
                f"Type: {query_type}"
            )

            return result

        except Exception as e:
            logger.error(
                f"RAG response generation failed: {str(e)}",
                exc_info=True
            )
            raise


# ==================== SINGLETON INSTANCE ====================

# Global RAG service instance - instantiated once on app startup
rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Get or create the RAG service singleton.

    This is used in FastAPI dependency injection.
    """
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service


def initialize_rag_service() -> RAGService:
    """Initialize the RAG service (called on app startup)."""
    global rag_service
    rag_service = RAGService()
    return rag_service
