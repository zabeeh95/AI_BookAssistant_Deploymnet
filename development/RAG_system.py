from constants import *

import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama


@st.cache_resource
def load_models():
    embed_model = SentenceTransformer(EMBEDDINGS_MODEL)
    reranker = CrossEncoder(RANKING_MODEL)

    with open("embeddings/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    index = faiss.read_index("embeddings/faiss_index.bin")

    return embed_model, reranker, chunks, index


embed_model, reranker, chunks, index = load_models()


def rewrite_query(query):
    prompt = f"""
    Rewrite the question to be clear and self-contained.

    Question:
    {query}
    
    Rewritten Question:
    """

    response = ollama.chat(model=SMALL_LLM,
                           messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]


def rag_system(query):
    better_query = rewrite_query(query)
    print("better query is ", better_query)

    search_query = "Represent this sentence for searching relevant passages: " + better_query
    query_vec = embed_model.encode([search_query], normalize_embeddings=True)

    distance_chunk, index_chunk = index.search(query_vec, k=10)
    retrieved_chunks = [chunks[i] for i in index_chunk[0]]

    pairs = [[better_query, chunk] for chunk in retrieved_chunks]
    scores = reranker.predict(pairs)

    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, retrieved_chunks),
                                                  key=lambda x: x[0],
                                                  reverse=True)]

    context = "\n\n".join(sorted_chunks[:3])

    prompt = prompt_for_LLM(context=context, query=better_query)

    return prompt
