"""
rag_core.py — RAG Pipeline Core Logic
=======================================
Provides the complete Retrieval-Augmented Generation pipeline for the
SmartHome Hub Assistant. This module is imported by app.py (Streamlit UI)
and test_suite.py (automated testing).

Public API:
    init_models(chat_model_name, embed_progress_cb, chat_progress_cb) -> dict
    answer_query(question, embedding_client, chat_client, top_k, stream_callback) -> dict
"""

import json
import math
import sqlite3
from pathlib import Path

from sdk_utils import init_sdk, load_model

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent
DB_FILE         = BASE_DIR / "knowledge_base.db"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"


# ------------------------------------------------------------------
# Vector Math
# ------------------------------------------------------------------
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Computes cosine similarity between two equal-length float vectors.
    Returns a score in the range [-1.0, 1.0], where 1.0 is identical.
    """
    dot         = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot / (magnitude_a * magnitude_b)


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------
def get_top_chunks(query: str,
                   embedding_client,
                   top_k: int = 5) -> list[dict]:
    """
    Embeds the query, fetches all stored vectors from SQLite, ranks
    them by cosine similarity, and returns the top_k matches.

    Parameters:
        query            : The user's natural-language question.
        embedding_client : Active Foundry Local embedding client.
        top_k            : Number of top-scoring chunks to return.

    Returns:
        List of dicts sorted by score descending:
        [{"id": int, "filename": str, "content": str, "score": float}, ...]
    """
    # Embed the query into a vector
    query_response = embedding_client.generate_embeddings([query])
    query_vector   = query_response.data[0].embedding

    # Fetch all stored chunks and their embedding vectors
    conn   = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    # Score each chunk against the query vector
    scored = []
    for row_id, filename, content, embedding_json in rows:
        doc_vector = json.loads(embedding_json)
        score = cosine_similarity(query_vector, doc_vector)
        scored.append({"id": row_id, "filename": filename,
                        "content": content, "score": score})

    # Return highest-scoring chunks
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ------------------------------------------------------------------
# Answer Generation
# ------------------------------------------------------------------
def expand_query(query: str, chat_client) -> str:
    """
    Rewrites the user's natural language question into optimized search keywords
    to bridge semantic gaps (e.g. converting 'guarantee' to 'warranty').
    """
    system_prompt = (
        "You are a search query optimizer.\n"
        "Convert the user's question into 3 to 5 search keywords or synonyms "
        "relevant to finding information in a technical product manual.\n"
        "CRITICAL: Map colloquial or informal terms to standard technical terms. "
        "For example, convert 'guarantee' or 'refund' to 'warranty', 'wires' to 'cables', 'net' to 'wifi network'.\n"
        "Output ONLY the optimized keywords separated by spaces. Do not write full sentences, notes, or explanations."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query}
    ]
    
    keywords = []
    token_count = 0
    # Strict low token limit for speed
    for chunk in chat_client.complete_streaming_chat(messages):
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                keywords.append(content)
                token_count += 1
                if token_count >= 15:
                    break
    
    expanded = "".join(keywords).strip()
    return expanded if expanded else query


# ------------------------------------------------------------------
# Answer Generation
# ------------------------------------------------------------------
def answer_query(question: str,
                 embedding_client,
                 chat_client,
                 top_k: int = 5,
                 stream_callback=None) -> dict:
    """
    Full RAG pipeline: Retrieve → Augment → Generate.

    Steps:
      A. Expand the query to standardize synonyms (e.g. guarantee -> warranty).
      B. Retrieve the top_k document chunks most relevant to the expanded question.
      C. Build a grounded system prompt containing only those chunks.
      D. Stream the chat model's response token-by-token.

    Parameters:
        question         : The user's natural-language question.
        embedding_client : Active Foundry Local embedding client.
        chat_client      : Active Foundry Local chat client.
        top_k            : Number of chunks to retrieve and provide as context.
        stream_callback  : Optional callable(str) invoked with the accumulated
                           response text after each token — use this to update
                           a UI in real-time.

    Returns:
        {
            "answer"  : str,        # The model's complete response
            "sources" : list[dict], # Retrieved chunks used as context
        }
    """
    # A — Expand query to resolve synonyms (e.g. guarantee -> warranty)
    expanded_query = expand_query(question, chat_client)
    
    # B — Retrieve relevant document chunks using the expanded query
    chunks = get_top_chunks(expanded_query, embedding_client, top_k=top_k)

    # B — Build grounded system prompt
    # Each chunk is prefixed with its source file name for citation
    context_lines = []
    for chunk in chunks:
        source = chunk["filename"].replace(".txt", "").replace("_", " ").title()
        context_lines.append(f"[Source: {source}]\n{chunk['content']}")
    context_text = "\n\n".join(context_lines)

    system_prompt = (
        "You are an intelligent document retrieval assistant.\n\n"
        "STRICT INSTRUCTIONS:\n"
        "1. Answer the user's question using ONLY the exact information provided "
        "in the Context section below. Do not use any outside knowledge.\n"
        "2. If the Context does not contain the answer, respond with EXACTLY: "
        "\"I don't have that information.\"\n"
        "3. Always cite which source document your answer comes from "
        "(e.g., 'According to the Warranty policy...').\n"
        "4. Keep your answer concise — no more than 3 sentences.\n\n"
        f"Context:\n{context_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]

    # C — Stream the response with a hard token cap to prevent runaway generation
    response_parts = []
    token_count    = 0
    MAX_TOKENS     = 200

    for chunk in chat_client.complete_streaming_chat(messages):
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                response_parts.append(content)
                token_count += 1

                if stream_callback:
                    stream_callback("".join(response_parts))

                if token_count >= MAX_TOKENS:
                    break

    return {
        "answer":  "".join(response_parts).strip(),
        "sources": chunks,
    }


# ------------------------------------------------------------------
# Model Initialization
# ------------------------------------------------------------------
def init_models(chat_model_name: str,
                embed_progress_cb=None,
                chat_progress_cb=None) -> dict:
    """
    Initializes the Foundry Local SDK and loads both the embedding model
    and the specified chat model. Designed to be called inside Streamlit's
    @st.cache_resource so initialization only happens once per model.

    Parameters:
        chat_model_name   : Catalog alias for the LLM (e.g. 'phi-3.5-mini').
        embed_progress_cb : Optional callable(float) for embedding download progress.
        chat_progress_cb  : Optional callable(float) for chat model download progress.

    Returns:
        {
            "embedding_model"  : model object,
            "embedding_client" : client object,
            "chat_model"       : model object,
            "chat_client"      : client object,
            "chat_model_name"  : str,
        }

    Raises:
        FileNotFoundError: If knowledge_base.db has not been built yet.
                           Run ingest.py first.
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"knowledge_base.db not found at:\n  {DB_FILE}\n\n"
            "Run the ingestion pipeline first:\n"
            "  python ingest.py"
        )

    manager = init_sdk("smarthome_hub_rag")

    embedding_model  = load_model(manager, EMBEDDING_MODEL,  "embedding model", embed_progress_cb)
    embedding_client = embedding_model.get_embedding_client()

    chat_model  = load_model(manager, chat_model_name, "chat model", chat_progress_cb)
    chat_client = chat_model.get_chat_client()

    return {
        "embedding_model":  embedding_model,
        "embedding_client": embedding_client,
        "chat_model":       chat_model,
        "chat_client":      chat_client,
        "chat_model_name":  chat_model_name,
    }
