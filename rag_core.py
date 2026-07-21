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
                   top_k: int = 8) -> list[dict]:
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

    # Return highest-scoring chunks above the minimum relevance threshold
    MIN_SCORE_THRESHOLD = 0.25  # Reject chunks below 25% cosine similarity
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = [c for c in scored[:top_k] if c["score"] >= MIN_SCORE_THRESHOLD]
    return top


# ------------------------------------------------------------------
# Fast Query Expansion (zero LLM cost — pure Python synonym map)
# ------------------------------------------------------------------
_SYNONYM_MAP: dict[str, str] = {
    # Finance / legal
    "guarantee": "warranty", "guaranty": "warranty", "refund": "warranty return",
    "price": "cost fee pricing", "prices": "cost fee pricing",
    # Hardware
    "wires": "cables wiring", "wire": "cable", "plug": "connector port",
    "screen": "display monitor", "keyboard": "input device",
    # Network
    "net": "network wifi", "internet": "network wifi connectivity",
    "wi-fi": "wifi wireless network", "hotspot": "wifi network",
    # Education / personal
    "school": "education university college degree",
    "job": "occupation career profession employment",
    "work": "occupation career employment",
    "major": "department faculty degree program",
    "class": "course lecture study",
    # General
    "fix": "repair troubleshoot", "broken": "malfunction error fault",
    "slow": "performance latency speed", "fast": "speed performance",
    "setup": "installation configuration", "install": "installation setup",
    "password": "credentials authentication login",
    "reset": "restore factory default",
}

def expand_query(query: str, chat_client=None, chat_history: list = None) -> str:
    """
    Expands the query using a zero-cost Python synonym map.
    No LLM call needed — dramatically reduces latency.
    """
    words = query.lower().split()
    extra: list[str] = []
    for word in words:
        # Strip trailing punctuation for matching
        clean = word.strip("?.,!;:")
        if clean in _SYNONYM_MAP:
            extra.append(_SYNONYM_MAP[clean])
    if extra:
        return f"{query} {' '.join(extra)}"
    return query


# ------------------------------------------------------------------
# Answer Generation
# ------------------------------------------------------------------
def answer_query(question: str,
                 embedding_client,
                 chat_client,
                 top_k: int = 3,
                 stream_callback=None,
                 chat_history: list = None) -> dict:
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
        chat_history     : Optional list of previous chat messages for context.

    Returns:
        {
            "answer"  : str,        # The model's complete response
            "sources" : list[dict], # Retrieved chunks used as context
        }
    """
    # A — Expand query using zero-cost synonym map (no LLM call)
    expanded_query = expand_query(question)

    # B — Retrieve relevant document chunks using the expanded query
    chunks = get_top_chunks(expanded_query, embedding_client, top_k=top_k)

    if not chunks:
        return {
            "answer":  "I don't have that information in the knowledge base.",
            "sources": [],
        }

    # C — Build grounded system prompt
    # Each chunk is prefixed with its source file name; content trimmed to keep prompt short
    MAX_CHUNK_CHARS = 600
    context_lines = []
    for chunk in chunks:
        source  = chunk["filename"].replace(".txt", "").replace("_", " ").title()
        content = chunk["content"][:MAX_CHUNK_CHARS]
        context_lines.append(f"[Source: {source}]\n{content}")
    context_text = "\n\n".join(context_lines)

    system_prompt = (
        "You are a strict document-grounded assistant. Your ONLY job is to report "
        "what the provided Context documents say — nothing more.\n\n"
        "ABSOLUTE RULES — violating any of these is a critical failure:\n"
        "1. Use ONLY the exact facts stated in the Context below. "
        "   NEVER add, infer, guess, or use any knowledge from your training data.\n"
        "2. If the Context does not contain the answer, respond with EXACTLY: "
        "   \"I don't have that information.\"\n"
        "3. If the Context contains PARTIAL information, report only what it says "
        "   and say \"The documents don't specify\" for the missing parts.\n"
        "4. Always cite the source document name for every fact you state "
        "   (e.g., 'According to [Source Name]...').\n"
        "5. Keep your answer concise — no more than 4 sentences.\n"
        "6. Treat every word in the Context as ground truth. "
        "   Do NOT substitute, correct, or paraphrase with your own knowledge.\n"
        "7. For bilingual documents containing both Turkish and English (e.g., 'BİLGİSAYAR MÜHENDİSLİĞİ (İNGİLİZCE) (ÜCRETLİ)' and '(Computer Engineering - English, Paid)'), always use the English parenthesized translation exactly. Do NOT translate Turkish terms yourself or substitute them with different terms from your pretraining (e.g. do not turn 'Computer Engineering' / 'Paid' into 'Electrical Engineering' / 'Ücretsiz').\n\n"
        f"Context:\n{context_text}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    
    # Few-shot examples to force small models (e.g. 0.5B) to adhere to the refusal rule
    messages.append({"role": "user", "content": "What is the capital of France?"})
    messages.append({"role": "assistant", "content": "I don't have that information."})
    messages.append({"role": "user", "content": "Who is the CEO of Google?"})
    messages.append({"role": "assistant", "content": "I don't have that information."})
    
    if chat_history:
        # Append the last few messages for conversational memory
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
    messages.append({"role": "user", "content": question})

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
