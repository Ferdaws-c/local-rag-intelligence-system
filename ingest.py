"""
ingest.py — Knowledge Base Ingestion Pipeline
================================================
Reads all .txt files from source_documents/, splits them into
paragraph-level chunks, generates vector embeddings using
qwen3-embedding-0.6b, and stores them in SQLite (knowledge_base.db).

Run this script once before launching the app, and re-run it
any time you add or change documents in source_documents/.

Usage:
    python ingest.py
"""

import json
import sqlite3
from pathlib import Path

from sdk_utils import init_sdk, load_model

# ------------------------------------------------------------------
# Configuration — all paths relative to this file's location
# ------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "source_documents"
DB_FILE       = BASE_DIR / "knowledge_base.db"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"


# ------------------------------------------------------------------
# Database Setup
# ------------------------------------------------------------------
def init_database(conn: sqlite3.Connection) -> None:
    """
    Drops and recreates the documents table for a clean rebuild.
    This ensures no stale data remains when documents are updated.
    """
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS documents")
    cursor.execute("""
        CREATE TABLE documents (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            filename   TEXT NOT NULL,
            content    TEXT NOT NULL,
            embedding  TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


# ------------------------------------------------------------------
# Chunking Logic
# ------------------------------------------------------------------
def load_and_chunk_documents(documents_dir: Path) -> list[dict]:
    """
    Reads every .txt file in documents_dir and splits each file
    on double-newline boundaries to produce paragraph chunks.

    Returns:
        List of dicts: [{"filename": str, "content": str}, ...]
    """
    all_chunks = []
    txt_files  = sorted(documents_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {documents_dir}")

    for filepath in txt_files:
        raw_text   = filepath.read_text(encoding="utf-8")
        raw_chunks = raw_text.split("\n\n")
        valid_chunks = [c.strip() for c in raw_chunks if c.strip()]

        print(f"  {filepath.name}: {len(valid_chunks)} chunks")
        for chunk in valid_chunks:
            all_chunks.append({"filename": filepath.name, "content": chunk})

    return all_chunks


# ------------------------------------------------------------------
# Embedding Generation
# ------------------------------------------------------------------
def generate_embeddings(chunks: list[dict], embedding_client) -> list[dict]:
    """
    Sends all chunk texts to the embedding model in a single batch request
    and attaches the resulting vectors back to each chunk dict.

    Returns:
        The same list with an 'embedding' key added to each dict.
    """
    texts    = [chunk["content"] for chunk in chunks]
    response = embedding_client.generate_embeddings(texts)

    for i, item in enumerate(response.data):
        chunks[i]["embedding"] = item.embedding

    return chunks


# ------------------------------------------------------------------
# Database Insertion
# ------------------------------------------------------------------
def insert_chunks(conn: sqlite3.Connection, chunks: list[dict]) -> None:
    """
    Inserts all chunks into the documents table.
    Embedding vectors are serialized as JSON strings for storage.
    """
    cursor = conn.cursor()
    for chunk in chunks:
        cursor.execute(
            "INSERT INTO documents (filename, content, embedding) VALUES (?, ?, ?)",
            (chunk["filename"], chunk["content"], json.dumps(chunk["embedding"]))
        )
    conn.commit()


# ------------------------------------------------------------------
# Verification Report
# ------------------------------------------------------------------
def print_report(conn: sqlite3.Connection) -> None:
    """Prints a summary of what was stored to verify ingestion success."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT filename, COUNT(*) FROM documents
        GROUP BY filename ORDER BY filename
    """)
    per_file = cursor.fetchall()

    print("\n" + "=" * 50)
    print("  INGESTION COMPLETE")
    print("=" * 50)
    print(f"  Total chunks stored : {total}")
    print(f"  Database            : {DB_FILE}")
    print(f"\n  Breakdown by file:")
    for filename, count in per_file:
        print(f"    {filename:<30} {count} chunks")
    print("=" * 50)


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------
def main():
    print("=" * 50)
    print("  SmartHome Hub — Ingestion Pipeline")
    print("=" * 50)

    # 1. Initialize SDK and load embedding model
    print("\n[1/4] Loading embedding model...")
    manager          = init_sdk("smarthome_hub_rag")
    embedding_model  = load_model(manager, EMBEDDING_MODEL, "embedding model")
    embedding_client = embedding_model.get_embedding_client()

    # 2. Read and chunk documents
    print(f"\n[2/4] Reading documents from: {DOCUMENTS_DIR}")
    chunks = load_and_chunk_documents(DOCUMENTS_DIR)
    print(f"  Total: {len(chunks)} chunks across all files")

    # 3. Generate embeddings
    print("\n[3/4] Generating embeddings (batch)...")
    chunks = generate_embeddings(chunks, embedding_client)

    # 4. Store in SQLite
    print(f"\n[4/4] Writing to database...")
    conn = sqlite3.connect(DB_FILE)
    try:
        init_database(conn)
        insert_chunks(conn, chunks)
        print_report(conn)
    finally:
        conn.close()
        embedding_model.unload()


if __name__ == "__main__":
    main()
