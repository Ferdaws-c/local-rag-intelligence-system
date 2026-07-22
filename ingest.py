"""
ingest.py — Knowledge Base Ingestion Pipeline
================================================
Reads .txt, .pdf, and .docx files from source_documents/, splits them into
paragraph-level chunks, generates vector embeddings using
qwen3-embedding-0.6b, and stores them in SQLite (knowledge_base.db).

Run this script once before launching the app, and re-run it
any time you add or change documents in source_documents/.

Usage:
    python ingest.py
"""

import json
import sqlite3
import time
from pathlib import Path

from sdk_utils import init_sdk, load_model, UNIFIED_APP_NAME

# ------------------------------------------------------------------
# Configuration — all paths relative to this file's location
# ------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent
DOCUMENTS_DIR   = BASE_DIR / "source_documents"
DB_FILE         = BASE_DIR / "knowledge_base.db"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


# ------------------------------------------------------------------
# Document Readers
# ------------------------------------------------------------------
def read_txt(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8")


def read_pdf(filepath: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(filepath))
        pages  = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())
    except Exception as e:
        print(f"  WARNING: Could not read PDF {filepath.name}: {e}")
        return ""


def read_docx(filepath: Path) -> str:
    try:
        import docx
        doc = docx.Document(str(filepath))
        lines = []
        for element in doc.element.body:
            tag = element.tag.lower()
            if tag.endswith('p'):
                p = docx.text.paragraph.Paragraph(element, doc)
                txt = p.text.strip()
                if txt:
                    lines.append(txt)
            elif tag.endswith('tbl'):
                t = docx.table.Table(element, doc)
                for row in t.rows:
                    row_vals = []
                    for cell in row.cells:
                        ct = cell.text.strip()
                        if ct and (not row_vals or ct != row_vals[-1]):
                            row_vals.append(ct)
                    if row_vals:
                        lines.append(" | ".join(row_vals))
        return "\n\n".join(lines)
    except Exception as e:
        print(f"  WARNING: Could not read DOCX {filepath.name}: {e}")
        return ""


def read_document(filepath: Path) -> str:
    """Dispatch to the correct reader based on file extension."""
    ext = filepath.suffix.lower()
    if ext == ".txt":
        return read_txt(filepath)
    elif ext == ".pdf":
        return read_pdf(filepath)
    elif ext == ".docx":
        return read_docx(filepath)
    return ""


# ------------------------------------------------------------------
# Database Setup
# ------------------------------------------------------------------
def init_database(conn: sqlite3.Connection) -> None:
    """
    Creates the documents table if it doesn't exist.
    """
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            filename   TEXT NOT NULL,
            content    TEXT NOT NULL,
            embedding  TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

def remove_document_from_db(conn: sqlite3.Connection, filename: str) -> None:
    """Deletes all chunks belonging to a specific file and completely wipes traces from disk."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()
    # Vacuum reclaims the freed space and completely erases deleted data from the SQLite file
    cursor.execute("VACUUM")
    conn.commit()

def rename_document(old_name: str, new_name: str) -> dict:
    """Instantly updates a document's filename in the database without re-embedding."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE documents SET filename = ? WHERE filename = ?", (new_name, old_name))
        conn.commit()
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_in_db = cursor.fetchone()[0]
        conn.close()
        return {"status": "ok", "total": total_in_db}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ------------------------------------------------------------------
# Chunking Logic
# ------------------------------------------------------------------
def load_and_chunk_documents(documents_dir: Path, target_filename: str = None) -> list[dict]:
    """
    Reads files in documents_dir and splits them into overlapping line-window chunks.
    Each chunk is ~15 lines with a 5-line overlap so no single-line fields (like
    formatted transcripts or student records) are ever isolated or lost.
    If target_filename is provided, it only reads that specific file.
    """
    CHUNK_LINES   = 15   # lines per chunk
    OVERLAP_LINES = 8    # lines shared between consecutive chunks (increased to prevent field-boundary splits)

    all_chunks = []

    if target_filename:
        all_files = [documents_dir / target_filename]
        if not all_files[0].exists():
            raise FileNotFoundError(f"Target file not found: {all_files[0]}")
    else:
        all_files = sorted(
            f for f in documents_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    if not all_files:
        raise FileNotFoundError(
            f"No supported files (.txt, .pdf, .docx) found in: {documents_dir}"
        )

    for filepath in all_files:
        raw_text = read_document(filepath)
        if not raw_text:
            continue

        lines = [l for l in raw_text.splitlines()]  # keep ALL lines, even blank ones for spacing
        non_empty_lines = [l.strip() for l in lines if l.strip()]

        if not non_empty_lines:
            continue

        # Use the first non-empty line as the document title prefix
        title = non_empty_lines[0]

        # Sliding window over non-empty lines
        step   = max(1, CHUNK_LINES - OVERLAP_LINES)
        chunks_for_file = []
        i = 0
        while i < len(non_empty_lines):
            window = non_empty_lines[i : i + CHUNK_LINES]
            chunk_text = "\n".join(window)
            # Prefix every chunk with the document title to improve embedding relevance
            # This ensures that even raw tables/lists are semantically linked to the file topic.
            context_prefix = f"[Document: {filepath.name}]\n"
            if window[0] != title:
                content_to_embed = f"{context_prefix}{title}\n{chunk_text}"
            else:
                content_to_embed = f"{context_prefix}{chunk_text}"
            chunks_for_file.append({"filename": filepath.name, "content": content_to_embed})
            i += step

        print(f"  {filepath.name}: {len(chunks_for_file)} chunks (sliding window)")
        all_chunks.extend(chunks_for_file)

    return all_chunks


# ------------------------------------------------------------------
# Embedding Generation
# ------------------------------------------------------------------
def generate_embeddings(chunks: list[dict], embedding_client, batch_size: int = 128, progress_callback=None) -> list[dict]:
    """
    Sends chunk texts to the embedding model in sub-batches of 128
    and attaches the resulting vectors back to each chunk dict.
    Updates progress_callback live and pings MemoryMonitor during each batch.

    Returns:
        The same list with an 'embedding' key added to each dict.
    """
    try:
        from sdk_utils import MemoryMonitor
        MemoryMonitor.ping()
    except Exception:
        pass

    texts = [chunk["content"] for chunk in chunks]
    total = len(texts)

    for start_idx in range(0, total, batch_size):
        try:
            from sdk_utils import MemoryMonitor
            MemoryMonitor.ping()
        except Exception:
            pass

        if progress_callback and total > 0:
            pct = 0.30 + 0.58 * (start_idx / total)
            try:
                progress_callback(pct, f"Embedding chunk {start_idx}/{total} ({int(start_idx/total*100)}%)...")
            except Exception:
                pass

        batch_texts = texts[start_idx : start_idx + batch_size]
        response = embedding_client.generate_embeddings(batch_texts)

        for j, item in enumerate(response.data):
            chunks[start_idx + j]["embedding"] = item.embedding

    try:
        from sdk_utils import MemoryMonitor
        MemoryMonitor.ping()
    except Exception:
        pass

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
# Callable API (used by app.py for in-app re-ingestion)
# ------------------------------------------------------------------
def clear_knowledge_base(progress_callback=None) -> dict:
    """
    Completely wipes all documents and embeddings from the knowledge base.
    Runs VACUUM afterward to physically erase all data from the SQLite file.

    Parameters:
        progress_callback : Optional callable(pct: float, label: str).
    """
    def _report(pct: float, label: str):
        if progress_callback:
            try:
                progress_callback(pct, label)
            except Exception:
                pass

    try:
        _report(0.0, "Connecting to database…")
        time.sleep(0.4)

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        _report(0.2, "Counting existing chunks…")
        time.sleep(0.4)
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_before = cursor.fetchone()[0]

        _report(0.45, f"Deleting {total_before} chunks…")
        time.sleep(0.5)
        cursor.execute("DELETE FROM documents")
        conn.commit()

        _report(0.70, "Running VACUUM to erase all traces…")
        time.sleep(0.5)
        cursor.execute("VACUUM")
        conn.commit()
        conn.close()

        _report(1.0, "Knowledge base cleared!")
        return {"status": "ok", "deleted": total_before}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_ingestion(embedding_client, progress_callback=None, target_file=None, is_delete=False) -> dict:
    """
    Programmatic entry point for triggering ingestion from the Streamlit app.
    
    Parameters:
        embedding_client  : Active Foundry Local embedding client.
        progress_callback : Optional callable(pct: float, label: str).
        target_file       : If set, only this file is processed (incremental).
        is_delete         : If True, target_file is removed from DB without embedding.
    """
    try:
        from sdk_utils import MemoryMonitor
        MemoryMonitor.set_busy(True)
        MemoryMonitor.ping()
    except Exception:
        pass

    def _report(pct: float, label: str):
        try:
            from sdk_utils import MemoryMonitor
            MemoryMonitor.ping()
        except Exception:
            pass
        if progress_callback:
            try:
                progress_callback(pct, label)
            except Exception:
                pass

    try:
        conn = sqlite3.connect(DB_FILE)
        init_database(conn)

        if is_delete and target_file:
            _report(0.5, f"Deleting {target_file} from database…")
            time.sleep(1.0) # Ensure the progress bar flashes in the UI before finishing
            remove_document_from_db(conn, target_file)
            
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_in_db = cursor.fetchone()[0]
            conn.close()
            
            _report(1.0, "Done!")
            return {"status": "ok", "total": total_in_db, "files": []}

        # For full rebuilds, clear everything first
        if not target_file:
            _report(0.0, "Clearing old database…")
            conn.cursor().execute("DROP TABLE IF EXISTS documents")
            init_database(conn)

        _report(0.05, "Reading & chunking documents…")
        chunks = load_and_chunk_documents(DOCUMENTS_DIR, target_filename=target_file)

        _report(0.30, f"Generating embeddings for {len(chunks)} chunks…")
        chunks = generate_embeddings(chunks, embedding_client, progress_callback=_report)

        _report(0.88, "Writing to database…")
        # If updating a single file, clear its old chunks first
        if target_file:
            remove_document_from_db(conn, target_file)
            
        insert_chunks(conn, chunks)
        
        # Get total chunks currently in DB for reporting
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_in_db = cursor.fetchone()[0]
        conn.close()

        _report(1.0, "Done!")
        return {"status": "ok", "total": total_in_db, "files": []}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        try:
            from sdk_utils import MemoryMonitor
            MemoryMonitor.set_busy(False)
            MemoryMonitor.ping()
        except Exception:
            pass


# ------------------------------------------------------------------
# Entry Point (CLI usage)
# ------------------------------------------------------------------
def main():
    print("=" * 50)
    print("  Local RAG Intelligence System — Ingestion Pipeline")
    print("=" * 50)

    # 1. Initialize SDK and load embedding model
    print("\n[1/4] Loading embedding model...")
    manager          = init_sdk(UNIFIED_APP_NAME)
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
        # Explicitly drop for CLI full rebuild
        conn.cursor().execute("DROP TABLE IF EXISTS documents")
        init_database(conn)
        insert_chunks(conn, chunks)
        print_report(conn)
    finally:
        conn.close()
        embedding_model.unload()


if __name__ == "__main__":
    main()
