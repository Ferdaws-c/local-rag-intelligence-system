# 📄 Comprehensive Project Report
## Local RAG Assistant: Ferdaws Qaem Profile
**Author:** Ferdaws Qaem  
**Program:** Local AI with Microsoft Foundry Local Summer School  
**Date:** July 2026

---

## 1. Executive Summary
This report documents the design, implementation, and evaluation of a fully offline, AI-powered Q&A assistant. Built using Retrieval-Augmented Generation (RAG) and Microsoft Foundry Local, the system is designed to answer questions about my academic and professional profile. The core achievement of this project is a production-ready assistant that runs entirely locally, requiring zero cloud dependencies or API keys, ensuring complete data privacy and offline capability.

## 2. Problem Statement
Recruiters, hiring managers, and academic advisors often have to sift through multiple disparate documents (transcripts, resumes, project overviews) to find specific information about a candidate. 

Without AI, this process is manual and time-consuming. However, generic AI models (like standard ChatGPT) hallucinate or lack the specific, personal knowledge required to answer questions about a private individual. The goal of this project was to build a system that can accurately answer queries based **only** on my personal documents, providing precise, cited answers in seconds.

## 3. Technology Stack & Methodology
The system architecture was designed around the principles of RAG to ground the AI in factual data.

- **AI Runtime:** Microsoft Foundry Local with registered `CUDAExecutionProvider` GPU acceleration.
- **Embedding Model:** `qwen3-embedding-0.6b` (CUDA GPU variant).
- **Language Models (LLMs):** Selectable between `phi-3.5-mini` (balanced/accurate), `phi-4-mini` (best quality), and `qwen2.5-0.5b` (fastest).
- **Database Stores:** SQLite `knowledge_base.db` (vector store) and `chat_history.db` (chat history & settings persistence).
- **Memory Engine:** 5-Stage Deep Memory Offloader with Win32 process tree working set trimming (`SetProcessWorkingSetSize` + `EmptyWorkingSet`).
- **User Interface:** Streamlit with token-by-token streaming, clean non-cited response formatting, collapsible `📄 Sources` expanders, and manual/auto-free memory controls.

## 4. Implementation Details
The project was executed in five primary phases:

### 4.1 Data Ingestion (`ingest.py`)
A custom ingestion pipeline processes source documents (`official_transcript.docx`, `academic_and_experience.docx`, `.txt`, `.pdf`). The script chunks text into paragraph-level segments to balance context completeness with retrieval precision. Each chunk is passed to the Foundry Local embedding model and stored in `knowledge_base.db` alongside its vector representation.

### 4.2 Semantic Retrieval & Zero-Cost Synonym Expansion (`rag_core.py`)
When a user submits a query, it undergoes fast zero-cost synonym expansion via a Python dictionary lookup. The query is embedded, and the system computes cosine similarity against all stored vectors in SQLite. Chunks scoring below a `0.25` similarity threshold are filtered out, and the top matches form the grounded context window.

### 4.3 Prompt Engineering & Presentation Rules
To prevent hallucination and ensure clean presentation:
1. Strict anti-hallucination rules mandate that if the context does not contain the answer, the model must output EXACTLY: `"I don't have that information."`
2. Inline citation clutter (e.g. `(Source: ...)` or `Reference: ...`) is forbidden in the LLM output and filtered via regex. All document citations are displayed exclusively inside collapsible `📄 Sources` expanders below each answer.

### 4.4 5-Stage Deep Memory Engine & Self-Healing State (`sdk_utils.py`)
To prevent ONNX Runtime GenAI and CUDA memory heaps from permanently occupying multi-gigabyte RAM/VRAM:
- **5-Stage Offload**: On manual trigger (`⚡ Free Memory Now`) or background idle timeout, the system unloads C++ models, clears Streamlit caches, executes triple-pass `gc.collect()`, flushes PyTorch CUDA caches, and uses Win32 `OpenProcess` + `EmptyWorkingSet` to trim physical RAM to **~1–2 MB** and GPU VRAM to **0%**.
- **Self-Healing State**: `get_active_models()` checks model loading status on every new query and automatically re-initializes GPU resources if models were previously offloaded.

### 4.5 Generation & Persistent UI (`app.py` & `chat_history.py`)
The response is streamed token-by-token. Chat sessions, user messages, and UI preferences (such as the Auto-Free Memory timer setting) are persisted to SQLite (`chat_history.db`) and survive terminal restarts.

## 5. System Testing & Evaluation
An automated test suite (`test_suite.py`) evaluates the assistant's accuracy, reliability, and anti-hallucination guardrails across 12 test queries.

**Test Results Summary (100% Pass Rate):**
- **In-Context Queries (6/6 Passed):** The system successfully extracted facts (e.g., student ID, languages spoken, Erasmus program details) and presented clean answers with source citations in the expander.
- **Out-of-Context Queries (3/3 Passed):** The system correctly refused to answer questions outside the knowledge base (e.g., "Who is the CEO of Microsoft?"), proving the efficacy of the anti-hallucination prompt.
- **Edge Cases (3/3 Passed):** The system gracefully handled vague inputs (e.g., single characters, simple greetings) by safely falling back to the refusal message.
- **Performance:** CUDA GPU acceleration delivered average token generation speeds of 15-20 tok/s on local hardware.

## 6. Lessons Learned
1. **CUDA Memory Heap Persistence:** ONNX Runtime GenAI C++ allocators retain pinned memory pages. Implementing Win32 `OpenProcess` working set trimming was necessary to return memory to the OS kernel.
2. **Atomic Singleton Watchdogs:** Running background idle threads in Streamlit required strict thread locking (`threading.Lock`) to prevent duplicate watchdog loops.
3. **Clean Output Separation:** Instructing the model to state direct facts while placing citations in collapsible UI expanders created a cleaner user experience.

## 7. Conclusion
The Local RAG Intelligence System provides a robust, fast, memory-efficient, and 100% offline solution for querying personal documents, backed by a 100% pass rate in automated testing and enterprise-grade CUDA memory management.
