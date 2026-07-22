# 🎓 Local RAG Intelligence System

An advanced, 100% offline AI-powered Q&A assistant built using **Retrieval-Augmented Generation (RAG)** and **Microsoft Foundry Local**. All AI inference and vector searches run entirely on-device with CUDA GPU acceleration — no cloud API keys or internet connection required.

🎬 **[Watch Live Project Demonstration Video](https://drive.google.com/file/d/1lNyXGFHMmbMe7b-jhc3UwWixI-JnVigW/view?usp=sharing)**

---

## 📋 Table of Contents
- [🎬 Live Demo Video](#-live-demo-video)
- [What Is This?](#what-is-this)
- [Key Features](#key-features)
- [Architecture & Memory Engine](#architecture--memory-engine)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Automated Testing](#automated-testing)
- [Performance & Hardware Requirements](#performance--hardware-requirements)
- [License](#license)

---

## 🎬 Live Demo Video

Watch a complete walkthrough and live demonstration of the system in action:

▶️ **[Watch the Video Demonstration on Google Drive](https://drive.google.com/file/d/1lNyXGFHMmbMe7b-jhc3UwWixI-JnVigW/view?usp=sharing)**

For the spoken video presentation guide and script, see **[DEMO_SCRIPT.md](DEMO_SCRIPT.md)**.

---

## What Is This?

This project is a production-grade local RAG application that combines:
- **Microsoft Foundry Local SDK** for running LLMs (`phi-3.5-mini`, `phi-4-mini`, `qwen2.5-0.5b`) and embedding models (`qwen3-embedding-0.6b`) on CUDA GPU acceleration.
- **SQLite Vector Store** with cosine similarity matching for document retrieval.
- **5-Stage Deep Memory Offloading Engine** that flushes VRAM to 0% and trims System RAM to ~1–2 MB after idle timeout or manual trigger.
- **Persistent Chat History & Settings** stored locally in `chat_history.db`.
- **Streamlit Modern Web Interface** featuring token-by-token streaming, self-healing model state verification, and clean collapsible source expanders.

---

## Key Features

- ⚡ **CUDA GPU Acceleration**: Automatically detects and registers `CUDAExecutionProvider` for high-throughput local token generation.
- 🧹 **5-Stage Memory Offloader**:
  - **Manual Offload**: `⚡ Free Memory Now` button in sidebar.
  - **Auto-Free Memory Timer**: Configurable timeout (`30s`, `2m`, `5m`, `30m`, `Keep`) that starts counting **only after response delivery** and offloads memory silently.
  - **Process Tree Memory Trimming**: Uses Win32 `OpenProcess` + `EmptyWorkingSet` to return System RAM to ~1–2 MB and GPU VRAM to 0%.
- 🛡️ **Self-Healing Model State**: Automatically detects offloaded models when a new question is typed, re-initializing GPU resources seamlessly without app crashes.
- 📄 **Clean Text Output**: Inline citations (`(Source: ...)` or `Reference: ...`) are stripped from the AI response and presented cleanly inside collapsible `📄 Sources` expanders.
- 💾 **Persistent Settings & History**: Chat history and Auto-Free timer preferences are saved to `chat_history.db` and persist across terminal restarts.

---

## 🎯 Grounded RAG Extraction Rules

The system enforces strict anti-hallucination and presentation rules across every prompt execution:

1. **Strict Context Grounding**: The LLM uses ONLY the exact facts written in the retrieved Context chunks; it is strictly forbidden to guess, extrapolate, or use outside knowledge.
2. **Deterministic Out-of-Context Refusal**: If the retrieved Context does not contain the answer, the assistant outputs EXACTLY: `"I don't have that information."`
3. **No Inline Citation Clutter**: The model is instructed to omit inline file references or source citations (e.g. `(Source: ...)` or `Reference: ...`). All document sources are cleanly structured and displayed below the answer in a dedicated collapsible `📄 Sources` UI expander.
4. **No Meta-Commentary or Disclaimers**: The LLM responds with direct, concise statements without preamble, notes, or parenthesized commentary.
5. **Zero-Cost Synonym Query Expansion**: Uses a fast Python dictionary lookup to expand domain synonyms before vector matching, ensuring high retrieval precision without extra LLM latency.
6. **Relevance Thresholding**: Rejects any retrieved vector chunks scoring below a 25% cosine similarity threshold (`score < 0.25`).

---

## Architecture & Memory Engine

```
 [User Prompt] ──► [Self-Healing State Check] ──► [CUDA Embedding Model] 
                                                        │
 [Collapsible UI Sources] ◄── [Streamed Response] ◄── [CUDA Chat LLM] ◄── [SQLite Vector Match]
                                                        │
                                                [Memory Offload Engine]
                                            (Unload C++ Models + Win32 Trim)
```

For complete technical specifications, see **[DOCUMENTATION.md](DOCUMENTATION.md)**.

---

## Project Structure

```
final_project/
│
├── app.py                  # Streamlit web interface & settings UI
├── rag_core.py             # RAG pipeline: vector search & prompt execution
├── sdk_utils.py            # Foundry Local SDK & 5-stage memory offload engine
├── chat_history.py         # Persistent chat session & settings SQLite storage
├── ingest.py               # Document chunking & embedding ingestion pipeline
├── test_suite.py           # Automated functional test runner
├── requirements.txt        # Python package dependencies
│
├── source_documents/       # Local knowledge base (.docx, .pdf, .txt supported)
├── chat_history.db         # Persistent chat session & preference database
└── knowledge_base.db       # Embedded vector database (generated via ingest.py)
```

---

## Setup & Installation

### Prerequisites
- Windows 10 / 11 (64-bit) with NVIDIA GPU (CUDA support recommended)
- Python 3.10 or later
- [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local) runtime installed

### 1. Clone & Navigate
```bash
git clone https://github.com/Ferdaws-c/local-rag-intelligence-system.git
cd local-rag-intelligence-system/final_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Ingest Documents into Vector Store
```bash
python ingest.py
```

---

## Running the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your web browser.

---

## Automated Testing

Run the 12-case automated functional test suite:
```bash
python test_suite.py
```
Results are saved to `test_results.txt`.

---

## Performance & Speed

Answer latency is dominated by the **local LLM**, not by retrieval. Every query prints a diagnostic line to the terminal running Streamlit, e.g.:

```
[perf] retrieval=0.41s  generation=18.7s  tokens=74  (4.0 tok/s)
```

- **`retrieval`** is the embedding + cosine search. This is nearly always < 1s and is *not* the problem.
- **`generation`** is the model producing the answer. This is where the seconds go. `tok/s` tells you your effective decode speed.

If `generation` is large, work through these in order of impact:

1. **Use hardware acceleration (biggest win, zero quality loss).** On CPU-only, a 3.8B model like `phi-3.5-mini` generates ~1.1 tokens/second. To enable NVIDIA GPU acceleration, `sdk_utils.py` automatically checks for and registers `CUDAExecutionProvider` via the Foundry Local SDK during app/script initialization (`manager.download_and_register_eps(["CUDAExecutionProvider"])`).
   
   - **One-time Process Launch Cost:** Registering CUDA EP is scoped per Python process and takes ~6-7 minutes on initial startup per process session (`streamlit run app.py`). Once registered, Streamlit's `@st.cache_resource` keeps the GPU-accelerated model warm for all subsequent queries.
   - **Verify Execution Providers & Model Variants:** Run the diagnostic script to check discoverable execution providers and active model variants for your system:
     ```bash
     python scratch/diagnose_gpu.py
     ```
     When registered, `diagnose_gpu.py` will report `ep=CUDAExecutionProvider` and `device=GPU` for selected catalog models (e.g. `Phi-3.5-mini-instruct-cuda-gpu:2`). Moving from CPU to GPU cuts query generation time dramatically.

2. **Pick a faster model in the sidebar.** `qwen2.5-0.5b` is ~7× smaller than `phi-3.5-mini` and responds far faster, at some cost to answer quality. Use it for quick demos; keep `phi-3.5-mini` (default) for graded accuracy.

3. **Keep the model warm.** The sidebar's *Auto-Free Memory* setting unloads the model after idle time. If it unloads between questions, the *next* question pays a reload penalty. During a demo, set it to **"Keep (Don't free)"** so every answer is measured against a warm model.

The pipeline itself is already tuned for speed without touching accuracy: the prompt instructions are compact, retrieved context is trimmed to its substantive lines, and generation is capped at a concise 3-sentence answer.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **SQLite over a dedicated vector DB** | Zero external dependencies; sufficient for a small knowledge base of ~30 chunks |
| **Paragraph-level chunking** | Balances retrieval precision and context completeness |
| **top_k = 5** | Captures both title paragraphs and content paragraphs in case they don't score equally |
| **200-token hard cap** | Prevents runaway CPU usage on small models given ambiguous or off-topic queries |
| **Streamlit over Flask/CLI** | Fastest path to a browser-based UI with minimal boilerplate; `@st.cache_resource` handles model caching elegantly |
| **Three selectable models** | Lets users trade response speed for answer quality without code changes |

---

## Limitations

- **Hallucination on small models**: `qwen2.5-0.5b` may occasionally ignore the "only use context" instruction.
- **English only**: The knowledge base and models are optimised for English.
- **Static knowledge base**: The assistant cannot answer about events or information not in the documents. Re-run `ingest.py` to update.
- **Single-machine only**: Designed for local use; not configured for multi-user cloud deployment.

---

## References

- [Microsoft Foundry Local GitHub](https://github.com/microsoft/Foundry-Local)
- [Building Your First Local RAG Application with Foundry Local](https://azurefeeds.com/2026/03/30/building-your-first-local-rag-application-with-foundry-local/)
- [Streamlit Documentation](https://docs.streamlit.io)
- Summer School Program: *Local AI with Microsoft Foundry Local* — Weeks 1–6
