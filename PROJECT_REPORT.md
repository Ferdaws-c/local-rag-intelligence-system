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

- **AI Runtime:** Microsoft Foundry Local (enables on-device LLM inference using CPU/GPU).
- **Embedding Model:** `qwen3-embedding-0.6b` (converts text to vector representations).
- **Language Models (LLMs):** Selectable between `qwen2.5-0.5b` (fast) and `phi-3.5-mini` (balanced/accurate).
- **Database:** SQLite (lightweight, serverless vector store).
- **User Interface:** Streamlit (rapid deployment of an interactive web application).
- **Similarity Metric:** Cosine Similarity.

## 4. Implementation Details
The project was executed in four primary phases:

### 4.1 Data Ingestion (`ingest.py`)
A custom ingestion pipeline was built to process five source documents (`official_transcript.txt`, `academic_and_experience.txt`, etc.). The script chunks the text into paragraph-level segments to balance context completeness with retrieval precision. Each chunk is passed to the Foundry Local embedding model and stored in a SQLite database (`knowledge_base.db`) alongside its vector representation.

### 4.2 Semantic Retrieval (`rag_core.py`)
When a user submits a query via the UI, the text is immediately embedded. The system retrieves all stored vectors from the SQLite database and calculates the cosine similarity against the query vector. The top 5 most relevant chunks (highest similarity scores) are retrieved to form the context window.

### 4.3 Prompt Engineering
To prevent hallucination—a critical requirement for an AI representing personal data—a strict system prompt was engineered. The retrieved chunks are injected into the prompt with explicit instructions:
> *"You are an AI assistant answering questions based ONLY on the provided context. If the answer is not contained in the context, you must reply 'I don't have that information'."*

### 4.4 Generation and User Interface (`app.py`)
The system passes the augmented prompt to the local chat model via the Foundry Local SDK. The response is streamed token-by-token to the Streamlit UI, mimicking the responsiveness of cloud-based LLMs. The UI also features session-based chat history, a sidebar to toggle between AI models on-the-fly, and an expandable panel showing the exact documents and similarity scores cited for the answer.

## 5. System Testing & Evaluation
An automated test suite (`test_suite.py`) was developed to evaluate the assistant's accuracy, reliability, and safeguard triggers. The suite executed 12 queries across three categories using the `phi-3.5-mini` model.

**Test Results Summary (100% Pass Rate):**
- **In-Context Queries (6/6 Passed):** The system successfully extracted facts (e.g., student ID, languages spoken, Erasmus program details) and accurately cited the source document.
- **Out-of-Context Queries (3/3 Passed):** The system correctly refused to answer questions outside the knowledge base (e.g., "Who is the CEO of Microsoft?"), proving the efficacy of the anti-hallucination prompt.
- **Edge Cases (3/3 Passed):** The system gracefully handled vague inputs (e.g., single characters, simple greetings) by safely falling back to the default refusal.
- **Performance:** Average response time was 14.2 seconds per query on standard hardware.

## 6. Lessons Learned
1. **Chunking Strategy is Critical:** Early iterations retrieved only the top 2 chunks, which often resulted in retrieving document titles rather than substance. Increasing to `top_k=5` drastically improved answer quality without overwhelming the model's context window.
2. **Strict Prompting:** Standard prompt instructions were insufficient for smaller models (like `qwen2.5-0.5b`). Using absolute terms ("STRICT", "EXACTLY") was necessary to enforce boundaries and prevent hallucination.
3. **Local AI Viability:** The project proved that running sophisticated AI pipelines locally is entirely feasible for production-like use cases. The privacy benefits of keeping data entirely on-device make this architecture highly suitable for personal or corporate knowledge bases.

## 7. Conclusion
The Local RAG Assistant successfully meets all requirements set out in the summer school program. It provides an intuitive, robust, and entirely offline solution for querying personal documents, backed by a 100% pass rate in automated testing. The system serves as a strong foundation for future enhancements, such as integrating more complex vector databases or processing additional file formats.
