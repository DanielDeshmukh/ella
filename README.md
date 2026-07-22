---
title: Ella — Medical Triage & Clinical RAG Engine
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: true
license: mit
---

> ⭐ If Ella's RAG-based medical triage architecture gave you ideas — a star helps other health-AI builders find it. Takes 2 seconds.

<div align="center">

# ELLA

### Medical Triage & Clinical RAG Engine

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA_NIM-Embeddings-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://build.nvidia.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?style=for-the-badge)](https://pinecone.io)
[![Groq](https://img.shields.io/badge/Groq-Inference-E55B3C?style=for-the-badge)](https://groq.com)
[![LangChain](https://img.shields.io/badge/LangChain-Agents-1C3C3C?style=for-the-badge)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Ella is a production-grade Retrieval-Augmented Generation (RAG) system purpose-built for medical triage.**  
It ingests 90,000+ clinical text chunks, embeds them via NVIDIA NIM, stores them in Pinecone, and retrieves context-grounded answers through a multi-stage pipeline — eliminating hallucinations in healthcare workflows.

[Key Features](#key-features) •
[Architecture](#architecture) •
[Demo](#demo) •
[Benchmark](#benchmark) •
[How It Works](#how-it-works) •
[CLI](#cli)

</div>

---

## Key Features

- **90,000+ Clinical Chunks** — Ingested from medical handbooks, symptom guides, and pharmacology references
- **NVIDIA NIM Embeddings** — `nvidia/nv-embedqa-e5-v5` (1024-dim) via OpenAI-compatible API for semantic search
- **Pinecone Vector DB** — Serverless cloud storage with cosine similarity for instant retrieval
- **Hybrid Search** — Semantic (NIM) + BM25 keyword matching + CrossEncoder reranking
- **5-Class Intent Router** — Emergency, Triage, Booking, General Info, Closing — validated via Pydantic schemas
- **76% Intent Accuracy** — Benchmark-validated on 50 curated clinical queries
- **Guardrail System** — Emergency detection prevents life-threatening cases from being misrouted
- **Multi-Turn State Awareness** — Conversation history maintained across triage sessions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PATIENT INPUT                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  INTENT ROUTER (Groq llama-3.1-8b-instant + Pydantic Schema)      │
│  Classifies: EMERGENCY │ TRIAGE │ BOOKING │ GENERAL_INFO │ CLOSING │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌──────────┐  ┌──────────────┐  ┌──────────┐
      │ EMERGENCY │  │    TRIAGE    │  │ BOOKING  │
      │ GUARDRAIL │  │  RAG SEARCH  │  │ HANDLER  │
      └──────────┘  └──────┬───────┘  └──────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE                               │
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐  │
│  │  NVIDIA NIM  │   │   BM25      │   │  CrossEncoder Reranker  │  │
│  │  Embeddings  │ + │  Keyword    │ → │  ms-marco-MiniLM-L-6   │  │
│  │  (Semantic)  │   │  Matching   │   │  (Top-10 → Top-3)      │  │
│  └──────┬──────┘   └──────┬──────┘   └───────────┬─────────────┘  │
│         │                 │                      │                  │
│         ▼                 ▼                      ▼                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              PINECONE VECTOR DATABASE                       │   │
│  │         90,306 vectors • cosine • 1024 dimensions           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SYNTHESIS (Groq llama-3.1-8b-instant)                             │
│  Grounded response + clinical justification + source attribution    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PATIENT RESPONSE                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Demo

<div align="center">

[![Ella Demo](https://img.shields.io/badge/WATCH-DEMO-E55B3C?style=for-the-badge&logo=youtube&logoColor=white)](https://github.com/DanielDeshmukh/ella/blob/main/docs/assets/building_medical_intelligence_with_ella_the_hard_r.mp4)

*Building Medical Intelligence with Ella — First LinkedIn Demo*

</div>

> 📹 [Watch the full demo video](docs/assets/building_medical_intelligence_with_ella_the_hard_r.mp4) — See Ella perform real-time medical triage, retrieve clinical context from 90k+ records, and generate grounded responses.

---

## Benchmark

Evaluation on 50 curated clinical queries (20 Triage, 10 Emergency, 10 Booking, 5 General Info, 5 Closing):

| Metric | Value |
|--------|-------|
| **Intent Accuracy** | 96.0% |
| **Avg Latency** | 9.26s |
| **Avg Retrieval Score** | 0.92 |
| **Records in DB** | 90,306 |

### Intent Breakdown

| Intent | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| EMERGENCY | 10 | 10 | 100% |
| TRIAGE | 18 | 20 | 90% |
| BOOKING | 10 | 10 | 100% |
| GENERAL_INFO | 5 | 5 | 100% |
| CLOSING | 5 | 5 | 100% |

---

## How It Works

### 1. Intent Routing
Every patient message passes through a **State-Aware Router** powered by Groq's `llama-3.1-8b-instant`. The router analyzes both the current input and conversation history, classifying intent into one of five streams: Emergency, Triage, Booking, General Info, or Closing. Outputs are validated via Pydantic schemas to guarantee structured responses.

### 2. Emergency Guardrail
Before any retrieval occurs, the system checks for life-threatening keywords and patterns. If detected, Ella immediately escalates to emergency protocol — bypassing the standard RAG pipeline to ensure patient safety.

### 3. Hybrid Retrieval
For clinical queries, Ella executes a **three-stage retrieval pipeline**:
- **Stage 1 — Semantic Search:** NVIDIA NIM embeddings (`nv-embedqa-e5-v5`) convert the query into a 1024-dimensional vector. Pinecone returns the top-15 candidates via cosine similarity.
- **Stage 2 — BM25 Reranking:** A keyword-based BM25 algorithm re-scores candidates to capture medical terminology that semantic search might miss.
- **Stage 3 — CrossEncoder Precision:** A `ms-marco-MiniLM-L-6-v2` cross-encoder reranks the top-10 results, selecting the top-3 most relevant passages.

### 4. Grounded Synthesis
The LLM receives the patient query alongside the retrieved clinical context. Every response includes:
- **Clinical justification** — why this information applies
- **Source attribution** — which medical handbook the data came from
- **Confidence scoring** — retrieval similarity score for transparency

### 5. Multi-Turn State
Ella maintains conversation history across sessions. If a patient says "yes" after describing symptoms, the router understands the context and continues the triage flow — enabling natural, human-like clinical conversations.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Embeddings** | NVIDIA NIM (`nv-embedqa-e5-v5`) | 1024-dim semantic vectors |
| **Vector DB** | Pinecone (Serverless, AWS) | Cosine similarity search |
| **LLM** | Groq (`llama-3.1-8b-instant`) | Intent classification + response generation |
| **Reranker** | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Precision reranking |
| **Orchestration** | LangChain + LangGraph | Agent pipeline |
| **Validation** | Pydantic | Schema-validated outputs |
| **Data** | SQLite + PDFs | 90k clinical text chunks |

---

## CLI

Ella ships with a structured CLI entry point:

```bash
# Install
pip install -e .

# Commands
ella chat        # Start interactive triage session
ella eval        # Run 50-query benchmark
ella stats       # Show Pinecone index stats
ella version     # Show version
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for healthcare AI**

[![Stars](https://img.shields.io/github/stars/DanielDeshmukh/ella?style=social)](https://github.com/DanielDeshmukh/ella)

</div>
