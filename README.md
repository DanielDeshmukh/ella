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
hardware: cpu-basic
---

> ⭐ If Ella's RAG-based medical triage architecture gave you ideas — a star helps other health-AI builders find it. Takes 2 seconds.

<div align="center">

# ELLA

### Medical Triage & Clinical RAG Engine

<br>

[![96% Accuracy](https://img.shields.io/badge/96%25_Accuracy-white?style=flat-square&labelColor=000000&color=000000&logo=checkmark&logoColor=white)](https://huggingface.co/spaces/Daniel2503/ella-medical)
[![90K Records](https://img.shields.io/badge/90K_Records-white?style=flat-square&labelColor=000000&color=000000&logo=database&logoColor=white)](https://huggingface.co/spaces/Daniel2503/ella-medical)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA_NIM-white?style=flat-square&labelColor=000000&color=000000&logo=nvidia&logoColor=white)](https://build.nvidia.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-white?style=flat-square&labelColor=000000&color=000000&logo=pinecone&logoColor=white)](https://pinecone.io)
[![Live Demo](https://img.shields.io/badge/Live_Demo-white?style=flat-square&labelColor=000000&color=000000&logo=gradio&logoColor=white)](https://huggingface.co/spaces/Daniel2503/ella-medical)
[![GitHub](https://img.shields.io/badge/Source_Code-white?style=flat-square&labelColor=000000&color=000000&logo=github&logoColor=white)](https://github.com/DanielDeshmukh/ella)
[![License](https://img.shields.io/badge/MIT_License-white?style=flat-square&labelColor=000000&color=000000&logo=open-source-initiative&logoColor=white)](https://opensource.org/licenses/MIT)

<br>

**Ella is a production-grade Retrieval-Augmented Generation (RAG) system purpose-built for medical triage.**  
She ingests 90,000+ clinical text chunks, embeds them via NVIDIA NIM, stores them in Pinecone, and retrieves context-grounded answers through a multi-stage pipeline — eliminating hallucinations in healthcare workflows.

[Key Features](#key-features) •
[Architecture](#architecture) •
[Live Demo](#live-demo) •
[Benchmark](#benchmark) •
[How It Works](#how-it-works)

</div>

---

## Key Features

- **90,000+ Clinical Chunks** — Ingested from medical handbooks, symptom guides, and pharmacology references
- **NVIDIA NIM Embeddings** — `nvidia/nv-embedqa-e5-v5` (1024-dim) via OpenAI-compatible API for semantic search
- **Pinecone Vector DB** — Serverless cloud storage with cosine similarity for instant retrieval
- **Hybrid Search** — Semantic (NIM) + BM25 keyword matching + CrossEncoder reranking
- **5-Class Intent Router** — Emergency, Triage, Booking, General Info, Closing — validated via Pydantic schemas
- **96% Intent Accuracy** — Benchmark-validated on 50 curated clinical queries
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

## Live Demo

<div align="center">

[![Ella Live Demo](https://img.shields.io/badge/TRY_ELLA_LIVE-6366f1?style=for-the-badge&logo=gradio&logoColor=white&labelColor=6366f1)](https://huggingface.co/spaces/Daniel2503/ella-medical)

**[→ Launch Ella on HuggingFace Spaces](https://huggingface.co/spaces/Daniel2503/ella-medical)**

</div>

Ella is deployed on HuggingFace Spaces with a custom dark medical UI. Type a medical question and see the full pipeline in action — intent classification, thought process, retrieved context, and grounded response.

---

## Benchmark

Evaluation on 50 curated clinical queries (20 Triage, 10 Emergency, 10 Booking, 5 General Info, 5 Closing):

| Metric | Value |
|--------|-------|
| **Intent Accuracy** | **96.0%** |
| **Avg Latency** | 9.26s |
| **Avg Retrieval Score** | 0.92 |
| **Records in DB** | 90,306 |

### Intent Breakdown

| Intent | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| EMERGENCY | 10 | 10 | **100%** |
| TRIAGE | 18 | 20 | 90% |
| BOOKING | 10 | 10 | **100%** |
| GENERAL_INFO | 5 | 5 | **100%** |
| CLOSING | 5 | 5 | **100%** |

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

Ella ships with a single command entry point:

```bash
# Install
pip install -e .

# Run
ella
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for healthcare AI**

[![Stars](https://img.shields.io/github/stars/DanielDeshmukh/ella?style=social)](https://github.com/DanielDeshmukh/ella)

</div>
