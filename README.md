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
[Benchmark](#benchmark) •
[Getting Started](#getting-started) •
[Roadmap](#roadmap)

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

## Benchmark

Evaluation on 50 curated clinical queries (20 Triage, 10 Emergency, 10 Booking, 5 General Info, 5 Closing):

| Metric | Value |
|--------|-------|
| **Intent Accuracy** | 76.0% |
| **Avg Latency** | 8.95s |
| **Avg Retrieval Score** | 0.92 |
| **Records in DB** | 90,306 |

### Intent Breakdown

| Intent | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| EMERGENCY | 9 | 10 | 90% |
| TRIAGE | 12 | 20 | 60% |
| BOOKING | 8 | 10 | 80% |
| GENERAL_INFO | 5 | 5 | 100% |
| CLOSING | 4 | 5 | 80% |

---

## Getting Started

### Prerequisites

- Python 3.11+
- [NVIDIA API Key](https://build.nvidia.com) (free tier available)
- [Pinecone API Key](https://pinecone.io)
- [Groq API Key](https://groq.com)

### Installation

```bash
git clone https://github.com/DanielDeshmukh/ella.git
cd ella
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:

```env
GROQ_API_KEY=gsk_xxxxx
GROQ_MODEL=llama-3.1-8b-instant
PINECONE_API_KEY=pcsk_xxxxx
NVIDIA_API_KEY=nvapi_xxxxx
```

### Data Ingestion

```bash
# Ingest PDFs into Pinecone
python -m src.engine.ingest

# Or run migration from old ChromaDB
python -m src.engine.migrate_chroma_to_pinecone
```

### Run Evaluation

```bash
python -m src.evaluation
```

### Start CLI

```bash
python -m src.main
```

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

## Project Structure

```
ella/
├── src/
│   ├── agents/
│   │   ├── router.py              # Intent classification (5-class)
│   │   └── guardrails.py          # Emergency detection
│   ├── engine/
│   │   ├── retriever.py           # Pinecone + NIM hybrid search
│   │   ├── nim_embeddings.py      # NVIDIA NIM API wrapper
│   │   ├── bm25_retriever.py      # BM25 keyword retrieval
│   │   ├── ingest.py              # PDF → Pinecone pipeline
│   │   └── migrate_chroma_to_pinecone.py  # Batch migration
│   ├── evaluation.py              # 50-query benchmark
│   └── main.py                    # CLI entry point
├── requirements.txt
├── .env                           # API keys (gitignored)
└── README.md
```

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **1–8** | Core RAG, Vector DB, Hybrid Search, Guardrails | ✅ Complete |
| **9–12** | PostgreSQL backend, Patient records, Memory | 🔜 Planned |
| **13–16** | Voice (Vapi/Twilio), STT/TTS, <1.5s latency | 🔜 Planned |
| **17–20** | WhatsApp/SMS, Admin dashboard, HIPAA compliance | 🔜 Planned |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for healthcare AI**

[![Stars](https://img.shields.io/github/stars/DanielDeshmukh/ella?style=social)](https://github.com/DanielDeshmukh/ella)

</div>
