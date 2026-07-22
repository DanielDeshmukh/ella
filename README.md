> ⭐ If Ella's RAG-based medical triage architecture gave you ideas — a star helps other health-AI builders find it. Takes 2 seconds.

<div align="center">

# ELLA

### Medical Triage & Clinical RAG Engine

<br>

[![PyPI](https://img.shields.io/badge/PyPI_ella--sdk-white?style=flat-square&labelColor=000000&color=000000&logo=pypi&logoColor=white)](https://pypi.org/project/ella-sdk/)
[![96% Accuracy](https://img.shields.io/badge/96%25_Accuracy-white?style=flat-square&labelColor=000000&color=000000&logo=checkmark&logoColor=white)](https://huggingface.co/spaces/Daniel2503/ella-medical)
[![90K Records](https://img.shields.io/badge/90K_Records-white?style=flat-square&labelColor=000000&color=000000&logo=database&logoColor=white)](https://huggingface.co/spaces/Daniel2503/ella-medical)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA_NIM-white?style=flat-square&labelColor=000000&color=000000&logo=nvidia&logoColor=white)](https://build.nvidia.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-white?style=flat-square&labelColor=000000&color=000000&logo=pinecone&logoColor=white)](https://pinecone.io)
[![Live Demo](https://img.shields.io/badge/Live_Demo-white?style=flat-square&labelColor=000000&color=000000&logo=gradio&logoColor=white)](https://huggingface.co/spaces/Daniel2503/ella-medical)
[![GitHub](https://img.shields.io/badge/Source_Code-white?style=flat-square&labelColor=000000&color=000000&logo=github&logoColor=white)](https://github.com/DanielDeshmukh/ella)

<br>

**Ella is a production-grade Retrieval-Augmented Generation (RAG) system purpose-built for medical triage.**
She ingests 90,000+ clinical text chunks, embeds them via NVIDIA NIM, stores them in Pinecone, and retrieves context-grounded answers through a multi-stage pipeline — eliminating hallucinations in healthcare workflows.

[Install](#install) •
[Quick Start](#quick-start) •
[Architecture](#architecture) •
[Live Demo](#live-demo) •
[Benchmark](#benchmark)

</div>

---

## Install

```bash
pip install ella-sdk
```

---

## Quick Start

```python
from ella_medical import Ella

client = Ella()
response = client.query("What are the symptoms of a heart attack?")

print(response.intent)         # "TRIAGE"
print(response.response)       # Grounded clinical response
```

---

## Usage

### Basic Query

```python
from ella_medical import Ella

client = Ella()
response = client.query("What are the symptoms of a heart attack?")

print(response.intent)              # "TRIAGE"
print(response.priority)            # Priority level
print(response.thought_process)     # Router's reasoning
print(response.response)            # Ella's response
print(response.retrieved_context)   # Retrieved medical documents
```

### Multi-Turn Conversation

```python
from ella_medical import Ella

client = Ella()

# First message
r1 = client.query("I have chest pain")

# Follow-up
r2 = client.query(
    "What about treatment options?",
    history=f"Patient: I have chest pain\nElla: {r1.response}"
)

print(r2.response)
```

### Context Manager

```python
from ella_medical import Ella

with Ella() as client:
    response = client.query("What are the symptoms of diabetes?")
    print(response.response)
```

### Response Object

```python
@dataclass
class QueryResponse:
    intent: str            # EMERGENCY | TRIAGE | BOOKING | GENERAL_INFO | CLOSING
    priority: str          # Priority level
    thought_process: str   # Router's reasoning
    justification: str     # Clinical justification
    response: str          # Ella's response
    retrieved_context: str # Retrieved medical documents
```

---

## Live Demo

<div align="center">

[![Try Ella Live](https://img.shields.io/badge/TRY_ELLA_LIVE-white?style=flat-square&labelColor=000000&color=000000&logo=gradio&logoColor=white)](https://huggingface.co/spaces/Daniel2503/ella-medical)

**[→ Launch Ella on HuggingFace Spaces](https://huggingface.co/spaces/Daniel2503/ella-medical)**

</div>

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

| Metric | Value |
|--------|-------|
| **Intent Accuracy** | **96.0%** |
| **Avg Latency** | 9.26s |
| **Avg Retrieval Score** | 0.92 |
| **Records in DB** | 90,306 |

| Intent | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| EMERGENCY | 10 | 10 | **100%** |
| TRIAGE | 18 | 20 | 90% |
| BOOKING | 10 | 10 | **100%** |
| GENERAL_INFO | 5 | 5 | **100%** |
| CLOSING | 5 | 5 | **100%** |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **SDK** | `ella-sdk` (PyPI) | Python client |
| **Embeddings** | NVIDIA NIM (`nv-embedqa-e5-v5`) | 1024-dim semantic vectors |
| **Vector DB** | Pinecone (Serverless, AWS) | Cosine similarity search |
| **LLM** | Groq (`llama-3.1-8b-instant`) | Intent classification + response generation |
| **Reranker** | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Precision reranking |
| **Orchestration** | LangChain + LangGraph | Agent pipeline |
| **Validation** | Pydantic | Schema-validated outputs |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for healthcare AI**

[![Stars](https://img.shields.io/github/stars/DanielDeshmukh/ella?style=social)](https://github.com/DanielDeshmukh/ella)

</div>
