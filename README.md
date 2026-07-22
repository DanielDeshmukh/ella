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

# ELLA

Medical Triage & Clinical RAG Engine

**96% intent accuracy** on 50 clinical queries | **90,306** medical records | NVIDIA NIM + Pinecone + Groq

## How it works

1. Patient types a medical question
2. Ella classifies intent (Emergency / Triage / Booking / General Info / Closing)
3. Hybrid retrieval searches 90k+ clinical text chunks via Pinecone
4. CrossEncoder reranks results for precision
5. LLM generates a grounded, source-attributed response

## Tech Stack

- **Embeddings:** NVIDIA NIM `nv-embedqa-e5-v5` (1024-dim)
- **Vector DB:** Pinecone (cosine similarity)
- **LLM:** Groq `llama-3.1-8b-instant`
- **Reranker:** CrossEncoder `ms-marco-MiniLM-L-6-v2`
