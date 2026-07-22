"""
Ella API — Medical Triage & Clinical RAG Engine
FastAPI endpoint for programmatic access
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agents.router import EllaRouter

app = FastAPI(
    title="Ella API",
    description="Medical Triage & Clinical RAG Engine",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = None


def get_router():
    global router
    if router is None:
        router = EllaRouter()
    return router


class QueryRequest(BaseModel):
    query: str
    history: str = ""


class QueryResponse(BaseModel):
    intent: str
    priority: str
    thought_process: str
    justification: str
    response: str
    retrieved_context: str


@app.get("/")
def root():
    return {
        "name": "Ella API",
        "version": "1.1.0",
        "description": "Medical Triage & Clinical RAG Engine",
        "accuracy": "96%",
        "records": 90306,
        "endpoints": {
            "POST /query": "Send a medical query",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    r = get_router()

    decision = r.route_request(req.query, history=req.history)
    context = getattr(decision, "retrieved_context", "")

    prompt = (
        "SYSTEM: You are ELLA, a clinical receptionist. Use the provided DOCUMENTS to guide the patient.\n"
        f"DOCUMENTS RETRIEVED:\n{context}\n\n"
        f"LATEST PATIENT INPUT: {req.query}\n\n"
        "STRICT PROTOCOL:\n"
        "1. INTEGRATE information naturally.\n"
        "2. NO REPETITION.\n"
        "3. BE CONCISE.\n"
        "4. Speak like a professional who knows the material by heart.\n"
        "5. If no documents are relevant, advise seeing a doctor."
    )

    final_res = r.raw_llm.invoke(prompt)
    output = final_res.content if hasattr(final_res, 'content') else str(final_res)

    return QueryResponse(
        intent=decision.intent,
        priority=decision.priority,
        thought_process=decision.thought_process,
        justification=decision.justification,
        response=output,
        retrieved_context=context,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
