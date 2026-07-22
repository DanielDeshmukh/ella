"""
Re-embed from SQLite using Ollama, store as numpy arrays for fast retrieval.
One-time build, then instant similarity search.
"""
import os
import sys
import json
import sqlite3
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

console = Console()

CHROMA_DB = str(ROOT_DIR / "src" / "data" / "vector_db" / "chroma.sqlite3")
CACHE_DIR = str(ROOT_DIR / "src" / "data" / "embed_cache")
EMBED_CACHE = os.path.join(CACHE_DIR, "vectors.npy")
META_CACHE = os.path.join(CACHE_DIR, "metadata.json")

def load_or_build_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    if os.path.exists(EMBED_CACHE) and os.path.exists(META_CACHE):
        console.print("[green]Loading cached embeddings...[/green]")
        vectors = np.load(EMBED_CACHE)
        with open(META_CACHE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        console.print(f"[green]✓ Loaded {len(vectors)} cached vectors[/green]")
        return vectors, metadata
    
    console.print("[cyan]No cache found. Building from SQLite + Ollama...[/cyan]")
    
    from langchain_ollama import OllamaEmbeddings
    embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")
    
    conn = sqlite3.connect(CHROMA_DB)
    cursor = conn.execute("SELECT id, embedding_id FROM embeddings ORDER BY id")
    rows = cursor.fetchall()
    total = len(rows)
    console.print(f"[bold]Found {total} records in SQLite[/bold]")
    
    # Load all metadata
    meta_cursor = conn.execute("SELECT id, key, string_value FROM embedding_metadata")
    all_meta = {}
    for row_id, key, value in meta_cursor.fetchall():
        if row_id not in all_meta:
            all_meta[row_id] = {}
        all_meta[row_id][key] = value
    conn.close()
    
    records = []
    for row_id, emb_id in rows:
        meta = all_meta.get(row_id, {})
        text = meta.get("chroma:document", "")
        source = meta.get("source", "")
        category = meta.get("category", "")
        page_label = meta.get("page_label", "")
        if text.strip():
            records.append({
                "id": emb_id,
                "text": text,
                "source": os.path.basename(source) if source else "unknown",
                "category": category,
                "page_label": page_label,
            })
    
    console.print(f"[bold]{len(records)} records with text. Embedding via Ollama...[/bold]")
    
    vectors = []
    metadata = []
    batch_size = 50
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding", total=len(records))
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            texts = [r["text"][:512] for r in batch]  # Truncate for embedding
            
            try:
                vecs = embeddings_model.embed_documents(texts)
                for j, (vec, rec) in enumerate(zip(vecs, batch)):
                    vectors.append(vec)
                    metadata.append({
                        "id": rec["id"],
                        "text": rec["text"],
                        "source": rec["source"],
                        "category": rec["category"],
                        "page_label": rec["page_label"],
                    })
            except Exception as e:
                console.print(f"[red]Batch error at {i}: {e}[/red]")
                # Try one by one
                for rec in batch:
                    try:
                        vec = embeddings_model.embed_query(rec["text"][:512])
                        vectors.append(vec)
                        metadata.append({
                            "id": rec["id"],
                            "text": rec["text"],
                            "source": rec["source"],
                            "category": rec["category"],
                            "page_label": rec["page_label"],
                        })
                    except Exception as e2:
                        console.print(f"[red]Skip: {str(e2)[:50]}[/red]")
            
            progress.update(task, advance=min(batch_size, len(records) - i))
    
    vectors = np.array(vectors, dtype=np.float32)
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors_norm = vectors / norms
    
    # Save cache
    np.save(EMBED_CACHE, vectors_norm)
    with open(META_CACHE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    
    console.print(f"[green]✓ Cached {len(vectors)} vectors to {CACHE_DIR}[/green]")
    return vectors_norm, metadata


class NumpyRetriever:
    def __init__(self):
        self.vectors, self.metadata = load_or_build_cache()
        self.count = len(self.metadata)
        console.print(f"[green]✓ Retriever ready with {self.count} records[/green]")
    
    def search(self, query_vector, k=5):
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        
        scores = np.dot(self.vectors, q)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            meta = self.metadata[idx]
            results.append({
                "id": meta["id"],
                "score": float(scores[idx]),
                "text": meta["text"],
                "source": meta["source"],
                "category": meta["category"],
            })
        return results


if __name__ == "__main__":
    retriever = NumpyRetriever()
    
    from langchain_ollama import OllamaEmbeddings
    embs = OllamaEmbeddings(model="mxbai-embed-large")
    
    test = "What is the standard dosage for Metformin?"
    qvec = embs.embed_query(test)
    results = retriever.search(qvec, k=3)
    
    print(f"\nQuery: {test}")
    for i, r in enumerate(results):
        print(f"\nRank {i+1} (score: {r['score']:.4f}):")
        print(f"  Source: {r['source']}")
        print(f"  Text: {r['text'][:200]}")
