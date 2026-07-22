"""
Fast BM25 Retriever - reads text from SQLite, no embeddings needed.
Uses rank_bm25 for retrieval. Instant startup.
"""
import os
import json
import sqlite3
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from rich.console import Console

console = Console()

CHROMA_DB = str(Path(__file__).resolve().parent.parent / "data" / "vector_db" / "chroma.sqlite3")
CACHE_DIR = str(Path(__file__).resolve().parent.parent / "data" / "bm25_cache")

class BM25Retriever:
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        tokenized_cache = os.path.join(CACHE_DIR, "tokenized.pkl")
        meta_cache = os.path.join(CACHE_DIR, "metadata.json")
        
        if os.path.exists(tokenized_cache) and os.path.exists(meta_cache):
            console.print("[green]Loading cached BM25 index...[/green]")
            with open(tokenized_cache, "rb") as f:
                self.tokenized_corpus = pickle.load(f)
            with open(meta_cache, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            console.print("[cyan]Building BM25 index from SQLite...[/cyan]")
            self._build_index()
            with open(tokenized_cache, "wb") as f:
                pickle.dump(self.tokenized_corpus, f)
            with open(meta_cache, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False)
            console.print(f"[green]Cached to {CACHE_DIR}[/green]")
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.count = len(self.metadata)
        console.print(f"[green]BM25 Retriever ready: {self.count} records[/green]\n")
    
    def _build_index(self):
        conn = sqlite3.connect(CHROMA_DB)
        
        cursor = conn.execute("SELECT id, embedding_id FROM embeddings ORDER BY id")
        rows = cursor.fetchall()
        
        meta_cursor = conn.execute("SELECT id, key, string_value FROM embedding_metadata")
        all_meta = {}
        for row_id, key, value in meta_cursor.fetchall():
            if row_id not in all_meta:
                all_meta[row_id] = {}
            all_meta[row_id][key] = value
        conn.close()
        
        self.tokenized_corpus = []
        self.metadata = []
        
        for row_id, emb_id in rows:
            meta = all_meta.get(row_id, {})
            text = meta.get("chroma:document", "")
            source = meta.get("source", "")
            category = meta.get("category", "")
            page_label = meta.get("page_label", "")
            
            if text.strip():
                tokens = text.lower().split()
                self.tokenized_corpus.append(tokens)
                self.metadata.append({
                    "id": emb_id,
                    "text": text,
                    "source": os.path.basename(source) if source else "unknown",
                    "category": category,
                    "page_label": page_label,
                })
        
        console.print(f"[bold]Indexed {len(self.metadata)} documents[/bold]")
    
    def search(self, query, k=5):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
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
    retriever = BM25Retriever()
    
    test_queries = [
        "What is the standard dosage for Metformin?",
        "symptoms of heart attack",
        "chest pain and difficulty breathing",
        "bye goodbye",
    ]
    
    for q in test_queries:
        results = retriever.search(q, k=3)
        print(f"\nQuery: {q}")
        for i, r in enumerate(results):
            print(f"  Rank {i+1} (score: {r['score']:.4f}): {r['text'][:150]}")
