"""
Raw ChromaDB Retriever - reads directly from binary segment files + SQLite.
No chromadb dependency needed.
"""
import struct
import sqlite3
import os
import numpy as np
from pathlib import Path

class RawChromaRetriever:
    def __init__(self, db_dir=None):
        if db_dir is None:
            db_dir = str(Path(__file__).resolve().parent.parent / "data" / "vector_db")
        
        self.db_dir = db_dir
        self.dim = 1024
        
        # Find segment folder
        segments = [d for d in os.listdir(db_dir) 
                    if os.path.isdir(os.path.join(db_dir, d)) and d != "__pycache__"]
        
        if not segments:
            raise FileNotFoundError(f"No segment folders found in {db_dir}")
        
        seg_dir = os.path.join(db_dir, segments[0])
        
        # Load vectors from binary
        data_path = os.path.join(seg_dir, "data_level0.bin")
        file_size = os.path.getsize(data_path)
        num_vectors = file_size // (self.dim * 4)
        
        print(f"Loading {num_vectors} vectors from {data_path}...")
        with open(data_path, "rb") as f:
            raw = f.read()
        self.vectors = np.frombuffer(raw, dtype=np.float32).reshape(num_vectors, self.dim)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.vectors_normalized = self.vectors / norms
        
        # Load metadata from SQLite
        conn = sqlite3.connect(os.path.join(db_dir, "chroma.sqlite3"))
        cursor = conn.execute("SELECT id, embedding_id FROM embeddings ORDER BY id")
        self.embeddings = cursor.fetchall()
        
        # Load all metadata
        self.metadata = {}
        cursor = conn.execute("SELECT id, key, string_value FROM embedding_metadata")
        for row_id, key, value in cursor.fetchall():
            if row_id not in self.metadata:
                self.metadata[row_id] = {}
            self.metadata[row_id][key] = value
        
        conn.close()
        
        # Build mapping: vector_index -> metadata
        # The vectors are stored in the same order as embeddings table (by id)
        self.id_to_vec_idx = {}
        for idx, (row_id, emb_id) in enumerate(self.embeddings):
            if idx < num_vectors:
                self.id_to_vec_idx[row_id] = idx
        
        print(f"Loaded {len(self.embeddings)} metadata entries, {num_vectors} vectors")
    
    def search(self, query_vector, k=5):
        """Cosine similarity search using numpy."""
        # Normalize query
        q_norm = np.linalg.norm(query_vector)
        if q_norm > 0:
            query_normalized = np.array(query_vector, dtype=np.float32) / q_norm
        else:
            query_normalized = np.array(query_vector, dtype=np.float32)
        
        # Cosine similarity
        scores = np.dot(self.vectors_normalized, query_normalized)
        
        # Top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.embeddings):
                row_id, emb_id = self.embeddings[idx]
                meta = self.metadata.get(row_id, {})
                results.append({
                    "id": emb_id,
                    "score": float(scores[idx]),
                    "text": meta.get("chroma:document", ""),
                    "source": meta.get("source", "unknown"),
                    "category": meta.get("category", ""),
                    "page_label": meta.get("page_label", ""),
                })
        
        return results
    
    def count(self):
        return len(self.embeddings)

if __name__ == "__main__":
    retriever = RawChromaRetriever()
    print(f"\nTotal records: {retriever.count()}")
    
    # Test query
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    test_query = "What is the standard dosage for Metformin?"
    print(f"\nTest query: {test_query}")
    query_vec = embeddings.embed_query(test_query)
    
    results = retriever.search(query_vec, k=3)
    for i, r in enumerate(results):
        print(f"\n  Rank {i+1} (score: {r['score']:.4f}):")
        print(f"    Source: {os.path.basename(r['source'])}")
        print(f"    Text: {r['text'][:200]}")
