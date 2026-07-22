import os
import torch
from pathlib import Path
from typing import List, Optional
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from rich.console import Console

console = Console()

PINECONE_INDEX = "ella-medical"
NAMESPACE = "medical-books"

class EllaRetriever:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        console.print(f"[bold green]✓[/bold green] [dim]Hardware Acceleration: {self.device.upper()}[/dim]")
        
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        
        console.print("[dim]Loading Phase 7 Reranker (MiniLM-L6)...[/dim]")
        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            device=self.device
        )

        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)
        
        existing = [idx.name for idx in self.pc.list_indexes()]
        if PINECONE_INDEX not in existing:
            console.print(f"[bold red]FATAL ERROR:[/bold red] Index '{PINECONE_INDEX}' not found in Pinecone")
            self.index = None
        else:
            self.index = self.pc.Index(PINECONE_INDEX)
            stats = self.index.describe_index_stats()
            count = stats.namespaces.get(NAMESPACE, {}).vector_count
            console.print(f"[bold green]✓[/bold green] [dim]Pinecone online with {count} records.[/dim]")
            
    def hybrid_search(self, query: str, k: int = 5):
        """
        Optimized Hybrid Search via Pinecone:
        - Slam-Dunk Detection: Skip Reranking if score > 0.9.
        - Hardware Accelerated Reranking.
        """
        if not self.index:
            return []

        query_vector = self.embeddings.embed_query(query)
        
        results = self.index.query(
            vector=query_vector,
            top_k=15,
            namespace=NAMESPACE,
            include_metadata=True
        )

        if not results.matches:
            return []

        candidates = []
        for match in results.matches:
            meta = match.metadata
            text = meta.get("text", "")
            source = meta.get("source", "unknown")
            doc = Document(
                page_content=text,
                metadata={"source": source, "score": match.score, **{k: v for k, v in meta.items() if k not in ["text", "source"]}}
            )
            candidates.append(doc)

        top_score = results.matches[0].score
        if top_score > 0.90:
            console.print(f"[dim]⚡ Golden Match Found ({top_score:.2f}). Skipping Rerank.[/dim]")
            return candidates[:k]

        tokenized_corpus = [doc.page_content.lower().split() for doc in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_results = bm25.get_top_n(query.lower().split(), candidates, n=10)

        model_inputs = [[query, doc.page_content] for doc in bm25_results]
        
        scores = self.reranker.predict(model_inputs, batch_size=10, show_progress_bar=False)
        
        scored_docs = sorted(zip(scores, bm25_results), key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_docs[:k]]

if __name__ == "__main__":
    retriever = EllaRetriever()
    test_query = "What is the standard dosage for Metformin?"
    
    console.print(f"\n[bold yellow]Testing Optimized Reranking for:[/bold yellow] '{test_query}'")
    results = retriever.hybrid_search(test_query, k=3)
    
    if not results:
        console.print("[red]No results found.[/red]")
    else:
        for i, doc in enumerate(results):
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            console.print(f"\n[bold cyan]RANK {i+1} | Source: {source}[/bold cyan]")
            console.print(f"[white]{doc.page_content[:300]}...[/white]")