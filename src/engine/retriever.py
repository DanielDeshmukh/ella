import os
import torch
from pathlib import Path
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from rich.console import Console

console = Console()

class EllaRetriever:
    def __init__(self):
        current_file_dir = Path(__file__).resolve().parent 
        self.vector_db_path = str(current_file_dir.parent / "data" / "vector_db")
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        console.print(f"[dim]Retriever targeting: {self.vector_db_path}[/dim]")
        console.print(f"[bold green]✓[/bold green] [dim]Hardware Acceleration: {self.device.upper()}[/dim]")
        
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        
        console.print("[dim]Loading Phase 7 Reranker (MiniLM-L6)...[/dim]")
        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            device=self.device
        )
        
        if not os.path.exists(self.vector_db_path):
            console.print(f"[bold red]FATAL ERROR:[/bold red] DB not found at {self.vector_db_path}")
            self.db = None
        else:
            self.db = Chroma(
                persist_directory=self.vector_db_path, 
                embedding_function=self.embeddings
            )
            count = self.db._collection.count()
            console.print(f"[bold green]✓[/bold green] [dim]System online with {count} records.[/dim]")
            
    def hybrid_search(self, query: str, k: int = 5):
        """
        Optimized Hybrid Search:
        - Slam-Dunk Detection: Skip Reranking if score > 0.9.
        - Hardware Accelerated Reranking.
        """
        if not self.db:
            return []

        results_with_scores = self.db.similarity_search_with_relevance_scores(query, k=15)
        
        if not results_with_scores:
            return []

        top_score = results_with_scores[0][1]
        if top_score > 0.90:
            console.print(f"[dim]⚡ Golden Match Found ({top_score:.2f}). Skipping Rerank.[/dim]")
            return [doc for doc, score in results_with_scores[:k]]

        candidates = [doc for doc, score in results_with_scores]

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