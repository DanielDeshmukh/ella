"""
NVIDIA NIM Embeddings - OpenAI-compatible API wrapper.
Uses nvidia/nv-embedqa-e5-v5 (1024 dimensions) for medical text.
"""
import os
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "nvidia/nv-embedqa-e5-v5"
NIM_DIMENSIONS = 1024

class NIMEmbeddings:
    def __init__(self, api_key: str = None, model: str = NIM_MODEL):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment or .env")
        self.model = model
        self.base_url = NIM_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        payload = {
            "input": [text],
            "model": self.model,
            "input_type": "query",
            "encoding_format": "float"
        }
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    
    def embed_batch(self, texts: List[str], input_type: str = "passage", batch_size: int = 128) -> List[List[float]]:
        """Embed multiple texts in batches for speed."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            payload = {
                "input": batch,
                "model": self.model,
                "input_type": input_type,
                "encoding_format": "float"
            }
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend([item["embedding"] for item in data["data"]])
        return all_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple document texts."""
        if not texts:
            return []
        
        payload = {
            "input": texts,
            "model": self.model,
            "input_type": "passage",
            "encoding_format": "float"
        }
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]


if __name__ == "__main__":
    embedder = NIMEmbeddings()
    
    test_query = "What are the symptoms of a heart attack?"
    query_vec = embedder.embed_query(test_query)
    print(f"Query: {test_query}")
    print(f"Dimensions: {len(query_vec)}")
    print(f"First 5 values: {query_vec[:5]}")
    
    test_docs = [
        "Chest pain and shortness of breath are key symptoms.",
        "Regular exercise improves cardiovascular health."
    ]
    doc_vecs = embedder.embed_documents(test_docs)
    print(f"\nEmbedded {len(doc_vecs)} documents")
    print(f"Doc 1 dimensions: {len(doc_vecs[0])}")
