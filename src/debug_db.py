import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings

load_dotenv()

PINECONE_INDEX = "ella-medical"
NAMESPACE = "medical-books"

def debug_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not found in .env")
        return

    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX not in existing:
        print(f"ERROR: Index '{PINECONE_INDEX}' not found. Available: {existing}")
        return

    index = pc.Index(PINECONE_INDEX)
    stats = index.describe_index_stats()

    print(f"\n=== Pinecone Index: {PINECONE_INDEX} ===")
    print(f"Dimension: {stats.dimension}")
    print(f"Total vectors: {stats.total_vector_count}")
    for ns, ns_stats in stats.namespaces.items():
        print(f"  Namespace '{ns}': {ns_stats.vector_count} vectors")

    if stats.total_vector_count > 0:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        query_vector = embeddings.embed_query("Metformin")
        results = index.query(
            vector=query_vector,
            top_k=1,
            namespace=NAMESPACE,
            include_metadata=True
        )
        if results.matches:
            match = results.matches[0]
            print(f"\nTop Result (score: {match.score:.4f}):")
            text = match.metadata.get("text", "N/A")
            print(f"  Source: {match.metadata.get('source', 'Unknown')}")
            print(f"  Text: {text[:300]}...")

if __name__ == "__main__":
    debug_pinecone()