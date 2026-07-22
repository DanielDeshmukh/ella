import os
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

load_dotenv()

PINECONE_INDEX = "ella-medical"
NAMESPACE = "medical-books"

def test_pinecone_ingest():
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    test_text = "Metformin is a medication used to treat type 2 diabetes. Standard starting dose is 500mg."
    vector = embeddings.embed_query(test_text)

    doc_id = f"test-{hashlib.md5(test_text.encode()).hexdigest()[:12]}"
    index.upsert(
        vectors=[{
            "id": doc_id,
            "values": vector,
            "metadata": {
                "source": "test_book.pdf",
                "category": "clinical",
                "text": test_text
            }
        }],
        namespace=NAMESPACE
    )

    print(f"Test ingestion complete. ID: {doc_id}")
    print("Now run debug_db.py to verify.")

if __name__ == "__main__":
    test_pinecone_ingest()