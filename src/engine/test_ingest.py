import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(os.path.dirname(current_dir), "data", "vector_db")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

test_docs = [
    Document(
        page_content="Metformin is a medication used to treat type 2 diabetes. Standard starting dose is 500mg.",
        metadata={"source": "test_book.pdf", "category": "clinical"}
    )
]

print(f"Targeting directory: {db_path}")
vector_db = Chroma.from_documents(
    documents=test_docs,
    embedding=embeddings,
    persist_directory=db_path
)

print(" Test ingestion complete. Now run debug_db.py again.")