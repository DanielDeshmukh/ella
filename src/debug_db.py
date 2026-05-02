import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

db_path = r"D:\Vs Code\VS code\ella\src\data\vector_db"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

if not os.path.exists(db_path):
    print(f" Error: Folder does not exist at {db_path}")
else:
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    count = db._collection.count()
    print(f" Total entries found: {count}")
    
    if count > 0:
        res = db.similarity_search("Metformin", k=1)
        print(f"Top Result: {res[0].page_content}")