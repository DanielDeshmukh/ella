import os
import pytesseract
import numpy as np  
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from rich.console import Console

TESSERACT_EXE = r"C:\Users\Daniel\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
console = Console()

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent if "src" in __file__ else Path(__file__).resolve().parent
DB_PATH = os.path.join(ROOT_DIR, "data", "vector_db")

print(f"DEBUG: System is looking for DB at: {DB_PATH}")

def sanitize_metadata(splits, file_name):
    """
    Scrubs complex metadata and adds project-specific tags.
    Prevents the 'dict in upsert' crash by flattening metadata.
    """
    clean_splits = []
    
    is_narrative = any(word in file_name.lower() for word in ["mortal", "story", "narrative", "kindness", "hurt"])
    category = "narrative" if is_narrative else "clinical"

    for chunk in splits:
        new_metadata = {
            "source": file_name,
            "category": category,
            "page_label": str(chunk.metadata.get("page_number", "unknown"))
        }

        for key, value in chunk.metadata.items():
            if key in ["coordinates", "points", "system", "layout_width", "layout_height"]:
                continue
            
            if hasattr(value, 'item'): 
                new_metadata[key] = value.item()
            elif isinstance(value, (str, int, float, bool)):
                new_metadata[key] = value
        
        chunk.metadata = new_metadata
        clean_splits.append(chunk)
    
    return clean_splits

def ingest_medical_book(file_path: str):
    """
    Handles extraction, cleaning, and vectorization for a single PDF.
    """
    file_name = os.path.basename(file_path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    vector_db_path = os.path.join(src_dir, "data", "vector_db")

    is_ocr_needed = "needs_ocr" in file_path
    strategy = "hi_res" if is_ocr_needed else "fast"
    
    console.print(f"\n[bold cyan] Ingesting:[/bold cyan] [white]{file_name}[/white] [dim]({strategy} mode)[/dim]")

    try:
        with console.status(f"[bold yellow]Reading PDF with {strategy}...[/bold yellow]", spinner="earth"):
            loader = UnstructuredPDFLoader(
                file_path,
                strategy=strategy,
                mode="elements",
                poppler_path=POPPLER_PATH,
                tesseract_executable=TESSERACT_EXE, 
                languages=["eng"] 
            )
            docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=120,
            strip_whitespace=True
        )
        splits = text_splitter.split_documents(docs)
        
        clean_splits = sanitize_metadata(splits, file_name)
        
        if not clean_splits:
            console.print(f"[bold yellow]  Skipping:[/bold yellow] No text extracted from {file_name}")
            return

        console.print(f"[blue] Chunks created:[/blue] {len(clean_splits)}")

        with console.status("[bold magenta]Storing in Chroma DB...[/bold magenta]", spinner="bouncingBall"):
            embeddings = OllamaEmbeddings(model="mxbai-embed-large")
            Chroma.from_documents(
                documents=clean_splits,
                embedding=embeddings,
                persist_directory=vector_db_path
            )

        console.print(f"[bold green] SUCCESS![/bold green] {file_name} added to Ella's brain.\n")

    except Exception as e:
        error_msg = str(e)
        if "400" in error_msg and "context length" in error_msg:
            console.print(f"[bold red] CONTEXT ERROR:[/bold red] {file_name} has a chunk too large for mxbai-embed.")
        else:
            console.print(f"[bold red] INGESTION CRASHED:[/bold red] {error_msg}")

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    src_folder = os.path.dirname(current_script_dir)
    base_books_dir = os.path.join(src_folder, "medical books")
    
    ocr_folder = os.path.join(base_books_dir, "needs_ocr")
    
    if os.path.exists(base_books_dir):
        for file in os.listdir(base_books_dir):
            full_path = os.path.join(base_books_dir, file)
            if os.path.isfile(full_path) and file.lower().endswith(".pdf"):
                ingest_medical_book(full_path)
    
    if os.path.exists(ocr_folder):
        console.print("\n[bold reverse yellow]  PROCESSING OCR QUEUE  [/bold reverse yellow]")
        for file in os.listdir(ocr_folder):
            full_path = os.path.join(ocr_folder, file)
            if file.lower().endswith(".pdf"):
                ingest_medical_book(full_path)
    else:
        console.print("[dim]No 'needs_ocr' folder found. Skipping high-res queue.[/dim]")