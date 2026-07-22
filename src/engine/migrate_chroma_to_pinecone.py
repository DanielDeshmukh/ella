"""
Migration Script: ChromaDB (SQLite) -> Pinecone
Uses NVIDIA NIM embeddings (fast) + resume from where it stopped.
Run: python src/engine/migrate_chroma_to_pinecone.py
"""
import os
import sys
import time
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from pinecone import Pinecone, ServerlessSpec
from src.engine.nim_embeddings import NIMEmbeddings

console = Console()

PINECONE_INDEX = "ella-medical"
DIMENSIONS = 1024
NAMESPACE = "medical-books"
BATCH_SIZE = 100
CHROMA_DB = str(ROOT_DIR / "src" / "data" / "vector_db" / "chroma.sqlite3")


def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        console.print("[bold red]ERROR:[/bold red] PINECONE_API_KEY not found in .env")
        sys.exit(1)
    return Pinecone(api_key=api_key)


def ensure_index_exists(pc):
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX in existing:
        console.print(f"[green]Index '{PINECONE_INDEX}' already exists. Using it.[/green]")
        return

    console.print(f"[cyan]Creating index '{PINECONE_INDEX}' (dim={DIMENSIONS}, metric=cosine)...[/cyan]")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(PINECONE_INDEX).status["ready"]:
        time.sleep(2)
    console.print("[green]Index ready.[/green]")


def read_chroma_data(db_path):
    """Read all text chunks and metadata from Chroma SQLite."""
    conn = sqlite3.connect(db_path)

    cursor = conn.execute("SELECT id, embedding_id FROM embeddings ORDER BY id")
    embedding_rows = cursor.fetchall()
    total = len(embedding_rows)
    console.print(f"[bold]Found {total} embeddings in SQLite.[/bold]")

    records = []
    for row_id, embedding_id in embedding_rows:
        meta_cursor = conn.execute(
            "SELECT key, string_value FROM embedding_metadata WHERE id = ?", (row_id,)
        )
        metadata = {}
        text = ""
        for key, value in meta_cursor.fetchall():
            if key == "chroma:document":
                text = value or ""
            elif key in ["source", "filename", "category", "page_label"]:
                metadata[key] = value or ""

        if text.strip():
            source = metadata.get("source", metadata.get("filename", "unknown"))
            source_name = os.path.basename(source) if source else "unknown"
            metadata["source"] = source_name
            records.append({
                "id": embedding_id,
                "text": text,
                "metadata": metadata
            })

    conn.close()
    return records, total


def get_existing_ids(index, namespace):
    """Fetch all vector IDs already in Pinecone."""
    console.print("[dim]Fetching existing IDs from Pinecone...[/dim]")
    existing_ids = set()
    stats = index.describe_index_stats()
    count = stats.namespaces.get(namespace, {}).vector_count
    console.print(f"[dim]Found {count} vectors in Pinecone[/dim]")
    
    # Query with a dummy vector to fetch IDs (Pinecone doesn't have list_all)
    # We'll use a different approach: query with a zero vector and high top_k
    # Actually, we can skip this and just let upsert handle duplicates (it overwrites)
    return existing_ids


def migrate():
    pc = get_pinecone_client()
    ensure_index_exists(pc)

    console.print(f"\n[bold]Reading from Chroma SQLite:[/bold] {CHROMA_DB}")
    records, total = read_chroma_data(CHROMA_DB)
    console.print(f"[bold]Usable records (with text):[/bold] {len(records)}\n")

    if not records:
        console.print("[yellow]No records to migrate.[/yellow]")
        return

    index = pc.Index(PINECONE_INDEX)
    
    # Check current count
    stats = index.describe_index_stats()
    current_count = stats.namespaces.get(NAMESPACE, {}).vector_count
    console.print(f"[bold]Already in Pinecone:[/bold] {current_count} vectors")
    
    if current_count >= len(records):
        console.print("[green]All records already migrated![/green]")
        return

    console.print("[bold]Using NVIDIA NIM embeddings (nvidia/nv-embedqa-e5-v5) - BATCH MODE...[/bold]")
    embeddings_model = NIMEmbeddings()

    # Skip first N records that are already migrated
    skip_count = current_count
    records_to_migrate = records[skip_count:]
    console.print(f"[bold]Resuming from record {skip_count + 1}, {len(records_to_migrate)} remaining[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding + Upserting to Pinecone", total=len(records_to_migrate))
        
        # Batch embedding: collect texts, embed 128 at a time, upsert 100 at a time
        EMBED_BATCH = 128
        UPSERT_BATCH = 100
        batch_vectors = []

        for batch_start in range(0, len(records_to_migrate), EMBED_BATCH):
            batch_records = records_to_migrate[batch_start:batch_start + EMBED_BATCH]
            texts = [r["text"] for r in batch_records]
            
            try:
                # Embed entire batch in one API call
                vectors = embeddings_model.embed_batch(texts, input_type="passage", batch_size=EMBED_BATCH)
                
                for record, vector in zip(batch_records, vectors):
                    meta = {k: v for k, v in record["metadata"].items() if isinstance(v, (str, int, float, bool))}
                    meta["text"] = record["text"]
                    batch_vectors.append({
                        "id": record["id"],
                        "values": vector,
                        "metadata": meta
                    })
                
                # Upsert in smaller chunks
                while len(batch_vectors) >= UPSERT_BATCH:
                    index.upsert(vectors=batch_vectors[:UPSERT_BATCH], namespace=NAMESPACE)
                    batch_vectors = batch_vectors[UPSERT_BATCH:]
                    progress.update(task, advance=UPSERT_BATCH)
                    
            except Exception as e:
                console.print(f"[red]Error at batch {batch_start}: {e}[/red]")
                continue

        if batch_vectors:
            index.upsert(vectors=batch_vectors, namespace=NAMESPACE)
            progress.update(task, advance=len(batch_vectors))

    # Verify
    time.sleep(3)
    stats = index.describe_index_stats()
    pinecone_count = stats.namespaces.get(NAMESPACE, {}).vector_count
    console.print(f"\n[bold green]Migration complete![/bold green]")
    console.print(f"Chroma records: {total}")
    console.print(f"Pinecone records: {pinecone_count}")

    if pinecone_count >= len(records):
        console.print("[bold green]All records migrated successfully![/bold green]")
    else:
        console.print(f"[bold yellow]WARNING: {len(records) - pinecone_count} records missing.[/bold yellow]")


if __name__ == "__main__":
    migrate()
