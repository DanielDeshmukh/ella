import os
import time
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()
console = Console()

def test_providers():
    console.print("\n[bold white]DIAGNOSTIC BOOT SEQUENCE[/bold white]\n")
    
    results = Table(show_header=True, header_style="bold cyan", border_style="dim")
    results.add_column("Provider")
    results.add_column("Status")
    results.add_column("Details / Model")

    key = os.getenv("GROQ_API_KEY")
    model_id = "llama-3.1-8b-instant"
    try:
        groq_llm = ChatGroq(model_name=model_id, groq_api_key=key)
        start = time.time()
        groq_llm.invoke("ping")
        latency = round(time.time() - start, 2)
        results.add_row("Groq", "[green]ONLINE[/green]", f"{latency}s ({model_id})")
    except Exception as e:
        results.add_row("Groq", "[red]OFFLINE[/red]", f"Error: {str(e)[:40]}")

    try:
        ollama_llm = ChatOllama(model="mistral:latest")
        start = time.time()
        ollama_llm.invoke("ping")
        latency = round(time.time() - start, 2)
        results.add_row("Ollama", "[green]ONLINE[/green]", f"{latency}s (Mistral)")
    except Exception as e:
        results.add_row("Ollama", "[red]OFFLINE[/red]", "Ensure Ollama app is open")

    console.print(results)

if __name__ == "__main__":
    test_providers()