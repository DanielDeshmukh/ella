"""
Ella CLI — Medical Triage & Clinical RAG Engine
Entry point: `ella` command after pip install
"""
import os
import sys
import time
import typer
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

app = typer.Typer(
    name="ella",
    help="Medical Triage & Clinical RAG Engine",
    add_completion=False,
)
console = Console()


def stream_ella_voice(text: str):
    """Streams output with a professional typing effect."""
    print(f"Ella > ", end="", flush=True)
    for char in text:
        print(char, end="", flush=True)
        time.sleep(0.01)
    print("\n")


@app.command()
def chat():
    """Start an interactive triage session with Ella."""
    from src.agents.router import EllaRouter
    from src.agents.guardrails import EmergencyGuardrail

    router = EllaRouter()
    chat_history = []

    console.print(Panel(
        "[bold white]ELLA CORE v1.1.0[/bold white]\n"
        "[green]Context-Aware Routing Active[/green]\n"
        "[blue]Hard-RAG Protocol: De-Robotized[/blue]\n"
        "[yellow]Session Ready[/yellow]",
        border_style="#333333"
    ))

    while True:
        try:
            user_input = console.input("[bold white]Patient > [/bold white]").strip()

            if user_input.lower() in ["exit", "quit", "bye", "goodbye", "tata"]:
                with console.status("[dim]Closing session...[/dim]"):
                    prompt = f"The patient said '{user_input}'. Give a brief, professional medical closing and wish them well."
                    final_farewell = router.raw_llm.invoke(prompt)
                    output = final_farewell.content if hasattr(final_farewell, 'content') else str(final_farewell)

                stream_ella_voice(output.strip())
                console.print("[bold red]Session Terminated.[/bold red]")
                raise typer.Exit()

            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-3:]])

            with console.status("[dim]Consulting Medical Handbooks...[/dim]"):
                decision = router.route_request(user_input, history=history_str)

            if decision.intent == "EMERGENCY":
                console.print(f"\n[bold red]RED FLAG:[/bold red] {decision.justification}")
                EmergencyGuardrail.trigger()

            console.print(f"\n[dim italic]Thought: {decision.thought_process}[/dim italic]")

            context = getattr(decision, "retrieved_context", "")

            prompt = (
                "SYSTEM: You are ELLA, a clinical receptionist. Use the provided DOCUMENTS to guide the patient.\n"
                f"CONVERSATION HISTORY:\n{history_str}\n\n"
                f"DOCUMENTS RETRIEVED:\n{context}\n\n"
                f"LATEST PATIENT INPUT: {user_input}\n\n"
                "STRICT PROTOCOL:\n"
                "1. INTEGRATE information naturally. Do NOT use robotic intros like 'Based on the documents' or 'According to page X'.\n"
                "2. NO REPETITION. Do not repeat the name of the book or your previous questions.\n"
                "3. BE CONCISE. Use the documents to identify the next logical triage step or warning.\n"
                "4. Speak like a professional human who knows the material by heart, not like an AI reading a search result.\n"
                "5. If no documents are relevant, stay in character but advise seeing a doctor for a definitive diagnosis."
            )

            final_res = router.raw_llm.invoke(prompt)
            output = final_res.content if hasattr(final_res, 'content') else str(final_res)

            stream_ella_voice(output.strip())

            chat_history.append({"role": "Patient", "content": user_input})
            chat_history.append({"role": "Ella", "content": output.strip()})
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

            if context:
                console.print(f"[dim]Verified via Medical Handbooks[/dim]")
            console.print("[dim]---[/dim]")

        except KeyboardInterrupt:
            console.print("\n[red]Session Force-Closed.[/red]")
            raise typer.Exit()
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]System Error: {e}[/red]")


@app.command()
def eval():
    """Run the 50-query evaluation benchmark."""
    console.print("[bold]Running Ella Evaluation Benchmark...[/bold]\n")
    os.system(f"{sys.executable} -m src.evaluation")


@app.command()
def stats():
    """Show Pinecone index statistics."""
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        console.print("[red]PINECONE_API_KEY not found in .env[/red]")
        raise typer.Exit()

    pc = Pinecone(api_key=api_key)
    index = pc.Index("ella-medical")
    stats_data = index.describe_index_stats()

    console.print(Panel(
        f"[bold]Pinecone Index: ella-medical[/bold]\n\n"
        f"Total Vectors: [green]{stats_data.total_vector_count}[/green]\n"
        f"Namespaces:\n" +
        "\n".join([f"  [dim]{ns}:[/dim] {data.vector_count} vectors" for ns, data in stats_data.namespaces.items()]),
        border_style="blue"
    ))


@app.command()
def version():
    """Show Ella version."""
    console.print("[bold white]Ella Core v1.1.0[/bold white]")
    console.print("[dim]Medical Triage & Clinical RAG Engine[/dim]")


if __name__ == "__main__":
    app()
