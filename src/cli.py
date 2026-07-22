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

console = Console()

CLOSING_PHRASES = [
    "bye", "goodbye", "tata", "see you", "see ya", "take care",
    "cya", "later", "gotta go", "i'm leaving", "i have to go",
    "that's all", "done", "end", "quit", "exit",
]


def stream_ella_voice(text: str):
    """Streams output with a professional typing effect."""
    print(f"\nElla > ", end="", flush=True)
    for char in text:
        print(char, end="", flush=True)
        time.sleep(0.012)
    print()


def is_closing(text: str) -> bool:
    """Check if input is a closing/farewell phrase."""
    lower = text.lower().strip().rstrip("!.?")
    return any(lower == phrase or lower.startswith(phrase) for phrase in CLOSING_PHRASES)


def launch_ella():
    """Main entry point — `ella` launches this directly."""
    from src.agents.router import EllaRouter
    from src.agents.guardrails import EmergencyGuardrail

    try:
        router = EllaRouter()
    except Exception as e:
        console.print(f"[red]Failed to initialize Ella: {e}[/red]")
        raise typer.Exit()

    chat_history = []

    console.print(Panel(
        "[bold white]ELLA[/bold white]\n"
        "[green]Medical Triage & Clinical RAG Engine[/green]\n"
        "[dim]Type 'bye' or 'exit' to end session[/dim]",
        border_style="#333333"
    ))

    while True:
        try:
            user_input = console.input("\n[bold white]You > [/bold white]").strip()

            if not user_input:
                continue

            if is_closing(user_input):
                try:
                    farewell_prompt = "Give a warm, brief medical farewell. Wish them well. 1-2 sentences max."
                    final_farewell = router.raw_llm.invoke(farewell_prompt)
                    output = final_farewell.content if hasattr(final_farewell, 'content') else str(final_farewell)
                    stream_ella_voice(output.strip())
                except Exception:
                    stream_ella_voice("Take care and stay healthy. Goodbye!")
                console.print("\n[dim]Session ended.[/dim]\n")
                break

            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-3:]])

            try:
                with console.status("[dim]Consulting Medical Handbooks...[/dim]"):
                    decision = router.route_request(user_input, history=history_str)
            except Exception:
                console.print("[red]Could not process input. Please try again.[/red]")
                continue

            if decision.intent == "EMERGENCY":
                console.print(f"\n[bold red]RED FLAG:[/bold red] {decision.justification}")
                try:
                    EmergencyGuardrail.trigger()
                except Exception:
                    pass

            context = getattr(decision, "retrieved_context", "")

            prompt = (
                "SYSTEM: You are ELLA, a clinical receptionist. Use the provided DOCUMENTS to guide the patient.\n"
                f"CONVERSATION HISTORY:\n{history_str}\n\n"
                f"DOCUMENTS RETRIEVED:\n{context}\n\n"
                f"LATEST PATIENT INPUT: {user_input}\n\n"
                "STRICT PROTOCOL:\n"
                "1. INTEGRATE information naturally. Do NOT use robotic intros.\n"
                "2. NO REPETITION. Do not repeat book names or previous questions.\n"
                "3. BE CONCISE. Identify the next logical triage step or warning.\n"
                "4. Speak like a professional who knows the material by heart.\n"
                "5. If no documents are relevant, advise seeing a doctor."
            )

            try:
                final_res = router.raw_llm.invoke(prompt)
                output = final_res.content if hasattr(final_res, 'content') else str(final_res)
                stream_ella_voice(output.strip())
            except Exception:
                stream_ella_voice("I'm having trouble processing that. Could you rephrase your question?")

            chat_history.append({"role": "Patient", "content": user_input})
            chat_history.append({"role": "Ella", "content": output.strip() if 'output' in dir() else ""})
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except KeyboardInterrupt:
            console.print("\n\n[dim]Session ended.[/dim]\n")
            break
        except EOFError:
            break
        except Exception:
            console.print("[red]Something went wrong. Please try again.[/red]")
            continue


def main():
    """Entry point: `ella` launches the triage session."""
    launch_ella()


@app.command()
def serve(port: int = 8000):
    """Start the Ella API server."""
    import uvicorn
    console.print(f"[green]Starting Ella API on port {port}...[/green]")
    console.print(f"[dim]Docs: http://localhost:{port}/docs[/dim]")
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
