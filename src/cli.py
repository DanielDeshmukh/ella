"""
Ella CLI — Medical Triage & Clinical RAG Engine
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.markdown import Markdown
from rich.rule import Rule
from rich import box

app = typer.Typer(name="ella", help="Medical Triage & Clinical RAG Engine", no_args_is_help=False)
console = Console()

CLOSING_PHRASES = [
    "bye", "goodbye", "tata", "see you", "see ya", "take care",
    "cya", "later", "gotta go", "i'm leaving", "i have to go",
    "that's all", "done", "end", "quit", "exit",
]

INTENT_COLORS = {
    "EMERGENCY": "bold red",
    "TRIAGE": "bold yellow",
    "BOOKING": "bold blue",
    "GENERAL_INFO": "bold cyan",
    "CLOSING": "bold dim",
}

INTENT_ICONS = {
    "EMERGENCY": "[bold red]\u26a0[/bold red]",
    "TRIAGE": "[bold yellow]\u25b2[/bold yellow]",
    "BOOKING": "[bold blue]\u25c6[/bold blue]",
    "GENERAL_INFO": "[bold cyan]\u2139[/bold cyan]",
    "CLOSING": "[dim]\u2716[/dim]",
}


def is_closing(text: str) -> bool:
    lower = text.lower().strip().rstrip("!.?")
    return any(lower == phrase or lower.startswith(phrase) for phrase in CLOSING_PHRASES)


def stream_text(text: str, speed: float = 0.008):
    """Character-by-character streaming."""
    for char in text:
        print(char, end="", flush=True)
        if char in ".!?\n":
            time.sleep(speed * 4)
        elif char == ",":
            time.sleep(speed * 2)
        else:
            time.sleep(speed)
    print()


def render_header():
    console.print()
    console.print(Rule(style="#2a3340"))
    header = Text()
    header.append("  ELLA", style="bold white")
    header.append("  ", style="default")
    header.append("Medical Triage & Clinical RAG Engine", style="dim")
    console.print(header)
    console.print(Rule(style="#2a3340"))
    console.print()


def render_response(response_text: str, intent: str, priority: str, sources: str):
    """Render a polished response panel."""
    # Intent badge
    color = INTENT_COLORS.get(intent, "white")
    icon = INTENT_ICONS.get(intent, "")
    badge = f"{icon} [{color}]{intent}[/{color}]"
    if priority:
        badge += f" [dim]{priority}[/dim]"

    # Response panel
    response_text = response_text.strip()
    console.print()
    console.print(Panel(
        response_text,
        title=badge,
        title_align="left",
        border_style="#2a3340",
        padding=(1, 2),
        box=box.ROUNDED,
    ))

    # Sources
    if sources and sources.strip():
        console.print()
        source_lines = [l.strip() for l in sources.split("\n") if l.strip()]
        source_table = Table(show_header=False, box=None, padding=(0, 1))
        source_table.add_column(style="dim")
        for line in source_lines[:3]:
            source_table.add_row(line)
        console.print(Panel(
            source_table,
            title="[dim]Sources[/dim]",
            title_align="left",
            border_style="#1e2530",
            padding=(0, 1),
            box=box.SIMPLE,
        ))

    console.print()


@app.command(invoke_without_command=True)
def main(ctx: typer.Context):
    """Start an interactive triage session."""
    if ctx.invoked_subcommand is not None:
        return

    from src.agents.router import EllaRouter
    from src.agents.guardrails import EmergencyGuardrail

    try:
        router = EllaRouter()
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        raise typer.Exit()

    chat_history = []
    render_header()
    console.print("  [dim]Type[/dim] [bold dim]'bye'[/bold dim] [dim]to end  \u00b7  [dim]Press[/dim] [bold dim]Ctrl+C[/bold dim] [dim]to exit[/dim]")
    console.print()

    while True:
        try:
            user_input = console.input("  [bold]\u2588 You >[/bold] ").strip()
            if not user_input:
                continue

            if is_closing(user_input):
                try:
                    farewell = router.raw_llm.invoke("Give a warm, brief medical farewell. 1 sentence.")
                    output = farewell.content if hasattr(farewell, 'content') else str(farewell)
                    console.print()
                    stream_text(f"  \u2022 {output.strip()}", speed=0.01)
                except Exception:
                    console.print("\n  \u2022 Take care and stay healthy. Goodbye!")
                console.print()
                console.print(Rule(style="#2a3340"))
                console.print("  [dim]Session ended.[/dim]")
                console.print()
                break

            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-3:]])

            with console.status("  [dim]Consulting medical handbooks...[/dim]", spinner="dots"):
                try:
                    decision = router.route_request(user_input, history=history_str)
                except Exception:
                    console.print("  [red]\u2717 Could not process. Try again.[/red]")
                    continue

            if decision.intent == "EMERGENCY":
                console.print()
                console.print(Panel(
                    f"[bold white]{decision.justification}[/bold white]",
                    title="[bold red]\u26a0 EMERGENCY[/bold red]",
                    title_align="left",
                    border_style="red",
                    padding=(0, 2),
                    box=box.DOUBLE,
                ))
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
                render_response(
                    response_text=output.strip(),
                    intent=decision.intent,
                    priority=decision.priority,
                    sources=context,
                )
            except Exception:
                console.print("  [red]Trouble processing. Please try again.[/red]")

            chat_history.append({"role": "Patient", "content": user_input})
            chat_history.append({"role": "Ella", "content": output.strip() if 'output' in dir() else ""})
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except KeyboardInterrupt:
            console.print("\n\n  [dim]Session ended.[/dim]\n")
            break
        except EOFError:
            break
        except Exception:
            console.print("  [red]Something went wrong.[/red]")
            continue


if __name__ == "__main__":
    app()
