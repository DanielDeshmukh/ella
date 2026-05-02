from rich.console import Console
from rich.panel import Panel

console = Console()

class EmergencyGuardrail:
    @staticmethod
    def trigger():
        console.print(Panel(
            "[bold red] EMERGENCY PROTOCOL ACTIVATED[/bold red]\n\n"
            "If you are experiencing a life-threatening emergency:\n"
            "1. [bold white]Call 108 or your local emergency services immediately.[/bold white]\n"
            "2. Do not wait for a response from this automated system.\n"
            "3. Stay on the line if you are currently at the clinic entrance.",
            title="[white]Action Required[/white]",
            border_style="red"
        ))
        return "Emergency instructions provided. Protocol closed."