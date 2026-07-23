"""Response models for Ella Medical SDK."""

from dataclasses import dataclass


@dataclass
class QueryResponse:
    """Response from an Ella query."""

    intent: str
    priority: str
    thought_process: str
    justification: str
    response: str
    retrieved_context: str

    def __repr__(self) -> str:
        return f"QueryResponse(intent='{self.intent}', priority='{self.priority}')"

    def __str__(self) -> str:
        return self.response

    def show(self):
        """Pretty-print the full response."""
        W = 60
        sep = "\033[38;2;42;51;64m" + "\u2500" * W + "\033[0m"
        dim = "\033[2m"
        bold = "\033[1m"
        reset = "\033[0m"
        cyan = "\033[38;2;45;212;191m"
        white = "\033[38;2;232;236;241m"
        gray = "\033[38;2;137;147;164m"
        dimgray = "\033[38;2;75;85;99m"

        # Intent badge
        icons = {"EMERGENCY": "\u26a0", "TRIAGE": "\u25b2", "BOOKING": "\u25c6", "GENERAL_INFO": "\u2139", "CLOSING": "\u2716"}
        icon = icons.get(self.intent, "\u25cf")
        badge = f"{cyan}{bold} {icon} {self.intent}{reset}"
        if self.priority:
            badge += f" {dimgray}{self.priority}{reset}"

        print(f"\n{sep}")
        print(f" {badge}")
        print(f"{sep}\n")

        # Response text with proper wrapping
        for line in self.response.split("\n"):
            line = line.strip()
            if not line:
                print()
                continue
            if line.startswith("- ") or line.startswith("\u2022"):
                print(f"  {cyan}\u25cf{reset}  {white}{line[2:]}{reset}")
            else:
                print(f"  {white}{line}{reset}")

        # Sources
        if self.retrieved_context and self.retrieved_context.strip():
        print(f"\n{sep}")
        print(f" {dimgray}{bold}Sources{reset}")
        rule = "\u2500" * W
        print(f"{dimgray}{rule}{reset}")
            for line in self.retrieved_context.strip().split("\n")[:3]:
                line = line.strip()
                if line.startswith("[Source:"):
                    print(f"  {dimgray}{line}{reset}")
                elif line:
                    # Truncate long lines
                    display = line[:80] + "..." if len(line) > 80 else line
                    print(f"  {dimgray}{display}{reset}")

        print(f"\n{sep}\n")
