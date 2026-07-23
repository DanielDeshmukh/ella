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
        sep = "-" * W
        bold = "\033[1m"
        reset = "\033[0m"
        cyan = "\033[36m"
        white = "\033[97m"
        dimgray = "\033[90m"

        icons = {
            "EMERGENCY": "!",
            "TRIAGE": "^",
            "BOOKING": "*",
            "GENERAL_INFO": "i",
            "CLOSING": "x",
        }
        icon = icons.get(self.intent, "o")
        badge = cyan + bold + " [" + icon + "] " + self.intent + reset
        if self.priority:
            badge += " " + dimgray + self.priority + reset

        print()
        print(sep)
        print(" " + badge)
        print(sep)
        print()

        for text_line in self.response.split("\n"):
            text_line = text_line.strip()
            if not text_line:
                print()
                continue
            if text_line.startswith("- ") or text_line.startswith("* "):
                print("  " + cyan + "*" + reset + "  " + white + text_line[2:] + reset)
            else:
                print("  " + white + text_line + reset)

        if self.retrieved_context and self.retrieved_context.strip():
            print()
            print(sep)
            print(" " + dimgray + bold + "Sources" + reset)
            print(dimgray + sep + reset)
            for src_line in self.retrieved_context.strip().split("\n")[:3]:
                src_line = src_line.strip()
                if src_line:
                    display = src_line[:80] + "..." if len(src_line) > 80 else src_line
                    print("  " + dimgray + display + reset)

        print()
        print(sep)
        print()
