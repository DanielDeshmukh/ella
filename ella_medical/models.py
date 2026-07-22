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
        lines = [
            f"\n  \u2500" * 30,
            f"  Intent:     {self.intent}",
            f"  Priority:   {self.priority}",
            f"  \u2500" * 30,
            f"\n{self.response}",
        ]
        if self.retrieved_context:
            lines.append(f"\n  \u2500" * 30)
            lines.append(f"  Sources:")
            for line in self.retrieved_context.strip().split("\n")[:3]:
                lines.append(f"  {line.strip()}")
        lines.append(f"  \u2500" * 30 + "\n")
        return "\n".join(lines)

    def show(self):
        """Pretty-print the full response."""
        INTENT_ICONS = {
            "EMERGENCY": "\u26a0",
            "TRIAGE": "\u25b2",
            "BOOKING": "\u25c6",
            "GENERAL_INFO": "\u2139",
            "CLOSING": "\u2716",
        }
        icon = INTENT_ICONS.get(self.intent, "\u25cf")
        sep = "\u2500" * 56

        print(f"\n{sep}")
        print(f"  {icon}  {self.intent}  {self.priority}")
        print(sep)
        print()
        print(self.response)

        if self.retrieved_context:
            print(f"\n{sep}")
            print("  Sources:")
            for line in self.retrieved_context.strip().split("\n")[:3]:
                print(f"  {line.strip()}")

        print(f"{sep}\n")
