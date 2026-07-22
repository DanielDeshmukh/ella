"""Response models for Ella Medical SDK."""

from dataclasses import dataclass, field
from typing import Optional


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
            f"  Intent:     {self.intent}",
            f"  Priority:   {self.priority}",
            f"  Response:   {self.response[:200]}{'...' if len(self.response) > 200 else ''}",
            f"  Context:    {self.retrieved_context[:100]}{'...' if len(self.retrieved_context) > 100 else ''}",
        ]
        return "\n".join(lines)

    def show(self):
        """Pretty-print the full response."""
        print("\n" + "=" * 60)
        print(f"  ELLA — {self.intent}")
        print("=" * 60)
        print(f"\n{self.response}")
        if self.retrieved_context:
            print(f"\n{'─' * 60}")
            print(f"  Sources:\n{self.retrieved_context}")
        print("=" * 60 + "\n")
