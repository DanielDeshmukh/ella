"""Response models for Ella Medical SDK."""

from dataclasses import dataclass
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
