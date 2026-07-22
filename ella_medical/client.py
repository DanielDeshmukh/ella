"""Ella Medical SDK — Client for Medical Triage & Clinical RAG Engine."""

from typing import Optional
import httpx

from ella_medical.models import QueryResponse

DEFAULT_BASE_URL = "https://daniel2503-ella-medical.hf.space"


class Ella:
    """Client for Ella Medical Triage & Clinical RAG Engine.

    Usage:
        from ella_medical import Ella

        client = Ella()
        response = client.query("What are the symptoms of a heart attack?")
        print(response.intent)
        print(response.response)

    With history:
        response = client.query(
            "What about treatment?",
            history="Patient: I have chest pain\\nElla: ..."
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ):
        """Initialize Ella client.

        Args:
            api_key: Optional API key (not required for hosted version).
            base_url: Base URL of the Ella API. Defaults to HuggingFace Space.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._history: list[dict] = []

    def query(
        self,
        message: str,
        history: Optional[str] = None,
    ) -> QueryResponse:
        """Send a medical query to Ella.

        Args:
            message: The patient's message or medical question.
            history: Optional conversation history string.

        Returns:
            QueryResponse with intent, priority, thought process, and response.
        """
        payload = {
            "data": [
                message,
                history or "",
            ]
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._client.post(
            f"{self.base_url}/api/predict",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

        result = response.json()
        data = result.get("data", [])

        return QueryResponse(
            intent=data[1] if len(data) > 1 else "",
            priority=data[1] if len(data) > 1 else "",
            thought_process=data[2] if len(data) > 2 else "",
            justification="",
            response="",
            retrieved_context=data[3] if len(data) > 3 else "",
        )

    def triage(self, symptoms: str) -> QueryResponse:
        """Convenience method for triage queries.

        Args:
            symptoms: Description of symptoms.

        Returns:
            QueryResponse with triage assessment.
        """
        return self.query(symptoms)

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
