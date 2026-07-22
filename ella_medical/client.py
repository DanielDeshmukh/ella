"""Ella Medical SDK — Client for Medical Triage & Clinical RAG Engine."""

from typing import Optional

from ella_medical.models import QueryResponse

DEFAULT_SPACE = "Daniel2503/ella-medical"


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
        space: str = DEFAULT_SPACE,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """Initialize Ella client.

        Args:
            api_key: Optional HuggingFace API key (not required for public spaces).
            space: HuggingFace Space ID. Defaults to "Daniel2503/ella-medical".
            base_url: Reserved for future self-hosted use.
            timeout: Request timeout in seconds.
        """
        from gradio_client import Client

        self._client = Client(space, token=api_key, verbose=False)

    def query(
        self,
        message: str,
        history: Optional[str] = None,
    ) -> QueryResponse:
        """Send a medical query to Ella.

        Args:
            message: The patient's message or medical question.
            history: Ignored — Gradio manages state internally.

        Returns:
            QueryResponse with intent, priority, thought process, and response.
        """
        result = self._client.predict(
            user_input=message,
            api_name="/process_query_gpu",
        )

        # result is a tuple: (response, intent, thought_process, context, cleared_input)
        data = result if isinstance(result, (list, tuple)) else [result]

        response_text = data[0] if len(data) > 0 else ""
        intent_raw = data[1] if len(data) > 1 else ""
        thought_process = data[2] if len(data) > 2 else ""
        context = data[3] if len(data) > 3 else ""

        # Parse "TRIAGE (P2)" into intent and priority
        intent = intent_raw
        priority = ""
        if "(" in intent_raw and ")" in intent_raw:
            intent = intent_raw.split("(")[0].strip()
            priority = intent_raw.split("(")[1].rstrip(")")

        return QueryResponse(
            intent=intent,
            priority=priority,
            thought_process=thought_process,
            justification="",
            response=response_text,
            retrieved_context=context,
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
        """Close the client."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
