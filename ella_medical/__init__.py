"""Ella Medical — Python SDK for Medical Triage & Clinical RAG"""

__version__ = "1.1.1"

from ella_medical.client import Ella
from ella_medical.models import QueryResponse

__all__ = ["Ella", "QueryResponse"]
