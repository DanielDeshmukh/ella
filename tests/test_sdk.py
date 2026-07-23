"""Tests for ella_medical SDK."""

import pytest
from unittest.mock import patch, MagicMock
from ella_medical.models import QueryResponse
from ella_medical import __version__


class TestQueryResponse:
    def test_repr(self):
        r = QueryResponse(
            intent="TRIAGE",
            priority="P2",
            thought_process="test",
            justification="test",
            response="test response",
            retrieved_context="test context",
        )
        assert "TRIAGE" in repr(r)
        assert "P2" in repr(r)

    def test_str_returns_response(self):
        r = QueryResponse(
            intent="TRIAGE",
            priority="P2",
            thought_process="",
            justification="",
            response="Heart attack symptoms include chest pain.",
            retrieved_context="",
        )
        assert str(r) == "Heart attack symptoms include chest pain."

    def test_show_captures_output(self, capsys):
        r = QueryResponse(
            intent="TRIAGE",
            priority="P2",
            thought_process="",
            justification="",
            response="Test response text.",
            retrieved_context="[Source: test.pdf]: test context",
        )
        r.show()
        captured = capsys.readouterr()
        assert "TRIAGE" in captured.out
        assert "P2" in captured.out
        assert "Test response text." in captured.out
        assert "Sources" in captured.out
        assert "test.pdf" in captured.out

    def test_show_without_context(self, capsys):
        r = QueryResponse(
            intent="BOOKING",
            priority="P3",
            thought_process="",
            justification="",
            response="Booking response.",
            retrieved_context="",
        )
        r.show()
        captured = capsys.readouterr()
        assert "BOOKING" in captured.out
        assert "Booking response." in captured.out
        assert "Sources" not in captured.out

    def test_show_bullet_points(self, capsys):
        r = QueryResponse(
            intent="TRIAGE",
            priority="P2",
            thought_process="",
            justification="",
            response="Symptoms:\n- Chest pain\n- Dizziness",
            retrieved_context="",
        )
        r.show()
        captured = capsys.readouterr()
        assert "Chest pain" in captured.out
        assert "Dizziness" in captured.out

    def test_show_truncates_long_sources(self, capsys):
        long_source = "[Source: very_long_document_name.pdf]: " + "x" * 100
        r = QueryResponse(
            intent="TRIAGE",
            priority="P2",
            thought_process="",
            justification="",
            response="test",
            retrieved_context=long_source,
        )
        r.show()
        captured = capsys.readouterr()
        assert "..." in captured.out

    def test_show_emergency_intent(self, capsys):
        r = QueryResponse(
            intent="EMERGENCY",
            priority="P1",
            thought_process="",
            justification="",
            response="Seek immediate help.",
            retrieved_context="",
        )
        r.show()
        captured = capsys.readouterr()
        assert "EMERGENCY" in captured.out

    def test_show_empty_response(self, capsys):
        r = QueryResponse(
            intent="TRIAGE",
            priority="P2",
            thought_process="",
            justification="",
            response="",
            retrieved_context="",
        )
        r.show()
        captured = capsys.readouterr()
        assert "TRIAGE" in captured.out


class TestVersion:
    def test_version_exists(self):
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_version_format(self):
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestImports:
    def test_import_ella(self):
        from ella_medical import Ella
        assert Ella is not None

    def test_import_query_response(self):
        from ella_medical import QueryResponse
        assert QueryResponse is not None
