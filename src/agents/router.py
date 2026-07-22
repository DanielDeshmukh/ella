import os
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, ConfigDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.engine.retriever import EllaRetriever

load_dotenv()

class RouteResponse(BaseModel):

    model_config = ConfigDict(extra='allow') 
    
    intent: Literal["EMERGENCY", "TRIAGE", "BOOKING", "GENERAL_INFO", "CLOSING"] = Field(
        description="The categorized intent."
    )
    priority: Literal["P1", "P2", "P3"] = Field(
        description="P1 (Life-threat), P2 (Urgent), or P3 (Routine)."
    )
    thought_process: str = Field(
        description="High-level reasoning for the intent."
    )
    justification: str = Field(
        description="A 1-sentence clinical justification."
    )

class EllaRouter:
    def __init__(self):
        self.raw_llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0,
        )
        self.structured_llm = self.raw_llm.with_structured_output(RouteResponse)
        self.retriever = EllaRetriever()

    def route_request(self, user_input: str, history: str = "") -> RouteResponse:
        """
        Routes the request by considering both current input and conversation history.
        """
        system_prompt = (
            "You are a medical triage router. Categorize the input based on the NEW message "
            "AND the PREVIOUS CONTEXT provided in the history.\n\n"
            f"RECENT HISTORY:\n{history}\n\n"
            "INSTRUCTIONS:\n"
            "1. If a medical investigation (e.g., snakebite, chest pain) is ongoing in the history, "
            "maintain 'TRIAGE' intent even for short/one-word follow-up answers.\n"
            "2. If the user is asking about symptoms, medications, dosages, side effects, "
            "medical conditions, diet for a condition, or any health-related question, "
            "the intent is 'TRIAGE' — it requires clinical context to answer properly.\n"
            "3. Use 'GENERAL_INFO' ONLY for completely non-medical questions (weather, jokes, "
            "general life advice unrelated to health).\n"
            "4. Use 'CLOSING' ONLY for farewells, goodbyes, or leaving phrases.\n"
            "5. Use 'BOOKING' for clinic logistics: hours, location, insurance, scheduling.\n"
            "6. Use 'EMERGENCY' for life-threatening situations: chest pain, severe bleeding, "
            "allergic reactions, choking, seizures, overdose, neck/spine injuries.\n\n"
            "KEY DISTINCTION — TRIAGE vs GENERAL_INFO:\n"
            "- 'What are the side effects of Lisinopril?' -> TRIAGE (medication question)\n"
            "- 'What is the standard dosage for Metformin?' -> TRIAGE (dosage question)\n"
            "- 'Can you explain what diabetes type 2 is?' -> TRIAGE (medical condition)\n"
            "- 'What foods should I avoid with high cholesterol?' -> TRIAGE (diet for condition)\n"
            "- 'How do I manage chronic back pain?' -> TRIAGE (pain management)\n"
            "- 'Can I take ibuprofen with blood thinners?' -> TRIAGE (drug interaction)\n"
            "- 'What is a healthy BMI range?' -> GENERAL_INFO (general wellness, no condition)\n"
            "- 'How much water should I drink daily?' -> GENERAL_INFO (general wellness)\n"
            "- 'What are your clinic hours?' -> BOOKING\n"
            "- 'My blood pressure is 180/110' -> EMERGENCY\n"
            "- 'I have chest pain when I breathe' -> TRIAGE\n"
            "- 'bye' -> CLOSING\n"
        )
        
        try:
            decision = self.structured_llm.invoke([
                ("system", system_prompt),
                ("human", f"New Patient Input: {user_input}")
            ])
        except Exception:
            decision = RouteResponse(
                intent="GENERAL_INFO",
                priority="P3",
                thought_process="Fallback triggered.",
                justification="Input complexity caused validation error."
            )

        decision.retrieved_context = ""
        
        search_query = f"{history[-200:]} {user_input}" if history else user_input

        if decision.intent in ["TRIAGE", "GENERAL_INFO", "EMERGENCY"]:
            docs = self.retriever.hybrid_search(search_query, k=3)
            if docs:
                decision.retrieved_context = "\n\n".join([
                    f"[Source: {os.path.basename(d.metadata.get('source', 'Manual'))}]: {d.page_content}"
                    for d in docs
                ])
        
        return decision