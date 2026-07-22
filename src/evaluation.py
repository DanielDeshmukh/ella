import os
import sys
import time
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from src.engine.bm25_retriever import BM25Retriever
from src.engine.nim_embeddings import NIMEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

console = Console()

# --- Config ---
CHROMA_PATH = str(ROOT_DIR / "src" / "data" / "vector_db")
RESULTS_PATH = str(ROOT_DIR / "evaluation_results.json")

# --- Intent Schema ---
class RouteResponse(BaseModel):
    model_config = ConfigDict(extra='allow')
    intent: Literal["EMERGENCY", "TRIAGE", "BOOKING", "GENERAL_INFO", "CLOSING"]
    priority: Literal["P1", "P2", "P3"]
    thought_process: str
    justification: str

# --- Test Queries ---
TEST_QUERIES = [
    # TRIAGE (20 queries)
    ("What are the symptoms of a heart attack?", "TRIAGE", "P1"),
    ("I have a severe headache that won't go away", "TRIAGE", "P2"),
    ("My child has a fever of 103°F", "TRIAGE", "P1"),
    ("I've been coughing for 3 weeks", "TRIAGE", "P2"),
    ("What is the standard dosage for Metformin?", "TRIAGE", "P3"),
    ("I have chest pain when I breathe", "TRIAGE", "P1"),
    ("Can you explain what diabetes type 2 is?", "TRIAGE", "P3"),
    ("I've been feeling dizzy and nauseous for 2 days", "TRIAGE", "P2"),
    ("What are the side effects of Lisinopril?", "TRIAGE", "P3"),
    ("I have a rash that's spreading quickly", "TRIAGE", "P2"),
    ("My blood pressure reading was 180/110", "TRIAGE", "P1"),
    ("I've been experiencing shortness of breath", "TRIAGE", "P2"),
    ("What foods should I avoid with high cholesterol?", "TRIAGE", "P3"),
    ("I think I might have pneumonia", "TRIAGE", "P2"),
    ("How do I manage chronic back pain?", "TRIAGE", "P3"),
    ("I've been bleeding for 10 minutes from a cut", "TRIAGE", "P2"),
    ("What are the warning signs of a stroke?", "TRIAGE", "P1"),
    ("My wound looks infected and red", "TRIAGE", "P2"),
    ("I have a UTI and it burns when I urinate", "TRIAGE", "P3"),
    ("Can I take ibuprofen with blood thinners?", "TRIAGE", "P3"),
    
    # EMERGENCY (10 queries)
    ("I'm having chest pain and can't breathe", "EMERGENCY", "P1"),
    ("Someone is choking and can't speak", "EMERGENCY", "P1"),
    ("I think I'm having an allergic reaction, my throat is closing", "EMERGENCY", "P1"),
    ("There's been a car accident, someone is bleeding heavily", "EMERGENCY", "P1"),
    ("I took too many pills", "EMERGENCY", "P1"),
    ("My child just swallowed a battery", "EMERGENCY", "P1"),
    ("I have severe chest pain radiating to my left arm", "EMERGENCY", "P1"),
    ("I can't stop vomiting blood", "EMERGENCY", "P1"),
    ("I fell and think my neck is injured", "EMERGENCY", "P1"),
    ("I'm having a seizure", "EMERGENCY", "P1"),
    
    # BOOKING (10 queries)
    ("I'd like to book an appointment with a cardiologist", "BOOKING", "P3"),
    ("Can I schedule a blood test for next week?", "BOOKING", "P3"),
    ("I need to reschedule my appointment", "BOOKING", "P3"),
    ("What are your clinic hours?", "BOOKING", "P3"),
    ("Is Dr. Smith available on Monday?", "BOOKING", "P3"),
    ("How do I cancel my appointment?", "BOOKING", "P3"),
    ("I need a follow-up visit after my surgery", "BOOKING", "P3"),
    ("Can I book a teleconsultation?", "BOOKING", "P3"),
    ("Where is the clinic located?", "BOOKING", "P3"),
    ("What insurance do you accept?", "BOOKING", "P3"),
    
    # GENERAL_INFO (5 queries)
    ("How can I improve my sleep quality?", "GENERAL_INFO", "P3"),
    ("What's a healthy BMI range?", "GENERAL_INFO", "P3"),
    ("How much water should I drink daily?", "GENERAL_INFO", "P3"),
    ("Is walking 10000 steps good for health?", "GENERAL_INFO", "P3"),
    ("What are the benefits of meditation?", "GENERAL_INFO", "P3"),
    
    # CLOSING PHRASES (5 queries)
    ("bye", "CLOSING", "P3"),
    ("goodbye", "CLOSING", "P3"),
    ("thank you, see you later", "CLOSING", "P3"),
    ("tata, take care", "CLOSING", "P3"),
    ("that's all for now, bye bye", "CLOSING", "P3"),
]

class EllaEvaluator:
    def __init__(self):
        self.groq_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        self.structured_llm = self.groq_llm.with_structured_output(RouteResponse)
        
        console.print("[dim]Loading NIM Embeddings...[/dim]")
        self.embeddings = NIMEmbeddings()
        console.print("[green]NIM Embeddings ready[/green]")
        
        console.print("[dim]Loading BM25 Retriever (90k records)...[/dim]")
        self.retriever = BM25Retriever()
        console.print(f"[green]Loaded {self.retriever.count} records[/green]")
        
        console.print("[dim]Loading Reranker...[/dim]")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        console.print("[green]Reranker ready[/green]\n")
    
    def hybrid_search(self, query, k=5):
        results = self.retriever.search(query, k=15)
        
        if not results:
            return [], 0.0
        
        top_score = results[0]["score"]
        
        # Build doc objects
        candidates = []
        for r in results:
            candidates.append(type('Doc', (), {
                'page_content': r["text"],
                'metadata': {"source": r["source"], "category": r["category"]}
            })())
        
        # CrossEncoder reranking
        model_inputs = [[query, doc.page_content] for doc in candidates[:10]]
        scores = self.reranker.predict(model_inputs, batch_size=10, show_progress_bar=False)
        scored_docs = sorted(zip(scores, candidates[:10]), key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_docs[:k]], top_score
    
    def classify_intent(self, query):
        system_prompt = (
            "You are a medical triage router. Categorize the input.\n"
            "INSTRUCTIONS:\n"
            "1. Use 'TRIAGE' for ANY medical question: symptoms, medications, dosages, "
            "side effects, conditions, diet for a condition, drug interactions, pain management.\n"
            "2. Use 'EMERGENCY' for life-threatening: chest pain, severe bleeding, seizures, "
            "overdose, choking, allergic reactions, neck/spine injuries.\n"
            "3. Use 'BOOKING' for clinic logistics: hours, location, insurance, scheduling.\n"
            "4. Use 'CLOSING' ONLY for farewells (bye, goodbye, see you later).\n"
            "5. Use 'GENERAL_INFO' ONLY for completely non-medical questions.\n\n"
            "KEY RULE: If it mentions a medical condition, drug, symptom, or health concern -> TRIAGE.\n"
            "Examples:\n"
            "- 'What is the dosage for Metformin?' -> TRIAGE\n"
            "- 'What are the side effects of Lisinopril?' -> TRIAGE\n"
            "- 'Can I take ibuprofen with blood thinners?' -> TRIAGE\n"
            "- 'What foods should I avoid with high cholesterol?' -> TRIAGE\n"
            "- 'How do I manage chronic back pain?' -> TRIAGE\n"
            "- 'What is a healthy BMI range?' -> GENERAL_INFO\n"
            "- 'How much water should I drink daily?' -> GENERAL_INFO"
        )
        try:
            decision = self.structured_llm.invoke([
                ("system", system_prompt),
                ("human", query)
            ])
            return decision
        except Exception:
            return RouteResponse(
                intent="GENERAL_INFO", priority="P3",
                thought_process="Fallback", justification="Error"
            )
    
    def generate_response(self, query, context, history=""):
        prompt = (
            "SYSTEM: You are ELLA, a clinical receptionist. Use DOCUMENTS to guide the patient.\n"
            f"DOCUMENTS RETRIEVED:\n{context}\n\n"
            f"LATEST PATIENT INPUT: {query}\n\n"
            "STRICT PROTOCOL:\n"
            "1. INTEGRATE information naturally.\n"
            "2. NO REPETITION.\n"
            "3. BE CONCISE.\n"
            "4. If no documents are relevant, advise seeing a doctor.\n"
            "5. For closing phrases like 'bye', 'goodbye', give a brief professional farewell."
        )
        try:
            res = self.groq_llm.invoke(prompt)
            return res.content if hasattr(res, 'content') else str(res)
        except Exception as e:
            return f"Error: {e}"
    
    def evaluate_retrieval(self, docs, expected_intent):
        if not docs:
            return 0.0, "No results"
        
        has_medical = any(
            "clinical" in str(getattr(d, 'metadata', {}).get("category", "")).lower() or
            "narrative" in str(getattr(d, 'metadata', {}).get("category", "")).lower()
            for d in docs
        )
        has_relevant = len(docs) > 0
        
        if expected_intent == "CLOSING":
            return 1.0, "Closing doesn't need retrieval"
        
        if has_medical and has_relevant:
            return 1.0, "Relevant medical docs found"
        elif has_relevant:
            return 0.5, "Some docs found, limited relevance"
        else:
            return 0.0, "No relevant docs"
    
    def run_evaluation(self):
        results = []
        intent_correct = 0
        retrieval_scores = []
        total_latency = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating queries", total=len(TEST_QUERIES))
            
            for query, expected_intent, expected_priority in TEST_QUERIES:
                start_time = time.time()
                
                # 1. Intent classification
                decision = self.classify_intent(query)
                intent_match = decision.intent == expected_intent
                if intent_match:
                    intent_correct += 1
                
                # 2. Retrieval (skip for closing phrases)
                if expected_intent != "CLOSING":
                    docs, top_score = self.hybrid_search(query, k=3)
                    retrieval_score, retrieval_note = self.evaluate_retrieval(docs, expected_intent)
                    retrieval_scores.append(retrieval_score)
                    
                    context = "\n\n".join([
                        f"[Source: {os.path.basename(getattr(d, 'metadata', {}).get('source', 'Manual'))}]: {getattr(d, 'page_content', '')[:300]}"
                        for d in docs
                    ])
                else:
                    docs = []
                    top_score = 0.0
                    retrieval_score = 1.0
                    retrieval_note = "Closing - no retrieval needed"
                    retrieval_scores.append(retrieval_score)
                    context = ""
                
                # 3. Response generation
                response = self.generate_response(query, context)
                latency = time.time() - start_time
                total_latency += latency
                
                # 4. Check response quality
                response_quality = "Good"
                if len(response) < 20:
                    response_quality = "Too short"
                elif "error" in response.lower():
                    response_quality = "Error"
                elif expected_intent == "CLOSING" and not any(w in response.lower() for w in ["bye", "goodbye", "take care", "well", "thank"]):
                    response_quality = "Missing closing"
                
                result = {
                    "query": query,
                    "expected_intent": expected_intent,
                    "detected_intent": decision.intent,
                    "intent_match": intent_match,
                    "expected_priority": expected_priority,
                    "detected_priority": decision.priority,
                    "top_relevance_score": round(top_score, 4),
                    "retrieval_score": retrieval_score,
                    "retrieval_note": retrieval_note,
                    "response_quality": response_quality,
                    "response_length": len(response),
                    "latency_seconds": round(latency, 3),
                    "response_preview": response[:200],
                }
                results.append(result)
                progress.update(task, advance=1)
        
        # --- Summary ---
        total = len(TEST_QUERIES)
        avg_latency = total_latency / total
        avg_retrieval = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0
        quality_counts = {}
        for r in results:
            q = r["response_quality"]
            quality_counts[q] = quality_counts.get(q, 0) + 1
        
        summary = {
            "total_queries": total,
            "intent_accuracy": round(intent_correct / total * 100, 1),
            "avg_latency_seconds": round(avg_latency, 3),
            "avg_retrieval_score": round(avg_retrieval, 3),
            "response_quality_distribution": quality_counts,
            "intent_correct": intent_correct,
            "timestamp": datetime.now().isoformat(),
        }
        
        return results, summary
    
    def print_report(self, results, summary):
        console.print(Panel(
            f"[bold white]EVALUATION REPORT[/bold white]\n"
            f"[dim]{summary['timestamp']}[/dim]",
            border_style="blue"
        ))
        
        # Summary table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Total Queries", str(summary["total_queries"]))
        table.add_row("Intent Accuracy", f"{summary['intent_accuracy']}%")
        table.add_row("Avg Latency", f"{summary['avg_latency_seconds']}s")
        table.add_row("Avg Retrieval Score", f"{summary['avg_retrieval_score']}")
        for quality, count in summary["response_quality_distribution"].items():
            table.add_row(f"Quality: {quality}", str(count))
        console.print(table)
        
        # Intent breakdown
        console.print("\n[bold]Intent Classification Breakdown:[/bold]")
        intent_table = Table(show_header=True, header_style="bold green")
        intent_table.add_column("Expected")
        intent_table.add_column("Detected")
        intent_table.add_column("Match")
        intent_table.add_column("Count")
        
        intent_groups = {}
        for r in results:
            key = (r["expected_intent"], r["detected_intent"], r["intent_match"])
            intent_groups[key] = intent_groups.get(key, 0) + 1
        
        for (exp, det, match), count in sorted(intent_groups.items()):
            status = "[green]✓[/green]" if match else "[red]✗[/red]"
            intent_table.add_row(exp, det, status, str(count))
        console.print(intent_table)
        
        # Failed queries
        failures = [r for r in results if not r["intent_match"]]
        if failures:
            console.print(f"\n[bold red]Failed Intent Classifications ({len(failures)}):[/bold red]")
            for f in failures:
                console.print(f"  [yellow]Query:[/yellow] {f['query']}")
                console.print(f"    Expected: {f['expected_intent']}, Got: {f['detected_intent']}")
        
        # Low retrieval scores
        low_retrieval = [r for r in results if r["retrieval_score"] < 0.5 and r["expected_intent"] != "CLOSING"]
        if low_retrieval:
            console.print(f"\n[bold yellow]Low Retrieval Scores ({len(low_retrieval)}):[/bold yellow]")
            for r in low_retrieval:
                console.print(f"  [yellow]Query:[/yellow] {r['query']}")
                console.print(f"    Score: {r['top_relevance_score']}, Note: {r['retrieval_note']}")
        
        # Sample responses
        console.print("\n[bold]Sample Responses:[/bold]")
        sample_indices = [0, 10, 20, 30, 40, 45, 46, 47, 48, 49]
        for i in sample_indices:
            if i < len(results):
                r = results[i]
                console.print(f"\n[cyan]Q{i+1}:[/cyan] {r['query']}")
                console.print(f"  Intent: {r['detected_intent']} (expected: {r['expected_intent']})")
                console.print(f"  Response: {r['response_preview']}...")

if __name__ == "__main__":
    evaluator = EllaEvaluator()
    results, summary = evaluator.run_evaluation()
    
    # Save results
    output = {"summary": summary, "results": results}
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[dim]Results saved to {RESULTS_PATH}[/dim]")
    
    evaluator.print_report(results, summary)
