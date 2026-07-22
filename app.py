"""
Ella — Medical Triage & Clinical RAG Engine
Gradio Demo for HuggingFace Spaces
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
import spaces
import torch
from src.agents.router import EllaRouter
from src.agents.guardrails import EmergencyGuardrail

router = None
chat_history = []


def initialize_ella():
    global router
    if router is None:
        router = EllaRouter()
    return router


@spaces.GPU
def process_query_gpu(user_input, history):
    """Wrapper that satisfies ZeroGPU requirement."""
    return process_query(user_input, history)


def process_query(user_input, history):
    """Process a query through Ella's pipeline and return structured output."""
    if not user_input.strip():
        return history, "", "", "", ""

    router = initialize_ella()
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])

    # Route
    try:
        decision = router.route_request(user_input, history=history_str)
    except Exception as e:
        return history, f"Error: {e}", "", "", ""

    # Build context
    context = getattr(decision, "retrieved_context", "")

    # Generate response
    prompt = (
        "SYSTEM: You are ELLA, a clinical receptionist. Use the provided DOCUMENTS to guide the patient.\n"
        f"CONVERSATION HISTORY:\n{history_str}\n\n"
        f"DOCUMENTS RETRIEVED:\n{context}\n\n"
        f"LATEST PATIENT INPUT: {user_input}\n\n"
        "STRICT PROTOCOL:\n"
        "1. INTEGRATE information naturally. Do NOT use robotic intros.\n"
        "2. NO REPETITION. Do not repeat book names or previous questions.\n"
        "3. BE CONCISE. Identify the next logical triage step or warning.\n"
        "4. Speak like a professional who knows the material by heart.\n"
        "5. If no documents are relevant, advise seeing a doctor."
    )

    try:
        final_res = router.raw_llm.invoke(prompt)
        output = final_res.content if hasattr(final_res, 'content') else str(final_res)
    except Exception:
        output = "I'm having trouble processing that. Please try again."

    # Format context for display
    context_display = context if context else "No relevant documents retrieved."

    # Update history
    history.append({"role": "Patient", "content": user_input})
    history.append({"role": "Ella", "content": output})

    # Format chat history for display
    chat_display = ""
    for msg in history[-6:]:
        role = msg["role"]
        content = msg["content"]
        chat_display += f"**{role}:** {content}\n\n"

    return history, chat_display, f"{decision.intent} ({decision.priority})", decision.thought_process, output, context_display


def clear_history():
    global chat_history
    chat_history = []
    return [], "", "", "", "", ""


# Custom CSS
css = """
.gradio-container { max-width: 900px !important; }
.intent-badge { 
    display: inline-block; 
    padding: 4px 12px; 
    border-radius: 20px; 
    font-weight: bold;
    font-size: 14px;
}
"""

# Build UI
with gr.Blocks(css=css, title="Ella — Medical Triage RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="margin-bottom: 5px;">ELLA</h1>
        <p style="color: #666; margin-top: 0;">Medical Triage & Clinical RAG Engine</p>
        <p style="font-size: 13px; color: #999;">
        Powered by NVIDIA NIM • Pinecone • Groq | 90,306 clinical records | 96% intent accuracy
        </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            chat_display = gr.Markdown(
                value="*Start a conversation by typing your medical question below.*",
                label="Chat",
                height=400,
            )
            user_input = gr.Textbox(
                placeholder="e.g., What are the symptoms of a heart attack?",
                label="Your Message",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            gr.Markdown("### Pipeline Output")
            intent_output = gr.Textbox(label="Intent", interactive=False)
            thought_output = gr.Textbox(label="Thought Process", lines=3, interactive=False)
            response_output = gr.Textbox(label="Ella's Response", lines=5, interactive=False)
            context_output = gr.Textbox(label="Retrieved Context", lines=5, interactive=False)

    gr.Markdown(
        """
        ---
        <div style="text-align: center; font-size: 12px; color: #999;">
        Built with NVIDIA NIM Embeddings • Pinecone Vector DB • Groq LLM • CrossEncoder Reranker<br>
        90,306 clinical text chunks • Hybrid retrieval • 96% intent accuracy
        </div>
        """
    )

    # State
    history_state = gr.State([])

    # Events
    submit_btn.click(
        process_query_gpu,
        inputs=[user_input, history_state],
        outputs=[history_state, chat_display, intent_output, thought_output, response_output, context_output],
    ).then(lambda: "", outputs=user_input)

    user_input.submit(
        process_query_gpu,
        inputs=[user_input, history_state],
        outputs=[history_state, chat_display, intent_output, thought_output, response_output, context_output],
    ).then(lambda: "", outputs=user_input)

    clear_btn.click(
        clear_history,
        outputs=[history_state, chat_display, intent_output, thought_output, response_output, context_output],
    )


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
