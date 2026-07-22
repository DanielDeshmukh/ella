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
    return process_query(user_input, history)


def process_query(user_input, history):
    if not user_input.strip():
        return history, "", "", "", "", ""

    router = initialize_ella()
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])

    try:
        decision = router.route_request(user_input, history=history_str)
    except Exception as e:
        return history, "", f"Error: {e}", "", "", ""

    context = getattr(decision, "retrieved_context", "")

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

    context_display = context if context else "No relevant documents retrieved."

    history.append({"role": "Patient", "content": user_input})
    history.append({"role": "Ella", "content": output})

    return history, output, f"{decision.intent} ({decision.priority})", decision.thought_process, output, context_display


def clear_all():
    return [], "", "", "", "", ""


css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container {
    background: #0f1117 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
}

.main-title {
    text-align: center;
    padding: 48px 0 32px;
}

.main-title h1 {
    font-size: 48px !important;
    font-weight: 700 !important;
    letter-spacing: 12px !important;
    color: #ffffff !important;
    margin: 0 !important;
    text-transform: uppercase;
}

.main-title p {
    font-size: 15px;
    color: #64748b;
    margin-top: 8px;
    letter-spacing: 3px;
    font-weight: 300;
}

.badges-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 20px;
    flex-wrap: wrap;
}

.badge-item {
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.8px;
    border: 1px solid #1e293b;
    background: transparent;
    color: #94a3b8;
    text-transform: uppercase;
}

.badge-item.green {
    border-color: #10b981;
    color: #10b981;
}

.divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 0 60px;
}

.chat-panel {
    background: #111318;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 24px;
    min-height: 500px;
}

.chat-msg {
    padding: 16px 20px;
    margin-bottom: 16px;
    border-radius: 10px;
    font-size: 14px;
    line-height: 1.7;
    color: #e2e8f0;
}

.chat-msg.patient {
    background: #1a1d2e;
    border-left: 3px solid #6366f1;
}

.chat-msg.ella {
    background: #111827;
    border-left: 3px solid #10b981;
}

.chat-msg strong {
    color: #f8fafc;
    font-weight: 600;
    display: block;
    margin-bottom: 6px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.pipeline-panel {
    background: #111318;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 24px;
    height: 100%;
}

.pipeline-panel h3 {
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 24px 0 !important;
    padding-bottom: 16px;
    border-bottom: 1px solid #1e293b;
}

.pipeline-card {
    background: #1a1d2e;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}

.pipeline-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 10px;
    display: block;
}

.pipeline-value {
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.6;
    word-wrap: break-word;
}

.input-row {
    background: #111318;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.input-row textarea {
    background: #1a1d2e !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    flex: 1 !important;
}

.input-row textarea:focus {
    border-color: #6366f1 !important;
    outline: none !important;
}

.send-btn {
    background: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 12px 28px !important;
    font-size: 14px !important;
    letter-spacing: 0.5px;
    min-width: 100px !important;
}

.send-btn:hover {
    background: #5558e6 !important;
}

.clear-btn {
    background: transparent !important;
    color: #64748b !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    font-size: 14px !important;
}

.clear-btn:hover {
    border-color: #334155 !important;
    color: #94a3b8 !important;
}

.footer-text {
    text-align: center;
    padding: 32px 0 24px;
    color: #334155;
    font-size: 12px;
    letter-spacing: 0.5px;
}

label {
    display: block !important;
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    margin-bottom: 6px !important;
}

textarea, input[type="text"] {
    background: #1a1d2e !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #6366f1 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

textarea::placeholder, input[type="text"]::placeholder {
    color: #475569 !important;
}
"""

with gr.Blocks(css=css, title="Ella — Medical Triage RAG", theme=gr.themes.Base()) as demo:
    # Header
    gr.HTML("""
        <div class="main-title">
            <h1>ELLA</h1>
            <p>Medical Triage & Clinical RAG Engine</p>
            <div class="badges-row">
                <span class="badge-item">NVIDIA NIM</span>
                <span class="badge-item">Pinecone</span>
                <span class="badge-item">Groq</span>
                <span class="badge-item green">90,306 records</span>
                <span class="badge-item green">96% accuracy</span>
            </div>
        </div>
        <hr class="divider">
    """)

    with gr.Row(equal_height=True):
        # Chat Column
        with gr.Column(scale=3):
            chat_display = gr.Markdown(
                value="*Start a conversation by typing your medical question below.*",
                elem_classes=["chat-panel"],
            )

            with gr.Row(elem_classes=["input-row"]):
                user_input = gr.Textbox(
                    placeholder="e.g., What are the symptoms of a heart attack?",
                    show_label=False,
                    lines=1,
                    max_lines=3,
                    scale=4,
                )
                submit_btn = gr.Button("Send", elem_classes=["send-btn"], scale=0)
                clear_btn = gr.Button("Clear", elem_classes=["clear-btn"], scale=0)

        # Pipeline Column
        with gr.Column(scale=2):
            gr.HTML('<div class="pipeline-panel"><h3>Pipeline Output</h3>')

            gr.HTML('<div class="pipeline-card"><span class="pipeline-label">Intent</span>')
            intent_output = gr.Textbox(show_label=False, interactive=False, elem_classes=["pipeline-value"])
            gr.HTML('</div>')

            gr.HTML('<div class="pipeline-card"><span class="pipeline-label">Thought Process</span>')
            thought_output = gr.Textbox(show_label=False, interactive=False, lines=3, elem_classes=["pipeline-value"])
            gr.HTML('</div>')

            gr.HTML('<div class="pipeline-card"><span class="pipeline-label">Retrieved Context</span>')
            context_output = gr.Textbox(show_label=False, interactive=False, lines=5, elem_classes=["pipeline-value"])
            gr.HTML('</div>')

            gr.HTML('</div>')

    gr.HTML("""
        <div class="footer-text">
            NVIDIA NIM Embeddings • Pinecone Vector DB • Groq LLM • CrossEncoder Reranker<br>
            90,306 clinical text chunks • Hybrid retrieval • 96% intent accuracy
        </div>
    """)

    # State
    history_state = gr.State([])

    def update_chat(history):
        if not history:
            return "*Start a conversation by typing your medical question below.*"
        html = ""
        for msg in history[-6:]:
            role = msg["role"].lower()
            html += f'<div class="chat-msg {role}"><strong>{msg["role"]}</strong>{msg["content"]}</div>'
        return html

    # Events
    submit_btn.click(
        process_query_gpu,
        inputs=[user_input, history_state],
        outputs=[history_state, chat_display, intent_output, thought_output, context_output],
    ).then(
        update_chat, inputs=[history_state], outputs=[chat_display]
    ).then(lambda: "", outputs=user_input)

    user_input.submit(
        process_query_gpu,
        inputs=[user_input, history_state],
        outputs=[history_state, chat_display, intent_output, thought_output, context_output],
    ).then(
        update_chat, inputs=[history_state], outputs=[chat_display]
    ).then(lambda: "", outputs=user_input)

    clear_btn.click(
        clear_all,
        outputs=[history_state, chat_display, intent_output, thought_output, context_output],
    )


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
