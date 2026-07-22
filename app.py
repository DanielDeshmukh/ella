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

    return history, output, f"{decision.intent} ({decision.priority})", decision.thought_process, context_display, ""


def clear_all():
    return [], "", "", "", "", ""


css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-base: #0a0d12;
    --bg-panel: #10141b;
    --bg-panel-raised: #151a23;
    --border-subtle: #1e2530;
    --border-strong: #2a3340;
    --text-primary: #e8ecf1;
    --text-secondary: #8993a4;
    --text-muted: #4b5563;
    --accent-primary: #2dd4bf;
    --accent-secondary: #38bdf8;
    --accent-patient: #38bdf8;
    --accent-ella: #2dd4bf;
    --accent-danger: #f87171;
}

.gradio-container {
    background: var(--bg-base) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
}

.header-bar {
    padding: 40px 24px 28px;
    text-align: center;
    border-bottom: 1px solid var(--border-subtle);
    background: linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-base) 100%);
}

.header-bar .brand-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 6px;
}

.header-bar .brand-mark {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    background: var(--accent-primary);
    box-shadow: 0 0 12px rgba(45, 212, 191, 0.6);
}

.header-bar h1 {
    font-size: 28px !important;
    font-weight: 600 !important;
    letter-spacing: 4px !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    text-transform: uppercase;
}

.header-bar p {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 6px;
    letter-spacing: 0.5px;
    font-weight: 400;
}

.badges-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-top: 22px;
    flex-wrap: wrap;
}

.badge-item {
    padding: 5px 12px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.4px;
    border: 1px solid var(--border-strong);
    background: var(--bg-panel-raised);
    color: var(--text-secondary);
}

.badge-item.metric {
    border-color: rgba(45, 212, 191, 0.35);
    color: var(--accent-primary);
    background: rgba(45, 212, 191, 0.06);
}

.content-row {
    padding: 28px 32px;
}

.chat-panel {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 22px !important;
    min-height: 480px;
}

.chat-msg {
    padding: 14px 18px;
    margin-bottom: 14px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.65;
    color: var(--text-primary);
    background: var(--bg-panel-raised);
    border: 1px solid var(--border-subtle);
    border-left: 2px solid var(--border-strong);
}

.chat-msg.patient {
    border-left-color: var(--accent-patient);
}

.chat-msg.ella {
    border-left-color: var(--accent-ella);
}

.chat-msg strong {
    color: var(--text-secondary);
    font-weight: 600;
    display: block;
    margin-bottom: 6px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.chat-msg.patient strong {
    color: var(--accent-patient);
}

.chat-msg.ella strong {
    color: var(--accent-ella);
}

.pipeline-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 22px;
    height: 100%;
}

.pipeline-panel h3 {
    color: var(--text-primary) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    margin: 0 0 20px 0 !important;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border-subtle);
}

.pipeline-card {
    background: var(--bg-panel-raised);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}

.pipeline-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--accent-secondary);
    margin-bottom: 8px;
    display: block;
}

.pipeline-value {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
    word-wrap: break-word;
}

.input-row {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    margin-top: 14px !important;
    display: flex;
    align-items: center;
    gap: 10px;
}

.input-row textarea {
    background: var(--bg-panel-raised) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text-primary) !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 11px 14px !important;
    flex: 1 !important;
}

.input-row textarea:focus {
    border-color: var(--accent-primary) !important;
    outline: none !important;
}

.send-btn {
    background: var(--accent-primary) !important;
    color: #06110f !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    padding: 11px 26px !important;
    font-size: 13px !important;
    letter-spacing: 0.4px;
    min-width: 96px !important;
}

.send-btn:hover {
    background: #26b8a4 !important;
}

.clear-btn {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    padding: 11px 18px !important;
    font-size: 13px !important;
}

.clear-btn:hover {
    border-color: var(--text-muted) !important;
    color: var(--text-primary) !important;
}

.footer-text {
    text-align: center;
    padding: 24px 0 28px;
    color: var(--text-muted);
    font-size: 11px;
    letter-spacing: 0.4px;
    line-height: 1.8;
    border-top: 1px solid var(--border-subtle);
    margin-top: 12px;
}

label {
    display: block !important;
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    margin-bottom: 6px !important;
}

textarea, input[type="text"] {
    background: var(--bg-panel-raised) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text-primary) !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--accent-primary) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(45, 212, 191, 0.15) !important;
}

textarea::placeholder, input[type="text"]::placeholder {
    color: var(--text-muted) !important;
}
"""

with gr.Blocks(css=css, title="Ella — Medical Triage RAG", theme=gr.themes.Base()) as demo:
    # Header
    gr.HTML("""
        <div class="header-bar">
            <div class="brand-row">
                <span class="brand-mark"></span>
                <h1>Ella</h1>
            </div>
            <p>Medical Triage &amp; Clinical RAG Engine</p>
            <div class="badges-row">
                <span class="badge-item">NVIDIA NIM</span>
                <span class="badge-item">Pinecone</span>
                <span class="badge-item">Groq</span>
                <span class="badge-item metric">90,306 records</span>
                <span class="badge-item metric">96% accuracy</span>
            </div>
        </div>
    """)

    with gr.Row(equal_height=True, elem_classes=["content-row"]):
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

            gr.HTML('<div class="pipeline-card"><span class="pipeline-label">Intent</span></div>')
            intent_output = gr.Textbox(show_label=False, interactive=False, elem_classes=["pipeline-value"])

            gr.HTML('<div class="pipeline-card"><span class="pipeline-label">Response</span></div>')
            response_output = gr.Textbox(show_label=False, interactive=False, lines=4, elem_classes=["pipeline-value"])

            gr.HTML('<div class="pipeline-card"><span class="pipeline-label">Thought Process</span></div>')
            thought_output = gr.Textbox(show_label=False, interactive=False, lines=3, elem_classes=["pipeline-value"])

            gr.HTML('<div class="pipeline-card"><span class="pipeline-label">Retrieved Context</span></div>')
            context_output = gr.Textbox(show_label=False, interactive=False, lines=5, elem_classes=["pipeline-value"])

            gr.HTML('</div>')

    gr.HTML("""
        <div class="footer-text">
            NVIDIA NIM Embeddings &bull; Pinecone Vector DB &bull; Groq LLM &bull; CrossEncoder Reranker<br>
            90,306 clinical text chunks &bull; Hybrid retrieval &bull; 96% intent accuracy
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
        outputs=[history_state, response_output, intent_output, thought_output, context_output, user_input],
    ).then(
        update_chat, inputs=[history_state], outputs=[chat_display]
    )

    user_input.submit(
        process_query_gpu,
        inputs=[user_input, history_state],
        outputs=[history_state, response_output, intent_output, thought_output, context_output, user_input],
    ).then(
        update_chat, inputs=[history_state], outputs=[chat_display]
    )

    clear_btn.click(
        clear_all,
        outputs=[history_state, chat_display, response_output, intent_output, thought_output, context_output],
    )


if __name__ == "__main__":
    demo.launch(ssr_mode=False)