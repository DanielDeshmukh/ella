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
        return history, "", "", "", ""

    router = initialize_ella()
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-3:]])

    try:
        decision = router.route_request(user_input, history=history_str)
    except Exception as e:
        return history, f"Error: {e}", "", "", ""

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


css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a2236;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.3);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #1e293b;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --radius: 12px;
}

.gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1100px !important;
    color: var(--text-primary) !important;
}

/* Header */
.header-section {
    text-align: center;
    padding: 40px 0 30px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 30px;
}

.header-section h1 {
    font-size: 42px !important;
    font-weight: 700 !important;
    letter-spacing: 8px !important;
    color: var(--text-primary) !important;
    margin-bottom: 8px !important;
    text-transform: uppercase;
}

.header-section .subtitle {
    font-size: 16px;
    color: var(--text-secondary);
    font-weight: 300;
    letter-spacing: 2px;
    margin-bottom: 16px;
}

.header-section .badges {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 16px;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.5px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-secondary);
}

.badge.accent {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(99, 102, 241, 0.1);
}

.badge.success {
    border-color: var(--success);
    color: var(--success);
    background: rgba(16, 185, 129, 0.1);
}

/* Chat Area */
.chat-area {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    min-height: 400px;
}

.chat-message {
    padding: 12px 16px;
    margin-bottom: 12px;
    border-radius: var(--radius);
    font-size: 14px;
    line-height: 1.6;
}

.chat-message.patient {
    background: rgba(99, 102, 241, 0.1);
    border-left: 3px solid var(--accent);
    color: var(--text-primary);
}

.chat-message.ella {
    background: var(--bg-card);
    border-left: 3px solid var(--success);
    color: var(--text-primary);
}

/* Input Area */
.input-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-top: 16px;
}

/* Pipeline Output */
.pipeline-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
}

.pipeline-section h3 {
    color: var(--text-primary) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 20px !important;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}

.pipeline-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

.pipeline-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
    display: block;
}

.pipeline-value {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* Buttons */
.primary-btn {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
    padding: 12px 24px !important;
    transition: all 0.2s !important;
}

.primary-btn:hover {
    background: #5558e6 !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
}

.secondary-btn {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
}

.secondary-btn:hover {
    border-color: var(--text-muted) !important;
    color: var(--text-primary) !important;
}

/* Textbox */
textarea, input[type="text"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
    outline: none !important;
}

/* Footer */
.footer-section {
    text-align: center;
    padding: 30px 0 20px;
    border-top: 1px solid var(--border);
    margin-top: 30px;
}

.footer-section p {
    font-size: 12px;
    color: var(--text-muted);
    letter-spacing: 0.5px;
}

/* Labels */
label, .label-wrap span {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px;
}

/* Remove Gradio defaults */
.gr-block { border: none !important; }
.gr-form { background: transparent !important; }
.gr-panel { background: transparent !important; border: none !important; }
"""

# Build UI
with gr.Blocks(css=css, title="Ella — Medical Triage RAG", theme=gr.themes.Base()) as demo:
    # Header
    gr.HTML("""
        <div class="header-section">
            <h1>ELLA</h1>
            <p class="subtitle">Medical Triage & Clinical RAG Engine</p>
            <div class="badges">
                <span class="badge accent">NVIDIA NIM</span>
                <span class="badge accent">Pinecone</span>
                <span class="badge accent">Groq</span>
                <span class="badge success">90,306 records</span>
                <span class="badge success">96% accuracy</span>
            </div>
        </div>
    """)

    with gr.Row():
        # Chat Column
        with gr.Column(scale=3):
            chat_display = gr.HTML("""
                <div class="chat-area">
                    <p style="color: var(--text-muted); font-style: italic; text-align: center; padding: 40px;">
                        Start a conversation by typing your medical question below.
                    </p>
                </div>
            """)

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="e.g., What are the symptoms of a heart attack?",
                    show_label=False,
                    lines=1,
                    max_lines=3,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=0, min_width=100)
                clear_btn = gr.Button("Clear", variant="secondary", scale=0, min_width=100)

        # Pipeline Column
        with gr.Column(scale=2):
            gr.HTML("""
                <div class="pipeline-section">
                    <h3>Pipeline Output</h3>

                    <div class="pipeline-item">
                        <span class="pipeline-label">Intent</span>
                        <div class="pipeline-value" id="intent-val">—</div>
                    </div>

                    <div class="pipeline-item">
                        <span class="pipeline-label">Thought Process</span>
                        <div class="pipeline-value" id="thought-val">—</div>
                    </div>

                    <div class="pipeline-item">
                        <span class="pipeline-label">Ella's Response</span>
                        <div class="pipeline-value" id="response-val">—</div>
                    </div>

                    <div class="pipeline-item">
                        <span class="pipeline-label">Retrieved Context</span>
                        <div class="pipeline-value" id="context-val" style="max-height: 200px; overflow-y: auto;">—</div>
                    </div>
                </div>
            """)

            intent_output = gr.Textbox(visible=False)
            thought_output = gr.Textbox(visible=False)
            response_output = gr.Textbox(visible=False)
            context_output = gr.Textbox(visible=False)

    # Footer
    gr.HTML("""
        <div class="footer-section">
            <p>Built with NVIDIA NIM Embeddings • Pinecone Vector DB • Groq LLM • CrossEncoder Reranker</p>
            <p style="margin-top: 4px;">90,306 clinical text chunks • Hybrid retrieval • 96% intent accuracy</p>
        </div>
    """)

    # State
    history_state = gr.State([])

    # Update pipeline display function
    def update_display(history, intent, thought, response, context):
        chat_html = '<div class="chat-area">'
        for msg in history[-6:]:
            role = msg["role"].lower()
            chat_html += f'<div class="chat-message {role}"><strong>{msg["role"]}:</strong> {msg["content"]}</div>'
        chat_html += '</div>'

        return chat_html, intent, thought, response, context

    def clear_all():
        return [], '<div class="chat-area"><p style="color: var(--text-muted); font-style: italic; text-align: center; padding: 40px;">Start a conversation by typing your medical question below.</p></div>', "", "", "", ""

    # Events
    submit_btn.click(
        process_query_gpu,
        inputs=[user_input, history_state],
        outputs=[history_state, intent_output, thought_output, response_output, context_output],
    ).then(
        update_display,
        inputs=[history_state, intent_output, thought_output, response_output, context_output],
        outputs=[chat_display, intent_output, thought_output, response_output, context_output],
    ).then(lambda: "", outputs=user_input)

    user_input.submit(
        process_query_gpu,
        inputs=[user_input, history_state],
        outputs=[history_state, intent_output, thought_output, response_output, context_output],
    ).then(
        update_display,
        inputs=[history_state, intent_output, thought_output, response_output, context_output],
        outputs=[chat_display, intent_output, thought_output, response_output, context_output],
    ).then(lambda: "", outputs=user_input)

    clear_btn.click(
        clear_all,
        outputs=[history_state, chat_display, intent_output, thought_output, response_output, context_output],
    )


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
