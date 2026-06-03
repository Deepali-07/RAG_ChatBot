import gradio as gr
import os
import tempfile
from rag_engine import RAGEngine

# Global RAG instance
rag_instance = None

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap');

body, .gradio-container {
    background: #090b10 !important;
    font-family: 'Outfit', sans-serif !important;
    color: #e2e8f0 !important;
}

/* Header */
.cosmic-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.cosmic-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 2.2rem !important;
    background: linear-gradient(135deg, #5b8cff 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 !important;
}
.cosmic-sub {
    color: #64748b;
    font-size: 0.8rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Panels */
.panel {
    background: #0f1318 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
}

/* Chatbot */
.chatbot-wrap .message {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    border-radius: 14px !important;
}
.chatbot-wrap .user .message {
    background: linear-gradient(135deg, #1e3a5f, #2d1f5e) !important;
    border: 1px solid rgba(91,140,255,0.3) !important;
    color: #e2e8f0 !important;
}
.chatbot-wrap .bot .message {
    background: #0f1318 !important;
    border: 1px solid #1e2530 !important;
    color: #e2e8f0 !important;
}

/* Inputs */
input[type="text"], input[type="password"], textarea {
    background: #0f1318 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
}
input[type="text"]:focus, input[type="password"]:focus, textarea:focus {
    border-color: #5b8cff !important;
    box-shadow: 0 0 0 2px rgba(91,140,255,0.2) !important;
}

/* Buttons */
.btn-primary {
    background: linear-gradient(135deg, #5b8cff, #a78bfa) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(91,140,255,0.4) !important;
}
.btn-secondary {
    background: #0f1318 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Status */
.status-ok {
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 20px;
    padding: 0.4rem 1rem;
    color: #34d399;
    font-size: 0.82rem;
    font-family: 'Space Mono', monospace;
    display: inline-block;
}
.status-idle {
    background: rgba(100,116,139,0.1);
    border: 1px solid rgba(100,116,139,0.2);
    border-radius: 20px;
    padding: 0.4rem 1rem;
    color: #64748b;
    font-size: 0.82rem;
    font-family: 'Space Mono', monospace;
    display: inline-block;
}

/* Labels */
label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Accordion */
.accordion {
    background: #0f1318 !important;
    border: 1px solid #1e2530 !important;
    border-radius: 10px !important;
}

/* Stack info box */
.stack-box {
    background: #0a0d12;
    border: 1px solid #1e2530;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.9;
    margin-top: 0.5rem;
}

/* File upload */
.upload-zone {
    border: 2px dashed #1e2530 !important;
    border-radius: 12px !important;
    background: #0a0d12 !important;
}
"""

def build_kb(groq_key, pdf_files, use_demo):
    global rag_instance
    if not groq_key or not groq_key.strip():
        return "⚠️ Please enter your Groq API key.", "<span class='status-idle'>● Idle</span>"
    if not pdf_files and not use_demo:
        return "⚠️ Upload PDFs or enable the astrophysics demo KB.", "<span class='status-idle'>● Idle</span>"
    try:
        rag = RAGEngine(groq_api_key=groq_key.strip())
        all_docs = []
        if pdf_files:
            # pdf_files is a list of file paths in Gradio
            import types
            class FakeFile:
                def __init__(self, path):
                    self.name = os.path.basename(path)
                    self._path = path
                def read(self):
                    with open(self._path, "rb") as f:
                        return f.read()
            fake_files = [FakeFile(f) for f in pdf_files]
            docs, _ = rag.load_pdfs(fake_files)
            all_docs.extend(docs)
        if use_demo:
            demo_docs, _ = rag.load_demo_astrophysics()
            all_docs.extend(demo_docs)
        total_chunks = rag.build_vectorstore(all_docs)
        rag_instance = rag
        doc_count = len(pdf_files) + (1 if use_demo else 0)
        msg = f"✅ Knowledge base ready! {doc_count} doc(s) · {total_chunks} chunks indexed."
        status = "<span class='status-ok'>● KB Active</span>"
        return msg, status
    except Exception as e:
        return f"❌ Error: {str(e)}", "<span class='status-idle'>● Idle</span>"


def respond(message, history):
    global rag_instance
    if not message.strip():
        return history, ""
    if rag_instance is None:
        history.append((message, "⚠️ Please build the knowledge base first using the panel on the left."))
        return history, ""
    try:
        answer, sources = rag_instance.query(message)
        if sources:
            sources_md = "\n\n📄 **Sources:** " + " · ".join(sources)
            answer = answer + sources_md
        history.append((message, answer))
    except Exception as e:
        history.append((message, f"❌ Error: {str(e)}"))
    return history, ""


def clear_chat():
    return [], ""


# ── Build Gradio UI ────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="CosmicRAG · Document Intelligence") as demo:

    gr.HTML("""
    <div class='cosmic-header'>
        <div class='cosmic-title'>🔭 CosmicRAG</div>
        <div class='cosmic-sub'>Retrieval-Augmented Generation · Document Intelligence</div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left panel ─────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            gr.HTML("<div style='color:#a78bfa; font-weight:600; font-size:0.9rem; margin-bottom:0.5rem;'>⚙️ Setup</div>")

            groq_key = gr.Textbox(
                label="Groq API Key",
                placeholder="gsk_... (free at console.groq.com)",
                type="password",
                elem_classes=["panel"],
            )

            pdf_files = gr.File(
                label="📄 Upload PDFs",
                file_types=[".pdf"],
                file_count="multiple",
                elem_classes=["upload-zone"],
            )

            use_demo = gr.Checkbox(
                label="🌌 Load astrophysics knowledge base (demo)",
                value=False,
            )

            build_btn = gr.Button(
                "⚡ Build Knowledge Base",
                variant="primary",
                elem_classes=["btn-primary"],
            )

            build_status = gr.HTML("<span class='status-idle'>● Idle</span>")
            build_msg = gr.Textbox(
                label="Build log",
                interactive=False,
                lines=2,
                elem_classes=["panel"],
            )

            gr.HTML("""
            <div class='stack-box'>
                <span style='color:#a78bfa; font-weight:600;'>Stack</span><br>
                LangChain · ChromaDB<br>
                HF Embeddings · Groq LLM<br>
                Gradio · Python 3.10+<br><br>
                <span style='color:#5b8cff; font-weight:600;'>Built by</span><br>
                Deepali · ML Engineer<br>
                <a href='https://github.com/Deepali-07' style='color:#5b8cff;'>GitHub</a> ·
                <a href='https://huggingface.co/DeepaliMadala' style='color:#5b8cff;'>HuggingFace</a>
            </div>
            """)

        # ── Right panel ────────────────────────────────────────────────────────
        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                value=[],
                height=520,
                label="",
                show_label=False,
                bubble_full_width=False,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=cosmicrag"),
                elem_classes=["chatbot-wrap"],
                placeholder="""
                <div style='text-align:center; padding:3rem; color:#334155;'>
                    <div style='font-size:3rem; margin-bottom:1rem;'>🌌</div>
                    <div style='font-size:1.1rem; color:#475569; font-weight:500;'>Your knowledge base awaits</div>
                    <div style='font-size:0.85rem; color:#334155; margin-top:0.5rem;'>
                        Build the KB on the left, then ask anything
                    </div>
                    <div style='margin-top:1.5rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap;'>
                        <span style='background:#0f172a; border:1px solid #1e2530; border-radius:20px; padding:0.3rem 0.8rem; font-size:0.78rem; color:#5b8cff;'>📡 Dark matter</span>
                        <span style='background:#0f172a; border:1px solid #1e2530; border-radius:20px; padding:0.3rem 0.8rem; font-size:0.78rem; color:#a78bfa;'>⭐ Stellar evolution</span>
                        <span style='background:#0f172a; border:1px solid #1e2530; border-radius:20px; padding:0.3rem 0.8rem; font-size:0.78rem; color:#34d399;'>🌀 Black holes</span>
                        <span style='background:#0f172a; border:1px solid #1e2530; border-radius:20px; padding:0.3rem 0.8rem; font-size:0.78rem; color:#fbbf24;'>📄 Summarize PDF</span>
                    </div>
                </div>
                """,
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask anything about your documents...",
                    show_label=False,
                    scale=5,
                    container=False,
                    elem_classes=["panel"],
                )
                send_btn = gr.Button(
                    "Send ↗",
                    scale=1,
                    variant="primary",
                    elem_classes=["btn-primary"],
                )
                clear_btn = gr.Button(
                    "🗑 Clear",
                    scale=1,
                    elem_classes=["btn-secondary"],
                )

    # ── Event wiring ───────────────────────────────────────────────────────────
    build_btn.click(
        fn=build_kb,
        inputs=[groq_key, pdf_files, use_demo],
        outputs=[build_msg, build_status],
    )

    send_btn.click(
        fn=respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box],
    )

    msg_box.submit(
        fn=respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box],
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg_box],
    )


if __name__ == "__main__":
    demo.launch()