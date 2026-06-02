import streamlit as st
import os
import time
from pathlib import Path
from rag_engine import RAGEngine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CosmicRAG · Intelligent Document Chat",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg:       #090b10;
    --surface:  #0f1318;
    --border:   #1e2530;
    --accent:   #5b8cff;
    --accent2:  #a78bfa;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --success:  #34d399;
    --gold:     #fbbf24;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Header */
.cosmic-header {
    text-align: center;
    padding: 2rem 0 1rem;
    position: relative;
}
.cosmic-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #5b8cff 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}
.cosmic-header p {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Stars background effect */
.stars-bg {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(1px 1px at 20% 30%, rgba(91,140,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 10%, rgba(167,139,250,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 50% 80%, rgba(52,211,153,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 10% 60%, rgba(91,140,255,0.2) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 70%, rgba(167,139,250,0.2) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* Chat messages */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 1rem 0;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #1e3a5f, #2d1f5e);
    border: 1px solid rgba(91,140,255,0.3);
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.2rem;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.6;
}
.msg-assistant {
    display: flex;
    justify-content: flex-start;
    margin: 1rem 0;
}
.msg-assistant .bubble {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 0.8rem 1.2rem;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.7;
}
.msg-assistant .avatar {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #5b8cff, #a78bfa);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    margin-right: 0.7rem;
    flex-shrink: 0;
    margin-top: 0.2rem;
}

/* Source citations */
.source-pill {
    display: inline-block;
    background: rgba(91,140,255,0.1);
    border: 1px solid rgba(91,140,255,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: var(--accent);
    margin: 0.3rem 0.2rem 0 0;
    font-family: 'Space Mono', monospace;
}

/* Status badge */
.status-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.8rem;
    color: var(--success);
    font-family: 'Space Mono', monospace;
}
.status-dot {
    width: 7px; height: 7px;
    background: var(--success);
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state h3 { color: var(--text); margin-bottom: 0.5rem; font-weight: 500; }

/* Streamlit overrides */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(91,140,255,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #5b8cff, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(91,140,255,0.4) !important;
}
.stFileUploader {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
}
.stFileUploader:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
}
div[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
.stProgress .st-bo { background: var(--accent) !important; }
h1,h2,h3,h4 { font-family: 'Outfit', sans-serif !important; color: var(--text) !important; }
.stMarkdown p { color: var(--text) !important; }
label { color: var(--muted) !important; font-size: 0.85rem !important; }
[data-testid="stSidebarContent"] { padding: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:1.5rem;'>
        <div style='font-size:2.5rem;'>🔭</div>
        <div style='font-family:"Space Mono",monospace; font-size:1rem;
                    background:linear-gradient(135deg,#5b8cff,#a78bfa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; font-weight:700;'>CosmicRAG</div>
        <div style='color:#64748b; font-size:0.75rem; letter-spacing:2px; margin-top:0.2rem;'>
            DOCUMENT INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("#### 🌌 Demo: Astrophysics KB")
    use_demo = st.toggle("Load astrophysics knowledge base", value=False)

    groq_key = st.text_input(
        "Groq API Key (free at console.groq.com)",
        type="password",
        placeholder="gsk_...",
    )

    if st.button("⚡ Build Knowledge Base", use_container_width=True):
        if not groq_key:
            st.error("Please enter a Groq API key.")
        elif not uploaded_files and not use_demo:
            st.error("Upload PDFs or enable the demo knowledge base.")
        else:
            with st.spinner("Chunking & embedding documents..."):
                try:
                    rag = RAGEngine(groq_api_key=groq_key)
                    all_docs = []

                    if uploaded_files:
                        docs, chunks = rag.load_pdfs(uploaded_files)
                        all_docs.extend(docs)

                    if use_demo:
                        demo_docs, demo_chunks = rag.load_demo_astrophysics()
                        all_docs.extend(demo_docs)

                    total_chunks = rag.build_vectorstore(all_docs)
                    st.session_state.rag = rag
                    st.session_state.docs_loaded = True
                    st.session_state.doc_count = len(uploaded_files) + (1 if use_demo else 0)
                    st.session_state.chunk_count = total_chunks
                    st.session_state.messages = []
                    st.success(f"✅ {total_chunks} chunks indexed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.docs_loaded:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Docs", st.session_state.doc_count)
        with col2:
            st.metric("🧩 Chunks", st.session_state.chunk_count)

        st.markdown("""
        <div class='status-badge'>
            <div class='status-dot'></div> KB Active
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style='margin-top:2rem; padding:1rem; background:#0f1318;
                border:1px solid #1e2530; border-radius:10px;
                font-size:0.78rem; color:#64748b; line-height:1.7;'>
        <b style='color:#a78bfa;'>Stack</b><br>
        LangChain · ChromaDB<br>
        HF Embeddings · Groq LLM<br>
        Streamlit · Python 3.10+<br><br>
        <b style='color:#5b8cff;'>Built by</b><br>
        Deepali · ML Engineer
    </div>
    """, unsafe_allow_html=True)

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='cosmic-header'>
    <h1>🔭 CosmicRAG</h1>
    <p>Retrieval-Augmented Generation · Document Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Chat display
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class='empty-state'>
            <div class='icon'>🌌</div>
            <h3>Your knowledge base awaits</h3>
            <p>Upload PDFs or load the astrophysics demo, add your Groq key,<br>
            then hit <b>Build Knowledge Base</b> to start chatting.</p>
            <br>
            <div style='display:flex; gap:0.8rem; justify-content:center; flex-wrap:wrap;'>
                <span class='source-pill'>📡 Ask about dark matter</span>
                <span class='source-pill'>⭐ Query stellar evolution</span>
                <span class='source-pill'>🌀 Explore black holes</span>
                <span class='source-pill'>📄 Summarize any PDF</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class='msg-user'>
                    <div class='bubble'>{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = ""
                if msg.get("sources"):
                    pills = "".join([
                        f"<span class='source-pill'>📄 {s}</span>"
                        for s in msg["sources"]
                    ])
                    sources_html = f"<div style='margin-top:0.6rem;'>{pills}</div>"

                st.markdown(f"""
                <div class='msg-assistant'>
                    <div class='avatar'>✦</div>
                    <div class='bubble'>
                        {msg['content']}
                        {sources_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ── Input bar ──────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "Ask anything about your documents",
        placeholder="e.g. What is the Chandrasekhar limit? / Summarize page 3...",
        label_visibility="collapsed",
        key="user_input",
    )
with col_btn:
    send = st.button("Send ↗", use_container_width=True)

if send and user_input:
    if not st.session_state.docs_loaded:
        st.warning("⚠️ Build the knowledge base first using the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Retrieving & reasoning..."):
            try:
                answer, sources = st.session_state.rag.query(user_input)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)}",
                    "sources": [],
                })
        st.rerun()