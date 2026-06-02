---
<<<<<<< HEAD
title: CosmicRAG — Intelligent Document Chat
emoji: 🔭
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
license: mit
short_description: short_description: Chat with PDFs using LangChain, ChromaDB and Groq
---

# 🔭 CosmicRAG — Retrieval-Augmented Generation Document Chat

An end-to-end RAG (Retrieval-Augmented Generation) chatbot that lets you **chat with your PDF documents** — or explore a built-in **astrophysics knowledge base**.

## ✨ Features

- 📄 **Multi-PDF upload** — drag & drop any PDFs
- 🌌 **Astrophysics demo KB** — pre-loaded knowledge on black holes, dark matter, gravitational waves, stellar evolution & more
- 🔍 **MMR retrieval** — Maximal Marginal Relevance for diverse, relevant context
- 📌 **Source citations** — every answer shows which document/page it came from
- ⚡ **Groq LLM** — blazing fast inference with LLaMA 3.1 (free tier available)
- 🆓 **HuggingFace Embeddings** — no OpenAI key needed

## 🛠 Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| LLM | Groq · LLaMA 3.1 8B Instant |
| Orchestration | LangChain |

## 🚀 How to Use

1. Get a **free Groq API key** at [console.groq.com](https://console.groq.com)
2. Upload PDFs **or** toggle the astrophysics demo KB
3. Click **Build Knowledge Base**
4. Ask anything!

## 🔧 Run Locally

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/cosmicrag
cd cosmicrag
pip install -r requirements.txt
streamlit run app.py
```

## 👩‍💻 Built By

**Deepali** · ML Engineer  
[GitHub](https://github.com/Deepali-07) · [HuggingFace](https://huggingface.co/Deepali-07)
=======
title: RAG ChatBot
emoji: 🚀
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.15.2
python_version: '3.13'
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> b95c8fc508cd0aec8768b9dbe6d4ef74b91d26be
