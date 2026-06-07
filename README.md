# Agentic AI Chatbot

An agentic chatbot built with **LangGraph**, **Groq LLM**, and **RAG** (Retrieval-Augmented Generation) for PDF document Q&A.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

## Features

- **General Chat** — Conversational AI powered by Groq (Llama 3.3)
- **Document Q&A (RAG)** — Upload PDFs, chunk & embed, retrieve relevant context, generate answers
- **LangGraph orchestration** — Agent workflow with retrieve → generate nodes
- **Vector storage** — pgVector (PostgreSQL) with automatic Chroma fallback when Docker is offline
- **Modern Streamlit UI** — Dark theme, mode switching, source citations

## Architecture

```
User Query → LangGraph → [Retrieve from Vector DB] → [Groq LLM Generate] → Answer + Sources
PDF Upload → Load → Chunk → Embed → Store (pgVector / Chroma)
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph |
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | FastEmbed (BGE-small) |
| Vector DB | pgVector / Chroma |
| PDF Processing | PyPDF |
| Frontend | Streamlit |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/surajs41/Agentic-AI-Chatbot.git
cd Agentic-AI-Chatbot
```

### 2. Create environment

```bash
conda create -n agentic-chatbot python=3.11 -y
conda activate agentic-chatbot
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key from [console.groq.com](https://console.groq.com).

### 4. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## RAG Mode (Optional)

For **pgVector** (PostgreSQL):

1. Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Run:

```bash
docker compose up -d
```

3. Refresh the app — sidebar will show **Connected to PostgreSQL + pgVector**

If Docker is not available, the app automatically uses **local Chroma** storage — RAG still works.

## Project Structure

```
├── app.py                      # Streamlit UI
├── agentic_chatbot_backend.py  # LangGraph chat & RAG graphs
├── rag_backend.py              # PDF ingest, embeddings, vector retrieval
├── docker-compose.yml          # PostgreSQL + pgVector
├── requirements.txt
├── test.py                     # CLI streaming test
└── non_stream.py               # Non-streaming Streamlit variant
```

## Usage

1. **General Chat** — Ask anything; Groq responds directly
2. **Document Q&A** — Upload PDF(s) → Ingest → Ask questions about the content
3. View **Retrieved Sources** under each RAG answer

## License

MIT
