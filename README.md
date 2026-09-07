# RAG Agent Demo (LangChain + LangGraph + OpenAI)

A minimal, working example of an agentic RAG (Retrieval-Augmented Generation) assistant:
a LangGraph tool-calling loop that lets an OpenAI chat model decide when to search a local
Chroma vector store built from a small set of sample docs (a fictional company,
"TechNova Gadgets" — product catalog, warranty, returns, shipping FAQ).

## Setup

```bash
cd <path-to-project>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY=sk-...`.

## Usage

1. Build the vector store from `data/*.md` (run once, or again after editing the docs):

```bash
python src/ingest.py
```

2. Chat with the agent:

```bash
python src/main.py
```

Example questions to try:
- "What's the battery life of the NovaBuds Pro?"
- "Can I return the NovaWatch after 20 days?"
- "Does the NovaCam Mini's warranty cover water damage?"
- "How much is expedited shipping and how fast is it?"

## How it works

- [`src/ingest.py`](src/ingest.py) — splits the markdown docs in `data/` and embeds them
  (`text-embedding-3-small`) into a local persisted Chroma collection (`chroma_db/`).
- [`src/agent.py`](src/agent.py) — defines a `search_knowledge_base` tool over that
  Chroma store, and a LangGraph `StateGraph` with two nodes: `agent` (calls
  `gpt-4o-mini` with the tool bound) and `tools` (executes the tool). A conditional
  edge (`tools_condition`) routes back and forth until the model has enough context
  to answer directly.
- [`src/main.py`](src/main.py) — a simple REPL that keeps conversation history across turns.

## Project status / next steps

This covers the core RAG agent only. A planned next phase will extend this agent to:
1. Read incoming WhatsApp messages.
2. Draft a suggested reply grounded in the knowledge base.
3. Only send the reply after explicit user confirmation (human-in-the-loop).

That part isn't built yet — it needs a decision on WhatsApp integration method
(official WhatsApp Business Cloud API vs. an unofficial library) before implementation.
