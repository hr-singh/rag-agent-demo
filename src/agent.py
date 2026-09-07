"""Agentic RAG agent: a LangGraph tool-calling loop over a Chroma retriever, backed by OpenAI."""
import pathlib

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parent.parent
PERSIST_DIR = str(ROOT / "chroma_db")
COLLECTION_NAME = "technova_docs"

SYSTEM_PROMPT = (
    "You are a helpful support assistant for TechNova Gadgets, a consumer electronics company. "
    "Always use the search_knowledge_base tool to find grounded information before answering "
    "questions about products, pricing, warranty, returns, or shipping. "
    "If the knowledge base doesn't contain the answer, say so honestly instead of guessing. "
    "Keep answers concise, and mention which policy or product the info came from when relevant."
)

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=PERSIST_DIR,
)


@tool
def search_knowledge_base(query: str) -> str:
    """Search TechNova Gadgets' product catalog, warranty, return, and shipping docs."""
    docs = _store.similarity_search(query, k=4)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_llm_with_tools = _llm.bind_tools([search_knowledge_base])


def _call_model(state: MessagesState):
    response = _llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


_graph_builder = StateGraph(MessagesState)
_graph_builder.add_node("agent", _call_model)
_graph_builder.add_node("tools", ToolNode([search_knowledge_base]))
_graph_builder.add_edge(START, "agent")
_graph_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
_graph_builder.add_edge("tools", "agent")

graph = _graph_builder.compile()


def new_conversation() -> list:
    """Starting message list for a fresh conversation."""
    return [SystemMessage(content=SYSTEM_PROMPT)]


def ask(question: str, history: list | None = None) -> tuple[str, list]:
    """Run one turn of the RAG agent. Returns (answer_text, updated_history)."""
    messages = list(history) if history else new_conversation()
    messages.append({"role": "user", "content": question})
    result = graph.invoke({"messages": messages})
    updated = result["messages"]
    answer = updated[-1].content
    return answer, updated


if __name__ == "__main__":
    answer, _ = ask("What is the battery life of the NovaBuds Pro and can I return them after 20 days?")
    print(answer)
