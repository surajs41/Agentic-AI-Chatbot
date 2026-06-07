from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import os

from rag_backend import DEFAULT_COLLECTIONS, format_sources, retrieve_context

load_dotenv()

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

RAG_SYSTEM_PROMPT = """You are a document Q&A assistant powered by RAG (Retrieval-Augmented Generation).

Answer the user's question using ONLY the context retrieved from uploaded documents.
If the answer is not present in the context, clearly say that the information was not found in the documents.
Keep answers concise, accurate, and well-structured. Use markdown and code blocks when helpful.

Retrieved context:
{context}
"""


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    sources: list
    collection: str


def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def rag_retrieve_node(state: RAGState):
    query = state["messages"][-1].content
    collection = state.get("collection") or "General"
    context, sources = retrieve_context(query, collection=collection)
    return {"context": context, "sources": sources}


def rag_generate_node(state: RAGState):
    system_message = SystemMessage(
        content=RAG_SYSTEM_PROMPT.format(context=state.get("context", ""))
    )
    response = llm.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


checkpoint = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpoint)

rag_graph = StateGraph(RAGState)
rag_graph.add_node("retrieve", rag_retrieve_node)
rag_graph.add_node("generate", rag_generate_node)
rag_graph.add_edge(START, "retrieve")
rag_graph.add_edge("retrieve", "generate")
rag_graph.add_edge("generate", END)
rag_chatbot = rag_graph.compile(checkpointer=checkpoint)

__all__ = ["chatbot", "rag_chatbot", "format_sources", "llm"]
