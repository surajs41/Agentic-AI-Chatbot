import os
import tempfile
from functools import lru_cache
from typing import BinaryIO

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pypdf import PdfReader
from sqlalchemy import create_engine, text

load_dotenv()

CONNECTION_STRING = os.getenv(
    "POSTGRES_CONNECTION_STRING",
    "postgresql+psycopg://postgres:postgres@localhost:5433/rag_chatbot",
)
COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION", "document_embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
VECTOR_STORE_MODE = os.getenv("VECTOR_STORE", "auto").lower()
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embedding.tolist() for embedding in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.embed([text])).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(EMBEDDING_MODEL)


def postgres_available() -> bool:
    try:
        engine = create_engine(CONNECTION_STRING)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_active_backend() -> str:
    if VECTOR_STORE_MODE == "chroma":
        return "chroma"
    if VECTOR_STORE_MODE == "pgvector":
        return "pgvector" if postgres_available() else "chroma"
    return "pgvector" if postgres_available() else "chroma"


@lru_cache(maxsize=1)
def _get_pgvector_store() -> VectorStore:
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )


@lru_cache(maxsize=1)
def _get_chroma_store() -> VectorStore:
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )


def get_vector_store() -> VectorStore:
    if get_active_backend() == "pgvector":
        return _get_pgvector_store()
    return _get_chroma_store()


def check_db_connection() -> tuple[bool, str]:
    backend = get_active_backend()
    if backend == "pgvector":
        return True, "Connected to PostgreSQL + pgVector"
    if VECTOR_STORE_MODE == "pgvector" and not postgres_available():
        return False, "PostgreSQL unavailable. Start Docker Desktop, then run: docker compose up -d"
    return True, f"Using local Chroma store (Docker offline — RAG still works)"


def load_pdf_documents(file_path: str, file_name: str) -> list[Document]:
    reader = PdfReader(file_path)
    documents = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_name, "page": page_index + 1},
                )
            )

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    chunks: list[Document] = []

    for document in documents:
        text = document.page_content
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=document.metadata.copy(),
                    )
                )
            if end >= len(text):
                break
            start = end - chunk_overlap

    return chunks


def ingest_pdf_file(uploaded_file: BinaryIO) -> tuple[int, str]:
    uploaded_file.seek(0)
    file_name = getattr(uploaded_file, "name", "uploaded.pdf")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        documents = load_pdf_documents(tmp_path, file_name)
        chunks = split_documents(documents)
        if not chunks:
            return 0, file_name

        get_vector_store().add_documents(chunks)
        return len(chunks), file_name
    finally:
        os.unlink(tmp_path)


def retrieve_context(query: str, k: int = 4) -> tuple[str, list[Document]]:
    docs = get_vector_store().similarity_search(query, k=k)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return context, docs


def format_sources(sources: list[Document]) -> list[dict]:
    formatted = []
    for index, doc in enumerate(sources, start=1):
        metadata = doc.metadata or {}
        preview = doc.page_content[:280].strip()
        if len(doc.page_content) > 280:
            preview += "..."
        formatted.append(
            {
                "index": index,
                "page": metadata.get("page", "N/A"),
                "source": metadata.get("source", "Uploaded PDF"),
                "preview": preview,
            }
        )
    return formatted
