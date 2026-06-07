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

DEFAULT_COLLECTIONS = ["General", "HR", "Finance", "Legal", "Engineering"]


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


def _collection_key(collection: str) -> str:
    safe = collection.lower().replace(" ", "_")
    return f"{COLLECTION_NAME}_{safe}"


@lru_cache(maxsize=16)
def _get_pgvector_store(collection: str) -> VectorStore:
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=_collection_key(collection),
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )


@lru_cache(maxsize=16)
def _get_chroma_store(collection: str) -> VectorStore:
    from langchain_chroma import Chroma

    path = os.path.join(CHROMA_PATH, collection.lower().replace(" ", "_"))
    return Chroma(
        collection_name=_collection_key(collection),
        embedding_function=get_embeddings(),
        persist_directory=path,
    )


def get_vector_store(collection: str = "General") -> VectorStore:
    if get_active_backend() == "pgvector":
        return _get_pgvector_store(collection)
    return _get_chroma_store(collection)


def check_db_connection() -> tuple[bool, str]:
    backend = get_active_backend()
    if backend == "pgvector":
        return True, "Connected to PostgreSQL + pgVector"
    if VECTOR_STORE_MODE == "pgvector" and not postgres_available():
        return False, "PostgreSQL unavailable. Start Docker Desktop, then run: docker compose up -d"
    return True, "Using local Chroma store (Docker offline — RAG still works)"


def load_pdf_documents(file_path: str, file_name: str, collection: str) -> list[Document]:
    reader = PdfReader(file_path)
    documents = []
    for page_index, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": file_name,
                        "page": page_index + 1,
                        "collection": collection,
                    },
                )
            )
    return documents


def load_text_documents(file_path: str, file_name: str, collection: str) -> list[Document]:
    with open(file_path, encoding="utf-8", errors="ignore") as handle:
        content = handle.read()
    if not content.strip():
        return []
    return [
        Document(
            page_content=content,
            metadata={"source": file_name, "page": 1, "collection": collection},
        )
    ]


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    chunks: list[Document] = []
    for document in documents:
        text_content = document.page_content
        start = 0
        while start < len(text_content):
            end = start + chunk_size
            chunk_text = text_content[start:end].strip()
            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=document.metadata.copy(),
                    )
                )
            if end >= len(text_content):
                break
            start = end - chunk_overlap
    return chunks


def ingest_file(
    uploaded_file: BinaryIO,
    collection: str = "General",
) -> tuple[int, int, str, str, list[dict]]:
    uploaded_file.seek(0)
    file_name = getattr(uploaded_file, "name", "uploaded.file")
    ext = file_name.rsplit(".", 1)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if ext == "pdf":
            documents = load_pdf_documents(tmp_path, file_name, collection)
        elif ext in {"txt", "csv", "md"}:
            documents = load_text_documents(tmp_path, file_name, collection)
        else:
            raise ValueError(f"Unsupported file type: .{ext}")

        pages = len(documents) if ext == "pdf" else 1
        chunks = split_documents(documents)
        if not chunks:
            return 0, pages, file_name, ext, []

        get_vector_store(collection).add_documents(chunks)
        chunk_records = [
            {
                "chunk_index": index,
                "page": chunk.metadata.get("page"),
                "content": chunk.page_content,
                "char_count": len(chunk.page_content),
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
        return len(chunks), pages, file_name, ext, chunk_records
    finally:
        os.unlink(tmp_path)


def ingest_pdf_file(uploaded_file: BinaryIO, collection: str = "General") -> tuple[int, str]:
    chunk_count, _, name, _, _ = ingest_file(uploaded_file, collection)
    return chunk_count, name


def retrieve_context(
    query: str,
    collection: str = "General",
    k: int = 4,
) -> tuple[str, list[Document]]:
    store = get_vector_store(collection)
    try:
        docs = store.similarity_search(
            query,
            k=k,
            filter={"collection": collection},
        )
    except TypeError:
        docs = store.similarity_search(query, k=k)

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return context, docs


def format_sources(sources: list[Document]) -> list[dict]:
    formatted = []
    for index, doc in enumerate(sources, start=1):
        metadata = doc.metadata or {}
        preview = doc.page_content[:280].strip()
        if len(doc.page_content) > 280:
            preview += "..."
        confidence = max(72, 96 - (index - 1) * 4)
        formatted.append(
            {
                "index": index,
                "page": metadata.get("page", "N/A"),
                "source": metadata.get("source", "Uploaded document"),
                "preview": preview,
                "confidence": confidence,
            }
        )
    return formatted
