"""
modules/rag_system.py – Retrieval-Augmented Generation (RAG) System
====================================================================
Pipeline
--------
1. Ingest PDF / TXT documents from a folder or via upload.
2. Extract text with pypdf; track source metadata and page numbers.
3. Split into overlapping chunks (char-level).
4. Embed chunks with Sentence Transformers (all-MiniLM-L6-v2).
5. Persist in ChromaDB under VECTOR_DB_DIR.
6. rag_retrieve(question, top_k) → list[dict] with text + source + page.

Public API
----------
ingest_documents(folder: Path | str) -> int
ingest_pdf_bytes(pdf_bytes: bytes, filename: str) -> int
rag_retrieve(question: str, top_k: int = 4) -> list[dict]
get_indexed_documents() -> list[str]
clear_collection() -> None
"""

from __future__ import annotations

import hashlib
import logging
import textwrap
from pathlib import Path
from typing import Any

from config import (
    VECTOR_DB_DIR,
    CLIMATE_DOCS_DIR,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAG_TOP_K,
)

logger = logging.getLogger(__name__)

# ── Lazy singletons (loaded on first use) ─────────────────────────────────────
_embedding_model = None
_chroma_client   = None
_collection      = None
_rag_available   = None   # None = not yet tested


def _get_embedding_model():
    global _embedding_model, _rag_available
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model '%s' loaded.", EMBEDDING_MODEL)
        _rag_available = True
        return _embedding_model
    except Exception as exc:
        _rag_available = False
        raise RuntimeError(
            f"Could not load embedding model '{EMBEDDING_MODEL}'. "
            "Install sentence-transformers and ensure internet access on first run. "
            f"Original error: {exc}"
        ) from exc


def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' opened at %s.",
            CHROMA_COLLECTION, VECTOR_DB_DIR
        )
        return _collection
    except Exception as exc:
        raise RuntimeError(
            f"Could not open ChromaDB at '{VECTOR_DB_DIR}'. "
            f"Original error: {exc}"
        ) from exc


def is_rag_available() -> bool:
    """Return True if both sentence-transformers and chromadb can be imported."""
    global _rag_available
    if _rag_available is not None:
        return _rag_available
    try:
        import sentence_transformers  # noqa: F401
        import chromadb               # noqa: F401
        _rag_available = True
    except ImportError:
        _rag_available = False
    return _rag_available


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_pdf_pages(pdf_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Returns list of dicts: {text, page_number, source}
    """
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({
                    "text":        text,
                    "page_number": i,
                    "source":      filename,
                })
        return pages
    except Exception as exc:
        raise RuntimeError(
            f"Failed to extract text from '{filename}': {exc}"
        ) from exc


def _extract_txt_file(path: Path) -> list[dict]:
    """Extract text from a plain-text file (treated as one 'page')."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [{"text": text, "page_number": 1, "source": path.name}]


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int   = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping character-level chunks."""
    if not text:
        return []
    chunks = []
    start  = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _make_chunk_id(source: str, page: int, chunk_idx: int) -> str:
    raw = f"{source}::{page}::{chunk_idx}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ── Ingestion ─────────────────────────────────────────────────────────────────

def _ingest_pages(pages: list[dict]) -> int:
    """
    Embed and store page chunks in ChromaDB.
    Skips chunks that are already present (by ID) to avoid duplicates.

    Returns the number of NEW chunks stored.
    """
    model      = _get_embedding_model()
    collection = _get_collection()

    new_count = 0
    for page in pages:
        chunks = _chunk_text(page["text"])
        for idx, chunk in enumerate(chunks):
            chunk_id = _make_chunk_id(page["source"], page["page_number"], idx)
            # Skip duplicates
            existing = collection.get(ids=[chunk_id])
            if existing and existing.get("ids"):
                continue
            embedding = model.encode(chunk).tolist()
            collection.add(
                ids        =[chunk_id],
                embeddings =[embedding],
                documents  =[chunk],
                metadatas  =[{
                    "source":      page["source"],
                    "page_number": page["page_number"],
                    "chunk_index": idx,
                }],
            )
            new_count += 1

    logger.info("Ingested %d new chunks.", new_count)
    return new_count


def ingest_documents(folder: Path | str | None = None) -> int:
    """
    Ingest all PDF and TXT files from *folder* (defaults to CLIMATE_DOCS_DIR).

    Returns the total number of new chunks stored.
    """
    folder = Path(folder or CLIMATE_DOCS_DIR)
    if not folder.exists():
        logger.warning("Document folder '%s' does not exist.", folder)
        return 0

    total = 0
    for filepath in folder.iterdir():
        suffix = filepath.suffix.lower()
        try:
            if suffix == ".pdf":
                pages = _extract_pdf_pages(filepath.read_bytes(), filepath.name)
            elif suffix in (".txt", ".md"):
                pages = _extract_txt_file(filepath)
            else:
                continue
            total += _ingest_pages(pages)
            logger.info("Ingested '%s'.", filepath.name)
        except Exception as exc:
            logger.error("Failed to ingest '%s': %s", filepath.name, exc)

    return total


def ingest_pdf_bytes(pdf_bytes: bytes, filename: str) -> int:
    """
    Ingest a PDF provided as raw bytes (e.g. from Streamlit file uploader).
    Sanitizes filename against path traversal and special characters.

    Returns the number of new chunks stored.
    """
    import re
    # Strip path separators to prevent directory traversal attacks
    safe_name = Path(filename).name
    # Keep only safe alphanumeric, underscore, dot, hyphen characters
    safe_name = re.sub(r"[^\w\.-]", "_", safe_name).strip() or "uploaded_document.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    pages = _extract_pdf_pages(pdf_bytes, safe_name)
    return _ingest_pages(pages)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def rag_retrieve(question: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """
    Retrieve the *top_k* most relevant passages for *question*.

    Returns
    -------
    list[dict], each with keys:
        text, source, page_number, relevance_score, chunk_index
    Returns [] if the collection is empty or RAG is unavailable.
    """
    if not question or not str(question).strip():
        return []

    # Bound top_k to prevent vector DB memory exhaustion
    try:
        top_k = int(top_k)
        if top_k < 1:
            top_k = RAG_TOP_K
    except (ValueError, TypeError):
        top_k = RAG_TOP_K
    top_k = min(top_k, 20)

    try:
        model      = _get_embedding_model()
        collection = _get_collection()
    except RuntimeError as exc:
        logger.error("RAG unavailable: %s", exc)
        return []

    count = collection.count()
    if count == 0:
        logger.info("rag_retrieve: collection is empty.")
        return []

    actual_k   = min(top_k, count)
    query_emb  = model.encode(question).tolist()

    try:
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return []

    passages = []
    docs      = results.get("documents",  [[]])[0]
    metas     = results.get("metadatas",  [[]])[0]
    distances = results.get("distances",  [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        # Convert cosine distance (0=identical, 2=opposite) → score (1=best)
        score = round(1 - (dist / 2), 4)
        passages.append({
            "text":            doc,
            "source":          meta.get("source", "unknown"),
            "page_number":     meta.get("page_number", "?"),
            "chunk_index":     meta.get("chunk_index", 0),
            "relevance_score": score,
        })

    logger.info(
        "rag_retrieve: returned %d passages for query='%s'.",
        len(passages), question[:60]
    )
    return passages


# ── Utility ───────────────────────────────────────────────────────────────────

def get_indexed_documents() -> list[str]:
    """Return a deduplicated list of source document names in the collection."""
    try:
        collection = _get_collection()
        all_meta   = collection.get(include=["metadatas"])
        sources    = {m.get("source", "") for m in (all_meta.get("metadatas") or [])}
        return sorted(s for s in sources if s)
    except Exception as exc:
        logger.error("get_indexed_documents failed: %s", exc)
        return []


def clear_collection() -> None:
    """Delete and recreate the ChromaDB collection (for testing / reset)."""
    global _collection
    try:
        client = _get_collection()   # ensures client exists
        _chroma_client.delete_collection(CHROMA_COLLECTION)
        _collection = _chroma_client.create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection cleared.")
    except Exception as exc:
        logger.error("clear_collection failed: %s", exc)


def get_collection_count() -> int:
    """Return the number of chunks in the collection."""
    try:
        return _get_collection().count()
    except Exception:
        return 0
