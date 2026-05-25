"""Build and manage a vector-enabled search index for processed documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from llama_index.core.text_splitter import SentenceSplitter
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from legal_dms.classifier.model import DocumentMetadata
from legal_dms.config.settings import settings
from legal_dms.ocr.engine import OcrResult


class RetrievedChunk(BaseModel):
    text: str
    score: float
    document_id: str
    document_type: str
    page_number: int
    primary_language: str


_embedding_model: SentenceTransformer | None = None
_collection: Any | None = None
_splitter: SentenceSplitter | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def _get_splitter() -> SentenceSplitter:
    global _splitter
    if _splitter is None:
        _splitter = SentenceSplitter(split_length=1000, chunk_overlap=150)
    return _splitter


def _get_collection() -> Any:
    global _collection
    if _collection is None:
        settings.vector_db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(settings.vector_db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(name="legal_dms")
    return _collection


def _embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
    return [list(vec) for vec in embeddings]


def _chunk_document(ocr_result: OcrResult) -> list[tuple[str, int]]:
    splitter = _get_splitter()
    chunks: list[tuple[str, int]] = []

    for page in ocr_result.pages:
        page_text = f"[PAGE {page.page_number}]\n{page.text.strip()}"
        if not page_text.strip():
            continue
        page_chunks = splitter.split_text(page_text)
        for chunk in page_chunks:
            chunks.append((chunk.strip(), page.page_number))

    return chunks


def index_document(document_id: str, ocr_result: OcrResult, metadata: DocumentMetadata) -> None:
    remove_document(document_id)
    chunks = _chunk_document(ocr_result)
    if not chunks:
        return

    documents = []
    metadatas = []
    ids = []

    for index, (text, page_number) in enumerate(chunks):
        ids.append(f"{document_id}_{index}")
        documents.append(text)
        metadatas.append(
            {
                "document_id": document_id,
                "document_type": metadata.document_type.value,
                "page_number": page_number,
                "primary_language": metadata.primary_language.value,
                "execution_date": metadata.execution_date.isoformat() if metadata.execution_date else "",
                "chunk_index": index,
            }
        )

    embeddings = _embed_texts(documents)
    collection = _get_collection()
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    collection.persist()


def remove_document(document_id: str) -> None:
    collection = _get_collection()
    try:
        collection.delete(where={"document_id": [document_id]})
        collection.persist()
    except Exception:
        pass


def search(query: str, document_id: str | None = None, top_k: int = 6) -> list[RetrievedChunk]:
    collection = _get_collection()
    query_embedding = _embed_texts([query])[0]
    kwargs: dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": top_k}
    if document_id:
        kwargs["where"] = {"document_id": document_id}

    results = collection.query(**kwargs)
    if not results or not results.get("ids"):
        return []

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    chunks: list[RetrievedChunk] = []

    for text, metadata, distance in zip(docs, metadatas, distances):
        score = 1.0 - distance if distance is not None else 0.0
        if score < 0:
            score = 0.0
        chunks.append(
            RetrievedChunk(
                text=text,
                score=round(score, 4),
                document_id=metadata["document_id"],
                document_type=metadata["document_type"],
                page_number=int(metadata["page_number"]),
                primary_language=metadata["primary_language"],
            )
        )

    return chunks
