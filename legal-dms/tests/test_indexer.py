from pathlib import Path

import pytest

from legal_dms.classifier.model import DocumentMetadata, DocumentType, Party, PrimaryLanguage
from legal_dms.indexer import index_document, remove_document, search, RetrievedChunk
from legal_dms.ocr.engine import OcrResult, PageResult
from legal_dms.config.settings import settings
from legal_dms.indexer import store as indexer_store


class FakeEmbedder:
    def encode(self, texts, convert_to_numpy=False, show_progress_bar=False):
        def encode_text(text: str):
            lower = text.lower()
            return [
                1.0 if "lease" in lower else 0.0,
                1.0 if "sale" in lower else 0.0,
                1.0 if "rera" in lower else 0.0,
            ]

        return [encode_text(text) for text in texts]


class FakeSplitter:
    def split_text(self, text: str):
        return [text]


class FakeCollection:
    def __init__(self):
        self.records = []

    def add(self, ids, documents, metadatas, embeddings, **kwargs):
        for _id, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            self.records.append({
                "id": _id,
                "document": doc,
                "metadata": meta,
                "embedding": emb,
            })

    def delete(self, where=None, **kwargs):
        if where is None:
            self.records = []
            return
        document_ids = where.get("document_id", [])
        self.records = [record for record in self.records if record["metadata"]["document_id"] not in document_ids]

    def query(self, query_embeddings, n_results=1, where=None, **kwargs):
        query_embedding = query_embeddings[0]
        candidates = []
        for record in self.records:
            if where is not None:
                document_ids = where.get("document_id", [])
                if record["metadata"]["document_id"] not in document_ids:
                    continue
            embedding = record["embedding"]
            distance = sum((a - b) ** 2 for a, b in zip(embedding, query_embedding)) ** 0.5
            candidates.append((distance, record))
        candidates.sort(key=lambda item: item[0])
        selected = candidates[:n_results]
        return {
            "ids": [[record["id"] for _, record in selected]],
            "documents": [[record["document"] for _, record in selected]],
            "metadatas": [[record["metadata"] for _, record in selected]],
            "distances": [[distance for distance, _ in selected]],
        }

    def persist(self):
        pass


def _make_ocr_result(doc_id: str, text: str) -> OcrResult:
    return OcrResult(
        document_path=Path(f"/tmp/{doc_id}.pdf"),
        pages=[PageResult(page_number=1, text=text, confidence=1.0, language="en", source="ocr")],
    )


def _make_metadata(document_type: DocumentType, language: PrimaryLanguage) -> DocumentMetadata:
    return DocumentMetadata(
        document_type=document_type,
        primary_language=language,
        parties=[Party(name="Party A", role="Party")],
        execution_date=None,
        jurisdiction="Mumbai",
        summary="Document summary.",
        key_clauses=["termination"],
        confidence=0.9,
    )


def test_index_search_remove_document(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "vector_db_path", tmp_path / "vector", raising=False)
    fake_collection = FakeCollection()
    monkeypatch.setattr(indexer_store, "_collection", None)
    monkeypatch.setattr(indexer_store, "_embedding_model", None)
    monkeypatch.setattr(indexer_store, "_splitter", None)
    monkeypatch.setattr(indexer_store, "_get_embedding_model", lambda: FakeEmbedder())
    monkeypatch.setattr(indexer_store, "_get_splitter", lambda: FakeSplitter())
    monkeypatch.setattr(indexer_store, "_get_collection", lambda: fake_collection)

    docs = [
        ("doc1", "This is a lease agreement document.", DocumentType.lease_agreement),
        ("doc2", "This is a sale deed document.", DocumentType.sale_deed),
        ("doc3", "This is a RERA filing document.", DocumentType.rera_filing),
    ]

    for doc_id, text, doc_type in docs:
        index_document(doc_id, _make_ocr_result(doc_id, text), _make_metadata(doc_type, PrimaryLanguage.en))

    lease_results = search("lease agreement", top_k=2)
    assert lease_results
    assert lease_results[0].document_id == "doc1"
    assert isinstance(lease_results[0], RetrievedChunk)

    filtered = search("lease agreement", document_id="doc2", top_k=2)
    assert filtered
    assert all(result.document_id == "doc2" for result in filtered)

    remove_document("doc1")
    post_remove = search("lease agreement", top_k=3)
    assert all(result.document_id != "doc1" for result in post_remove)
