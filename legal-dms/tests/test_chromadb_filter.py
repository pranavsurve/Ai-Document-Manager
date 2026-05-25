import sys
from pathlib import Path

# Ensure imports work when tests are run from workspace root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_dms.indexer import store


def test_search_uses_string_document_id(monkeypatch):
    captured = {}

    # stub embedding generator
    monkeypatch.setattr(store, "_embed_texts", lambda texts: [[0.1, 0.2, 0.3]])

    class FakeCollection:
        def query(self, **kwargs):
            captured.update(kwargs)
            # Return minimal empty results matching store.search expectations
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    monkeypatch.setattr(store, "_get_collection", lambda: FakeCollection())

    # Call search with a document_id string and ensure no exception
    chunks = store.search("some query", document_id="doc-123", top_k=2)

    assert captured, "Collection.query was not called"
    assert "where" in captured, "where filter not passed to collection.query"
    assert isinstance(captured["where"], dict)
    # Ensure the document_id value is a string, not a list (regression)
    assert captured["where"].get("document_id") == "doc-123"
    assert "query_embeddings" in captured
    assert chunks == [], "Expected empty results from fake collection"
