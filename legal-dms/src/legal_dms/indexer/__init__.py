"""Indexer package for Legal DMS."""

from legal_dms.indexer.store import RetrievedChunk, index_document, remove_document, search

__all__ = ["RetrievedChunk", "index_document", "remove_document", "search"]
