"""Expose conversational document access and retrieval through an LLM-driven chat layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from pydantic import BaseModel, Field

from legal_dms.chat.prompts import SYSTEM_PROMPT
from legal_dms.common.llm import get_llm_client
from legal_dms.indexer import RetrievedChunk, search


class ChatResponse(BaseModel):
    """Response object returned from the chat assistant."""

    answer_text: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    confidence: str
    generated_at: datetime


def _build_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for chunk in chunks:
        document_id_short = chunk.document_id[:8]
        lines.append(f"[Doc {document_id_short} p.{chunk.page_number}] {chunk.text}")
    return "\n".join(lines)


def _build_user_prompt(question: str, context: str) -> str:
    return (
        "Context:\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer only with information from the context above."
    )


def _is_refusal(answer_text: str) -> bool:
    normalized = answer_text.strip().lower()
    refusal_phrases = [
        "do not have enough information",
        "do not have enough",
        "cannot answer",
        "cannot find",
        "not enough information",
        "not contained",
        "do not know",
        "no information",
        "unable to answer",
    ]
    return any(phrase in normalized for phrase in refusal_phrases)


def _ensure_disclaimer(answer_text: str) -> str:
    normalized = answer_text.strip()
    if normalized.endswith("This is legal information, not legal advice"):
        return normalized
    if normalized.endswith("."):
        return f"{normalized} This is legal information, not legal advice"
    return f"{normalized}. This is legal information, not legal advice"


def _confidence_label(answer_text: str, top_score: float) -> str:
    if _is_refusal(answer_text):
        return "low"
    if top_score >= 0.85:
        return "high"
    if top_score >= 0.65:
        return "medium"
    return "low"


def answer(question: str, document_id: str | None = None, top_k: int = 6) -> ChatResponse:
    chunks = search(question, document_id=document_id, top_k=top_k)
    if not chunks:
        answer_text = (
            "I do not have enough information to answer that question based on the provided documents. "
            "This is legal information, not legal advice"
        )
        return ChatResponse(
            answer_text=answer_text,
            sources=[],
            confidence="low",
            generated_at=datetime.now(timezone.utc),
        )

    context = _build_context(chunks)
    prompt = _build_user_prompt(question, context)
    llm = get_llm_client()
    answer_text = llm.complete(SYSTEM_PROMPT, prompt).strip()
    answer_text = _ensure_disclaimer(answer_text)
    refusal = _is_refusal(answer_text)
    sources = [] if refusal else chunks
    confidence = _confidence_label(answer_text, chunks[0].score if chunks else 0.0)

    return ChatResponse(
        answer_text=answer_text,
        sources=sources,
        confidence=confidence,
        generated_at=datetime.now(timezone.utc),
    )


def _stream_tokens(text: str) -> Iterator[str]:
    for token in text.split():
        yield token


def answer_stream(question: str, document_id: str | None = None, top_k: int = 6) -> Iterator[object]:
    chunks = search(question, document_id=document_id, top_k=top_k)
    if not chunks:
        refusal_text = (
            "I do not have enough information to answer that question based on the provided documents. "
            "This is legal information, not legal advice"
        )
        for token in _stream_tokens(refusal_text):
            yield token
        yield {"type": "sources", "sources": []}
        return

    context = _build_context(chunks)
    prompt = _build_user_prompt(question, context)
    llm = get_llm_client()
    answer_fragments = []

    for fragment in llm.stream(SYSTEM_PROMPT, prompt):
        if not fragment:
            continue
        answer_fragments.append(fragment)
        for token in _stream_tokens(fragment):
            yield token

    answer_text = _ensure_disclaimer("".join(answer_fragments).strip())
    if not answer_text.endswith("This is legal information, not legal advice"):
        for token in _stream_tokens("This is legal information, not legal advice"):
            yield token

    refusal = _is_refusal(answer_text)
    sources = [] if refusal else chunks
    yield {"type": "sources", "sources": sources}
