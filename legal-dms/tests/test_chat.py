from legal_dms.chat import ChatResponse, answer, answer_stream
from legal_dms.indexer import RetrievedChunk


class FakeLlmClient:
    def __init__(self, complete_text: str, stream_chunks: list[str] | None = None) -> None:
        self.complete_text = complete_text
        self.stream_chunks = stream_chunks or [complete_text]

    def complete(self, system: str, user: str, response_schema=None, temperature: float = 0.2):
        return self.complete_text

    def stream(self, system: str, user: str, temperature: float = 0.2):
        yield from self.stream_chunks


def _make_chunk(document_id: str, page_number: int, text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        document_id=document_id,
        document_type="lease_agreement",
        page_number=page_number,
        primary_language="en",
    )


def test_chat_answer_refuses_on_irrelevant_context(monkeypatch):
    chunks = [_make_chunk("doc123456789", 1, "This is unrelated content.", score=0.1)]
    monkeypatch.setattr("legal_dms.chat.agent.search", lambda question, document_id=None, top_k=6: chunks)
    monkeypatch.setattr("legal_dms.chat.agent.get_llm_client", lambda: FakeLlmClient(
        "I do not have enough information to answer that question based on the provided context."
    ))

    response = answer("What is the rent amount?", document_id="doc123456789")

    assert "not enough information" in response.answer_text.lower()
    assert response.sources == []
    assert response.confidence == "low"
    assert response.answer_text.endswith("This is legal information, not legal advice")


def test_chat_answer_returns_sources_for_non_refusal(monkeypatch):
    chunks = [_make_chunk("docabcdef01", 5, "The rent is $1,000 per month.", score=0.92)]
    monkeypatch.setattr("legal_dms.chat.agent.search", lambda question, document_id=None, top_k=6: chunks)
    monkeypatch.setattr("legal_dms.chat.agent.get_llm_client", lambda: FakeLlmClient(
        "The lease states the rent is $1,000 per month. This is legal information, not legal advice"
    ))

    response = answer("How much is the rent?", document_id="docabcdef01")

    assert isinstance(response, ChatResponse)
    assert response.sources == chunks
    assert response.confidence == "high"
    assert response.answer_text.endswith("This is legal information, not legal advice")


def test_chat_answer_stream_yields_tokens_then_sources(monkeypatch):
    chunks = [_make_chunk("docfeed123", 2, "The tenant must pay $500.", score=0.88)]
    monkeypatch.setattr("legal_dms.chat.agent.search", lambda question, document_id=None, top_k=6: chunks)
    monkeypatch.setattr(
        "legal_dms.chat.agent.get_llm_client",
        lambda: FakeLlmClient(
            "The tenant must pay $500. This is legal information, not legal advice",
            stream_chunks=["The ", "tenant ", "must ", "pay ", "$500."]
        ),
    )

    stream = list(answer_stream("What is the tenant obligated to pay?", document_id="docfeed123"))

    assert stream[:-1] == ["The", "tenant", "must", "pay", "$500."]
    assert isinstance(stream[-1], dict)
    assert stream[-1]["type"] == "sources"
    assert stream[-1]["sources"] == chunks
