from unittest.mock import Mock, patch

from legal_dms.classifier.model import (
    DocumentMetadata,
    DocumentType,
    Party,
    PrimaryLanguage,
    classify,
)
from legal_dms.classifier.prompts import SYSTEM_PROMPT


def _sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_type=DocumentType.lease_agreement,
        primary_language=PrimaryLanguage.en,
        parties=[Party(name="ABC Builders Pvt. Ltd.", role="Lessor")],
        execution_date=None,
        jurisdiction="Mumbai",
        summary="A lease agreement between two parties.",
        key_clauses=["termination"],
        confidence=0.85,
    )


def test_classify_success_calls_llm_and_returns_metadata():
    mock_client = Mock()
    mock_client.complete.return_value = _sample_metadata()

    with patch("legal_dms.classifier.model.get_llm_client", return_value=mock_client):
        result = classify("This is a legal document.", "en")

    assert result.document_type == DocumentType.lease_agreement
    mock_client.complete.assert_called_once()
    call_args = mock_client.complete.call_args[0]
    assert call_args[0] == SYSTEM_PROMPT
    assert call_args[1].startswith("Detected OCR language: en")


def test_classify_low_confidence_overrides_document_type_and_logs_original_guess():
    low_confidence_metadata = DocumentMetadata(
        document_type=DocumentType.sale_deed,
        primary_language=PrimaryLanguage.en,
        parties=[Party(name="Rajesh Kumar", role="Seller")],
        execution_date=None,
        jurisdiction="Pune",
        summary="A sale deed for property transfer.",
        key_clauses=["indemnity"],
        confidence=0.55,
    )
    mock_client = Mock()
    mock_client.complete.return_value = low_confidence_metadata

    with patch("legal_dms.classifier.model.get_llm_client", return_value=mock_client), patch(
        "legal_dms.classifier.model.logger.info"
    ) as mock_info:
        result = classify("Some legal text.", "en", document_path="/tmp/doc.pdf")

    assert result.document_type == DocumentType.other
    mock_info.assert_called_once()
    assert mock_info.call_args[0][1] == DocumentType.sale_deed
    assert mock_info.call_args[0][2] == 0.55
    assert mock_info.call_args[0][3] == "/tmp/doc.pdf"


def test_classify_truncates_oversize_input_before_passing_to_llm():
    long_text = "A" * 15000
    mock_client = Mock()
    mock_client.complete.return_value = _sample_metadata()

    with patch("legal_dms.classifier.model.get_llm_client", return_value=mock_client):
        result = classify(long_text)

    assert result.document_type == DocumentType.lease_agreement
    assert mock_client.complete.called
    _, user_prompt, *_ = mock_client.complete.call_args[0]
    assert "[... MIDDLE CONTENT REMOVED DUE TO LENGTH ...]" in user_prompt
    assert user_prompt.startswith("A" * 10)
    assert user_prompt.endswith("A" * 10)
