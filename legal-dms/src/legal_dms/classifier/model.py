"""Use LLM-driven classification to assign labels and metadata to incoming documents."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, constr, field_validator

from legal_dms.common.llm import get_llm_client
from legal_dms.common.logging import get_logger
from legal_dms.classifier.prompts import SYSTEM_PROMPT

logger = get_logger(__name__)

MAX_PROMPT_CHARS = 12000
TRUNCATION_MARKER = "\n\n[... MIDDLE CONTENT REMOVED DUE TO LENGTH ...]\n\n"


class DocumentType(str, Enum):
    sale_deed = "sale_deed"
    lease_agreement = "lease_agreement"
    rera_filing = "rera_filing"
    fssai_certificate = "fssai_certificate"
    seven_twelve_extract = "seven_twelve_extract"
    power_of_attorney = "power_of_attorney"
    noc = "noc"
    other = "other"


class PrimaryLanguage(str, Enum):
    en = "en"
    hi = "hi"
    mr = "mr"
    mixed = "mixed"


class Party(BaseModel):
    name: constr(strip_whitespace=True, min_length=1)
    role: constr(strip_whitespace=True, min_length=1)


class DocumentMetadata(BaseModel):
    document_type: DocumentType
    primary_language: PrimaryLanguage
    parties: list[Party]
    execution_date: date | None = None
    jurisdiction: str | None = None
    summary: constr(strip_whitespace=True, min_length=1)
    key_clauses: list[constr(strip_whitespace=True, min_length=1)] | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("summary")
    def summary_word_limit(cls, value: str) -> str:
        words = value.split()
        if len(words) > 200:
            raise ValueError("summary must be 200 words or fewer")
        return value


def _truncate_text(ocr_text: str) -> str:
    if len(ocr_text) <= MAX_PROMPT_CHARS:
        return ocr_text

    available = MAX_PROMPT_CHARS - len(TRUNCATION_MARKER)
    first_len = int(available * 0.6)
    last_len = available - first_len
    return f"{ocr_text[:first_len]}{TRUNCATION_MARKER}{ocr_text[-last_len:]}"


def classify(
    ocr_text: str,
    ocr_language: str | None = None,
    document_path: str | None = None,
) -> DocumentMetadata:
    truncated_text = _truncate_text(ocr_text)
    user_prompt = truncated_text
    if ocr_language:
        user_prompt = f"Detected OCR language: {ocr_language}\n\n{truncated_text}"

    client = get_llm_client()
    metadata = client.complete(
        SYSTEM_PROMPT,
        user_prompt,
        response_schema=DocumentMetadata,
        temperature=0.2,
    )

    if metadata.confidence < 0.6:
        original_guess = metadata.document_type
        metadata.document_type = DocumentType.other
        logger.info(
            "Low confidence classification override: original_type=%s confidence=%s document_path=%s",
            original_guess,
            metadata.confidence,
            document_path or "unknown",
        )

    return metadata
