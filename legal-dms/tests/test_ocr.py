from pathlib import Path

import pytest

from legal_dms.ocr.engine import OcrSource, extract_text


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _fixture_path(name: str) -> Path:
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(f"Skipping OCR test because fixture is missing: {path}")
    return path


def test_ocr_pdf_text_layer():
    fixture = _fixture_path("text_layer.pdf")
    result = extract_text(fixture)

    assert result.document_path == fixture
    assert result.pages
    assert all(page.source == OcrSource.text_layer for page in result.pages)
    assert result.overall_confidence == 1.0


def test_ocr_scanned_english():
    fixture = _fixture_path("scanned_en.png")
    result = extract_text(fixture)

    assert result.document_path == fixture
    assert result.pages
    assert result.pages[0].source == OcrSource.ocr
    assert result.pages[0].language == "en"
    assert result.pages[0].confidence >= 0.0


def test_ocr_scanned_marathi():
    fixture = _fixture_path("scanned_mr.png")
    result = extract_text(fixture)

    assert result.document_path == fixture
    assert result.pages
    assert result.pages[0].source == OcrSource.ocr
    assert result.pages[0].language in {"mr", "hi"}
    assert result.pages[0].confidence >= 0.0
