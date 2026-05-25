"""Extract text from scanned documents and images for downstream classification and indexing.

This module provides a best-effort OCR implementation that prefers existing PDF text layers
when available and falls back to EasyOCR for scanned pages or image inputs. Language detection
on each page is a heuristic signal based on Unicode ranges and a small Devanagari wordlist,
not a full language identification system.
"""

import tempfile
from enum import Enum
from pathlib import Path
from typing import List

import fitz
from pydantic import BaseModel, Field

from easyocr import Reader

from legal_dms.config.settings import settings


class OcrSource(str, Enum):
    text_layer = "text_layer"
    ocr = "ocr"


class PageResult(BaseModel):
    page_number: int
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    language: str
    source: OcrSource


class OcrResult(BaseModel):
    document_path: Path
    pages: List[PageResult]

    @property
    def overall_confidence(self) -> float:
        if not self.pages:
            return 0.0
        total = sum(page.confidence for page in self.pages)
        return round(total / len(self.pages), 4)


_reader: Reader | None = None


def _get_reader() -> Reader:
    global _reader
    if _reader is None:
        languages = settings.ocr_languages
        _reader = Reader(languages, gpu=False)
    return _reader


MARATHI_WORDS = {"आहे", "माझे", "आम्ही", "होत", "करा"}
HINDI_WORDS = {"है", "क्या", "आप", "हूँ", "करें"}


def _detect_language(text: str) -> str:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return "en"


    devanagari_chars = [ch for ch in chars if "\u0900" <= ch <= "\u097F"]
    devanagari_ratio = len(devanagari_chars) / len(chars)

    if devanagari_ratio < 0.4:
        return "en"

    tokens = {token for token in text.split()}
    marathi_hits = len(tokens & MARATHI_WORDS)
    hindi_hits = len(tokens & HINDI_WORDS)
    if marathi_hits >= hindi_hits:
        return "mr"
    return "hi"


def _text_layer_coverage(page: fitz.Page) -> float:
    page_area = page.rect.width * page.rect.height
    text_blocks = [block for block in page.get_text("blocks") if block[6] == 0]
    text_area = sum((block[2] - block[0]) * (block[3] - block[1]) for block in text_blocks)
    if page_area <= 0:
        return 0.0
    return text_area / page_area


def _ocr_image(image_path: Path, page_number: int) -> PageResult:
    reader = _get_reader()
    results = reader.readtext(str(image_path), detail=1)
    lines = [result[1] for result in results]
    confidences = [result[2] / 100.0 for result in results] if results else [0.0]
    text = "\n".join(lines).strip()
    confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    language = _detect_language(text)
    return PageResult(
        page_number=page_number,
        text=text,
        confidence=confidence,
        language=language,
        source=OcrSource.ocr,
    )


def extract_text(file_path: Path) -> OcrResult:
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".pdf":
        document = fitz.open(file_path)
        pages: List[PageResult] = []
        for index, page in enumerate(document, start=1):
            if _text_layer_coverage(page) >= 0.3:
                page_text = page.get_text("text").strip()
                language = _detect_language(page_text)
                pages.append(
                    PageResult(
                        page_number=index,
                        text=page_text,
                        confidence=1.0,
                        language=language,
                        source=OcrSource.text_layer,
                    )
                )
            else:
                pix = page.get_pixmap(dpi=300)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                pix.save(temp_path)
                try:
                    pages.append(_ocr_image(temp_path, index))
                finally:
                    if temp_path.exists():
                        temp_path.unlink()
        return OcrResult(document_path=file_path, pages=pages)

    return OcrResult(document_path=file_path, pages=[_ocr_image(file_path, 1)])
