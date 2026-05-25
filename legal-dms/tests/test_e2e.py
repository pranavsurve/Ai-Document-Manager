import json
import re
import sys
from pathlib import Path

# Ensure the package root is discoverable when running tests directly from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datetime import date

import pytest

# Lazy imports to avoid initialization overhead
LEASE_TEXT = """Lease Agreement
This lease agreement sets forth the termination conditions and renewal terms for the tenant."""
SALE_DEED_TEXT = """विक्री दस्तऐवज
हा विक्री करार मालमत्ता हस्तांतरण बद्दल आहे."""
RERA_TEXT = """RERA Compliance Filing
This is a RERA compliance document for an apartment project."""


class FakeLlmClient:
    def complete(self, system: str, user: str, response_schema=None, temperature: float = 0.2):
        from legal_dms.classifier.model import DocumentMetadata, DocumentType, Party, PrimaryLanguage
        from datetime import date

        text = user.lower()
        if response_schema == DocumentMetadata:
            if "lease" in text or "termination" in text:
                return DocumentMetadata(
                    document_type=DocumentType.lease_agreement,
                    primary_language=PrimaryLanguage.en,
                    parties=[Party(name="Lessor", role="Landlord"), Party(name="Lessee", role="Tenant")],
                    execution_date=date(2025, 1, 15),
                    jurisdiction="Mumbai",
                    summary="A lease agreement describing terms, termination conditions, and renewal rights.",
                    key_clauses=["term", "termination", "renewal"],
                    confidence=0.92,
                )
            if "sale" in text or "विक्री" in text or "deed" in text:
                return DocumentMetadata(
                    document_type=DocumentType.sale_deed,
                    primary_language=PrimaryLanguage.mr,
                    parties=[Party(name="Seller", role="Vendor"), Party(name="Buyer", role="Purchaser")],
                    execution_date=date(2025, 2, 20),
                    jurisdiction="Pune",
                    summary="A sale deed capturing the transfer of property in Marathi.",
                    key_clauses=["consideration", "transfer", "title"],
                    confidence=0.87,
                )
            if "rera" in text or "compliance" in text:
                return DocumentMetadata(
                    document_type=DocumentType.rera_filing,
                    primary_language=PrimaryLanguage.en,
                    parties=[Party(name="Developer", role="Developer")],
                    execution_date=date(2025, 3, 10),
                    jurisdiction="Mumbai",
                    summary="A RERA filing for project compliance and regulatory disclosures.",
                    key_clauses=["registration", "compliance", "project details"],
                    confidence=0.94,
                )
            raise ValueError("Unknown classification input")

        if "termination" in text:
            return "The lease termination conditions are spelled out in the lease agreement. This is legal information, not legal advice"
        return "I do not have enough information to answer that question based on the provided documents. This is legal information, not legal advice"

    def stream(self, system: str, user: str, temperature: float = 0.2):
        text = user.lower()
        if "termination" in text:
            yield "The lease termination conditions are spelled out in the lease agreement. This is legal information, not legal advice"
        else:
            yield "I do not have enough information to answer that question based on the provided documents. This is legal information, not legal advice"


class FakeSentenceTransformer:
    def encode(self, texts, convert_to_numpy=False, show_progress_bar=False):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeSplitter:
    def split_text(self, text: str):
        return [text[:1000]]


def _create_text_pdf(path: Path, text: str) -> None:
    import fitz
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    for line in text.splitlines():
        c.drawString(72, y, line)
        y -= 20
    c.save()


def _create_image_pdf(path: Path, text: str) -> None:
    import fitz

    temp_pdf = path.parent / f"{path.stem}_source.pdf"
    _create_text_pdf(temp_pdf, text)
    source_doc = fitz.open(str(temp_pdf))
    page = source_doc[0]
    pix = page.get_pixmap(dpi=150)
    image_doc = fitz.open()
    image_doc.new_page(width=page.rect.width, height=page.rect.height)
    image_doc[0].insert_image(page.rect, pixmap=pix)
    image_doc.save(str(path))
    source_doc.close()
    image_doc.close()
    temp_pdf.unlink()


@pytest.fixture
def temp_paths(tmp_path: Path, monkeypatch):
    from legal_dms.config.settings import settings

    inbox = tmp_path / "inbox"
    library = tmp_path / "library"
    work = tmp_path / "work"
    vector_db = tmp_path / "vector_db"
    inbox.mkdir()
    library.mkdir()
    work.mkdir()
    vector_db.mkdir()

    monkeypatch.setattr(settings, "inbox_path", inbox, raising=False)
    monkeypatch.setattr(settings, "library_path", library, raising=False)
    monkeypatch.setattr(settings, "work_path", work, raising=False)
    monkeypatch.setattr(settings, "vector_db_path", vector_db, raising=False)
    monkeypatch.setattr(settings, "auto_confirm", True, raising=False)

    return {
        "inbox": inbox,
        "library": library,
        "work": work,
        "vector_db": vector_db,
    }


@pytest.fixture(autouse=False)
def fake_ml_client_and_index(monkeypatch):
    import legal_dms.classifier.model as classifier_model
    import legal_dms.chat.agent as chat_agent
    import legal_dms.common.llm as common_llm
    from legal_dms.indexer import store

    fake = FakeLlmClient()
    monkeypatch.setattr(common_llm, "get_llm_client", lambda: fake)
    common_llm._client_cache.clear()
    monkeypatch.setattr(classifier_model, "get_llm_client", lambda: fake)
    monkeypatch.setattr(chat_agent, "get_llm_client", lambda: fake)
    monkeypatch.setattr(store, "_get_embedding_model", lambda: FakeSentenceTransformer())
    monkeypatch.setattr(store, "_get_splitter", lambda: FakeSplitter())
    monkeypatch.setattr(store, "_collection", None, raising=False)

    original_get_collection = store._get_collection

    def adapted_get_collection():
        collection = original_get_collection()
        if not hasattr(collection, "persist"):
            setattr(collection, "persist", lambda: None)
        return collection

    monkeypatch.setattr(store, "_get_collection", adapted_get_collection)
    yield


def test_end_to_end_pipeline_and_chat(temp_paths, fake_ml_client_and_index, monkeypatch):
    from legal_dms.classifier.model import DocumentType
    from legal_dms.pipeline import Pipeline, PipelineStage
    from legal_dms.ocr.engine import PageResult, OcrResult
    from legal_dms.chat import answer
    from legal_dms.indexer import store

    lease_pdf = temp_paths["inbox"] / "lease_agreement.pdf"
    sale_deed_pdf = temp_paths["inbox"] / "sale_deed.pdf"
    rera_pdf = temp_paths["inbox"] / "rera_filing.pdf"
    _create_text_pdf(lease_pdf, LEASE_TEXT)
    _create_image_pdf(sale_deed_pdf, SALE_DEED_TEXT)
    _create_text_pdf(rera_pdf, RERA_TEXT)

    from legal_dms import pipeline

    def fake_extract_text(path: Path) -> OcrResult:
        if path.name == lease_pdf.name:
            return OcrResult(
                document_path=path,
                pages=[PageResult(page_number=1, text=LEASE_TEXT, confidence=0.96, language="en", source="text_layer")],
            )
        if path.name == sale_deed_pdf.name:
            return OcrResult(
                document_path=path,
                pages=[PageResult(page_number=1, text=SALE_DEED_TEXT, confidence=0.90, language="mr", source="ocr")],
            )
        if path.name == rera_pdf.name:
            return OcrResult(
                document_path=path,
                pages=[PageResult(page_number=1, text=RERA_TEXT, confidence=0.95, language="en", source="text_layer")],
            )
        raise AssertionError("Unexpected document in OCR path")

    monkeypatch.setattr(pipeline, "extract_text", fake_extract_text)

    pipeline_instance = Pipeline()
    lease_job = pipeline_instance.process(lease_pdf)
    sale_job = pipeline_instance.process(sale_deed_pdf)
    rera_job = pipeline_instance.process(rera_pdf)

    for job, expected_type in [
        (lease_job, DocumentType.lease_agreement),
        (sale_job, DocumentType.sale_deed),
        (rera_job, DocumentType.rera_filing),
    ]:
        assert job.current_stage == PipelineStage.done
        dest = Path(job.artifacts["destination_path"])
        assert dest.exists()
        assert dest.parent.name == "2025"
        assert dest.parent.parent.name == expected_type.value
        assert dest.parent.parent.parent == temp_paths["library"]
        assert dest.suffix == ".pdf"

        sidecar = dest.with_suffix(".json")
        assert sidecar.exists()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert isinstance(data["document_id"], str) and data["document_id"]
        assert data["metadata"]["document_type"] == expected_type.value
        assert data["metadata"]["confidence"] >= 0.0

    # verify every document is indexed with retrievable chunks
    for job in [lease_job, sale_job, rera_job]:
        chunks = store.search("document", document_id=job.document_id, top_k=2)
        assert chunks
        assert all(chunk.document_id == job.document_id for chunk in chunks)

    # query chat about termination conditions and confirm the lease is cited
    response = answer("What are the termination conditions?", document_id=None)
    assert "termination" in response.answer_text.lower()
    assert "lease agreement" in response.answer_text.lower()
    assert any(chunk.document_id == lease_job.document_id for chunk in response.sources)


def test_ui_css_palette_and_import():
    """Smoke test: verify CSS string is non-empty and colors are from approved palette."""
    import sys
    import importlib.util
    from legal_dms.common import ui_theme

    # Load styles.py module directly without running app.py
    styles_path = Path(__file__).parent.parent / "src/legal_dms/ui/styles.py"
    spec = importlib.util.spec_from_file_location("ui_styles", styles_path)
    ui_styles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui_styles)

    css = ui_styles.get_base_css()
    assert css.strip(), "CSS string must not be empty"
    assert len(css) > 100, "CSS string should be substantial"

    # Verify only approved colors appear in UI files
    approved = {ui_theme.INK.lower(), ui_theme.MUTED.lower(), ui_theme.SURFACE.lower(), ui_theme.ACCENT.lower()}
    ui_files = Path(__file__).parent.parent.joinpath("src/legal_dms/ui").glob("*.py")
    for path in ui_files:
        if path.name.startswith("__"):
            continue
        text = path.read_text(encoding="utf-8")
        # find all hex codes (6-digit)
        hex_codes = set(re.findall(r"#[0-9A-Fa-f]{6}", text))
        for hex_code in hex_codes:
            assert hex_code.lower() in approved, f"Disallowed palette color {hex_code} found in {path.name}"
