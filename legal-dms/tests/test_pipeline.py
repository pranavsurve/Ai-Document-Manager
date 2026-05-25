"""Test suite for the document processing pipeline."""

from pathlib import Path
from datetime import date

import pytest

from legal_dms.config.settings import settings
from legal_dms.pipeline import Pipeline, PipelineJob, PipelineStage
from legal_dms.classifier.model import DocumentMetadata, DocumentType, Party, PrimaryLanguage
from legal_dms.ocr.engine import OcrResult, PageResult
from legal_dms.organizer.manager import MovePlan


class FakeLlmClient:
    """Fake LLM client for testing."""

    def complete(self, system_prompt, user_prompt, response_schema=None, temperature=None):
        """Return a fake classification result."""
        if response_schema == DocumentMetadata:
            return DocumentMetadata(
                document_type=DocumentType.lease_agreement,
                primary_language=PrimaryLanguage.en,
                parties=[
                    Party(name="Lessor", role="Landlord"),
                    Party(name="Lessee", role="Tenant"),
                ],
                execution_date=date(2025, 1, 15),
                jurisdiction="Mumbai",
                summary="This is a lease agreement between lessor and lessee for commercial property.",
                key_clauses=["rent", "term", "termination", "renewal"],
                confidence=0.95,
            )
        return None


class FakeSentenceTransformer:
    """Fake embedding model."""

    def encode(self, texts, convert_to_numpy=False, show_progress_bar=False):
        """Return simple fake embeddings."""
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeSplitter:
    """Fake text splitter."""

    def split_text(self, text: str):
        return [text[:1000]]  # Simple fake split


class FakeCollection:
    """Fake vector collection for indexing."""

    def __init__(self):
        self.records = []

    def add(self, ids, documents, metadatas, embeddings, **kwargs):
        for _id, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            self.records.append({"id": _id, "document": doc, "metadata": meta, "embedding": emb})

    def delete(self, where=None, **kwargs):
        if where is None:
            self.records = []
        else:
            doc_ids = where.get("document_id", [])
            self.records = [r for r in self.records if r["metadata"]["document_id"] not in doc_ids]

    def query(self, query_embeddings, n_results=1, where=None, **kwargs):
        candidates = []
        for record in self.records:
            if where:
                doc_ids = where.get("document_id", [])
                if record["metadata"]["document_id"] not in doc_ids:
                    continue
            emb = record["embedding"]
            distance = sum((a - b) ** 2 for a, b in zip(emb, query_embeddings[0])) ** 0.5
            candidates.append((distance, record))
        candidates.sort()
        return {
            "ids": [[r["id"] for _, r in candidates[:n_results]]],
            "documents": [[r["document"] for _, r in candidates[:n_results]]],
            "metadatas": [[r["metadata"] for _, r in candidates[:n_results]]],
            "distances": [[d for d, _ in candidates[:n_results]]],
        }

    def persist(self):
        pass


def create_test_document(inbox_path: Path, document_id: str = "test_doc_001") -> Path:
    """Create a simple test PDF-like file."""
    pdf_path = inbox_path / f"{document_id}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n%fake pdf content")
    return pdf_path


@pytest.fixture
def temp_dirs(tmp_path: Path, monkeypatch):
    """Create temporary directories for testing."""
    inbox_path = tmp_path / "inbox"
    work_path = tmp_path / "work"
    library_path = tmp_path / "library"
    vector_db_path = tmp_path / "vector_db"

    inbox_path.mkdir(parents=True, exist_ok=True)
    work_path.mkdir(parents=True, exist_ok=True)
    library_path.mkdir(parents=True, exist_ok=True)
    vector_db_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "inbox_path", inbox_path, raising=False)
    monkeypatch.setattr(settings, "work_path", work_path, raising=False)
    monkeypatch.setattr(settings, "library_path", library_path, raising=False)
    monkeypatch.setattr(settings, "vector_db_path", vector_db_path, raising=False)

    return {
        "inbox": inbox_path,
        "work": work_path,
        "library": library_path,
        "vector_db": vector_db_path,
    }


@pytest.fixture
def mock_llm_and_embeddings(monkeypatch):
    """Mock LLM client and embedding model."""
    from legal_dms.common import llm
    from legal_dms.indexer import store

    monkeypatch.setattr(llm, "get_llm_client", lambda: FakeLlmClient())
    monkeypatch.setattr(store, "_get_embedding_model", lambda: FakeSentenceTransformer())
    monkeypatch.setattr(store, "_get_splitter", lambda: FakeSplitter())
    monkeypatch.setattr(store, "_get_collection", lambda: FakeCollection())
    monkeypatch.setattr(store, "_collection", None)


def test_full_pipeline_processing(temp_dirs, mock_llm_and_embeddings, monkeypatch):
    """Test processing a document through all pipeline stages."""
    # Create a test document
    pdf_path = create_test_document(temp_dirs["inbox"])
    
    # Mock OCR to return simple result
    def fake_extract_text(path):
        return OcrResult(
            document_path=path,
            pages=[
                PageResult(
                    page_number=1,
                    text="This is a lease agreement between landlord and tenant for commercial space.",
                    confidence=0.9,
                    language="en",
                    source="ocr",
                )
            ],
        )

    # Patch extract_text in the pipeline module where it was imported
    from legal_dms import pipeline
    monkeypatch.setattr(pipeline, "extract_text", fake_extract_text)

    # Mock organizer plan_move
    def fake_plan_move(source, metadata):
        return MovePlan(
            source_path=source,
            destination_path=temp_dirs["library"] / "lease_agreement" / "2025" / "test_doc_001.pdf",
            document_id="test_doc_001",
            metadata=metadata,
        )

    # Patch plan_move in the pipeline module where it was imported
    monkeypatch.setattr(pipeline, "plan_move", fake_plan_move)

    # Mock organizer execute_move
    def fake_execute_move(plan, confirmed):
        plan.destination_path.parent.mkdir(parents=True, exist_ok=True)
        plan.destination_path.write_bytes(plan.source_path.read_bytes())
        return plan.destination_path, plan.sidecar_path

    # Patch execute_move in the pipeline module where it was imported
    monkeypatch.setattr(pipeline, "execute_move", fake_execute_move)

    # Mock classify
    def fake_classify(ocr_text, ocr_language=None):
        return DocumentMetadata(
            document_type=DocumentType.lease_agreement,
            primary_language=PrimaryLanguage.en,
            parties=[
                Party(name="Lessor", role="Landlord"),
                Party(name="Lessee", role="Tenant"),
            ],
            execution_date=date(2025, 1, 15),
            jurisdiction="Mumbai",
            summary="This is a lease agreement between lessor and lessee for commercial property.",
            key_clauses=["rent", "term", "termination", "renewal"],
            confidence=0.95,
        )

    monkeypatch.setattr(pipeline, "classify", fake_classify)

    # Mock index_document
    def fake_index_document(document_id, ocr_result, metadata):
        pass

    monkeypatch.setattr(pipeline, "index_document", fake_index_document)

    # Create and run pipeline with auto-confirm
    monkeypatch.setattr(settings, "auto_confirm", True, raising=False)
    pipeline_instance = pipeline.Pipeline()
    job = pipeline_instance.process(pdf_path)

    # Verify final state
    assert job.current_stage == pipeline.PipelineStage.done
    assert job.error is None
    assert "ocr_result" in job.artifacts
    assert "metadata" in job.artifacts
    assert "move_plan" in job.artifacts
    assert "destination_path" in job.artifacts

    # Verify job was saved
    loaded_job = pipeline.PipelineJob.load(job.document_id)
    assert loaded_job is not None
    assert loaded_job.current_stage == pipeline.PipelineStage.done


def test_pipeline_resume_after_crash(temp_dirs, mock_llm_and_embeddings, monkeypatch):
    """Test resuming pipeline after a simulated crash after classify stage."""
    # Create a test document
    pdf_path = create_test_document(temp_dirs["inbox"], "resume_test_001")
    
    from legal_dms import pipeline
    
    # Mock all the functions we'll need
    def fake_extract_text(path):
        return OcrResult(
            document_path=path,
            pages=[
                PageResult(
                    page_number=1,
                    text="This is a lease agreement document.",
                    confidence=0.9,
                    language="en",
                    source="ocr",
                )
            ],
        )

    def fake_plan_move(source, metadata):
        return MovePlan(
            source_path=source,
            destination_path=temp_dirs["library"] / "lease_agreement" / "2025" / "resumed.pdf",
            document_id="resume_test_001",
            metadata=metadata,
        )

    def fake_execute_move(plan, confirmed):
        plan.destination_path.parent.mkdir(parents=True, exist_ok=True)
        plan.destination_path.write_bytes(plan.source_path.read_bytes())
        return plan.destination_path, plan.sidecar_path

    def fake_classify(ocr_text, ocr_language=None):
        return DocumentMetadata(
            document_type=DocumentType.lease_agreement,
            primary_language=PrimaryLanguage.en,
            parties=[Party(name="Lessor", role="Landlord"), Party(name="Lessee", role="Tenant")],
            execution_date=date(2025, 1, 15),
            jurisdiction="Mumbai",
            summary="This is a lease agreement between lessor and lessee for commercial property.",
            key_clauses=["rent", "term", "termination", "renewal"],
            confidence=0.95,
        )

    def fake_index_document(document_id, ocr_result, metadata):
        pass

    monkeypatch.setattr(pipeline, "extract_text", fake_extract_text)
    monkeypatch.setattr(pipeline, "plan_move", fake_plan_move)
    monkeypatch.setattr(pipeline, "execute_move", fake_execute_move)
    monkeypatch.setattr(pipeline, "classify", fake_classify)
    monkeypatch.setattr(pipeline, "index_document", fake_index_document)

    # Process up to plan stage
    pipeline_inst = pipeline.Pipeline()
    job1 = pipeline.PipelineJob(source_path=pdf_path, document_id="resume_test_001")
    job1.save()

    while job1.current_stage != pipeline.PipelineStage.plan:
        pipeline_inst._run_stage(job1)
        job1.save()
    assert job1.current_stage == pipeline.PipelineStage.plan

    # Now simulate crash and resume
    monkeypatch.setattr(settings, "auto_confirm", True, raising=False)
    pipeline2 = pipeline.Pipeline()
    job2 = pipeline2.process(pdf_path, resume_from_document_id="resume_test_001")

    # Verify we continued from where we left off
    assert job2.current_stage == pipeline.PipelineStage.done
    assert job2.error is None
    assert job2.document_id == "resume_test_001"


def test_pipeline_with_custom_confirmation(temp_dirs, mock_llm_and_embeddings, monkeypatch):
    """Test pipeline with custom confirmation callback."""
    pdf_path = create_test_document(temp_dirs["inbox"], "confirm_test_001")
    
    from legal_dms import pipeline
    
    def fake_extract_text(path):
        return OcrResult(
            document_path=path,
            pages=[
                PageResult(
                    page_number=1,
                    text="This is a lease agreement.",
                    confidence=0.9,
                    language="en",
                    source="ocr",
                )
            ],
        )

    def fake_plan_move(source, metadata):
        return MovePlan(
            source_path=source,
            destination_path=temp_dirs["library"] / "lease_agreement" / "2025" / "confirmed.pdf",
            document_id="confirm_test_001",
            metadata=metadata,
        )

    def fake_execute_move(plan, confirmed):
        plan.destination_path.parent.mkdir(parents=True, exist_ok=True)
        plan.destination_path.write_bytes(plan.source_path.read_bytes())
        return plan.destination_path, plan.sidecar_path

    def fake_classify(ocr_text, ocr_language=None):
        return DocumentMetadata(
            document_type=DocumentType.lease_agreement,
            primary_language=PrimaryLanguage.en,
            parties=[Party(name="Lessor", role="Landlord"), Party(name="Lessee", role="Tenant")],
            execution_date=date(2025, 1, 15),
            jurisdiction="Mumbai",
            summary="This is a lease agreement between lessor and lessee for commercial property.",
            key_clauses=["rent", "term", "termination", "renewal"],
            confidence=0.95,
        )

    def fake_index_document(document_id, ocr_result, metadata):
        pass

    monkeypatch.setattr(pipeline, "extract_text", fake_extract_text)
    monkeypatch.setattr(pipeline, "plan_move", fake_plan_move)
    monkeypatch.setattr(pipeline, "execute_move", fake_execute_move)
    monkeypatch.setattr(pipeline, "classify", fake_classify)
    monkeypatch.setattr(pipeline, "index_document", fake_index_document)

    # Custom confirmation callback
    confirmation_calls = []

    def confirm_callback(job) -> bool:
        confirmation_calls.append(job)
        return True

    pipeline_inst = pipeline.Pipeline(confirmation_callback=confirm_callback)
    job = pipeline_inst.process(pdf_path)

    # Verify confirmation was called
    assert len(confirmation_calls) == 1
    assert confirmation_calls[0].document_id == job.document_id
    assert job.current_stage == pipeline.PipelineStage.done


def test_pipeline_rejection_on_confirmation(temp_dirs, mock_llm_and_embeddings, monkeypatch):
    """Test pipeline stops when confirmation is rejected."""
    pdf_path = create_test_document(temp_dirs["inbox"], "reject_test_001")
    
    from legal_dms import pipeline

    def fake_extract_text(path):
        return OcrResult(
            document_path=path,
            pages=[
                PageResult(
                    page_number=1,
                    text="This is a lease agreement.",
                    confidence=0.9,
                    language="en",
                    source="ocr",
                )
            ],
        )

    def fake_plan_move(source, metadata):
        return MovePlan(
            source_path=source,
            destination_path=temp_dirs["library"] / "lease_agreement" / "2025" / "rejected.pdf",
            document_id="reject_test_001",
            metadata=metadata,
        )

    def fake_execute_move(plan, confirmed):
        plan.destination_path.parent.mkdir(parents=True, exist_ok=True)
        plan.destination_path.write_bytes(plan.source_path.read_bytes())
        return plan.destination_path, plan.sidecar_path

    def fake_classify(ocr_text, ocr_language=None):
        return DocumentMetadata(
            document_type=DocumentType.lease_agreement,
            primary_language=PrimaryLanguage.en,
            parties=[Party(name="Lessor", role="Landlord"), Party(name="Lessee", role="Tenant")],
            execution_date=date(2025, 1, 15),
            jurisdiction="Mumbai",
            summary="This is a lease agreement between lessor and lessee for commercial property.",
            key_clauses=["rent", "term", "termination", "renewal"],
            confidence=0.95,
        )

    def fake_index_document(document_id, ocr_result, metadata):
        pass

    monkeypatch.setattr(pipeline, "extract_text", fake_extract_text)
    monkeypatch.setattr(pipeline, "plan_move", fake_plan_move)
    monkeypatch.setattr(pipeline, "execute_move", fake_execute_move)
    monkeypatch.setattr(pipeline, "classify", fake_classify)
    monkeypatch.setattr(pipeline, "index_document", fake_index_document)

    # Rejection callback
    def reject_callback(job) -> bool:
        return False

    pipeline_inst = pipeline.Pipeline(confirmation_callback=reject_callback)

    # Process should raise an exception when confirmation is rejected
    with pytest.raises(ValueError, match="User rejected"):
        pipeline_inst.process(pdf_path)
