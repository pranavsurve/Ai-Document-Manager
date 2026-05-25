"""Document processing pipeline with multi-stage orchestration and resumability."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from legal_dms.classifier.model import DocumentMetadata, classify
from legal_dms.common.logging import get_logger
from legal_dms.config.settings import settings
from legal_dms.indexer import index_document
from legal_dms.ocr.engine import extract_text
from legal_dms.organizer.manager import plan_move, execute_move

logger = get_logger(__name__)


class PipelineStage(str, Enum):
    """Pipeline processing stages in order."""

    ocr = "ocr"
    classify = "classify"
    plan = "plan"
    await_confirmation = "await_confirmation"
    execute = "execute"
    index = "index"
    done = "done"


STAGE_ORDER = [
    PipelineStage.ocr,
    PipelineStage.classify,
    PipelineStage.plan,
    PipelineStage.await_confirmation,
    PipelineStage.execute,
    PipelineStage.index,
    PipelineStage.done,
]


class PipelineJob(BaseModel):
    """Represents the state of a document as it moves through the pipeline."""

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: Path
    current_stage: PipelineStage = PipelineStage.ocr
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifacts: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    def work_dir(self) -> Path:
        """Get the work directory for this job."""
        return settings.work_path / self.document_id

    def save(self) -> None:
        """Save job state to disk."""
        work_dir = self.work_dir()
        work_dir.mkdir(parents=True, exist_ok=True)
        job_file = work_dir / "job.json"
        data = json.loads(self.model_dump_json())
        job_file.write_text(json.dumps(data, indent=2, default=str))

    @staticmethod
    def load(document_id: str) -> Optional[PipelineJob]:
        """Load job state from disk."""
        job_file = settings.work_path / document_id / "job.json"
        if not job_file.exists():
            return None
        try:
            data = json.loads(job_file.read_text())
            data["source_path"] = Path(data["source_path"])
            return PipelineJob(**data)
        except Exception as e:
            logger.error(f"Failed to load job {document_id}: {e}")
            return None

    def write_artifact(self, stage: PipelineStage, artifact_name: str, data: Any) -> None:
        """Write an artifact from a stage to disk."""
        work_dir = self.work_dir()
        work_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = work_dir / f"{stage.value}_{artifact_name}"
        if isinstance(data, dict) or isinstance(data, list):
            artifact_file.with_suffix(".json").write_text(json.dumps(data, indent=2, default=str))
        else:
            artifact_file.with_suffix(".txt").write_text(str(data))

    def read_artifact(self, stage: PipelineStage, artifact_name: str) -> Optional[Any]:
        """Read an artifact from disk."""
        work_dir = self.work_dir()
        json_file = work_dir / f"{stage.value}_{artifact_name}.json"
        txt_file = work_dir / f"{stage.value}_{artifact_name}.txt"

        if json_file.exists():
            try:
                return json.loads(json_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to read artifact {json_file}: {e}")
                return None

        if txt_file.exists():
            try:
                return txt_file.read_text()
            except Exception as e:
                logger.warning(f"Failed to read artifact {txt_file}: {e}")
                return None

        return None

    def move_to_stage(self, next_stage: PipelineStage) -> None:
        """Transition to the next stage."""
        self.current_stage = next_stage
        self.updated_at = datetime.now(timezone.utc)
        self.error = None
        self.save()


class Pipeline:
    """Orchestrates document processing through all pipeline stages."""

    def __init__(self, confirmation_callback: Optional[Callable[[PipelineJob], bool]] = None):
        """Initialize the pipeline.

        Args:
            confirmation_callback: Called to confirm execution plans. If None, uses AUTO_CONFIRM setting.
        """
        self.confirmation_callback = confirmation_callback or self._default_confirmation_callback

    def _default_confirmation_callback(self, job: PipelineJob) -> bool:
        """Default confirmation uses the AUTO_CONFIRM setting."""
        return settings.auto_confirm

    def process(self, source_path: Path, resume_from_document_id: Optional[str] = None) -> PipelineJob:
        """Process a document through the entire pipeline.

        Args:
            source_path: Path to the document to process.
            resume_from_document_id: If provided, resume an existing job.

        Returns:
            The completed PipelineJob.
        """
        if resume_from_document_id:
            job = PipelineJob.load(resume_from_document_id)
            if not job:
                logger.warning(f"Could not resume job {resume_from_document_id}, creating new")
                job = PipelineJob(source_path=source_path)
        else:
            job = PipelineJob(source_path=source_path)

        job.save()
        logger.info(f"Starting pipeline for {job.document_id} from {source_path}")

        while job.current_stage != PipelineStage.done:
            try:
                self._run_stage(job)
                job.save()
            except Exception as e:
                logger.error(f"Error in stage {job.current_stage.value}: {e}", exc_info=True)
                job.error = str(e)
                job.save()
                raise

        return job

    def _run_stage(self, job: PipelineJob) -> None:
        """Run the current stage and transition to the next."""
        stage = job.current_stage
        logger.info(f"Running stage: {stage.value}")

        if stage == PipelineStage.ocr:
            self._stage_ocr(job)
        elif stage == PipelineStage.classify:
            self._stage_classify(job)
        elif stage == PipelineStage.plan:
            self._stage_plan(job)
        elif stage == PipelineStage.await_confirmation:
            self._stage_await_confirmation(job)
        elif stage == PipelineStage.execute:
            self._stage_execute(job)
        elif stage == PipelineStage.index:
            self._stage_index(job)
        elif stage == PipelineStage.done:
            pass
        else:
            raise ValueError(f"Unknown stage: {stage}")

        # Move to next stage
        current_idx = STAGE_ORDER.index(job.current_stage)
        if current_idx + 1 < len(STAGE_ORDER):
            job.move_to_stage(STAGE_ORDER[current_idx + 1])

    def _stage_ocr(self, job: PipelineJob) -> None:
        """Extract text from the document."""
        logger.info(f"OCR: Extracting text from {job.source_path}")
        ocr_result = extract_text(job.source_path)
        job.artifacts["ocr_result"] = ocr_result.model_dump()
        job.write_artifact(PipelineStage.ocr, "result", ocr_result.model_dump())
        # Also save plain text OCR result for easier access
        ocr_text = "\n".join(page.text for page in ocr_result.pages)
        job.write_artifact(PipelineStage.ocr, "text", ocr_text)

    def _stage_classify(self, job: PipelineJob) -> None:
        """Classify the document and extract metadata."""
        logger.info(f"Classify: Analyzing document {job.document_id}")
        ocr_result_dict = job.artifacts.get("ocr_result")
        if not ocr_result_dict:
            raise ValueError("OCR result not found in artifacts")

        from legal_dms.ocr.engine import OcrResult

        ocr_result = OcrResult(**ocr_result_dict)
        ocr_text = "\n".join(page.text for page in ocr_result.pages)
        primary_language = ocr_result.pages[0].language if ocr_result.pages else None
        metadata = classify(ocr_text, ocr_language=primary_language)
        job.artifacts["metadata"] = metadata.model_dump()
        job.write_artifact(PipelineStage.classify, "metadata", metadata.model_dump())

    def _stage_plan(self, job: PipelineJob) -> None:
        """Plan the document movement to the library."""
        logger.info(f"Plan: Creating move plan for {job.document_id}")
        metadata_dict = job.artifacts.get("metadata")
        if not metadata_dict:
            raise ValueError("Metadata not found in artifacts")

        metadata = DocumentMetadata(**metadata_dict)
        move_plan = plan_move(job.source_path, metadata)
        job.artifacts["move_plan"] = move_plan.model_dump()
        job.write_artifact(PipelineStage.plan, "move_plan", move_plan.model_dump())

    def _stage_await_confirmation(self, job: PipelineJob) -> None:
        """Wait for user confirmation to execute the plan."""
        logger.info(f"AwaitConfirmation: Requesting confirmation for {job.document_id}")
        if not self.confirmation_callback(job):
            raise ValueError("User rejected the move plan")
        logger.info(f"AwaitConfirmation: Plan confirmed for {job.document_id}")

    def _stage_execute(self, job: PipelineJob) -> None:
        """Execute the move plan."""
        logger.info(f"Execute: Moving {job.document_id} to library")
        move_plan_dict = job.artifacts.get("move_plan")
        if not move_plan_dict:
            raise ValueError("Move plan not found in artifacts")

        from legal_dms.organizer.manager import MovePlan

        move_plan = MovePlan(**move_plan_dict)
        # Get OCR text from artifact
        ocr_text = job.read_artifact(PipelineStage.ocr, "text")
        if ocr_text is None:
            # Fallback: compute from OCR result
            ocr_result_dict = job.artifacts.get("ocr_result")
            if ocr_result_dict:
                from legal_dms.ocr.engine import OcrResult
                ocr_result = OcrResult(**ocr_result_dict)
                ocr_text = "\n".join(page.text for page in ocr_result.pages)
        destination, sidecar = execute_move(move_plan, confirmed=True, ocr_text=ocr_text)
        job.artifacts["destination_path"] = str(destination)
        job.write_artifact(PipelineStage.execute, "destination", str(destination))

    def _stage_index(self, job: PipelineJob) -> None:
        """Index the document for search."""
        logger.info(f"Index: Indexing {job.document_id}")
        ocr_result_dict = job.artifacts.get("ocr_result")
        metadata_dict = job.artifacts.get("metadata")

        if not ocr_result_dict or not metadata_dict:
            raise ValueError("OCR result or metadata not found in artifacts")

        from legal_dms.ocr.engine import OcrResult

        ocr_result = OcrResult(**ocr_result_dict)
        metadata = DocumentMetadata(**metadata_dict)
        index_document(job.document_id, ocr_result, metadata)
        logger.info(f"Index: Indexed {job.document_id}")
