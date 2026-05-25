"""Apply folder structure and document routing rules to keep the library organized."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from legal_dms.classifier.model import DocumentMetadata
from legal_dms.config.settings import settings
from legal_dms.common.logging import get_logger

logger = get_logger(__name__)


class OrganizerNotConfirmedError(Exception):
    pass


class MovePlan(BaseModel):
    source_path: Path
    destination_path: Path
    document_id: str
    metadata: DocumentMetadata
    sidecar_path: Path | None = None
    conflict: str | None = None

    @model_validator(mode="after")
    def set_sidecar_path(self) -> MovePlan:
        if self.sidecar_path is None:
            self.sidecar_path = self.destination_path.with_suffix(".json")
        return self


def _slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.lower()
    ascii_name = re.sub(r"[^a-z0-9]+", "-", ascii_name)
    ascii_name = ascii_name.strip("-")
    return ascii_name or "unknown"


def _party_slug(metadata: DocumentMetadata) -> str:
    party_names = [party.name for party in metadata.parties[:2]]
    if not party_names:
        return "unknown"
    slug_parts = [_slugify(name) for name in party_names]
    return "-".join(slug_parts)


def _compute_document_id(source: Path) -> str:
    hash_obj = hashlib.sha256()
    with source.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _destination_folder(metadata: DocumentMetadata, execution_date: date) -> Path:
    return settings.library_path / metadata.document_type.value / str(execution_date.year)


def plan_move(source: Path, metadata: DocumentMetadata) -> MovePlan:
    source = source.expanduser()
    document_id = _compute_document_id(source)

    execution_date = metadata.execution_date or date.today()
    prefix = execution_date.strftime("%Y-%m")
    folder = _destination_folder(metadata, execution_date)
    folder.mkdir(parents=True, exist_ok=True)

    slugified_parties = _party_slug(metadata)
    base_name = f"{prefix}-{metadata.document_type.value}-{slugified_parties}.pdf"
    destination = folder / base_name
    conflict = None

    if destination.exists():
        suffix = document_id[:6]
        destination = folder / f"{prefix}-{metadata.document_type.value}-{slugified_parties}-{suffix}.pdf"
        conflict = f"Destination exists. Using suffix {suffix}."

    return MovePlan(
        source_path=source,
        destination_path=destination,
        document_id=document_id,
        metadata=metadata,
        conflict=conflict,
    )


def execute_move(plan: MovePlan, confirmed: bool, ocr_text: str | None = None) -> tuple[Path, Path]:
    if not confirmed:
        raise OrganizerNotConfirmedError("Move must be confirmed before execution.")

    plan.destination_path.parent.mkdir(parents=True, exist_ok=True)
    final_destination = Path(shutil.move(str(plan.source_path), str(plan.destination_path)))

    sidecar_data = {
        "metadata": plan.metadata.model_dump(mode="json"),
        "document_id": plan.document_id,
        "original_filename": plan.source_path.name,
        "destination_path": str(final_destination),
    }
    if ocr_text is not None:
        sidecar_data["ocr_text"] = ocr_text

    plan.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with plan.sidecar_path.open("w", encoding="utf-8") as sidecar_file:
        json.dump(sidecar_data, sidecar_file, ensure_ascii=False, indent=2)

    return final_destination, plan.sidecar_path
