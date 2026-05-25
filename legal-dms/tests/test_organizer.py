import json
from datetime import date as date_cls
from pathlib import Path
from unittest.mock import patch

import pytest

from legal_dms.classifier.model import (
    DocumentMetadata,
    DocumentType,
    Party,
    PrimaryLanguage,
)
from legal_dms.config.settings import settings
from legal_dms.organizer.manager import (
    MovePlan,
    OrganizerNotConfirmedError,
    execute_move,
    plan_move,
)


def make_metadata(document_type: DocumentType, execution_date=None):
    return DocumentMetadata(
        document_type=document_type,
        primary_language=PrimaryLanguage.en,
        parties=[Party(name="ABC Builders Pvt. Ltd.", role="Lessor")],
        execution_date=execution_date,
        jurisdiction="Mumbai",
        summary="A legal document summary.",
        key_clauses=["termination"],
        confidence=0.8,
    )


def test_plan_move_and_execute_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "library_path", tmp_path / "library", raising=False)
    monkeypatch.setattr(settings, "work_path", tmp_path / "work", raising=False)

    source = tmp_path / "source.pdf"
    source.write_bytes(b"dummy pdf content")
    metadata = make_metadata(DocumentType.lease_agreement, execution_date=date_cls(2025, 5, 20))

    plan = plan_move(source, metadata)

    assert plan.conflict is None
    assert plan.destination_path.parent == tmp_path / "library" / "lease_agreement" / "2025"
    assert plan.destination_path.name.startswith("2025-05-lease_agreement-")

    final_dest, sidecar_path = execute_move(plan, confirmed=True)

    assert final_dest.exists()
    assert not source.exists()
    assert sidecar_path.exists()

    with sidecar_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["document_id"] == plan.document_id
    assert payload["original_filename"] == "source.pdf"
    assert payload["ocr_text_path"] == str(settings.work_path / "source.txt")
    assert payload["metadata"]["document_type"] == "lease_agreement"


def test_plan_move_uses_current_month_when_execution_date_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "library_path", tmp_path / "library", raising=False)

    source = tmp_path / "source.pdf"
    source.write_bytes(b"dummy pdf content")
    metadata = make_metadata(DocumentType.noc, execution_date=None)

    fake_date = date_cls(2025, 4, 1)
    with patch("legal_dms.organizer.manager.date") as mock_date:
        mock_date.today.return_value = fake_date
        plan = plan_move(source, metadata)

    assert plan.destination_path.name.startswith("2025-04-noc-")
    assert plan.destination_path.parent.name == "2025"


def test_plan_move_slugifies_devanagari_and_emoji(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "library_path", tmp_path / "library", raising=False)

    source = tmp_path / "source.pdf"
    source.write_bytes(b"dummy pdf content")
    metadata = DocumentMetadata(
        document_type=DocumentType.sale_deed,
        primary_language=PrimaryLanguage.mixed,
        parties=[
            Party(name="राम😀 खत्री", role="Seller"),
            Party(name="A&B Co.", role="Buyer"),
        ],
        execution_date=date_cls(2025, 1, 10),
        jurisdiction="Mumbai",
        summary="A sale deed summary.",
        key_clauses=["indemnity"],
        confidence=0.8,
    )

    plan = plan_move(source, metadata)

    assert "unknown-a-b-co" in plan.destination_path.name
    assert "😀" not in plan.destination_path.name


def test_plan_move_appends_suffix_on_existing_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "library_path", tmp_path / "library", raising=False)

    source = tmp_path / "source.pdf"
    content = b"collision content"
    source.write_bytes(content)
    metadata = make_metadata(DocumentType.sale_deed, execution_date=date_cls(2025, 5, 20))

    plan = plan_move(source, metadata)
    plan.destination_path.parent.mkdir(parents=True, exist_ok=True)
    (plan.destination_path.parent / plan.destination_path.name).write_text("existing file", encoding="utf-8")

    plan_with_conflict = plan_move(source, metadata)

    assert plan_with_conflict.conflict is not None
    assert plan_with_conflict.destination_path.name.endswith(f"-{plan_with_conflict.document_id[:6]}.pdf")


def test_execute_move_refuses_without_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "library_path", tmp_path / "library", raising=False)
    monkeypatch.setattr(settings, "work_path", tmp_path / "work", raising=False)

    source = tmp_path / "source.pdf"
    source.write_bytes(b"dummy pdf content")
    metadata = make_metadata(DocumentType.power_of_attorney, execution_date=date_cls(2025, 7, 14))

    plan = plan_move(source, metadata)

    with pytest.raises(OrganizerNotConfirmedError):
        execute_move(plan, confirmed=False)
