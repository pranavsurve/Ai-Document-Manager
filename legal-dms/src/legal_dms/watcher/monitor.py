"""Watch local folders for incoming documents and trigger the ingestion pipeline."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from legal_dms.common.logging import get_logger
from legal_dms.config.settings import settings
from legal_dms.pipeline import Pipeline
from legal_dms.watcher.file_observer import FileObserver

logger = get_logger(__name__)

_observer: Optional[FileObserver] = None
_observer_lock = threading.Lock()


def start_observer(callback=None) -> None:
    """Start the file observer if not already started."""
    global _observer
    with _observer_lock:
        if _observer is not None:
            logger.warning("Observer already started")
            return

        logger.info("Starting file observer")
        _observer = FileObserver(settings.inbox_path)
        if callback:
            _observer.set_callback(callback)
        _observer.start()


def stop_observer() -> None:
    """Stop the file observer if running."""
    global _observer
    with _observer_lock:
        if _observer is None:
            logger.warning("Observer not started")
            return
        logger.info("Stopping file observer")
        _observer.stop()
        _observer = None


def _default_callback(file_path: Path) -> None:
    """Default callback: process the file with the pipeline."""
    logger.info(f"Processing file via pipeline: {file_path}")
    pipeline = Pipeline()
    try:
        pipeline.process(file_path)
        logger.info(f"Successfully processed {file_path}")
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}", exc_info=True)


def start_default_observer() -> None:
    """Start the observer with the default callback."""
    start_observer(callback=_default_callback)


if __name__ == "__main__":
    # For testing: run the observer directly
    start_default_observer()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_observer()