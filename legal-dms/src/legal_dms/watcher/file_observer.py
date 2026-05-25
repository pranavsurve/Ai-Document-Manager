"""Watch for incoming documents in the inbox and enqueue them for processing."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from queue import Queue
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from legal_dms.common.logging import get_logger
from legal_dms.config.settings import settings

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp"}
DEBOUNCE_STABLE_TIME = 2.0  # seconds


class DebounceTracker:
    """Track file sizes over time to detect when they've stabilized."""

    def __init__(self):
        self.file_stats: dict[Path, tuple[int, float]] = {}  # path -> (size, timestamp)

    def check_stable(self, file_path: Path, current_time: float) -> bool:
        """Return True if file has been stable for DEBOUNCE_STABLE_TIME seconds."""
        if not file_path.exists():
            return False

        try:
            current_size = file_path.stat().st_size
        except OSError:
            return False

        if file_path not in self.file_stats:
            self.file_stats[file_path] = (current_size, current_time)
            return False

        prev_size, first_seen = self.file_stats[file_path]

        if current_size != prev_size:
            self.file_stats[file_path] = (current_size, current_time)
            return False

        elapsed = current_time - first_seen
        if elapsed >= DEBOUNCE_STABLE_TIME:
            del self.file_stats[file_path]
            return True

        return False

    def cleanup(self, file_path: Path) -> None:
        """Remove a file from tracking."""
        self.file_stats.pop(file_path, None)


class FileObserverHandler(FileSystemEventHandler):
    """Handle filesystem events and enqueue stable files."""

    def __init__(self, queue: Queue[Path], debounce: DebounceTracker):
        self.queue = queue
        self.debounce = debounce

    def on_created(self, event) -> None:
        """Called when a file is created."""
        if event.is_dir:
            return
        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        logger.info(f"Detected new file: {file_path}")

    def on_modified(self, event) -> None:
        """Called when a file is modified; check for stability."""
        if event.is_dir:
            return
        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        if self.debounce.check_stable(file_path, time.time()):
            logger.info(f"File stable: {file_path}, enqueueing for processing")
            self.queue.put(file_path)


class FileObserver:
    """Observe the inbox directory and manage the worker thread."""

    def __init__(self, inbox_path: Optional[Path] = None):
        self.inbox_path = inbox_path or settings.inbox_path
        self.queue: Queue[Path] = Queue()
        self.debounce = DebounceTracker()
        self.observer = Observer()
        self.observer.schedule(
            FileObserverHandler(self.queue, self.debounce),
            str(self.inbox_path),
            recursive=False,
        )
        self.worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callback = None

    def set_callback(self, callback) -> None:
        """Set the callback to be called for each enqueued file."""
        self._callback = callback

    def start(self) -> None:
        """Start observing the inbox."""
        logger.info(f"Starting file observer on {self.inbox_path}")
        self.observer.start()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=False)
        self.worker_thread.start()

    def stop(self) -> None:
        """Stop observing and shutdown gracefully."""
        logger.info("Stopping file observer")
        self._stop_event.set()
        self.observer.stop()
        self.observer.join(timeout=5)
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def _process_queue(self) -> None:
        """Process files from the queue in a single worker thread."""
        while not self._stop_event.is_set():
            try:
                file_path = self.queue.get(timeout=1.0)
                if self._callback:
                    try:
                        self._callback(file_path)
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            except Exception:
                continue
