from __future__ import annotations

import itertools
import sys
import threading
import time
from typing import Optional

_COLOR_DIM = "\033[90m"
_COLOR_RESET = "\033[0m"


class Spinner:
    """Simple spinner helper to avoid log interleaving."""

    def __init__(self, message: str) -> None:
        self.message = message
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "Spinner":
        spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

        def run() -> None:
            while not self._stop_event.is_set():
                sym = next(spinner)
                sys.stdout.write(f"\r{_COLOR_DIM}{sym} {self.message}...{_COLOR_RESET}")
                sys.stdout.flush()
                time.sleep(0.1)
            sys.stdout.write("\r   \r\n")
            sys.stdout.flush()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
