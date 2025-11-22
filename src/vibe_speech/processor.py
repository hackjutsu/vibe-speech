from __future__ import annotations

import re
from typing import Optional

from .config import ProcessingConfig


def process_text(config: ProcessingConfig, text: str) -> str:
    if config.mode == "raw":
        result = text
    elif config.mode == "cleanup":
        result = _cleanup(text)
    elif config.mode == "summary":
        # Placeholder: in future, call a local summarizer or compression step.
        result = _cleanup(text)
    else:  # pragma: no cover
        result = text

    if config.max_chars > 0:
        result = result[: config.max_chars]
    return result


def _cleanup(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Simple sentence-ish spacing
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text

