from __future__ import annotations

import re
from typing import Optional

from typing import Optional

from .config import ProcessingConfig
from .rewriter import LocalLLMRewriter, RewriteResult


def process_text(config: ProcessingConfig, text: str, rewriter: Optional[LocalLLMRewriter] = None) -> str:
    if config.mode == "raw":
        result = text
    elif config.mode == "cleanup":
        result = _cleanup(text)
    elif config.mode == "summary":
        # Placeholder: in future, call a local summarizer or compression step.
        result = _cleanup(text)
    elif config.mode == "correct":
        result = _correct(text)
        if rewriter and rewriter.config.enabled:
            rewrite = rewriter.rewrite(result)
            result = rewrite.text
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


def _correct(text: str) -> str:
    """Lightweight grammatical cleanup without external dependencies."""
    text = _cleanup(text)
    if not text:
        return ""

    # Capitalize standalone "i" and common contractions.
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\bi'm\b", "I'm", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi've\b", "I've", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi'll\b", "I'll", text, flags=re.IGNORECASE)
    text = re.sub(r"\bim\b", "I'm", text, flags=re.IGNORECASE)

    # Capitalize first character if needed.
    text = text.strip()
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]

    # Ensure sentence-ending punctuation.
    if text and text[-1] not in ".!?":
        text = text + "."

    return text
