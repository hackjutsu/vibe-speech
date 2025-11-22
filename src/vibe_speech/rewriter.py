from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .config import RewriterConfig

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    text: str
    used_rewriter: bool


class LocalLLMRewriter:
    """Optional local LLM rewriter using llama.cpp-compatible models."""

    def __init__(self, config: RewriterConfig) -> None:
        self.config = config
        self._llama = None
        self._llama_error: Optional[str] = None

    def _load(self) -> None:
        if self._llama or self._llama_error:
            return
        if not self.config.enabled:
            self._llama_error = "Disabled"
            return
        if not self.config.model_path:
            self._llama_error = "No model_path configured"
            logger.warning("Rewriter enabled but no model_path set; skipping rewriter.")
            return
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            self._llama_error = f"llama-cpp import failed: {exc}"
            logger.warning("Rewriter unavailable: %s", self._llama_error)
            return
        try:
            self._llama = Llama(
                model_path=self.config.model_path,
                n_ctx=2048,
                logits_all=False,
            )
            logger.info("Local LLM rewriter loaded from %s", self.config.model_path)
        except Exception as exc:  # pragma: no cover - runtime dependency
            self._llama_error = f"Llama load failed: {exc}"
            logger.error("Failed to load rewriter model: %s", exc)

    def rewrite(self, text: str) -> RewriteResult:
        if not text.strip():
            return RewriteResult(text="", used_rewriter=False)
        self._load()
        if not self._llama:
            return RewriteResult(text=text, used_rewriter=False)

        prompt = (
            f"{self.config.system_prompt}\n"
            f"Input: {text.strip()}\n"
            "Rewrite:"
        )
        try:
            output = self._llama(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stop=["Input:"],
            )
            result = output.get("choices", [{}])[0].get("text", "").strip()
            if not result:
                return RewriteResult(text=text, used_rewriter=False)
            return RewriteResult(text=result, used_rewriter=True)
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.error("Rewriter failed: %s", exc)
            return RewriteResult(text=text, used_rewriter=False)
