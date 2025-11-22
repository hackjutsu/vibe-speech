from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from urllib import request, parse, error
import json

from .config import RewriterConfig

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    text: str
    used_rewriter: bool


class LocalLLMRewriter:
    """Optional rewriter using either llama.cpp (local GGUF) or Ollama HTTP."""

    def __init__(self, config: RewriterConfig) -> None:
        self.config = config
        self._llama = None
        self._llama_error: Optional[str] = None

    def _load(self) -> None:
        if self.config.provider != "local":
            return
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

    def _rewrite_via_ollama(self, text: str) -> RewriteResult:
        if not self.config.model:
            logger.warning("Ollama provider enabled but no model specified; skipping.")
            return RewriteResult(text=text, used_rewriter=False)
        prompt = (
            f"{self.config.system_prompt}\n"
            f"Input: {text.strip()}\n"
            "Rewrite:"
        )
        body = json.dumps(
            {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }
        ).encode("utf-8")
        url = self.config.ollama_url.rstrip("/") + "/api/generate"
        req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            payload = json.loads(data.decode("utf-8"))
            result = payload.get("response", "") or payload.get("text", "")
            result = result.strip()
            if not result:
                return RewriteResult(text=text, used_rewriter=False)
            return RewriteResult(text=result, used_rewriter=True)
        except error.URLError as exc:  # pragma: no cover - runtime dependency
            logger.error("Ollama request failed: %s", exc)
            return RewriteResult(text=text, used_rewriter=False)
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.error("Ollama parse failed: %s", exc)
            return RewriteResult(text=text, used_rewriter=False)

    def rewrite(self, text: str) -> RewriteResult:
        if not text.strip():
            return RewriteResult(text="", used_rewriter=False)
        if not self.config.enabled:
            return RewriteResult(text=text, used_rewriter=False)
        if self.config.provider == "ollama":
            return self._rewrite_via_ollama(text)
        # default to local llama.cpp
        return self._rewrite_via_llama(text)

    # Backward-compatible name
    def _rewrite_via_llama(self, text: str) -> RewriteResult:
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
