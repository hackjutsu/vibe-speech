from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional
from urllib import error, request

from .config import AssistantConfig

logger = logging.getLogger(__name__)


@dataclass
class AssistantReply:
    text: str
    duration_s: float
    used_assistant: bool = True


class AssistantError(RuntimeError):
    """Raised when the assistant backend fails."""


class LLMAssistant:
    """LLM-backed assistant that can run locally via llama.cpp or call Ollama."""

    def __init__(self, config: AssistantConfig) -> None:
        self.config = config
        self._llama = None
        self._llama_error: Optional[str] = None
        self._history: list[tuple[str, str]] = []

    def _load_local(self) -> None:
        if self.config.provider != "local":
            return
        if self._llama or self._llama_error:
            return
        if not self.config.enabled:
            self._llama_error = "Disabled"
            return
        if not self.config.model_path:
            self._llama_error = "No model_path configured"
            logger.warning("Assistant enabled but no model_path set; skipping assistant.")
            return
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            self._llama_error = f"llama-cpp import failed: {exc}"
            logger.warning("Assistant unavailable: %s", self._llama_error)
            return
        try:
            self._llama = Llama(
                model_path=self.config.model_path,
                n_ctx=2048,
                logits_all=False,
            )
            logger.info("Assistant model loaded from %s", self.config.model_path)
        except Exception as exc:  # pragma: no cover - runtime dependency
            self._llama_error = f"Llama load failed: {exc}"
            logger.error("Failed to load assistant model: %s", exc)

    def respond(self, user_text: str) -> AssistantReply:
        if not user_text.strip():
            raise AssistantError("Empty user text provided to assistant.")
        if not self.config.enabled:
            raise AssistantError("Assistant is disabled in configuration.")
        if self.config.provider == "ollama":
            reply = self._respond_via_ollama(user_text)
        else:
            reply = self._respond_via_llama(user_text)
        self._remember_turn(user_text, reply.text)
        return reply

    def _build_prompt(self, user_text: str) -> str:
        persona = self.config.personality.strip()
        lines: list[str] = [self.config.system_prompt, f"Personality: {persona}", ""]

        if self.config.history_length > 0 and self._history:
            for past_user, past_reply in self._history[-self.config.history_length :]:
                lines.append(f"User: {past_user.strip()}")
                lines.append(f"Assistant: {past_reply.strip()}")
                lines.append("")

        lines.append(f"User: {user_text.strip()}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _remember_turn(self, user_text: str, reply_text: str) -> None:
        if self.config.history_length <= 0:
            return
        self._history.append((user_text.strip(), reply_text.strip()))
        if len(self._history) > self.config.history_length:
            self._history = self._history[-self.config.history_length :]

    def _respond_via_ollama(self, user_text: str) -> AssistantReply:
        if not self.config.model:
            raise AssistantError("Ollama provider enabled but no model specified for assistant.")
        prompt = self._build_prompt(user_text)
        start = time.perf_counter()
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
                raise AssistantError("Ollama returned empty assistant response.")
            duration = time.perf_counter() - start
            return AssistantReply(text=result, duration_s=duration)
        except error.URLError as exc:  # pragma: no cover - runtime dependency
            raise AssistantError(f"Ollama request failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise AssistantError(f"Ollama parse failed: {exc}") from exc

    def _respond_via_llama(self, user_text: str) -> AssistantReply:
        self._load_local()
        if not self._llama:
            raise AssistantError(self._llama_error or "llama.cpp backend not available.")
        prompt = self._build_prompt(user_text)
        start = time.perf_counter()
        try:
            output = self._llama(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stop=["User:"],
            )
            result = output.get("choices", [{}])[0].get("text", "").strip()
            if not result:
                raise AssistantError("llama.cpp returned empty assistant response.")
            duration = time.perf_counter() - start
            return AssistantReply(text=result, duration_s=duration)
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.error("Assistant failed: %s", exc)
            raise AssistantError(f"llama.cpp assistant failed: {exc}") from exc
