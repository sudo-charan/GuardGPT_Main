import html
import logging
import os
from typing import Optional
import requests

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
DEFAULT_TIMEOUT = 120
DEFAULT_TEMPERATURE = 0.2


class LlamaBackend:
    """Ollama backend REST client."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

        try:
            self.timeout = int(timeout or os.getenv("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            self.timeout = DEFAULT_TIMEOUT

        try:
            self.temperature = float(
                temperature
                if temperature is not None
                else os.getenv("OLLAMA_TEMPERATURE", DEFAULT_TEMPERATURE)
            )
        except (TypeError, ValueError):
            self.temperature = DEFAULT_TEMPERATURE

        self._session = requests.Session()
        from urllib.parse import urlparse
        if urlparse(self.base_url).hostname in {"127.0.0.1", "localhost", "::1"}:
            self._session.trust_env = False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        *,
        schema=None,
        temperature=None,
        history=None,
    ) -> str:
        """Generate text. ``history`` is a list of ``{"role", "content"}`` turns.

        Without history the single-turn ``/api/generate`` endpoint is used (the
        reviewer and rewriter are single-turn on purpose). With history the
        conversation goes through ``/api/chat`` so the model sees earlier turns.
        """
        prompt = str(prompt).strip() if prompt else ""
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        use_chat = bool(history)
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature if temperature is None else temperature,
                        "num_predict": 768, "num_ctx": 8192,
                        # Discourages the degenerate token-repetition loops seen with
                        # smaller/general-purpose models under grammar-constrained JSON
                        # decoding (e.g. an enum array cycling the same 2-3 values).
                        "repeat_penalty": 1.3, "repeat_last_n": 64},
        }

        if use_chat:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": str(system_prompt)})
            for turn in history:
                role = turn.get("role") if isinstance(turn, dict) else None
                content = str(turn.get("content", "")).strip() if isinstance(turn, dict) else ""
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": prompt})
            payload["messages"] = messages
        else:
            payload["prompt"] = prompt
            if system_prompt:
                payload["system"] = str(system_prompt)
        if schema is not None:
            payload["format"] = schema

        url = f"{self.base_url}/api/chat" if use_chat else f"{self.base_url}/api/generate"

        try:
            response = self._session.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as error:
            raise ConnectionError(
                "Cannot reach Ollama. Start Ollama and run python main.py --status."
            ) from error
        except Exception as error:
            raise RuntimeError("Ollama generation failed or timed out.") from error

        if response.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP {response.status_code}. Check installed models.")

        data = response.json()
        if use_chat:
            message = data.get("message") if isinstance(data, dict) else None
            text = message.get("content") if isinstance(message, dict) else None
        else:
            text = data.get("response") if isinstance(data, dict) else None
        if not isinstance(text, str):
            raise RuntimeError("Invalid Ollama response format.")
        if data.get("done") is not True or data.get("done_reason") == "length":
            raise RuntimeError("Ollama response was incomplete or exceeded the token limit.")
        # Some local models emit HTML entities ("&amp;"); this is plain-text chat.
        generated_text = html.unescape(text).strip() if "&" in text else text.strip()

        if not generated_text:
            raise RuntimeError("Empty response received from backend.")

        return generated_text

    def is_available(self) -> bool:
        try:
            return self._session.get(f"{self.base_url}/api/tags", timeout=5).status_code == 200
        except Exception:
            return False
