"""Shared Groq LLM client used by every agent.

One client instance is created by the app and passed to each agent, so the
Groq SDK is initialised exactly once. The client provides:

* generate()       -> plain text completion
* generate_json()  -> JSON completion validated against a Pydantic model

Robustness features: request timeout, SDK retries for transient errors,
one corrective retry for malformed JSON, JSON extraction from messy output,
an in-memory response cache (cost control) and user-friendly error messages.
The API key is never logged, displayed or stored.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict
from typing import Annotated, Any, TypeVar, get_args, get_origin

from dotenv import load_dotenv
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

load_dotenv()

# A model that was generally available on Groq when this project was written.
# Override with the GROQ_MODEL secret / environment variable at any time.
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Reasoning models spend output tokens on hidden reasoning before the answer:
# keep reasoning effort low and add headroom so the JSON is not truncated.
REASONING_MODEL_PREFIXES = ("openai/gpt-oss",)
REASONING_TOKEN_HEADROOM = 1500

MISSING_KEY_MESSAGE = "Groq API key is not configured. Add GROQ_API_KEY to Streamlit Secrets (or to a local .env file)."

SHARED_RULES = """
GENERAL RULES (industrial maintenance decision support):
- Use ONLY the data supplied in the user message. Never invent measurements, maintenance history, part numbers or equipment details.
- Clearly separate provided facts from assumptions; label assumptions as such.
- Express uncertainty honestly ("possible", "likely", "requires verification"). You are NOT issuing a certified engineering diagnosis.
- Safety first: never suggest bypassing interlocks, trips, guards or safety systems. For dangerous conditions defer to site emergency procedures and qualified personnel.
- Platform-calculated indicators use demo/configurable thresholds: treat them as screening aids, not standards.
- Be concise: short phrases, no filler. 'reasoning' fields are brief justifications citing evidence, never step-by-step internal thoughts.
"""

JSON_INSTRUCTIONS = (
    "\nOUTPUT FORMAT: reply with ONE valid JSON object only (no markdown, no extra text) using exactly these keys. "
    "Where a template value lists options separated by '|', pick one option. Template:\n"
)


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from Streamlit secrets first, then environment variables (.env)."""
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            if value:
                return str(value).strip()
    except Exception:
        pass  # no secrets.toml, or not running inside Streamlit
    value = os.getenv(name)
    return value.strip() if value else default


# --------------------------------------------------------------------------- #
# Lenient field types: LLMs sometimes return a list where a string is expected,
# a string where a list is expected, or "4 hours" instead of 4. These coercers
# keep validation strict about *shape* without crashing on harmless variations.
# --------------------------------------------------------------------------- #
def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "; ".join(s for s in (_to_str(v) for v in value) if s)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_to_str(v)}" for k, v in value.items())
    return str(value)


def _to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, dict):
        return [f"{k}: {_to_str(v)}" for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if isinstance(item, dict):
                text = " - ".join(_to_str(v) for v in item.values() if v not in (None, ""))
            else:
                text = _to_str(item)
            if text:
                items.append(text)
        return items
    return [str(value)]


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else 0.0


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "yes", "y", "1", "required"}


Text = Annotated[str, BeforeValidator(_to_str)]
TextList = Annotated[list[str], BeforeValidator(_to_str_list)]
Number = Annotated[float, BeforeValidator(_to_float)]
Flag = Annotated[bool, BeforeValidator(_to_bool)]


class AgentOutput(BaseModel):
    """Base class for every agent's structured output."""

    model_config = ConfigDict(extra="ignore")


ModelT = TypeVar("ModelT", bound=BaseModel)


def compact(data: Any) -> str:
    """Compact JSON for prompts (fewer tokens than pretty-printed JSON)."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False, default=str)


def _template(model: type[BaseModel]) -> dict:
    """Build a JSON skeleton from a Pydantic model, using field descriptions as hints."""
    out: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        annotation = field.annotation
        hint = field.description or "string"
        if annotation is bool:
            out[name] = False
        elif annotation in (int, float):
            out[name] = 0
        elif get_origin(annotation) is list:
            args = get_args(annotation)
            inner = args[0] if args else str
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                out[name] = [_template(inner)]
            else:
                out[name] = [hint]
        else:
            out[name] = hint
    return out


def schema_template(model: type[BaseModel]) -> str:
    return json.dumps(_template(model), ensure_ascii=False)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences, prose and trailing commas."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in the model response")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))


class LLMError(RuntimeError):
    """Controlled, user-presentable LLM failure."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class GroqClient:
    """Thin, reusable wrapper around the Groq Python SDK."""

    _CACHE_SIZE = 64

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 60.0):
        self.model = model or get_setting("GROQ_MODEL") or DEFAULT_MODEL
        self.calls = 0
        self.cache_hits = 0
        self.init_error = ""
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._client = None
        key = api_key or get_setting("GROQ_API_KEY")
        if key:
            try:
                from groq import Groq

                # max_retries: the SDK retries timeouts, connection errors, 429 and 5xx with backoff.
                self._client = Groq(api_key=key, timeout=timeout, max_retries=3)
            except Exception as exc:  # never include the key in messages
                self.init_error = f"Could not initialise the Groq client ({type(exc).__name__})."

    @property
    def configured(self) -> bool:
        return self._client is not None

    def check_connection(self) -> tuple[bool, str]:
        """Lightweight connectivity/auth check (lists models, no tokens used)."""
        if not self.configured:
            return False, self.init_error or MISSING_KEY_MESSAGE
        try:
            self._client.models.list()
            return True, "Connected"
        except Exception as exc:
            return False, self._friendly_error(exc)[0]

    # ------------------------------------------------------------------ #
    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        """Return the raw text completion. Raises LLMError on failure."""
        if not self.configured:
            raise LLMError(self.init_error or MISSING_KEY_MESSAGE)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.model.startswith(REASONING_MODEL_PREFIXES):
            kwargs["reasoning_effort"] = "low"
            kwargs["max_tokens"] = max_tokens + REASONING_TOKEN_HEADROOM
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            message, retryable = self._friendly_error(exc)
            raise LLMError(message, retryable) from None
        self.calls += 1
        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not text:
            raise LLMError("The model returned an empty response.", retryable=True)
        return text

    def generate_json(
        self,
        system: str,
        user: str,
        schema: type[ModelT],
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> ModelT:
        """Return a validated Pydantic object.

        Strategy: JSON mode -> extract JSON -> validate. On failure, retry once
        with the validation problem fed back to the model, then raise LLMError.
        """
        system_full = system.strip() + "\n" + SHARED_RULES + JSON_INSTRUCTIONS + schema_template(schema)
        cache_key = hashlib.sha256(f"{self.model}|{max_tokens}|{system_full}|{user}".encode()).hexdigest()
        if cache_key in self._cache:
            try:
                result = schema.model_validate(extract_json(self._cache[cache_key]))
                self.cache_hits += 1
                return result
            except (ValueError, ValidationError):
                self._cache.pop(cache_key, None)

        prompt = user
        problem = ""
        for attempt in range(2):
            try:
                raw = self.generate(system_full, prompt, max_tokens=max_tokens, temperature=temperature, json_mode=True)
            except LLMError as exc:
                if exc.retryable and attempt == 0:
                    problem = str(exc)
                    continue
                raise
            try:
                data = extract_json(raw)
                if not isinstance(data, dict):
                    raise ValueError("Top-level JSON value is not an object")
                result = schema.model_validate(data)
            except (ValueError, ValidationError) as exc:
                problem = f"{type(exc).__name__}: {str(exc)[:300]}"
                prompt = (
                    f"{user}\n\nYour previous reply could not be parsed ({problem}). "
                    "Return ONLY the corrected JSON object that follows the template."
                )
                continue
            self._cache[cache_key] = raw
            while len(self._cache) > self._CACHE_SIZE:
                self._cache.popitem(last=False)
            return result
        raise LLMError(f"The model did not return valid structured output after a retry. Details: {problem}")

    # ------------------------------------------------------------------ #
    def _friendly_error(self, exc: Exception) -> tuple[str, bool]:
        """Translate SDK exceptions into (message, retryable). Never echoes the API key."""
        try:
            import groq
        except ImportError:
            return f"Unexpected LLM error ({type(exc).__name__}).", False
        detail = ""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                detail = str(err.get("message", ""))[:200]
        if isinstance(exc, groq.AuthenticationError):
            return "Groq rejected the API key (authentication failed). Check GROQ_API_KEY in Streamlit Secrets.", False
        if isinstance(exc, groq.NotFoundError):
            return f"Groq model '{self.model}' was not found. Set GROQ_MODEL to a model currently listed in the Groq console.", False
        if isinstance(exc, groq.RateLimitError):
            return "Groq rate limit reached (free tier). Wait about a minute and run the workflow again, or set GROQ_MODEL to a model with higher limits.", False
        if isinstance(exc, groq.APITimeoutError):
            return "The Groq request timed out. Please try again.", True
        if isinstance(exc, groq.APIConnectionError):
            return "Could not reach the Groq API. Check the network connection and try again.", True
        if isinstance(exc, groq.BadRequestError):
            lowered = detail.lower()
            if "decommission" in lowered or ("model" in lowered and "not" in lowered):
                return f"Groq rejected model '{self.model}': {detail} Set GROQ_MODEL to a current model.", False
            # json_validate_failed and similar output problems are worth one retry
            return f"Groq rejected the request or the model output ({detail or 'bad request'}).", True
        if isinstance(exc, groq.APIStatusError):
            return f"Groq API error (HTTP {exc.status_code}). {detail}".strip(), exc.status_code >= 500
        return f"Unexpected LLM error ({type(exc).__name__}).", False
