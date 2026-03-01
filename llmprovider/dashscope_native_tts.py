from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Union

import requests


@dataclass(frozen=True)
class DashScopeNativeTTSConfig:
    """
    DashScope native Qwen TTS (NOT OpenAI-compatible).

    CN endpoint:
      https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
    Intl endpoint:
      https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation

    Recommended models (2025):
      - qwen3-tts-flash
      - qwen3-tts-instruct-flash (supports "instructions")
    """

    api_key: str

    # Native endpoint
    endpoint: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    )

    model: str = "qwen3-tts-flash"
    voice: str = "Cherry"
    language_type: str = "Chinese"
    sample_rate: int = 24000  # reference only

    timeout_s: int = 120
    max_retries: int = 5
    retry_backoff_base_s: float = 1.2

    # streaming output (SSE) - only affects request header + response handling
    stream: bool = False

    # instruction control (only instruct model supports it)
    instructions: Optional[str] = None
    optimize_instructions: bool = False


DashScopeSynthesizeResult = Union[Dict[str, Any], Iterable[str]]


class DashScopeNativeTTSClient:
    """
    Thin HTTP client for DashScope native Qwen TTS.

    - `session` injectable for tests
    - `sleep_fn` injectable for retry tests
    """

    def __init__(
        self,
        cfg: DashScopeNativeTTSConfig,
        *,
        session: Optional[requests.Session] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if not cfg.api_key:
            raise ValueError("DashScope API key is required (env DASHSCOPE_API_KEY).")
        self.cfg = cfg
        self._session = session or requests.Session()
        self._sleep = sleep_fn

    @staticmethod
    def from_env() -> "DashScopeNativeTTSClient":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")

        # Region detection (default CN)
        region = os.environ.get("DASHSCOPE_REGION", "cn").lower()
        if region in ("intl", "sg", "singapore", "global"):
            default_endpoint = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        else:
            default_endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        endpoint = os.environ.get("DASHSCOPE_TTS_ENDPOINT", default_endpoint).strip()

        # Backward compatibility: allow using DASHSCOPE_TTS_MODEL/VOICE as native config too.
        model = (
            os.environ.get("DASHSCOPE_NATIVE_TTS_MODEL", "").strip()
            or os.environ.get("DASHSCOPE_TTS_MODEL", "").strip()
            or "qwen3-tts-flash"
        )
        # If user still has old compatible-mode default in .env
        if model == "qwen3-tts":
            model = "qwen3-tts-flash"

        voice = (
            os.environ.get("DASHSCOPE_NATIVE_TTS_VOICE", "").strip()
            or os.environ.get("DASHSCOPE_TTS_VOICE", "").strip()
            or "Cherry"
        )
        if voice == "female":
            voice = "Cherry"

        language_type = os.environ.get("DASHSCOPE_TTS_LANGUAGE", "Chinese").strip()
        sample_rate = int(os.environ.get("DASHSCOPE_TTS_SAMPLE_RATE", "24000"))
        timeout_s = int(os.environ.get("DASHSCOPE_TIMEOUT_S", "120"))
        max_retries = int(os.environ.get("DASHSCOPE_MAX_RETRIES", "5"))
        stream = os.environ.get("DASHSCOPE_TTS_STREAM", "false").lower() == "true"

        instructions = os.environ.get("DASHSCOPE_TTS_INSTRUCTIONS")
        if instructions:
            instructions = instructions.strip() or None
        optimize_instructions = (
            os.environ.get("DASHSCOPE_OPTIMIZE_INSTRUCTIONS", "false").lower() == "true"
        )

        return DashScopeNativeTTSClient(
            DashScopeNativeTTSConfig(
                api_key=api_key,
                endpoint=endpoint,
                model=model,
                voice=voice,
                language_type=language_type,
                sample_rate=sample_rate,
                timeout_s=timeout_s,
                max_retries=max_retries,
                retry_backoff_base_s=float(
                    os.environ.get("DASHSCOPE_RETRY_BACKOFF_BASE_S", "1.2")
                ),
                stream=stream,
                instructions=instructions,
                optimize_instructions=optimize_instructions,
            )
        )

    @staticmethod
    def _headers(
        api_key: str, *, stream: bool, extra: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if stream:
            h["X-DashScope-SSE"] = "enable"
        if extra:
            h.update(extra)
        return h

    def _post_with_retries(
        self,
        url: str,
        *,
        json: Dict[str, Any],
        stream: bool,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        last_exception: Optional[BaseException] = None
        headers = self._headers(
            self.cfg.api_key, stream=stream, extra=extra_headers or None
        )

        for attempt in range(self.cfg.max_retries):
            try:
                r = self._session.post(
                    url,
                    headers=headers,
                    json=json,
                    timeout=self.cfg.timeout_s,
                    stream=stream,
                )
                r.raise_for_status()
                return r
            except requests.exceptions.HTTPError as e:
                # Do NOT retry on deterministic client errors (except 429 rate limit).
                resp = getattr(e, "response", None)
                status = getattr(resp, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last_exception = e
            except requests.exceptions.RequestException as e:
                last_exception = e

            if attempt < self.cfg.max_retries - 1:
                sleep_time = self.cfg.retry_backoff_base_s * (2**attempt)
                self._sleep(sleep_time)

        raise last_exception or RuntimeError("TTS request failed after max retries")

    def synthesize_native(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        language_type: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> DashScopeSynthesizeResult:
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "input": {
                "text": text,
                "voice": voice or self.cfg.voice,
                "language_type": language_type or self.cfg.language_type,
            },
        }

        if instructions or self.cfg.instructions:
            payload["input"]["instructions"] = instructions or self.cfg.instructions
        if self.cfg.optimize_instructions:
            payload["input"]["optimize_instructions"] = True

        r = self._post_with_retries(
            self.cfg.endpoint,
            json=payload,
            stream=self.cfg.stream,
        )

        if self.cfg.stream:
            return self._iter_sse_lines(r)
        return r.json()

    # Backward-compatible alias
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language_type: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> DashScopeSynthesizeResult:
        return self.synthesize_native(
            text,
            voice=voice,
            language_type=language_type,
            instructions=instructions,
        )

    @staticmethod
    def _iter_sse_lines(response: requests.Response) -> Iterator[str]:
        for line in response.iter_lines():
            if line:
                yield line.decode("utf-8", errors="replace")

    @staticmethod
    def get_audio_url(response: Dict[str, Any]) -> Optional[str]:
        try:
            url = response.get("output", {}).get("audio", {}).get("url")
            return str(url) if url else None
        except Exception:
            return None

    def download_audio(self, url: str, output_path: str) -> str:
        r = self._session.get(url, timeout=60)
        r.raise_for_status()

        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
        return str(p)

    def download_audio_bytes(self, url: str) -> bytes:
        r = self._session.get(url, timeout=60)
        r.raise_for_status()
        return r.content


__all__ = [
    "DashScopeNativeTTSConfig",
    "DashScopeNativeTTSClient",
    "DashScopeSynthesizeResult",
]
