from __future__ import annotations
import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
from dotenv import load_dotenv


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    url: str
    status_code: int
    content_type: str
    detail: str


def _iter_unique(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def probe_models(
    api_key: str, base_urls: list[str], timeout_s: int = 15
) -> list[ProbeResult]:
    """
    Does NOT call TTS (no cost). Only probes the OpenAI-compatible /models endpoints.

    DashScope quirk:
    - If base URL already ends with "/v1", then "{base}/models" works, and
      "{base}/v1/models" becomes "/v1/v1/models" (404).
    - If base URL is ".../compatible-mode" (no /v1), then ".../compatible-mode/models"
      is typically 404; correct is ".../compatible-mode/v1/models".
    """
    results: list[ProbeResult] = []

    def _paths_for_base(base: str) -> list[str]:
        b = base.rstrip("/")
        paths: list[str] = []

        if b.endswith("/v1"):
            paths.append("/models")
        else:
            # Try the correct DashScope-compatible-mode path first
            if b.endswith("/compatible-mode"):
                paths.append("/v1/models")
            # Generic fallbacks
            paths.append("/models")
            paths.append("/v1/models")

        # de-dup keep order
        out: list[str] = []
        seen: set[str] = set()
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    for base in base_urls:
        for p in _paths_for_base(base):
            url = base.rstrip("/") + p
            try:
                r = requests.get(url, headers=_headers(api_key), timeout=timeout_s)
                ctype = (r.headers.get("Content-Type") or "").lower()
                body_preview = (r.text or "")[:200].replace("\n", "\\n")
                results.append(
                    ProbeResult(
                        ok=200 <= r.status_code < 300,
                        url=url,
                        status_code=r.status_code,
                        content_type=ctype,
                        detail=body_preview,
                    )
                )
            except Exception as e:
                results.append(
                    ProbeResult(
                        ok=False,
                        url=url,
                        status_code=0,
                        content_type="",
                        detail=f"EXC: {e}",
                    )
                )
    return results


def _extract_audio_b64(obj: Any) -> str:
    # same logic as llmprovider
    if isinstance(obj, dict):
        if isinstance(obj.get("audio"), str):
            return obj["audio"]
        data = obj.get("data")
        if isinstance(data, list) and data:
            item0 = data[0]
            if isinstance(item0, dict) and isinstance(item0.get("audio"), str):
                return item0["audio"]
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            item0 = choices[0]
            if isinstance(item0, dict) and isinstance(item0.get("audio"), str):
                return item0["audio"]
        output = obj.get("output")
        if isinstance(output, dict) and isinstance(output.get("audio"), str):
            return output["audio"]
    return ""


def list_api_v1_models(
    api_key: str,
    filter_text: str = "",
    page_size: int = 50,
    max_pages: int = 20,
    timeout_s: int = 20,
) -> list[str]:
    """
    Query DashScope native model list:
      GET https://dashscope.aliyuncs.com/api/v1/models?page_no=1&page_size=XX

    Returns matched model ids (strings).
    """
    url = "https://dashscope.aliyuncs.com/api/v1/models"
    filter_text = (filter_text or "").strip().lower()

    matched: list[str] = []
    page_no = 1
    total = None

    while page_no <= max_pages:
        params = {"page_no": page_no, "page_size": page_size}
        r = requests.get(
            url, headers=_headers(api_key), params=params, timeout=timeout_s
        )
        if r.status_code >= 400:
            raise RuntimeError(
                f"GET {url} failed {r.status_code}: {(r.text or '')[:300]}"
            )
        obj = r.json()
        output = obj.get("output") if isinstance(obj, dict) else None
        if not isinstance(output, dict):
            break

        if total is None:
            total = output.get("total")

        models = output.get("models")
        if not isinstance(models, list) or not models:
            break

        for m in models:
            if not isinstance(m, dict):
                continue
            mid = m.get("model") or m.get("id") or ""
            mid = str(mid)
            if not mid:
                continue
            if not filter_text or filter_text in mid.lower():
                matched.append(mid)

        # stop early if we've already collected some matches and filter is specific
        if filter_text and len(matched) >= 30:
            break

        page_no += 1

    # de-dup keep order
    return _iter_unique(matched)


def try_tts(
    api_key: str,
    base_urls: list[str],
    model: str,
    voice: str,
    out_file: Path,
    timeout_s: int = 60,
) -> list[ProbeResult]:
    """
    Calls TTS with minimal text (WILL incur cost if succeeds).
    Tries a few likely endpoints to locate the correct route, saves MP3 if successful.
    """
    endpoints = ["/audio/speech", "/v1/audio/speech"]
    text = "你好，这是一段连通性测试语音。"

    results: list[ProbeResult] = []
    for base in base_urls:
        for ep in endpoints:
            url = base.rstrip("/") + ep
            payload = {
                "model": model,
                "input": text,
                "voice": voice,
                "response_format": "mp3",
            }
            try:
                r = requests.post(
                    url,
                    headers=_headers(api_key),
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=timeout_s,
                )
                ctype = (r.headers.get("Content-Type") or "").lower()

                if 200 <= r.status_code < 300:
                    audio_bytes: Optional[bytes] = None

                    if "audio/" in ctype or "application/octet-stream" in ctype:
                        audio_bytes = r.content
                    else:
                        # maybe json
                        try:
                            obj = r.json()
                        except Exception:
                            obj = None
                        if obj is not None:
                            b64 = _extract_audio_b64(obj)
                            if b64:
                                audio_bytes = base64.b64decode(b64)

                    if audio_bytes:
                        out_file.parent.mkdir(parents=True, exist_ok=True)
                        out_file.write_bytes(audio_bytes)
                        results.append(
                            ProbeResult(
                                ok=True,
                                url=url,
                                status_code=r.status_code,
                                content_type=ctype,
                                detail=f"OK. Wrote {len(audio_bytes)} bytes to {out_file}",
                            )
                        )
                        return results  # stop at first success

                    results.append(
                        ProbeResult(
                            ok=False,
                            url=url,
                            status_code=r.status_code,
                            content_type=ctype,
                            detail=f"2xx but cannot extract audio. First 200 bytes: {r.content[:200]!r}",
                        )
                    )
                else:
                    results.append(
                        ProbeResult(
                            ok=False,
                            url=url,
                            status_code=r.status_code,
                            content_type=ctype,
                            detail=(r.text or "")[:200].replace("\n", "\\n"),
                        )
                    )
            except Exception as e:
                results.append(
                    ProbeResult(
                        ok=False,
                        url=url,
                        status_code=0,
                        content_type="",
                        detail=f"EXC: {e}",
                    )
                )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DashScope (Aliyun) connectivity probe for OpenAI-compatible mode and optional TTS attempt."
    )
    parser.add_argument(
        "--do-tts",
        action="store_true",
        help="Actually call TTS (WILL incur cost if succeeds). Default: false (0 cost).",
    )
    parser.add_argument(
        "--list-native-models",
        action="store_true",
        help="List native (api/v1) models that match --filter (no cost). Useful to find correct TTS model id.",
    )
    parser.add_argument(
        "--filter",
        default="sambert",
        help="Filter text for --list-native-models (default: sambert).",
    )
    parser.add_argument(
        "--out",
        default="audio_zh_test/probe.mp3",
        help="Output mp3 path when --do-tts succeeds (default: audio_zh_test/probe.mp3).",
    )
    args = parser.parse_args()

    load_dotenv()

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print("ERROR: missing DASHSCOPE_API_KEY in .env", file=sys.stderr)
        return 2

    # Keep probe output "clean": only include known-working OpenAI-compatible bases.
    # (The "/compatible-mode" base typically 404s on /models; "/api/v1" is native mode.)
    env_base = os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()

    base_urls = _iter_unique(
        [
            env_base,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ]
    )

    model = os.environ.get("DASHSCOPE_TTS_MODEL", "qwen3-tts").strip() or "qwen3-tts"
    voice = os.environ.get("DASHSCOPE_TTS_VOICE", "female").strip() or "female"

    print("=== DashScope OpenAI-compatible probe ===")
    print("Base URLs:")
    for b in base_urls:
        print(" -", b)
    print("Model:", model)
    print("Voice:", voice)
    print()

    print("== Probe /models (no cost) ==")
    models_results = probe_models(api_key=api_key, base_urls=base_urls)
    for r in models_results:
        tag = "OK " if r.ok else "ERR"
        print(f"[{tag}] {r.status_code:>3} GET {r.url} | {r.content_type} | {r.detail}")
    print()

    if args.list_native_models:
        print("== List api/v1 models (no cost) ==")
        try:
            matches = list_api_v1_models(api_key=api_key, filter_text=args.filter)
            if matches:
                print(f"Matched {len(matches)} model ids (filter={args.filter!r}):")
                for m in matches[:200]:
                    print(" -", m)
            else:
                print(
                    f"No matches for filter={args.filter!r}. Try --filter tts / --filter speech / --filter bert."
                )
        except Exception as e:
            print(f"Failed to list api/v1 models: {e}", file=sys.stderr)
            return 2
        print()

    if not args.do_tts:
        print("== Skip TTS call (0 cost) ==")
        print("Run: python tools/dashscope_tts_probe.py --do-tts")
        return 0

    print("== Try TTS (WILL incur cost if succeeds) ==")
    out_file = Path(args.out)
    tts_results = try_tts(
        api_key=api_key,
        base_urls=base_urls,
        model=model,
        voice=voice,
        out_file=out_file,
    )
    for r in tts_results:
        tag = "OK " if r.ok else "ERR"
        print(
            f"[{tag}] {r.status_code:>3} POST {r.url} | {r.content_type} | {r.detail}"
        )

    any_ok = any(r.ok for r in tts_results)
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
