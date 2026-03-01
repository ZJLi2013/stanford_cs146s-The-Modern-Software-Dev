import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import requests

from llmprovider.dashscope_native_tts import (
    DashScopeNativeTTSClient,
    DashScopeNativeTTSConfig,
)


def _mock_response(
    *,
    status_code: int = 200,
    headers: Optional[Dict[str, str]] = None,
    content: bytes = b"",
    json_obj: Optional[Any] = None,
    iter_lines: Optional[List[bytes]] = None,
) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.headers = headers or {"Content-Type": "application/json"}
    r.content = content
    r.text = "" if json_obj is None else str(json_obj)

    def raise_for_status():
        if status_code >= 400:
            e = requests.HTTPError(f"{status_code}")
            e.response = r  # attach response for callers to inspect (404 handling)
            raise e

    r.raise_for_status.side_effect = raise_for_status

    if json_obj is not None:
        r.json.return_value = json_obj
    else:
        r.json.side_effect = ValueError("no json")

    r.iter_lines.return_value = iter_lines or []
    return r


class TestDashScopeNativeTTSClient(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = DashScopeNativeTTSConfig(
            api_key="test-key",
            endpoint="https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            model="qwen3-tts-flash",
            voice="Cherry",
            language_type="Chinese",
            timeout_s=3,
            max_retries=3,
            retry_backoff_base_s=1.2,
            stream=False,
        )

    def test_from_env_default_cn(self) -> None:
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "k"}, clear=True):
            c = DashScopeNativeTTSClient.from_env()
            self.assertIn("dashscope.aliyuncs.com", c.cfg.endpoint)
            self.assertNotIn("dashscope-intl", c.cfg.endpoint)

    def test_from_env_intl_default_endpoint(self) -> None:
        with patch.dict(
            os.environ,
            {"DASHSCOPE_API_KEY": "k", "DASHSCOPE_REGION": "intl"},
            clear=True,
        ):
            c = DashScopeNativeTTSClient.from_env()
            self.assertIn("dashscope-intl.aliyuncs.com", c.cfg.endpoint)

    def test_synthesize_native_success_non_stream(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.post.return_value = _mock_response(json_obj={"ok": True})

        client = DashScopeNativeTTSClient(self.cfg, session=session)
        out = client.synthesize_native("你好", voice="Serena")
        self.assertEqual(out, {"ok": True})

        session.post.assert_called_once()
        args, kwargs = session.post.call_args
        self.assertEqual(args[0], self.cfg.endpoint)
        self.assertEqual(kwargs["timeout"], self.cfg.timeout_s)
        self.assertEqual(kwargs["stream"], False)

        headers = kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertNotIn("X-DashScope-SSE", headers)

        payload = kwargs["json"]
        self.assertEqual(payload["model"], self.cfg.model)
        self.assertEqual(payload["input"]["voice"], "Serena")
        self.assertEqual(payload["input"]["language_type"], self.cfg.language_type)

    def test_synthesize_native_retry_then_success(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.post.side_effect = [
            requests.RequestException("boom"),
            _mock_response(json_obj={"ok": True}),
        ]
        sleep = MagicMock()

        client = DashScopeNativeTTSClient(self.cfg, session=session, sleep_fn=sleep)
        out = client.synthesize_native("hello")
        self.assertEqual(out, {"ok": True})

        self.assertEqual(session.post.call_count, 2)
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], self.cfg.retry_backoff_base_s)

    def test_synthesize_native_fail_after_max_retries(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.post.side_effect = requests.RequestException("nope")
        sleep = MagicMock()

        client = DashScopeNativeTTSClient(self.cfg, session=session, sleep_fn=sleep)
        with self.assertRaises(requests.RequestException):
            client.synthesize_native("hello")

        self.assertEqual(session.post.call_count, self.cfg.max_retries)
        self.assertEqual(sleep.call_count, self.cfg.max_retries - 1)

    def test_synthesize_native_streaming_returns_lines(self) -> None:
        cfg = DashScopeNativeTTSConfig(
            **{**self.cfg.__dict__, "stream": True, "max_retries": 1}
        )
        session = MagicMock(spec=requests.Session)
        session.post.return_value = _mock_response(iter_lines=[b"line1", b"line2"])

        client = DashScopeNativeTTSClient(cfg, session=session)
        out = client.synthesize_native("hello")
        self.assertEqual(list(out), ["line1", "line2"])

        kwargs = session.post.call_args.kwargs
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["headers"]["X-DashScope-SSE"], "enable")

    def test_get_audio_url(self) -> None:
        url = DashScopeNativeTTSClient.get_audio_url(
            {"output": {"audio": {"url": "u"}}}
        )
        self.assertEqual(url, "u")

    def test_download_audio_writes_file(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(
            status_code=200, headers={"Content-Type": "audio/mpeg"}, content=b"abc"
        )

        client = DashScopeNativeTTSClient(self.cfg, session=session)

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "a" / "b.mp3"
            out = client.download_audio("https://audio", str(out_path))
            self.assertTrue(Path(out).exists())
            self.assertEqual(Path(out).read_bytes(), b"abc")


if __name__ == "__main__":
    unittest.main()
