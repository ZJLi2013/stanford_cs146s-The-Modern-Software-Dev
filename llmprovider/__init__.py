"""
llmprovider: thin provider layer for LLM / TTS integrations.

This repo currently uses it for Aliyun DashScope Qwen3-TTS.
- tools/tts_synthesize_transcripts_zh.py uses the native Qwen3-TTS HTTP endpoint via DashScopeNativeTTSClient.
"""

from .dashscope_native_tts import DashScopeNativeTTSClient, DashScopeNativeTTSConfig

__all__ = ["DashScopeNativeTTSClient", "DashScopeNativeTTSConfig"]
