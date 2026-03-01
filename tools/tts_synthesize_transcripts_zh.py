from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List
from urllib.parse import urlparse

# Ensure repo root is on sys.path when running "python tools/xxx.py"
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
from tqdm import tqdm

from llmprovider.dashscope_native_tts import DashScopeNativeTTSClient


@dataclass(frozen=True)
class Segment:
    idx: int
    text: str


def _extract_zh_full_text(md: str) -> str:
    """
    Extract content after heading "## 中文翻译（全文）" to end-of-file.
    Falls back to whole file if heading not found.
    """
    marker = "## 中文翻译（全文）"
    i = md.find(marker)
    if i < 0:
        return md.strip()
    return md[i + len(marker) :].strip()


def _strip_markdown(s: str) -> str:
    # remove code blocks
    s = re.sub(r"```[\s\S]*?```", "", s)
    # inline code
    s = re.sub(r"`([^`]*)`", r"\1", s)
    # links [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # bold/italic markers
    s = s.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    # headings and hr
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*---+\s*$", "", s, flags=re.MULTILINE)
    # collapse whitespace
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _split_into_sentences(text: str) -> List[str]:
    # split on common Chinese punctuation while keeping punctuation
    parts = re.split(r"(?<=[。！？；：\?!.;:])\s*", text)
    return [p.strip() for p in parts if p and p.strip()]


def _chunk_text(text: str, max_chars: int) -> List[str]:
    """
    Chunk by paragraphs then sentence-level to keep each chunk <= max_chars.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n".join(buf).strip())
        buf = []
        buf_len = 0

    for para in paras:
        if len(para) <= max_chars:
            if buf_len + len(para) + (1 if buf else 0) <= max_chars:
                buf.append(para)
                buf_len += len(para) + (1 if buf_len else 0)
            else:
                flush()
                buf.append(para)
                buf_len = len(para)
            continue

        # long paragraph -> sentence split
        flush()
        sentences = _split_into_sentences(para)
        cur = ""
        for sent in sentences:
            if not cur:
                cur = sent
                continue
            if len(cur) + 1 + len(sent) <= max_chars:
                cur = f"{cur} {sent}"
            else:
                chunks.append(cur.strip())
                cur = sent
        if cur:
            chunks.append(cur.strip())

    flush()
    return [c for c in chunks if c.strip()]


def _segments_from_md(md_text: str, max_chars: int) -> List[Segment]:
    body = _extract_zh_full_text(md_text)
    body = _strip_markdown(body)
    chunks = _chunk_text(body, max_chars=max_chars)
    return [Segment(idx=i + 1, text=t) for i, t in enumerate(chunks)]


def _safe_stem(p: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", p.stem)


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _require_ffmpeg() -> str:
    """
    Return ffmpeg executable path.

    On Windows, `winget install Gyan.FFmpeg` may add a command alias / Links entry
    but the current shell might not pick up PATH changes until restart.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    # winget common location (command line alias)
    winget_link = (
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Microsoft"
        / "WinGet"
        / "Links"
        / "ffmpeg.exe"
    )
    if winget_link.exists():
        return str(winget_link)

    raise RuntimeError(
        "ffmpeg not found. Install it (e.g. `winget install --id Gyan.FFmpeg -e`) and/or restart your shell."
    )


def _concat_mp3_with_ffmpeg(segment_paths: List[Path], out_mp3: Path) -> None:
    """
    Concatenate multiple MP3 files into a single MP3 using ffmpeg concat demuxer.
    This does stream copy (-c copy), so it's fast and avoids re-encoding.
    """
    ffmpeg = _require_ffmpeg()

    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg concat file format: each line: file 'ABS_PATH'
    # Use forward slashes to avoid escaping backslashes on Windows.
    concat_file = out_mp3.with_suffix(".concat.txt")
    lines = []
    for p in segment_paths:
        ap = p.resolve().as_posix()
        lines.append(f"file '{ap}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Use re-encode instead of stream copy to avoid non-monotonic DTS issues when concatenating
    # MP3 segments generated from separate TTS calls.
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(out_mp3),
    ]
    subprocess.run(cmd, check=True)


def synthesize_dir(
    transcripts_dir: Path,
    out_dir: Path,
    max_chars: int,
    overwrite: bool,
    pattern: str = "*.md",
    max_files: int = 0,
    max_segments_per_file: int = 0,
    dry_run: bool = False,
    single_mp3: bool = True,
    keep_segments: bool = True,
) -> None:
    # This repo uses DashScope native Qwen TTS (NOT OpenAI-compatible).
    # DashScope OpenAI-compatible TTS route may 404 depending on account/region, so we removed it.
    client = None if dry_run else DashScopeNativeTTSClient.from_env()

    md_files = sorted(transcripts_dir.glob(pattern))
    if max_files and max_files > 0:
        md_files = md_files[:max_files]
    out_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": "dashscope-tts",
        "base_url": os.environ.get(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "endpoint_native": os.environ.get(
            "DASHSCOPE_TTS_ENDPOINT",
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        ),
        "model": os.environ.get("DASHSCOPE_TTS_MODEL", "qwen3-tts"),
        "voice": os.environ.get("DASHSCOPE_TTS_VOICE", "female"),
        "format": "mp3",
        "sample_rate": int(os.environ.get("DASHSCOPE_TTS_SAMPLE_RATE", "24000")),
        "max_chars": max_chars,
        "single_mp3": single_mp3,
        "keep_segments": keep_segments,
        "files": [],
    }

    for md_path in md_files:
        md_text = md_path.read_text(encoding="utf-8")
        segments = _segments_from_md(md_text, max_chars=max_chars)

        stem = _safe_stem(md_path)

        # segments are generated under: out_dir/{stem}/seg_XXXX_xxxxxxxx.mp3
        week_out_dir = out_dir / stem
        week_out_dir.mkdir(parents=True, exist_ok=True)

        merged_mp3_path = out_dir / f"{stem}.mp3"

        manifest = {
            "source_md": str(md_path).replace("\\", "/"),
            "segments_dir": str(week_out_dir).replace("\\", "/"),
            "merged_mp3": str(merged_mp3_path).replace("\\", "/") if single_mp3 else "",
            "segments": [],
        }

        if max_segments_per_file and max_segments_per_file > 0:
            segments = segments[:max_segments_per_file]

        segment_paths: List[Path] = []

        for seg in tqdm(segments, desc=f"{md_path.name}", unit="seg"):
            seg_hash = _sha1(seg.text)
            seg_name = f"seg_{seg.idx:04d}_{seg_hash[:8]}.mp3"
            seg_path = week_out_dir / seg_name

            if dry_run:
                # no API call
                pass
            else:
                if seg_path.exists() and not overwrite:
                    pass
                else:
                    assert client is not None
                    resp = client.synthesize_native(seg.text)
                    if not isinstance(resp, dict):
                        raise RuntimeError(
                            "Unexpected streaming response in native mode; please keep stream disabled."
                        )
                    url = client.get_audio_url(resp)
                    if not url:
                        raise RuntimeError(
                            f"Native TTS returned no audio url. Response: {str(resp)[:300]}"
                        )

                    ext = Path(urlparse(url).path).suffix
                    if not ext:
                        ext = ".bin"
                    src_path = week_out_dir / f"{seg_path.stem}_src{ext}"

                    client.download_audio(url, str(src_path))

                    if src_path.suffix.lower() == ".mp3":
                        seg_path.write_bytes(src_path.read_bytes())
                    else:
                        ffmpeg = _require_ffmpeg()
                        cmd = [
                            ffmpeg,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(src_path),
                            "-vn",
                            "-codec:a",
                            "libmp3lame",
                            "-q:a",
                            "4",
                            str(seg_path),
                        ]
                        subprocess.run(cmd, check=True)

                    # keep workspace clean
                    try:
                        src_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            manifest["segments"].append(
                {
                    "idx": seg.idx,
                    "chars": len(seg.text),
                    "sha1": seg_hash,
                    "file": seg_name,
                }
            )
            segment_paths.append(seg_path)

        # Merge into one mp3 per document
        if single_mp3:
            if dry_run:
                # create empty placeholder only to show expected output path
                merged_mp3_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                if merged_mp3_path.exists() and not overwrite:
                    pass
                else:
                    _concat_mp3_with_ffmpeg(segment_paths, merged_mp3_path)

        # Optionally clean up segments directory
        if single_mp3 and (not keep_segments) and (not dry_run):
            # remove segment files only; keep manifest for traceability
            for p in segment_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        _write_json(week_out_dir / "manifest.json", manifest)

        index["files"].append(
            {
                "source": str(md_path).replace("\\", "/"),
                "segments_dir": str(week_out_dir).replace("\\", "/"),
                "segments": len(segments),
                "mp3": str(merged_mp3_path).replace("\\", "/") if single_mp3 else "",
            }
        )

    _write_json(out_dir / "index.json", index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize Chinese transcripts (transcripts_zh/*.md) into mp3 via DashScope Qwen3-TTS."
    )
    parser.add_argument(
        "--transcripts-dir",
        default="transcripts_zh",
        help="Input markdown directory (default: transcripts_zh)",
    )
    parser.add_argument(
        "--out-dir",
        default="audio_zh",
        help="Output directory (default: audio_zh)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=int(os.environ.get("TTS_MAX_CHARS", "800")),
        help="Max characters per TTS request (default: 800 or env TTS_MAX_CHARS).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mp3 files (segments and merged).",
    )
    parser.add_argument(
        "--pattern",
        default="*.md",
        help="Glob pattern under transcripts dir (default: *.md).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of md files to process (0 means no limit).",
    )
    parser.add_argument(
        "--max-segments-per-file",
        type=int,
        default=0,
        help="Limit number of segments per file (0 means no limit). Useful for low-cost testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call TTS API (no cost). Only output index/manifest structure.",
    )
    parser.add_argument(
        "--single-mp3",
        action="store_true",
        help="Generate one merged mp3 per document (default: true).",
    )
    parser.add_argument(
        "--no-single-mp3",
        action="store_true",
        help="Disable merged mp3 output (only keep per-segment mp3).",
    )
    parser.add_argument(
        "--keep-segments",
        action="store_true",
        help="Keep per-segment mp3 files (default: true).",
    )
    parser.add_argument(
        "--no-keep-segments",
        action="store_true",
        help="After merging, delete per-segment mp3 files.",
    )
    args = parser.parse_args()

    load_dotenv()  # load .env if present

    single_mp3 = True
    if args.no_single_mp3:
        single_mp3 = False
    elif args.single_mp3:
        single_mp3 = True

    keep_segments = True
    if args.no_keep_segments:
        keep_segments = False
    elif args.keep_segments:
        keep_segments = True

    if single_mp3 and not args.dry_run:
        _require_ffmpeg()

    synthesize_dir(
        transcripts_dir=Path(args.transcripts_dir),
        out_dir=Path(args.out_dir),
        max_chars=args.max_chars,
        overwrite=args.overwrite,
        pattern=args.pattern,
        max_files=args.max_files,
        max_segments_per_file=args.max_segments_per_file,
        dry_run=args.dry_run,
        single_mp3=single_mp3,
        keep_segments=keep_segments,
    )


if __name__ == "__main__":
    main()
