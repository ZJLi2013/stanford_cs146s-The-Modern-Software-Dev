# tools 使用说明（DashScope native Qwen TTS）

本仓库已**统一改为使用 DashScope 原生（native）Qwen TTS 接口**做中文音频合成，不再依赖 OpenAI-compatible TTS 路由（该路由在部分账号/地域会 404）。

## 1) 环境变量

在仓库根目录创建/编辑 `.env`：

```ini
DASHSCOPE_API_KEY=sk-xxxx

# 可选：地域（默认 cn）。intl/sg/singapore/global 会使用国际站 endpoint
DASHSCOPE_REGION=cn

# 可选：覆盖 native endpoint
# DASHSCOPE_TTS_ENDPOINT=https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation

# native 模型/音色（建议显式设置）
DASHSCOPE_NATIVE_TTS_MODEL=qwen3-tts-flash
DASHSCOPE_NATIVE_TTS_VOICE=Cherry

# 语言类型
DASHSCOPE_TTS_LANGUAGE=Chinese

# 每次请求最大字符数（用于分段）
TTS_MAX_CHARS=800
```

## 2) 依赖

- Python 依赖：`pip install -r requirements.txt`
- 合并/转码依赖：`ffmpeg`
  - Windows 推荐：`winget install --id Gyan.FFmpeg -e`

脚本会优先从 PATH 找 `ffmpeg`，如果当前 shell 没刷新 PATH，会尝试：
`C:\Users\<you>\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe`

## 3) 将 transcripts_zh 的 md 转 mp3（每文档一个 mp3）

脚本：`tools/tts_synthesize_transcripts_zh.py`

### 3.1 先 dry-run（0 成本）

只输出目录结构与 manifest/index，不调用 TTS：

```powershell
python tools/tts_synthesize_transcripts_zh.py --dry-run --max-files 1 --max-segments-per-file 2
```

### 3.2 生成单个文档（会产生费用）

以 `wk01_zh.md` 为例：

```powershell
python tools/tts_synthesize_transcripts_zh.py --pattern wk01_zh.md --overwrite
```

输出：
- `audio_zh/wk01_zh.mp3`：该文档的**单个合并 mp3**
- `audio_zh/wk01_zh/seg_XXXX_XXXXXXXX.mp3`：分段 mp3（用于失败重试/降低单次输入长度）
- `audio_zh/wk01_zh/manifest.json`：分段信息
- `audio_zh/wk01_zh.concat.txt`：ffmpeg concat 临时文件（可删除）

### 3.3 全量生成所有 wk*_zh.md（会产生费用）

```powershell
python tools/tts_synthesize_transcripts_zh.py --pattern wk*_zh.md --overwrite
```

## 4) 常用参数

- `--max-chars`：单次 TTS 输入字符上限（默认 `TTS_MAX_CHARS` 或 800）
- `--max-files`：最多处理多少个 md
- `--max-segments-per-file`：每个 md 最多生成多少段（省钱测试）
- `--overwrite`：覆盖已生成的分段/合并 mp3
- `--dry-run`：不调用 API（0 成本）
