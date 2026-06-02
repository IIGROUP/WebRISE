import os
import json
import asyncio
import base64
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
from datetime import datetime
import sys

load_dotenv()

# =========================
# ===== USER CONFIG =======
# =========================

ROOT_INPUT_DIR = (
    os.getenv("ROOT_INPUT_DIR")
    or os.getenv("WEBRISE_DATA_ROOT")
    or ""
)
TASK_MANIFEST = os.getenv("TASK_MANIFEST", "").strip()
# Both overridable via env vars so experimental runs (e.g. a different
# prompt) can land in a sibling tree without clobbering prior outputs:
#   OUTPUT_DIR=../inference_outputs LOG_DIR=../inference_logs python generate_html.py ...
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "../inference_outputs")
LOG_DIR    = os.getenv("LOG_DIR",    "log")

# 只跑特定 case（逗号分隔多个），例如:
#   CASE_FILTER=D02_S03_T036 python generate_html.py --force
#   CASE_FILTER=D02_S03_T036,D01_S01_T005 python generate_html.py
_env_case_filter = os.getenv("CASE_FILTER", "").strip()
CASE_FILTER = {c.strip() for c in _env_case_filter.split(",") if c.strip()}

# 只跑特定 (case, modality) 组合，逗号分隔，例如:
#   TASK_MODE_FILTER=D02_S03_T036:Text,D04_S12_T171:Video python generate_html.py --force
_env_task_mode_filter = os.getenv("TASK_MODE_FILTER", "").strip()
TASK_MODE_FILTER = set()
for item in [x.strip() for x in _env_task_mode_filter.split(",") if x.strip()]:
    if ":" in item:
        case_id, mode = item.split(":", 1)
    elif "/" in item:
        case_id, mode = item.split("/", 1)
    else:
        raise ValueError(
            f"Invalid TASK_MODE_FILTER item {item!r}; expected CASE:MODE"
        )
    TASK_MODE_FILTER.add((case_id.strip(), mode.strip()))

# 输入模态配置。可通过环境变量覆盖，逗号分隔:
#   ENABLE_MODALITIES=Text,MD,Sketch,Image,Video python generate_html.py
# 模态说明:
#   Text  — 纯文本 prompt（只传 input.json 的 user_prompt）
#   MD    — 文本 + 任务描述 markdown 文件
#   Sketch — 文本 + 草图截图
#   Image — 文本 + 页面截图
#   Video — 文本 + 交互视频 + 可选 stitched assets 拼图
_env_modalities = os.getenv("ENABLE_MODALITIES", "").strip()
if _env_modalities:
    ENABLE_MODALITIES = [m.strip() for m in _env_modalities.split(",") if m.strip()]
else:
    ENABLE_MODALITIES = ["Text", "MD", "Sketch", "Image", "Video"]

_OUTPUT_FORMAT_RULE = (
    "- Output ONLY the complete HTML document starting with `<!DOCTYPE html>` "
    "and ending with `</html>`. Do NOT wrap the response in markdown code "
    "fences (no ```html or ```), do NOT include any explanation, prose, or "
    "commentary before or after the HTML. The raw response will be saved "
    "directly as an `.html` file."
)

FIXED_RULE_CONTRACT = f"""
[GLOBAL RULE CONTRACT]
- Do not use native alert(), confirm() or prompt() functions. All notifications, pop-ups and dialogs must be rendered as custom DOM elements (such as modals, tooltips) inside the HTML body.
- All images used must be from Unsplash.
- All initial content (lists, feeds, history panels) must be rendered synchronously on page load. Do NOT use setTimeout, requestAnimationFrame, or skeleton loaders for initial data. Mock data must be present in DOM before first paint.
- If a requirement explicitly involves multi-user, collaborative, remote, or cross-client behavior, implement only the specific required collaborative state, indicators, or deterministic in-page simulation controls needed for that behavior. Do NOT invent unrelated participant-joining flows, generic multi-user demo controls, or auto-triggered simulations unless explicitly required.
- Do NOT pre-populate participant lists, message threads, or shared collaborative state unless the prompt or Test Data Contract explicitly specifies that initial state.
{_OUTPUT_FORMAT_RULE}
"""

FIXED_RULE_CONTRACT_VIDEO = f"""
[GLOBAL RULE CONTRACT]
- Do not use native alert(), confirm() or prompt() functions. All notifications, pop-ups and dialogs must be rendered as custom DOM elements (such as modals, tooltips) inside the HTML body.
- All initial content (lists, feeds, history panels) must be rendered synchronously on page load. Do NOT use setTimeout, requestAnimationFrame, or skeleton loaders for initial data. Mock data must be present in DOM before first paint.
- If a requirement explicitly involves multi-user, collaborative, remote, or cross-client behavior, implement only the specific required collaborative state, indicators, or deterministic in-page simulation controls needed for that behavior. Do NOT invent unrelated participant-joining flows, generic multi-user demo controls, or auto-triggered simulations unless explicitly required.
- Do NOT pre-populate participant lists, message threads, or shared collaborative state unless the prompt or Test Data Contract explicitly specifies that initial state.
- If a stitched-assets image is provided, every `<img>` src MUST start with the literal prefix `__PLACEHOLDER_ASSETS_BASE_DIR__/` followed by the exact filename shown under the corresponding asset in that stitched-assets image (e.g. `src="__PLACEHOLDER_ASSETS_BASE_DIR__/asset001.png"`). Do NOT use external URLs and do NOT invent filenames — pick only from the labelled assets.
- If no stitched-assets image is provided, it means this task does not require image assets. Do NOT invent `__PLACEHOLDER_ASSETS_BASE_DIR__` paths, asset filenames, or external image URLs.
{_OUTPUT_FORMAT_RULE}
"""

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
USE_STREAM = os.getenv("USE_STREAM", "1").strip().lower() not in ("0", "false", "no", "off")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# 去掉 prompt 中的 Test Data Contract 段落（实验开关）:
#   NO_CONTRACT=1 python generate_html.py
# 适用于探究：去掉 contract 后模型是否生成更多样化的实现
NO_CONTRACT = os.getenv("NO_CONTRACT", "0").strip() not in ("0", "false", "no")

def _parse_optional_int_limit(name: str, default: int) -> Optional[int]:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in ("none", "no", "off", "unlimited"):
        return None
    value = int(raw)
    if value <= 0:
        return None
    return value


def _limit_label(value: Optional[int]) -> str:
    return "none" if value is None else str(value)


# Video frame-extraction settings (for models without native video input)
VIDEO_FPS = 1                 # sample rate
VIDEO_MAX_FRAMES = _parse_optional_int_limit("VIDEO_MAX_FRAMES", 64)
# Optional provider-level cap for total image parts. Disabled by default so the
# video rule is exactly: sample at 1 fps, then uniformly downsample to 64 frames.
VIDEO_MAX_IMAGE_PARTS = _parse_optional_int_limit("VIDEO_MAX_IMAGE_PARTS", 0)
VIDEO_CLIP_SECONDS = float(os.getenv("VIDEO_CLIP_SECONDS", "0") or "0")

# Which prompt field to use from Video/input.json.
# Keep the req-bearing prompt as the default; use
# Defaults to the video-only prompt; use --video-prompt-field user_prompt for
# the variant that also lists remaining display requirements.
VIDEO_PROMPT_FIELD = os.getenv("VIDEO_PROMPT_FIELD", "video_only_user_prompt").strip() or "video_only_user_prompt"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


# =========================
# ===== PROGRESS TRACKER ==
# =========================

class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.done = 0
        self.success = 0
        self.failed = 0
        self.retry = 0
        self.active = 0
        self.lock = asyncio.Lock()

    async def start_task(self):
        async with self.lock:
            self.active += 1
            self.print_status()

    async def end_task(self, success=True):
        async with self.lock:
            self.active -= 1
            self.done += 1
            if success:
                self.success += 1
            else:
                self.failed += 1
            self.print_status()

    async def add_retry(self):
        async with self.lock:
            self.retry += 1
            self.print_status()

    def print_status(self):
        remaining = self.total - self.done
        msg = (
            f"\r[RUNNING] Total:{self.total} | Done:{self.done} | "
            f"Success:{self.success} | Failed:{self.failed} | "
            f"Retry:{self.retry} | Active:{self.active} | Remaining:{remaining}"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()


# =========================
# ===== LOAD ENV ==========
# =========================

_api_keys_raw = os.getenv("API_KEYS") or os.getenv("OPENAI_API_KEY") or ""
API_KEYS = [k.strip() for k in _api_keys_raw.split(",") if k.strip()]
BASE_URL = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = (os.getenv("MODEL_NAME") or "").strip()

clients = []


def configure_clients() -> None:
    global clients
    if not API_KEYS:
        raise RuntimeError("Set API_KEYS or OPENAI_API_KEY before running inference.")
    if not MODEL_NAME:
        raise RuntimeError("Set MODEL_NAME before running inference.")
    clients = [AsyncOpenAI(api_key=k.strip(), base_url=BASE_URL) for k in API_KEYS]

# =========================
# ===== MEDIA CONFIG ======
# =========================
def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "object", "native"}


_IMG_OBJ = _env_flag("WEBRISE_IMAGE_URL_OBJECT", False)
_VIDEO_OBJ = _env_flag("WEBRISE_VIDEO_URL_OBJECT", False)
_VIDEO_NATIVE = os.getenv("WEBRISE_VIDEO_INPUT_MODE", "native").strip().lower() != "frames"
_SHOW_STARTUP_CONFIG = not any(arg in ("-h", "--help") for arg in sys.argv[1:])
if _SHOW_STARTUP_CONFIG:
    print(f"  [media-config] image_obj={_IMG_OBJ}  video_native={_VIDEO_NATIVE}  video_obj={_VIDEO_OBJ}", flush=True)
    print(
        f"  [request-config] stream={USE_STREAM}  idle_timeout={REQUEST_TIMEOUT:g}s  "
        f"max_retries={MAX_RETRIES}  video_clip_seconds={VIDEO_CLIP_SECONDS:g}  "
        f"video_max_frames={_limit_label(VIDEO_MAX_FRAMES)}  "
        f"video_max_image_parts={_limit_label(VIDEO_MAX_IMAGE_PARTS)}",
        flush=True,
    )


def _wrap_image_url(b64_data_url: str) -> any:
    return {"url": b64_data_url} if _IMG_OBJ else b64_data_url

def _wrap_video_url(b64_data_url: str) -> any:
    return {"url": b64_data_url} if _VIDEO_OBJ else b64_data_url


# =========================
# ===== UTIL ==============
# =========================

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def encode_file_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# =========================
# ===== TASK BUILD ========
# =========================

def build_tasks(video_prompt_field: str = VIDEO_PROMPT_FIELD):
    """Build one task per (case, modality) pair.

    Modality contracts:
      Text  — pure prompt, no file attachment
      MD    — prompt + .md file (task description document)
      Sketch — prompt + sketch image (.png/.jpg)
      Image — prompt + screenshot image (.png/.jpg)
      Video — prompt + .mp4 video + optional stitched_assets.png

    Video tasks always read the system prompt from Video/input.json and read
    the user prompt from `video_prompt_field`.
    """
    _CONTRACT_SEP = "\nTest Data Contract:"

    def _strip_contract(prompt: str) -> str:
        idx = prompt.find(_CONTRACT_SEP)
        return prompt[:idx].rstrip() if idx != -1 else prompt

    def _resolve_existing_path(raw: str, bases: list[Path]) -> Path:
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p.resolve()
        for base in bases:
            cand = (base / p).resolve()
            if cand.exists():
                return cand
        return (REPO_ROOT / p).resolve()

    def _manifest_sources() -> list[tuple[Path, set[str]]]:
        if not TASK_MANIFEST:
            return []

        manifest_path = _resolve_existing_path(
            TASK_MANIFEST,
            [Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent, REPO_ROOT],
        )
        if not manifest_path.exists():
            raise FileNotFoundError(f"TASK_MANIFEST not found: {manifest_path}")

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            raw_items = data.get("tasks") or data.get("task_ids") or data.get("anomalies") or []
        else:
            raise ValueError("TASK_MANIFEST must be a JSON object or array")

        groups: dict[Path, set[str]] = {}
        for item in raw_items:
            if isinstance(item, str):
                case = item
                data_root = ROOT_INPUT_DIR
            elif isinstance(item, dict):
                case = (
                    item.get("task_id")
                    or item.get("case")
                    or item.get("case_id")
                    or item.get("id")
                )
                data_root = (
                    item.get("data_root")
                    or item.get("input_root")
                    or item.get("root_input_dir")
                    or item.get("root")
                    or ROOT_INPUT_DIR
                )
            else:
                continue
            if not case:
                continue
            case = str(case).strip()
            if CASE_FILTER and case not in CASE_FILTER:
                continue
            root = _resolve_existing_path(
                str(data_root),
                [Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent, REPO_ROOT, manifest_path.parent],
            )
            groups.setdefault(root, set()).add(case)

        return sorted(groups.items(), key=lambda x: str(x[0]))

    manifest_sources = _manifest_sources()
    if manifest_sources:
        sources = manifest_sources
    else:
        if not ROOT_INPUT_DIR:
            raise ValueError(
                "Set ROOT_INPUT_DIR or WEBRISE_DATA_ROOT to the task data root, "
                "or provide TASK_MANIFEST."
            )
        sources = [(Path(ROOT_INPUT_DIR).resolve(), set())]

    tasks = []

    for root, exact_cases in sources:
        if not root.exists():
            continue

        for case_dir in sorted(root.iterdir()):
            if not case_dir.is_dir():
                continue
            if exact_cases and case_dir.name not in exact_cases:
                continue
            if not exact_cases and CASE_FILTER and not any(f in case_dir.name for f in CASE_FILTER):
                continue

            for mode in ENABLE_MODALITIES:
                mode_dir = case_dir / mode
                if not mode_dir.exists():
                    continue

                input_json_path = mode_dir / "input.json"
                if not input_json_path.exists():
                    continue
                input_json = read_json(input_json_path)

                raw_user_prompt = input_json["user_prompt"]
                user_prompt = _strip_contract(raw_user_prompt) if NO_CONTRACT else raw_user_prompt

                if mode == "Text":
                    # Pure text: only input.json's prompt, no file attachment.
                    tasks.append({
                        "case":          case_dir.name,
                        "mode":          mode,
                        "system_prompt": input_json["system_prompt"] + FIXED_RULE_CONTRACT,
                        "user_prompt":   user_prompt,
                        "file_path":     None,
                        "root_input_dir": str(root),
                    })

                elif mode == "Video":
                    # Video + optional stitched assets image.
                    video_files = [f for f in mode_dir.iterdir() if f.suffix == ".mp4"]
                    if not video_files:
                        continue
                    video_path = video_files[0]
                    stitched_path = case_dir / "assets" / "stitched_assets.png"
                    sys_p = input_json["system_prompt"] + FIXED_RULE_CONTRACT_VIDEO
                    raw_video_prompt = input_json.get(video_prompt_field)
                    if raw_video_prompt is None:
                        raise KeyError(
                            f"{input_json_path} has no prompt field {video_prompt_field!r}"
                        )
                    usr_p = _strip_contract(raw_video_prompt) if NO_CONTRACT else raw_video_prompt
                    tasks.append({
                        "case":          case_dir.name,
                        "mode":          mode,
                        "system_prompt": sys_p,
                        "user_prompt":   usr_p,
                        "file_path":     str(video_path),
                        "assets_path":   str(stitched_path) if stitched_path.exists() else None,
                        "video_prompt_field": video_prompt_field,
                        "root_input_dir": str(root),
                    })

                else:
                    # MD / Sketch / Image — prompt + one attachment file.
                    # MD looks for .md; Sketch/Image look for .png/.jpg/.jpeg
                    if mode == "MD":
                        exts = [".md"]
                    else:
                        exts = [".png", ".jpg", ".jpeg"]
                    file_path = None
                    for f in mode_dir.iterdir():
                        if f.suffix.lower() in exts:
                            file_path = f
                            break
                    if file_path is None:
                        continue
                    tasks.append({
                        "case":          case_dir.name,
                        "mode":          mode,
                        "system_prompt": input_json["system_prompt"] + FIXED_RULE_CONTRACT,
                        "user_prompt":   user_prompt,
                        "file_path":     str(file_path),
                        "root_input_dir": str(root),
                    })

    if TASK_MODE_FILTER:
        tasks = [
            t for t in tasks
            if (t["case"], t["mode"]) in TASK_MODE_FILTER
        ]

    return tasks


# =========================
# ===== VIDEO → FRAMES ====
# =========================

def extract_video_frames(video_path: str, fps: int = VIDEO_FPS,
                         max_frames: Optional[int] = VIDEO_MAX_FRAMES) -> List[str]:
    """Extract frames at `fps` from a video.

    Videos exceeding `max_frames` extracted frames are uniformly downsampled
    to meet this limit. Returns base64-encoded PNG strings. Uses ffmpeg; if
    unavailable, raises RuntimeError.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found — required for video frame extraction")
    if max_frames is not None and max_frames <= 0:
        return []

    with tempfile.TemporaryDirectory() as td:
        out_pattern = os.path.join(td, "frame_%04d.png")
        # First pass: extract at the requested fps
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}", out_pattern],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        frames = sorted(Path(td).glob("frame_*.png"))
        # Preserve coverage across the whole video when the 1-fps sample exceeds
        # the request budget.
        if max_frames is not None and len(frames) > max_frames:
            if max_frames == 1:
                frames = [frames[0]]
            else:
                indices = [
                    round(i * (len(frames) - 1) / (max_frames - 1))
                    for i in range(max_frames)
                ]
                frames = [frames[i] for i in indices]
        return [encode_file_base64(str(f)) for f in frames]


def clip_video_for_request(video_path: str, seconds: float, out_dir: Path) -> str:
    """Create a temporary first-N-seconds MP4 for request packing."""
    if seconds <= 0:
        return video_path
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found — required for VIDEO_CLIP_SECONDS")

    src = Path(video_path)
    out = out_dir / f"{src.stem}_first_{seconds:g}s.mp4"
    fast_cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-t", f"{seconds:g}",
        "-map", "0",
        "-c", "copy",
        "-movflags", "+faststart",
        str(out),
    ]
    try:
        subprocess.run(
            fast_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        # Some files cannot be stream-copied cleanly at arbitrary cut points.
        # Fall back to a broadly-compatible transcode.
        transcode_cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-t", f"{seconds:g}",
            "-map", "0:v:0",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out),
        ]
        subprocess.run(
            transcode_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return str(out)


# =========================
# ===== LOGGING ===========
# =========================

# Resolved once per run by _init_log_path().
_LOG_PATH: Optional[Path] = None


def _find_model_logs() -> list[Path]:
    """Find all jsonl logs for this MODEL_NAME, sorted by filename timestamp."""
    log_dir = Path(LOG_DIR)
    if not log_dir.is_dir():
        return []
    return sorted(log_dir.glob(f"{MODEL_NAME}_*.jsonl"))


def _find_latest_log() -> Optional[Path]:
    """Find the most recent jsonl for this MODEL_NAME (by filename timestamp)."""
    candidates = _find_model_logs()
    return candidates[-1] if candidates else None


def _parse_success_keys(log_paths: list[Path]) -> set:
    """Read jsonl logs and return {(case, mode)} for all 'success' entries."""
    done = set()
    for log_path in log_paths:
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("status") == "success":
                    done.add((entry["case"], entry["mode"]))
        except Exception:
            pass
    return done


def _init_log_path(force: bool) -> Path:
    """Determine the log file path for this run.
    --force → new file (new timestamp).
    resume  → latest existing file (append to it).
    """
    global _LOG_PATH
    ensure_dir(LOG_DIR)
    if force:
        _LOG_PATH = Path(LOG_DIR) / f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    else:
        existing = _find_latest_log()
        _LOG_PATH = existing if existing else Path(LOG_DIR) / f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    return _LOG_PATH


def save_log(entry: Dict):
    ensure_dir(LOG_DIR)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


import re as _re


def _format_exception(exc: Exception) -> str:
    """Return a useful one-line exception string for jsonl logs.

    Some exceptions, especially asyncio.TimeoutError, stringify to an empty
    string, which makes retry logs look like they failed without a reason.
    """
    name = type(exc).__name__
    text = str(exc).strip()
    if text:
        return f"{name}: {text}"
    return f"{name}: {repr(exc)}"


def clean_html_response(text: str) -> str:
    """Defensive post-processing: strip markdown code fences and any
    pre/post prose so the saved file is a self-contained HTML document.
    Falls back to the original text if no recognisable HTML frame is found."""
    if not text:
        return text
    s = text.strip()

    # 1) Strip an outer markdown fence if the whole response is wrapped.
    #    Matches ```html ... ```, ``` ... ``` (with or without language tag).
    m = _re.match(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n```\s*$", s, flags=_re.DOTALL)
    if m:
        s = m.group(1).strip()

    # 2) If the response has prose before/after the HTML, extract the block
    #    from the first <!DOCTYPE html> (case-insensitive) to the last </html>.
    lower = s.lower()
    doctype_idx = lower.find("<!doctype html")
    close_idx = lower.rfind("</html>")
    if doctype_idx != -1 and close_idx != -1 and close_idx > doctype_idx:
        return s[doctype_idx : close_idx + len("</html>")]

    # 3) If no DOCTYPE but there's <html>...</html>, take that slice.
    html_open = lower.find("<html")
    if html_open != -1 and close_idx != -1 and close_idx > html_open:
        return s[html_open : close_idx + len("</html>")]

    # 4) Give up and return whatever we have after fence stripping.
    return s


def validate_html_response(html: str) -> None:
    """Reject empty or incomplete HTML before writing an output file."""
    s = (html or "").strip()
    if not s:
        raise ValueError("empty HTML response after cleaning")
    lower = s.lower()
    if "<html" not in lower or "</html>" not in lower:
        raise ValueError("response does not contain a complete <html>...</html> document")


def extract_usage(response):
    try:
        u = response.usage
        return {
            "prompt": u.prompt_tokens,
            "completion": u.completion_tokens,
            "total": u.total_tokens
        }
    except:
        return {
            "prompt": None,
            "completion": None,
            "total": None
        }


def _usage_to_dict(usage):
    if not usage:
        return {
            "prompt": None,
            "completion": None,
            "total": None
        }
    return {
        "prompt": getattr(usage, "prompt_tokens", None),
        "completion": getattr(usage, "completion_tokens", None),
        "total": getattr(usage, "total_tokens", None)
    }


def _chunk_text(chunk) -> str:
    try:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", "") if delta is not None else ""
        if content is None:
            return ""
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)
    except Exception:
        return ""


async def _collect_stream(stream):
    """Collect a streaming chat completion.

    REQUEST_TIMEOUT is an idle timeout: initial stream creation and every
    chunk wait must make progress within this window, but long generations can
    exceed it as long as chunks keep arriving.
    """
    parts = []
    usage = None
    ait = stream.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(ait.__anext__(), timeout=REQUEST_TIMEOUT)
        except StopAsyncIteration:
            break
        usage = getattr(chunk, "usage", None) or usage
        text = _chunk_text(chunk)
        if text:
            parts.append(text)
    return "".join(parts), _usage_to_dict(usage)


async def create_chat_completion(client, messages):
    if USE_STREAM:
        stream = await asyncio.wait_for(
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
            ),
            timeout=REQUEST_TIMEOUT,
        )
        html_raw, tokens = await _collect_stream(stream)
        return html_raw or "", tokens

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        ),
        timeout=REQUEST_TIMEOUT,
    )
    return response.choices[0].message.content or "", extract_usage(response)


def _build_video_native_content(video_path: str, assets_path: Optional[str]) -> list:
    """Content parts using native video_url. Format (bare-string vs object)
    is controlled by _wrap_video_url / _wrap_image_url (per-model config)."""
    video_b64 = encode_file_base64(video_path)
    parts = [{
        "type": "video_url",
        "video_url": _wrap_video_url(f"data:video/mp4;base64,{video_b64}")
    }]
    if assets_path:
        parts.append({
            "type": "image_url",
            "image_url": _wrap_image_url(f"data:image/png;base64,{encode_file_base64(assets_path)}")
        })
    return parts


def _build_video_frames_content(video_path: str, assets_path: Optional[str]) -> list:
    """Content parts using 1-fps frame extraction. Works everywhere.
    Image format (bare-string vs object) follows _wrap_image_url."""
    parts = []
    asset_slots = 1 if assets_path else 0
    limits = []
    if VIDEO_MAX_FRAMES is not None:
        limits.append(VIDEO_MAX_FRAMES)
    if VIDEO_MAX_IMAGE_PARTS is not None:
        limits.append(max(0, VIDEO_MAX_IMAGE_PARTS - asset_slots))
    frame_budget = min(limits) if limits else None
    frames = extract_video_frames(video_path, max_frames=frame_budget)
    for fb in frames:
        parts.append({"type": "image_url", "image_url": _wrap_image_url(f"data:image/png;base64,{fb}")})
    if assets_path:
        parts.append({
            "type": "image_url",
            "image_url": _wrap_image_url(f"data:image/png;base64,{encode_file_base64(assets_path)}")
        })
    return parts


def _summarize_media_parts(content: list, used_video_mode: Optional[str],
                           assets_path: Optional[str] = None) -> dict:
    types = [p.get("type") for p in content if isinstance(p, dict)]
    video_parts = [p for p in content if isinstance(p, dict) and p.get("type") == "video_url"]
    image_parts = [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]
    asset_attached = bool(assets_path)
    frame_count = None
    if used_video_mode == "frames":
        frame_count = len(image_parts) - (1 if asset_attached else 0)
    return {
        "model_config": {
            "image_url_format": "object" if _IMG_OBJ else "bare-string",
            "video_native": _VIDEO_NATIVE,
            "video_url_format": "object" if _VIDEO_OBJ else "bare-string",
        },
        "video_mode": used_video_mode,
        "content_part_types": types,
        "video_part_count": len(video_parts),
        "image_part_count": len(image_parts),
        "frame_count": frame_count,
        "video_frame_limits": {
            "video_max_frames": _limit_label(VIDEO_MAX_FRAMES),
            "video_max_image_parts": _limit_label(VIDEO_MAX_IMAGE_PARTS),
        },
        "assets_attached": asset_attached,
        "assets_path": assets_path,
    }


# =========================
# ===== CORE CALL =========
# =========================

async def call_mllm(task, clients, sem, tracker: ProgressTracker):

    async with sem:
        print(f"\n  [→] {task['case']}  [{task['mode']}]  starting...", flush=True)
        await tracker.start_task()

        start_time = time.time()

        attempts = max(1, MAX_RETRIES)
        last_error = None
        for attempt in range(attempts):
            client = clients[attempt % len(clients)]
            html_raw = ""

            try:
                file_path = task.get("file_path")
                suffix = Path(file_path).suffix if file_path else None
                content = []
                used_video_mode: Optional[str] = None
                assets_path: Optional[str] = None
                media_summary: dict = {}

                if file_path is None:
                    pass

                elif suffix == ".md":
                    content.append({"type": "text", "text": read_text(file_path)})

                elif suffix in [".png", ".jpg", ".jpeg"]:
                    content.append({
                        "type": "image_url",
                        "image_url": _wrap_image_url(f"data:image/png;base64,{encode_file_base64(file_path)}")
                    })

                elif suffix == ".mp4":
                    assets_path = task.get("assets_path")
                    video_path_for_request = task["file_path"]
                    if VIDEO_CLIP_SECONDS > 0:
                        with tempfile.TemporaryDirectory() as td:
                            video_path_for_request = clip_video_for_request(
                                task["file_path"],
                                VIDEO_CLIP_SECONDS,
                                Path(td),
                            )
                            if _VIDEO_NATIVE:
                                content = _build_video_native_content(video_path_for_request, assets_path)
                                used_video_mode = "native"
                            else:
                                content = _build_video_frames_content(video_path_for_request, assets_path)
                                used_video_mode = "frames"
                    elif _VIDEO_NATIVE:
                        content = _build_video_native_content(video_path_for_request, assets_path)
                        used_video_mode = "native"
                    else:
                        content = _build_video_frames_content(video_path_for_request, assets_path)
                        used_video_mode = "frames"
                    media_summary = _summarize_media_parts(
                        content, used_video_mode, assets_path
                    )
                    if VIDEO_CLIP_SECONDS > 0:
                        media_summary["video_clip_seconds"] = VIDEO_CLIP_SECONDS
                        media_summary["video_source_path"] = task["file_path"]

                messages = [
                    {"role": "system", "content": task["system_prompt"]},
                    {"role": "user", "content": [{"type": "text", "text": task["user_prompt"]}] + content}
                ]

                html_raw, tokens = await create_chat_completion(client, messages)
                html = clean_html_response(html_raw)
                validate_html_response(html)

                save_dir = Path(OUTPUT_DIR) / MODEL_NAME / task["mode"]
                ensure_dir(save_dir)

                output_path = save_dir / f"{MODEL_NAME}_{task['case']}_{task['mode']}.html"

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html)

                latency = time.time() - start_time

                # ===== 记录成功日志（含原始回复）=====
                success_entry = {
                    "time": now(),
                    "case": task["case"],
                    "mode": task["mode"],
                    "status": "success",
                    "attempt": attempt + 1,
                    "latency": round(latency, 3),
                    "tokens": tokens,
                    "output_path": str(output_path),
                    "video_mode": used_video_mode,
                    "media_summary": media_summary,
                    "stream": USE_STREAM,
                    "request_timeout": REQUEST_TIMEOUT,
                    "max_retries": MAX_RETRIES,
                    "root_input_dir": task.get("root_input_dir") or ROOT_INPUT_DIR,
                    "video_prompt_field": task.get("video_prompt_field"),
                    "raw_response": html_raw,
                    "cleaned": html != html_raw,
                    "error": None
                }
                save_log(success_entry)

                print(f"\n  [✔] {task['case']}  [{task['mode']}]  done ({latency:.2f}s)")

                await tracker.end_task(success=True)
                return

            except Exception as e:
                error_text = _format_exception(e)
                last_error = error_text
                await tracker.add_retry()

                # ===== 记录失败尝试 =====
                retry_entry = {
                    "time": now(),
                    "case": task["case"],
                    "mode": task["mode"],
                    "status": "retry",
                    "attempt": attempt + 1,
                    "latency": round(time.time() - start_time, 3),
                    "tokens": None,
                    "output_path": None,
                    "video_mode": used_video_mode,
                    "media_summary": media_summary,
                    "stream": USE_STREAM,
                    "request_timeout": REQUEST_TIMEOUT,
                    "max_retries": MAX_RETRIES,
                    "root_input_dir": task.get("root_input_dir") or ROOT_INPUT_DIR,
                    "video_prompt_field": task.get("video_prompt_field"),
                    "error": error_text
                }
                save_log(retry_entry)

                await asyncio.sleep(2 ** attempt)

        # ===== 最终失败 =====
        save_log({
            "time": now(),
            "case": task["case"],
            "mode": task["mode"],
            "status": "failed",
            "attempt": attempts,
            "latency": round(time.time() - start_time, 3),
            "tokens": None,
            "output_path": None,
            "video_mode": None,
            "media_summary": {},
            "stream": USE_STREAM,
            "request_timeout": REQUEST_TIMEOUT,
            "max_retries": MAX_RETRIES,
            "root_input_dir": task.get("root_input_dir") or ROOT_INPUT_DIR,
            "video_prompt_field": None,
            "error": f"All retries failed: {last_error}" if last_error else "All retries failed"
        })

        print(f"\n  [✖] {task['case']}  [{task['mode']}]  failed")
        await tracker.end_task(success=False)


# =========================
# ===== SCHEDULER =========
# =========================

async def run_all(force: bool = False,
                  video_prompt_field: str = VIDEO_PROMPT_FIELD,
                  preview_prompts: bool = False):

    all_tasks = build_tasks(
        video_prompt_field=video_prompt_field,
    )
    log_path = _init_log_path(force)
    if NO_CONTRACT:
        print("  [NO_CONTRACT=1] Test Data Contract 已从所有 user_prompt 中移除")
    print(f"  Video prompt field: {video_prompt_field}")
    if TASK_MANIFEST:
        roots = sorted({t.get("root_input_dir") or ROOT_INPUT_DIR for t in all_tasks})
        print(f"  Task manifest: {TASK_MANIFEST}")
        print(f"  Manifest roots: {len(roots)}")
        for root in roots:
            print(f"    - {root}")

    # ── Resume: skip already-successful tasks (only if output file still exists) ──
    if not force:
        resume_logs = _find_model_logs()
        done = _parse_success_keys(resume_logs)
        save_base = Path(OUTPUT_DIR) / MODEL_NAME
        def _output_exists(t: dict) -> bool:
            p = save_base / t["mode"] / f"{MODEL_NAME}_{t['case']}_{t['mode']}.html"
            return p.exists()
        tasks = [t for t in all_tasks
                 if (t["case"], t["mode"]) not in done or not _output_exists(t)]
        skipped = len(all_tasks) - len(tasks)
        if skipped:
            print(
                f"\n  Resume mode: {skipped} already succeeded "
                f"(from {len(resume_logs)} log(s), appending to {log_path.name}), skipping"
            )
    else:
        tasks = all_tasks
        skipped = 0

    # ── Print task summary breakdown ──
    from collections import Counter
    mode_counts = Counter(t["mode"] for t in tasks)
    case_count = len(set(t["case"] for t in tasks))
    print(f"\n  Tasks: {len(tasks)} to run  ({case_count} cases)")
    if skipped:
        print(f"  (skipped {skipped} already done)")
    for m in ENABLE_MODALITIES:
        c = mode_counts.get(m, 0)
        if c:
            print(f"    {m:8s}: {c}")
    print(f"  Log: {log_path}")
    print()

    if not tasks:
        print("  Nothing to run — all tasks already succeeded!")
        return

    if preview_prompts:
        print("\n===== PROMPT PREVIEW =====")
        for t in tasks:
            if t["mode"] != "Video":
                continue
            print(f"\n--- {t['case']} [{t['mode']}] ---")
            print(f"video: {t['file_path']}")
            if t.get("assets_path"):
                print(f"assets: {t['assets_path']}")
            print(
                "media format: "
                f"video={'native' if _VIDEO_NATIVE else 'frames'}; "
                f"video_url={'object' if _VIDEO_OBJ else 'bare-string'}; "
                f"image_url={'object' if _IMG_OBJ else 'bare-string'}; "
                f"prompt_field={t.get('video_prompt_field')}"
            )
            print("[system_prompt]")
            print(t["system_prompt"])
            print("[user_prompt]")
            print(t["user_prompt"])
        print("\n===== END PREVIEW =====")
        return

    tracker = ProgressTracker(len(tasks))
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    await asyncio.gather(*[
        call_mllm(t, clients, sem, tracker)
        for t in tasks
    ])

    print("\n\n===== ALL DONE =====")


# =========================
# ===== MAIN ==============
# =========================

def main():
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Re-run all tasks from scratch (new log file). "
                        "Default: resume from latest log, skip successes.")
    p.add_argument("--video-prompt-field", default=VIDEO_PROMPT_FIELD,
                   help="Video/input.json prompt field to use for Video mode "
                        "(default: video_only_user_prompt).")
    p.add_argument("--preview-prompts", action="store_true",
                   help="Build tasks and print Video prompts, then exit "
                        "without calling the model.")
    args = p.parse_args()
    if not args.preview_prompts:
        configure_clients()
    asyncio.run(run_all(force=args.force,
                        video_prompt_field=args.video_prompt_field,
                        preview_prompts=args.preview_prompts))


if __name__ == "__main__":
    main()
