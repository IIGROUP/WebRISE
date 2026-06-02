#!/usr/bin/env python3
"""Build Image-modality input directories for WebRISE tasks.

Steps:
  select      choose the most representative agent-eval screenshot
  screenshot  copy the selected screenshot into Image/
  tag         label which explicit requirements are visible in the image
  input       write Image/input.json with the remaining text requirements
  all         run all steps in order

Usage:
  python build_image_inputs.py --data-root path/to/dataset/data
  python build_image_inputs.py TASK_ID --data-root path/to/dataset/data --force
  python build_image_inputs.py --step tag --data-root path/to/dataset/data
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import shutil
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = os.environ.get("WEBRISE_DATA_ROOT")

API_KEY = (
    os.environ.get("OPENAI_API_KEY", "").strip()
)
BASE_URL = (
    os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
)
DEFAULT_MODEL = os.environ.get("WEBRISE_IMAGE_MODEL", "").strip()

TARGET_NAME = "Image"

# Maximum number of screenshots sent to the selection model.
MAX_SELECT_IMAGES = 12

# Suffix replacement for modality-specific system prompts.
_OLD_SUFFIX_PATTERNS = [
    "the page layout must strictly follow the MD documentation.",
    "strictly follow the MD documentation.",
    "strictly follow the MD documentation",
]
_NEW_SUFFIX = (
    "the webpage layout and appearance should reference the provided image example."
)


def patch_system_prompt(original: str) -> str:
    for pat in _OLD_SUFFIX_PATTERNS:
        if pat in original:
            return original.replace(pat, _NEW_SUFFIX)
    return original.rstrip(". ") + ". " + _NEW_SUFFIX


def find_largest_screenshot(screenshots_dir: Path) -> Path | None:
    pngs = list(screenshots_dir.glob("*.png"))
    return max(pngs, key=lambda p: p.stat().st_size) if pngs else None


def load_explicit_reqs_from_icg(task_dir: Path) -> list[dict]:
    """Load explicit requirements from task_dir/icg.json."""
    icg_path = task_dir / "icg.json"
    if not icg_path.exists():
        return []
    data = json.loads(icg_path.read_text(encoding="utf-8"))
    return [
        {"full_id": r["req_id"], "content_en": r["content_en"]}
        for r in data.get("requirements_coverage", [])
        if r.get("intent") == "explicit"
    ]


def find_screenshots_dir(task_dir: Path) -> Path | None:
    """Find the most suitable agent_eval screenshots directory."""
    agent_eval = task_dir / "agent_eval"
    if not agent_eval.exists():
        return None
    subdirs = sorted([d for d in agent_eval.iterdir() if d.is_dir()])
    for candidates in (
        [d for d in subdirs if d.name.endswith("_base")],
        [d for d in subdirs if not d.name.endswith("_base")],
    ):
        for d in reversed(candidates):
            ss = d / "screenshots"
            if ss.exists() and any(ss.glob("*.png")):
                return ss
    return None


def split_user_prompt(user_prompt: str) -> tuple[str, str, str]:
    """Return (intro, requirements section, contract section)."""
    m = re.search(r"\n\nTest Data Contract:", user_prompt)
    if m:
        before   = user_prompt[:m.start()]
        contract = user_prompt[m.start() + 2:]
    else:
        before   = user_prompt
        contract = ""

    dot = before.find(". ")
    if dot != -1:
        intro = before[: dot + 1]
        reqs  = before[dot + 2:].strip()
    else:
        intro = before
        reqs  = ""
    return intro, reqs, contract


def _encode_image(path: Path) -> tuple[str, str]:
    b64  = base64.b64encode(path.read_bytes()).decode()
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return mime, b64


def _candidate_screenshots(screenshots_dir: Path, max_n: int) -> list[Path]:
    post = sorted(screenshots_dir.glob("*_post.png"))
    init = screenshots_dir / "S0_initial.png"

    candidates: list[Path] = []
    if init.exists():
        candidates.append(init)
    candidates.extend(post)

    if not candidates:
        candidates = sorted(screenshots_dir.glob("*.png"))

    if len(candidates) <= max_n:
        return candidates

    step  = (len(candidates) - 1) / (max_n - 1)
    idxes = sorted({round(i * step) for i in range(max_n)})
    return [candidates[i] for i in idxes]


def _call_vlm(client, model: str, system: str,
              image_parts: list[dict], text: str,
              max_tokens: int = 8192) -> str | None:
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": image_parts + [{"type": "text", "text": text}],
        },
    ]
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                messages=messages,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [attempt {attempt}/3] API error: {e}")
    return None


# Step 0: select the representative screenshot.

_SELECT_SYSTEM = """\
You are a UI analyst. You will be shown multiple webpage screenshots (labeled [1], [2], ...) \
and a list of explicit UI requirements.

Your task: choose the SINGLE screenshot that is most representative — i.e., the one whose \
visible content covers the greatest number of the explicit requirements listed below.

Prefer screenshots that show:
- Filled-in interactive states (forms with data, open modals, expanded panels) over empty/initial states
- The most distinct UI components mentioned in the requirements
- Clear, rich content rather than blank or loading states
- For create / compose / edit / draft features: prefer the state AFTER clicking the compose or
  edit button, where the editor form / rich-text area is fully open and populated with content —
  NOT the initial entry point (button only) or an empty form
- Screens that reveal the maximum variety of interactive controls (dropdowns, toggles, input
  fields, action buttons) that appear in the requirements

Return ONLY valid JSON, no markdown fences, no explanation:
{
  "selected_index": <1-based integer>,
  "selected_filename": "<exact filename>",
  "reason": "<one or two sentences in English explaining why this screenshot best covers the requirements>"
}
"""


def step_select(task_dir: Path, task_id: str,
                explicit_reqs: list[dict], model: str,
                dry_run: bool, force: bool) -> str | None:
    target_dir = task_dir / TARGET_NAME
    sel_path   = target_dir / "screenshot_selection.json"
    screenshots_dir = find_screenshots_dir(task_dir)

    if sel_path.exists() and not force:
        data  = json.loads(sel_path.read_text(encoding="utf-8"))
        fname = data.get("selected_filename", "")
        print(f"    [skip] screenshot_selection.json exists → {fname}")
        return fname

    if screenshots_dir is None:
        print(f"    [WARN] no agent_eval screenshots found, skip select step")
        return None

    candidates = _candidate_screenshots(screenshots_dir, MAX_SELECT_IMAGES)
    if not candidates:
        print(f"    [WARN] no candidate screenshots found")
        return None

    if len(candidates) == 1:
        fname = candidates[0].name
        print(f"    [select] only 1 candidate → {fname}")
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            sel_path.write_text(
                json.dumps({
                    "selected_index": 1,
                    "selected_filename": fname,
                    "reason": "Only one candidate screenshot available.",
                    "candidates": [c.name for c in candidates],
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return fname

    reqs_str = "\n".join(
        f"  {r['full_id']}: {r['content_en']}"
        for r in explicit_reqs
    )
    user_text = (
        f"Task: {task_id}\n\n"
        f"Explicit requirements:\n{reqs_str}\n\n"
        f"Screenshots provided (in order): "
        + ", ".join(f"[{i+1}] {c.name}" for i, c in enumerate(candidates))
    )

    image_parts: list[dict] = []
    for i, path in enumerate(candidates):
        mime, b64 = _encode_image(path)
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
        image_parts.append({"type": "text", "text": f"[{i+1}] {path.name}"})

    print(f"    [select] calling {model} with {len(candidates)} screenshots ...")
    if dry_run:
        print(f"    [dry-run] would save screenshot_selection.json")
        return None

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    raw = _call_vlm(client, model, _SELECT_SYSTEM, image_parts, user_text)
    if raw is None:
        print(f"    [ERROR] select API call failed")
        return None

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    [ERROR] select: invalid JSON: {e}")
        print(f"    raw: {raw[:200]}")
        return None

    fname  = result.get("selected_filename", "")
    reason = result.get("reason", "")
    print(f"    [select] chose: {fname}  — {reason}")

    result["candidates"] = [c.name for c in candidates]
    target_dir.mkdir(parents=True, exist_ok=True)
    sel_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return fname


# Step 1: copy the selected screenshot.

def step_screenshot(task_dir: Path, dry_run: bool, force: bool) -> Path | None:
    target_dir  = task_dir / TARGET_NAME
    sel_path    = target_dir / "screenshot_selection.json"
    screenshots_dir = find_screenshots_dir(task_dir)

    chosen_name: str | None = None
    if sel_path.exists():
        sel_data    = json.loads(sel_path.read_text(encoding="utf-8"))
        chosen_name = sel_data.get("selected_filename")

    if chosen_name and screenshots_dir:
        src = screenshots_dir / chosen_name
        if not src.exists():
            print(f"    [WARN] selected screenshot not found: {chosen_name}, fallback to largest")
            chosen_name = None

    if not chosen_name:
        if screenshots_dir is None:
            print(f"    [WARN] no agent_eval screenshots/, skip screenshot step")
            return None
        src = find_largest_screenshot(screenshots_dir)
        if src is None:
            print(f"    [WARN] no PNG in screenshots/")
            return None
        chosen_name = src.name
    else:
        src = screenshots_dir / chosen_name

    dest = target_dir / chosen_name
    if dest.exists() and not force:
        print(f"    [skip] screenshot already exists: {chosen_name}")
        return dest

    size_kb = src.stat().st_size // 1024
    print(f"    [screenshot] {chosen_name} ({size_kb} KB) -> {TARGET_NAME}/")
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return dest


# Step 2: label explicit requirements visible in the screenshot.

_TAG_SYSTEM = """\
You are a careful UI analyst. You will be shown a webpage screenshot and a list of explicit UI \
requirements (with IDs and English descriptions). You will also receive the English requirements \
text section from the code-generation prompt.

For each explicit requirement, decide whether a developer could FULLY INFER it just by looking \
at the screenshot — i.e., the requirement is visually self-evident.

Rules:
- visible_in_image = true  → the requirement describes something clearly observable in the screenshot
  (the UI element exists, its type is apparent, its label/option names are readable, its layout is clear)
- visible_in_image = false → the requirement describes behavior, data contracts, state changes, counts,
  triggers, or any information that CANNOT be read directly from a single static screenshot

ALWAYS mark visible_in_image = false — regardless of what the screenshot shows:
1. Error boundary requirements: requirements that enumerate multiple distinct error types, failure
   codes, or boundary conditions. Even if the screenshot shows ONE error state, the full set of
   error conditions and their specific triggers is not inferrable from a static image.
2. Trigger conditions: requirements that describe WHAT action, input value, or state CAUSES an
   error or special behavior (e.g. "upload fails when file exceeds 10 MB", "network error on
   third retry"). The trigger logic is behavioral, not visual.
3. Validation rules with specific thresholds: requirements specifying exact numeric limits,
   character counts, or boundary values (e.g. "title must be ≤ 20 characters").
4. Multi-case enumerations: any requirement that lists ≥ 2 named cases/scenarios, even if
   one case is currently shown in the screenshot.

Also provide a filtered version of the English requirements text:
- Remove sentences that correspond to requirements where visible_in_image = true
- Keep sentences that correspond to requirements where visible_in_image = false
- Preserve the original sentence wording exactly for kept sentences; do NOT paraphrase
- If all requirements are visible, return an empty string for filtered_reqs_en

Return ONLY valid JSON, no markdown fences, no explanation.
{
  "requirements": [
    {
      "req_id": "<full_id>",
      "content_en": "<original English — copy exactly>",
      "visible_in_image": true or false,
      "reason": "<one sentence in English>"
    }
  ],
  "filtered_reqs_en": "<English requirements sentences that are NOT visually evident; empty string if none>"
}
"""


def step_tag(task_dir: Path, task_id: str,
             explicit_reqs: list[dict], model: str,
             dry_run: bool, force: bool) -> dict | None:
    from openai import OpenAI

    target_dir = task_dir / TARGET_NAME
    vis_path   = target_dir / "req_visibility.json"

    if vis_path.exists() and not force:
        print(f"    [skip] req_visibility.json exists")
        return json.loads(vis_path.read_text(encoding="utf-8"))

    # Prefer the copied Image screenshot; otherwise fall back to agent_eval.
    pngs = sorted(target_dir.glob("*.png")) if target_dir.exists() else []
    if pngs:
        screenshot_path = max(pngs, key=lambda p: p.stat().st_size)
    else:
        screenshots_dir = find_screenshots_dir(task_dir)
        screenshot_path = find_largest_screenshot(screenshots_dir) if screenshots_dir else None
        if screenshot_path is None:
            print(f"    [WARN] no screenshot found for tagging, skip")
            return None

    text_input_path = task_dir / "Text" / "input.json"
    if not text_input_path.exists():
        print(f"    [WARN] no Text/input.json, skip tag step")
        return None
    text_data = json.loads(text_input_path.read_text(encoding="utf-8"))
    _, reqs_en, _ = split_user_prompt(text_data["user_prompt"])

    reqs_list_str = "\n".join(
        f"  {r['full_id']}: {r['content_en']}" for r in explicit_reqs
    )
    user_text = (
        f"Task: {task_id}\n\n"
        f"Explicit requirements (with IDs):\n{reqs_list_str}\n\n"
        f"English requirements section from the text prompt:\n{reqs_en}"
    )

    mime, b64 = _encode_image(screenshot_path)
    image_parts = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    ]

    print(f"    [tag] calling {model} with {screenshot_path.name} ...")
    if dry_run:
        print(f"    [dry-run] would save req_visibility.json")
        return None

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    result = None
    for attempt in range(1, 4):
        raw = _call_vlm(client, model, _TAG_SYSTEM, image_parts, user_text)
        if raw is None:
            continue
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            result = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            print(f"    [attempt {attempt}/3] invalid JSON: {e}")
            if attempt == 3:
                print(f"    [ERROR] giving up; raw: {raw[:200]}")
                return None

    if result is None:
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    vis_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    kept  = sum(1 for r in result.get("requirements", []) if not r.get("visible_in_image"))
    total = len(result.get("requirements", []))
    print(f"    [tag] {kept}/{total} reqs kept (not visible in screenshot)")
    return result


# Step 3: write Image/input.json.

def step_input(task_dir: Path, dry_run: bool, force: bool) -> bool:
    target_dir = task_dir / TARGET_NAME
    input_path = target_dir / "input.json"

    if input_path.exists() and not force:
        print(f"    [skip] input.json exists")
        return True

    text_input_path = task_dir / "Text" / "input.json"
    vis_path        = target_dir / "req_visibility.json"

    if not text_input_path.exists():
        print(f"    [WARN] no Text/input.json, skip input step")
        return False
    if not vis_path.exists():
        print(f"    [WARN] no req_visibility.json, run --step tag first")
        return False

    text_data = json.loads(text_input_path.read_text(encoding="utf-8"))
    vis_data  = json.loads(vis_path.read_text(encoding="utf-8"))

    intro, _, contract = split_user_prompt(text_data["user_prompt"])
    filtered_reqs_en  = vis_data.get("filtered_reqs_en", "").strip()

    if filtered_reqs_en:
        new_user_prompt = intro + " " + filtered_reqs_en
    else:
        new_user_prompt = intro
    if contract:
        new_user_prompt = new_user_prompt.rstrip() + "\n\n" + contract

    reqs_data = vis_data.get("requirements", [])
    kept_ids  = [r["req_id"] for r in reqs_data if not r.get("visible_in_image")]
    drop_ids  = [r["req_id"] for r in reqs_data if r.get("visible_in_image")]

    pngs = sorted(target_dir.glob("*.png")) if target_dir.exists() else []
    if not pngs:
        print(f"    [WARN] no PNG in {TARGET_NAME}/, run --step screenshot first")
        return False
    chosen_name = max(pngs, key=lambda p: p.stat().st_size).name

    orig_len = len(text_data["user_prompt"])
    new_len  = len(new_user_prompt)
    pct      = 100 - int(new_len / orig_len * 100) if orig_len else 0

    new_input = {
        "task_id":       text_data["task_id"],
        "task_name":     text_data["task_name"],
        "system_prompt": patch_system_prompt(text_data["system_prompt"]),
        "user_prompt":   new_user_prompt,
        "images":        [chosen_name],
        "_req_kept":     kept_ids,
        "_req_dropped":  drop_ids,
    }

    print(f"    [input] kept {len(kept_ids)} reqs, dropped {len(drop_ids)}; "
          f"prompt {orig_len} -> {new_len} chars (-{pct}%)  image={chosen_name}")
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        input_path.write_text(
            json.dumps(new_input, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return True


def process_task(task_dir: Path, steps: list[str], model: str,
                 dry_run: bool, force: bool) -> bool:
    task_id = task_dir.name
    print(f"[{task_id}]")

    explicit_reqs = load_explicit_reqs_from_icg(task_dir)
    if not explicit_reqs:
        if any(s in steps for s in ("select", "tag")):
            print(f"    [WARN] no icg.json or no explicit reqs, skipping select/tag")

    ok = True

    if "select" in steps:
        if explicit_reqs:
            fname = step_select(task_dir, task_id, explicit_reqs, model, dry_run, force)
            if fname is None and not dry_run:
                ok = False
        else:
            print(f"    [skip] select: no explicit reqs")

    if "screenshot" in steps:
        png = step_screenshot(task_dir, dry_run, force)
        if png is None:
            existing = list((task_dir / TARGET_NAME).glob("*.png")) if (task_dir / TARGET_NAME).exists() else []
            if not existing:
                ok = False

    if "tag" in steps:
        if explicit_reqs:
            result = step_tag(task_dir, task_id, explicit_reqs, model, dry_run, force)
            if result is None and not dry_run:
                ok = False
        else:
            print(f"    [skip] tag: no explicit reqs")

    if "input" in steps:
        if not step_input(task_dir, dry_run, force):
            ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Build Image modality input directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task_ids", nargs="*",
                        help="Task IDs to process; defaults to all tasks")
    parser.add_argument("--step", default="all",
                        choices=["select", "screenshot", "tag", "input", "all"],
                        help="Pipeline step to run; default: all")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Vision-language model. Can also be set with WEBRISE_IMAGE_MODEL.")
    parser.add_argument("--data-root", dest="data_root",
                        default=DEFAULT_DATA_ROOT,
                        help="Task data root containing task folders. Can also be set with WEBRISE_DATA_ROOT.")
    parser.add_argument("--dir", dest="data_root", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers; default: 4")
    args = parser.parse_args()

    if not args.data_root:
        parser.error("--data-root is required unless WEBRISE_DATA_ROOT is set.")
    data_root = Path(args.data_root).expanduser()
    data_root = data_root if data_root.is_absolute() else (Path.cwd() / data_root)
    data_root = data_root.resolve()
    if not data_root.exists():
        print(f"ERROR: data root not found: {data_root}")
        sys.exit(1)

    steps = (
        ["select", "screenshot", "tag", "input"]
        if args.step == "all"
        else [args.step]
    )

    if args.task_ids:
        task_dirs = []
        for tid in args.task_ids:
            td = data_root / tid
            if not td.is_dir():
                print(f"ERROR: task dir not found: {td}")
            else:
                task_dirs.append(td)
    else:
        task_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    if not task_dirs:
        print("ERROR: no task dirs found")
        sys.exit(1)

    print(f"data_root={data_root}")
    print(f"target={TARGET_NAME}  steps={steps}  dry_run={args.dry_run}  tasks={len(task_dirs)}\n")

    ok_count = fail_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {
            exe.submit(process_task, td, steps, args.model, args.dry_run, args.force): td
            for td in task_dirs
        }
        for fut in concurrent.futures.as_completed(futures):
            try:
                if fut.result():
                    ok_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"[ERROR] {futures[fut].name}: {e}")
                fail_count += 1

    print(f"\nDone: ok={ok_count} fail/skip={fail_count} total={len(task_dirs)}")


if __name__ == "__main__":
    main()
