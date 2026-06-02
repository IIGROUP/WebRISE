import base64
import argparse
import concurrent.futures
import json
import mimetypes
import os
import shutil
from pathlib import Path

from openai import OpenAI
from playwright.sync_api import sync_playwright


# Screenshot GT HTML pages.
def _task_allowed(task_id: str, task_ids: set[str] | None) -> bool:
    return not task_ids or task_id in task_ids


def _screenshot_one(args: tuple) -> None:
    name, html_path, output_path = args
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(f"file://{html_path.resolve()}")
            page.wait_for_timeout(1000)
            page.screenshot(path=str(output_path), full_page=True)
            browser.close()
        print(f"[screenshot] {name} <- {html_path.name}", flush=True)
    except Exception as e:
        print(f"[error] {name}: {e}", flush=True)


def screenshot_html_folders(input_root: Path, output_dir: Path,
                             task_ids: set[str] | None = None, workers: int = 4,
                             overwrite: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for subdir in sorted(input_root.iterdir()):
        if not subdir.is_dir():
            continue
        if not _task_allowed(subdir.name, task_ids):
            continue
        html_files = list(subdir.glob("*.html"))
        if not html_files:
            print(f"[skip] {subdir.name}: no HTML file found")
            continue
        output_path = output_dir / f"{subdir.name}.png"
        if output_path.exists() and not overwrite:
            print(f"[skip] {subdir.name}: screenshot exists -> {output_path.name}")
            continue
        if output_path.exists():
            output_path.unlink()
        pending.append((subdir.name, html_files[0], output_path))

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
            list(exe.map(_screenshot_one, pending))


# Build MD/Sketch input.json files from icg.json.
def _process_icg_one(args: tuple) -> None:
    subdir, md_system_prompt, sketch_system_prompt, overwrite = args
    icg_path = subdir / "icg.json"
    if not icg_path.exists():
        print(f"[skip] {subdir.name}: icg.json not found")
        return
    try:
        md_dir = subdir / "MD"
        sketch_dir = subdir / "Sketch"
        md_dir.mkdir(exist_ok=True)
        sketch_dir.mkdir(exist_ok=True)

        md_input_path = md_dir / "input.json"
        sketch_input_path = sketch_dir / "input.json"

        if md_input_path.exists() and sketch_input_path.exists() and not overwrite:
            print(f"[skip] {subdir.name}: MD/Sketch input.json files already exist")
            return

        with open(icg_path, "r", encoding="utf-8") as f:
            icg_data = json.load(f)

        user_prompt = str(icg_data.get("input_text", "")).strip()
        if not user_prompt:
            raise ValueError("icg.json is missing a non-empty input_text field")

        output_task_id = icg_data.get("task_id", subdir.name)
        output_task_name = icg_data.get("task_name", "") or subdir.name

        if overwrite or not md_input_path.exists():
            with open(md_input_path, "w", encoding="utf-8") as f:
                json.dump({"task_id": output_task_id, "task_name": output_task_name,
                           "system_prompt": md_system_prompt, "user_prompt": user_prompt},
                          f, ensure_ascii=False, indent=2)
            print(f"[done] {subdir.name} -> MD/input.json")
        else:
            print(f"[skip] {subdir.name}: MD/input.json exists")

        if overwrite or not sketch_input_path.exists():
            with open(sketch_input_path, "w", encoding="utf-8") as f:
                json.dump({"task_id": output_task_id, "task_name": output_task_name,
                           "system_prompt": sketch_system_prompt, "user_prompt": user_prompt},
                          f, ensure_ascii=False, indent=2)
            print(f"[done] {subdir.name} -> Sketch/input.json")
        else:
            print(f"[skip] {subdir.name}: Sketch/input.json exists")

    except Exception as e:
        print(f"[error] {subdir.name}: {e}")


def process_icg_and_translate(input_root: Path, task_ids: set[str] | None = None,
                               workers: int = 4, overwrite: bool = False):
    md_system_prompt = (
        "You are an expert in writing front-end HTML web page code. "
        "The web pages you create must comply with the requirements and data contracts, "
        "and the page layout must strictly follow the MD documentation."
    )
    sketch_system_prompt = (
        "You are an expert in writing front-end HTML web code. "
        "Your generated web pages must conform to the requirements and data contracts, "
        "and the page layout must strictly follow the input sketch."
    )
    pending = [
        (subdir, md_system_prompt, sketch_system_prompt, overwrite)
        for subdir in sorted(input_root.iterdir())
        if subdir.is_dir() and _task_allowed(subdir.name, task_ids)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
        list(exe.map(_process_icg_one, pending))


def call_vlm(api_key, base_url, model, prompt, image_path):
    client = OpenAI(api_key=api_key, base_url=base_url)
    img_base64 = base64.b64encode(image_path.read_bytes()).decode()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


# Generate Sketch images.
def _sketch_one(args: tuple) -> None:
    img_path, target_root, api_key, base_url, model, prompt, overwrite = args
    name = img_path.stem
    target_dir = target_root / name / "Sketch"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{name}_sketch.png"

    if output_path.exists() and not overwrite:
        print(f"[skip] {name}: Sketch image exists -> {output_path.name}", flush=True)
        return
    if output_path.exists():
        output_path.unlink()

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        mime_type, _ = mimetypes.guess_type(str(img_path))
        if mime_type is None:
            mime_type = "image/png"

        with open(img_path, "rb") as f:
            result = client.images.edit(
                model=model,
                image=(img_path.name, f, mime_type),
                prompt=prompt,
                size="1024x1024",
            )

        if not result or not hasattr(result, "data"):
            raise RuntimeError("Image edit API returned an empty or invalid response")

        image_bytes = base64.b64decode(result.data[0].b64_json)
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        print(f"[done] {name} -> {output_path.name}", flush=True)
    except Exception as e:
        print(f"[error] {img_path.name}: {e}", flush=True)


def generate_sketch_images(image_dir: Path, target_root: Path,
                            api_key, base_url, model, prompt,
                            task_ids: set[str] | None = None, workers: int = 4,
                            overwrite: bool = False):
    pending = [
        (img_path, target_root, api_key, base_url, model, prompt, overwrite)
        for img_path in sorted(image_dir.glob("*.png"))
        if _task_allowed(img_path.stem, task_ids)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
        list(exe.map(_sketch_one, pending))


# Generate Markdown layout notes.
def _text_md_one(args: tuple) -> None:
    img_path, target_root, api_key, base_url, model, prompt, overwrite = args
    name = img_path.stem
    target_dir = target_root / name / "Text"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{name}_text.md"

    if output_path.exists() and not overwrite:
        print(f"[skip] {name}: Markdown exists -> {output_path.name}", flush=True)
        return
    if output_path.exists():
        output_path.unlink()

    try:
        print(f"[start] {name} -> Markdown", flush=True)
        md_text = call_vlm(api_key, base_url, model, prompt, img_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"[done] {name} -> {output_path.name}", flush=True)
    except Exception as e:
        print(f"[error] {img_path.name}: {e}", flush=True)


def generate_text_md(image_dir: Path, target_root: Path,
                     api_key, base_url, model, prompt,
                     task_ids: set[str] | None = None, workers: int = 4,
                     overwrite: bool = False):
    pending = [
        (img_path, target_root, api_key, base_url, model, prompt, overwrite)
        for img_path in sorted(image_dir.glob("*.png"))
        if _task_allowed(img_path.stem, task_ids)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
        list(exe.map(_text_md_one, pending))


def move_text_md_to_md_folder(input_root: Path, task_ids: set[str] | None = None,
                               overwrite: bool = False):
    for subdir in input_root.iterdir():
        if not subdir.is_dir():
            continue
        if not _task_allowed(subdir.name, task_ids):
            continue

        text_dir = subdir / "Text"
        md_dir = subdir / "MD"

        if not text_dir.exists():
            print(f"[skip] {subdir.name}: Text directory not found")
            continue

        md_files = list(text_dir.glob("*.md"))
        if not md_files:
            print(f"[skip] {subdir.name}: no markdown file in Text")
            continue

        md_dir.mkdir(exist_ok=True)

        for md_file in md_files:
            target_path = md_dir / md_file.name
            if target_path.exists() and not overwrite:
                print(f"[skip] {subdir.name}: MD/{md_file.name} exists")
                continue
            try:
                if target_path.exists():
                    target_path.unlink()
                shutil.move(str(md_file), str(target_path))
                print(f"[done] {subdir.name}: {md_file.name} -> MD/")
            except Exception as e:
                print(f"[error] {subdir.name}: failed to move {md_file.name}: {e}")


def copy_text_input_to_t_only(input_root: Path, overwrite: bool = False,
                               task_ids: set[str] | None = None):
    for subdir in input_root.iterdir():
        if not subdir.is_dir():
            continue
        if not _task_allowed(subdir.name, task_ids):
            continue

        source_path = subdir / "MD" / "input.json"
        if not source_path.exists():
            print(f"[skip] {subdir.name}: MD/input.json not found")
            continue

        target_dir = subdir / "Text"
        target_dir.mkdir(exist_ok=True)
        target_path = target_dir / "input.json"

        if target_path.exists() and not overwrite:
            print(f"[skip] {subdir.name}: Text/input.json exists")
            continue

        try:
            shutil.copy2(source_path, target_path)
            print(f"[done] {subdir.name} -> Text/input.json")
        except Exception as e:
            print(f"[error] {subdir.name}: {e}")


def replace_system_prompt_in_t_only(input_root: Path, new_system_prompt: str,
                                     task_ids: set[str] | None = None):
    for subdir in input_root.iterdir():
        if not subdir.is_dir():
            continue
        if not _task_allowed(subdir.name, task_ids):
            continue

        input_path = subdir / "Text" / "input.json"
        if not input_path.exists():
            print(f"[skip] {subdir.name}: Text/input.json not found")
            continue

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            old_prompt = data.get("system_prompt", "")
            if old_prompt == new_system_prompt:
                print(f"[skip] {subdir.name}: Text/input.json already has target system_prompt")
                continue

            data["system_prompt"] = new_system_prompt
            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"[done] {subdir.name} | old_len={len(old_prompt)} -> new_len={len(new_system_prompt)}")
        except Exception as e:
            print(f"[error] {subdir.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate modality inputs from task folders")
    parser.add_argument("--tasks", nargs="+",
                        help="Only process these task ids, e.g. D08_S28_T501 D08_S28_T502")
    parser.add_argument("--workers", type=int, default=4,
                        help="Max parallel workers (default: 4)")
    parser.add_argument("--data-root", dest="data_root",
                        default=os.environ.get("WEBRISE_DATA_ROOT"),
                        help="Task data root containing task folders. Can also be set with WEBRISE_DATA_ROOT.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate existing screenshots, modality input.json files, sketches, and markdown files")
    args = parser.parse_args()
    task_ids = set(args.tasks) if args.tasks else None
    workers = args.workers

    if not args.data_root:
        parser.error("--data-root is required unless WEBRISE_DATA_ROOT is set.")
    data_root_arg = Path(args.data_root).expanduser()
    input_root = data_root_arg if data_root_arg.is_absolute() else (Path.cwd() / data_root_arg)
    input_root = input_root.resolve()
    if not input_root.exists():
        parser.error(f"data root not found: {input_root}")
    screenshot_dir = Path(__file__).resolve().parent / "input_img"
    target_root = input_root

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
    )
    base_url = (
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    )
    sketch_model = os.environ.get("WEBRISE_SKETCH_MODEL", "").strip()
    md_model = os.environ.get("WEBRISE_MD_MODEL", "").strip()

    sketch_prompt = """
    You are a layout abstraction engine for image-to-image transformation.

    Your task is to convert the given real webpage screenshot into a very low-fidelity hand-drawn wireframe sketch.

    [Core Goal]
    Preserve the coarse layout structure with very sparse internal hints.
    Remove all semantic meaning and nearly all visual detail.

    [CRITICAL - Structural Fidelity (HARD CONSTRAINT)]
    - The overall layout structure MUST strictly match the input image
    - The number of major layout sections MUST be identical
    - Each large visual region must be preserved as a separate block
    - Maintain spatial relationships (top, bottom, left, right, columns, grids)
    - Do NOT merge, remove, rearrange, or invent any major section

    [CONTROLLED ABSTRACTION - SLIGHTLY STRONGER]
    - Remove all small UI components and fine details
    - DO NOT draw icons, avatars, badges, or controls
    - DO NOT draw individual items (no per-card or per-list detail)
    - Do not draw any text content (including graphic text)
    - Repeated elements MUST be collapsed into a single coarse pattern

    - Internal hints are allowed but MUST be extremely limited:
      - At most 1-2 short horizontal lines per major region
      - At most one simple inner block (if necessary)
      - Prefer empty regions over detailed ones

    - Avoid representing internal hierarchy explicitly
    - Avoid multiple small elements inside any region

    [STRICT REMOVAL]
    - No icons
    - No circles (avatars)
    - No small rectangles or buttons
    - No labels or symbols
    - No text or numbers
    - No fine-grained UI components

    [Visual Simplification Rules]
    - image -> large empty rectangle
    - text -> very few short horizontal lines (minimal, sparse)
    - lists/cards -> single grouped block (not repeated items)
    - complex UI -> one simplified block or empty region

    [Style Requirements]
    - extremely low fidelity
    - rough hand-drawn sketch
    - messy, uneven lines
    - black and white only
    - no colors, no shading, no gradients
    - no text in any form.

    [IMPORTANT BALANCE]
    - Major layout blocks: MUST match exactly
    - Internal hints: VERY sparse and minimal
    - Prefer under-representation over over-detail

    [Output Expectation]
    Generate a rough sketch that preserves the layout,
    avoiding any detailed or component-level structure.
    """
    md_prompt = """
    You are a webpage coarse-layout parser.

    Given a webpage screenshot, output a HUMAN-READABLE LOW-FIDELITY LAYOUT SKETCH in markdown.

    The markdown must balance:
    - human readability (natural, clean, easy to scan)
    - structural stability (ASCII layout must render correctly)
    - machine extractability (light tags for DSL extraction)

    ====================
    [CORE GOAL]
    ====================

    Capture only:
    - major page regions
    - vertical stacking
    - coarse horizontal structure
    - repeated patterns (feeds, cards, grids)

    Ignore detailed content.

    ====================
    [OUTPUT FORMAT - CRITICAL]
    ====================

    1. Output valid markdown.
    2. Each section must follow this structure:

       (A) A natural language heading
       (B) A fenced code block containing the ASCII layout

    3. NEVER place ASCII layout outside a code block.
    4. NEVER mix ASCII with normal markdown text.
    5. Use exactly triple backticks ``` for each block.
    6. Do NOT put any explanation inside the code block.

    ====================
    [SECTION HEADER FORMAT]
    ====================

    Use:

    ## <Natural Name> [role=<role>; span=<span>]

    Examples:
    ## Top bar [role=header; span=full-width]
    ## Main feed [role=list_area; span=centered-column]

    Rules:
    - Natural name must be human-friendly
    - Tag must be compact and stable
    - Do NOT use Region 1 / Region 2
    - Do NOT use schema-like naming

    Allowed role:
    header, sidebar, main, footer, hero, toolbar, content, form, notice, action_area, list_area, grid_area, modal

    Allowed span:
    full-width, centered-column, left-column, right-column, two-column, three-column, stacked

    ====================
    [ASCII LAYOUT RULES]
    ====================

    Inside each code block:

    1. Use only:
       +  -  |  [  ]  :

    2. Keep each row on ONE line (no wrapping).
    3. Keep width moderate (avoid very long lines).
    4. Maintain consistent spacing.
    5. Use simple, natural labels.

    ====================
    [CONTENT STYLE INSIDE BOX]
    ====================

    Use natural coarse labels:

    GOOD:
    - logo
    - search
    - cover
    - avatar
    - name / intro
    - actions
    - stats
    - list
    - cards
    - image
    - text
    - meta

    BAD:
    - title block
    - text block
    - image area
    - repeated list items
    - action button (too mechanical)

    ====================
    [REPEATED STRUCTURE RULE]
    ====================

    For feeds/lists:

    - Show ONE representative pattern
    - Indicate repetition using wording like:
      "repeated cards" or "..."

    Example:

    +----------------------------------------+
    | repeated cards                         |
    | [ image ]  title / meta                |
    |            short text                  |
    |            actions                     |
    +----------------------------------------+

    DO NOT list many duplicated rows.

    ====================
    [SECTION COUNT]
    ====================

    - Keep 3 to 6 sections total
    - Merge similar areas
    - Prefer fewer, clearer blocks

    ====================
    [NATURALNESS RULE]
    ====================

    The result should read like a human wireframe note, not a schema.

    GOOD:
    "Top bar", "Filter row", "Main results"

    BAD:
    "content container", "data region", "item structure"

    ====================
    [FINAL CHECK]
    ====================

    Before output:

    - ASCII is inside code blocks? (MANDATORY)
    - Layout will not break in markdown preview?
    - Sections are natural and readable?
    - Tags exist but are minimal?
    - Repeated items merged?

    If not, revise.
    """

    print(f"[config] input_root={input_root}")
    print(f"[config] tasks={'all' if task_ids is None else len(task_ids)} workers={workers} force={args.force}")

    screenshot_html_folders(input_root, screenshot_dir, task_ids=task_ids, workers=workers,
                            overwrite=args.force)

    process_icg_and_translate(input_root=input_root, task_ids=task_ids, workers=workers,
                              overwrite=args.force)

    generate_sketch_images(
        screenshot_dir, target_root, api_key, base_url, sketch_model, sketch_prompt,
        task_ids=task_ids, workers=workers, overwrite=args.force,
    )

    generate_text_md(
        screenshot_dir, target_root, api_key, base_url, md_model, md_prompt,
        task_ids=task_ids, workers=workers, overwrite=args.force,
    )

    move_text_md_to_md_folder(input_root=input_root, task_ids=task_ids, overwrite=args.force)

    copy_text_input_to_t_only(input_root=input_root, overwrite=args.force, task_ids=task_ids)

    replace_system_prompt_in_t_only(
        input_root=input_root,
        new_system_prompt=(
            "You are an expert in writing front-end HTML web page code. "
            "The web pages you create must comply with the requirements and data contracts, "
            "and on the premise of ensuring requirement fulfillment, complying with constraints and data contracts, efforts shall be made to achieve an aesthetically pleasing interface with restrained and harmonious color schemes, clear hierarchical layout, and refined and consistent details."
        ),
        task_ids=task_ids,
    )


if __name__ == "__main__":
    main()
