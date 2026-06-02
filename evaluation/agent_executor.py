"""Agent-loop executor - drives an ICG transition to completion via LLM decisions.

Replaces precompiled selector scripts with an online loop:

    Observation (indexed DOM text)  →  LLM (Thought + Action)  →  Executor
              ▲                                                       │
              └───────────────── page changes ────────────────────────┘

Reuses:
  - DOM tree extraction / indexing from dom_observation.py
  - Per-action Playwright execution from executor.py

Entry point:
    run_agent_transition(page, transition, client, ...) -> AgentRunResult

The result carries per-turn trajectory, the selector of the last concrete
action (for `[AFTER] The target ...` observation), and an overall status.
"""

from __future__ import annotations

import json
import re
import time
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from playwright.async_api import Page

import config
from executor import execute_action
from dom_observation import (
    _annotate_new_elements,
    _extract_dom_tree,
    _extract_interactive_elements,
    _flat_tree_to_llm_text,
    _validate_unique_selector,
)

_RETRY_DELAYS = [3, 8, 20]


# ── Prompts ───────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """\
Imagine you are a robot browsing the web to execute a web-testing task. In each \
iteration, you will receive an indexed DOM observation (elements prefixed with \
`[N]` or `*[N]`, where `*[N]` marks elements newly appeared this turn). Then \
follow the guidelines and choose ONE of the following actions:
1. Click an element.
2. Dismiss a modal/overlay by clicking outside it.
3. Double-click or right-click an element, or long-press it for N milliseconds.
4. Hover an element.
5. Type text into a non-date input at the current cursor position (use Clear first to replace all content; use `Press [N]; Control+End` first if you want to append to the end of existing content).
6. Set a native `<input type=date>` to a specific ISO date.
7. Clear an input, or blur (remove focus from) it.
8. Choose an option on a native `<select>` by its visible label.
9. Check or uncheck a checkbox.
10. Press a key on an element (Enter, Escape, Tab, ArrowDown, ...).
11. Scroll the window or a specific container (up/down/top/bottom/left/right).
12. Drag an element to another element by index, or by a pixel offset.
13. Drag a range slider to a target numeric value.
14. Click at absolute page coordinates — use when you need to interact with a specific position on the page but no element at that position appears in the observation list (e.g. clicking on a blank area of the canvas to deselect the selected element).
15. Upload one or more files by picking path(s) from the Available Test Assets table.
16. Wait for N milliseconds (for animations or async settles).
17. Refresh the page.
18. Select a text fragment inside an element (for contenteditable / rich text editors).
19. Go back to the previous page (browser back navigation).
20. Reset the page to the initial state of this task (use when previous steps caused unrecoverable errors).
21. Done — end the loop when the task is complete.

Correspondingly, Action MUST STRICTLY follow the format:
- Click [N]  OR  Click [N]; <count>
- Dismiss
- DoubleClick [N]
- RightClick [N]
- LongPress [N]; <ms>
- Hover [N]
- Input [N]; <text>
- InputDate [N]; <YYYY-MM-DD>
- Clear [N]
- Blur [N]
- Select [N]; <option label>
- Check [N] / Uncheck [N]
- Press [N]; <key>  OR  Press [N]; <key>; <count>
- Scroll [N or WINDOW]; <up|down|top|bottom|left|right>
- Drag [N]; [M]  OR  Drag [N]; offset_x=<px>,offset_y=<px>
- DragRange [N]; <target numeric value>
- ClickAt; x=<px> y=<px>
- Upload [N]; <file_path>  OR  Upload [N]; <file_path>|<file_path>|...
- SelectText [N]; <text to select>
- Wait; <ms>
- Refresh
- GoBack
- Reset
- Done

Key Guidelines You MUST follow:
1. Execute only ONE action per iteration.
   `Click [N]; <count>` is still one action, used only when the task explicitly requires repeating the exact same click on the exact same target (for example adding the same item up to a displayed stock count). Do not use click repeat for exploratory clicking.
2. Use indices from the latest observation — they change between turns.
3. For a native `<select>`, always use `Select`, never `Click`. For custom dropdowns (non-`<select>`), `Click` the trigger, then `Click` the option on the next turn.
4. For a native `<input type=date>`, always use `InputDate [N]; YYYY-MM-DD`, never regular `Input`. Use the ISO date requested by the task, for example `InputDate [8]; 2026-05-07`.
5. For `Upload`, pick file path(s) from the Available Test Assets table — do NOT invent paths. To upload multiple files at once, join paths with `|` in one Upload action, for example `Upload [N]; ./test_assets/upload_image_1.png|./test_assets/upload_image_2.png`. If the task asks for an oversized / over-limit image upload without naming a specific size, use `./test_assets/oversize_50mb.png`.
6. Use `DragRange [N]; value` only for range-slider / numeric-slider scenarios, especially when the task asks to adjust a slider or verify real-time slider updates. Do NOT use `DragRange` for canvas dragging, crop boxes, movable cards, scrollbars, or other non-slider drags; use `Drag` for those.
7. If the needed element is not in the observation, reveal it first (scroll a container, click a tab, hover a menu).
8. Do NOT use `Hover` as a no-op wait — use `Wait; <ms>` for timed settles.
9. Avoid repeating the same action if the page remains unchanged — try something else. **If the exact same action (same type AND same target index) has been executed 3 times in a row without change, the expected outcome is likely not implemented — call `Done` immediately instead of retrying.** If the task requires a known fixed number of repeated clicks on the same target, prefer `Click [N]; <count>` instead of spending multiple turns.
10. Do NOT use `Press [N]; Enter` as a substitute for `Click [N]`. If clicking an element has no effect, the element is likely not the right target — try a different element or approach instead.
11. **Attempt / prevention tasks** — when the task says to "attempt" an operation and expects prevention, identify the exact target first. If that exact target has no visible enabled control, is marked as current/disabled/unavailable, or its control is absent, that is a valid prevented state: call `Done`. Do NOT click a sibling, nearby, or similarly labeled control for a different item just to force an action.
12. Emit `Action: Done` only after the task's user action has fully executed.
   If a task requires a temporary intermediate state before returning to the original/restored state (for example add then remove, expand then collapse, open then close, undo then redo), do NOT call `Done` after only restoring the final state. First ensure the required intermediate state actually occurred; if it did not, retry the missing step with the correct action.
13. **Cursor control in text fields** — use `Press` with these keys:
   - End of line: `Press [N]; End`
   - Start of line: `Press [N]; Home`
   - End of document: `Press [N]; Control+End`
   - Start of document: `Press [N]; Control+Home`
   - Select all: `Press [N]; Control+a`
   - Indent a line: first `Press [N]; Home`, then `Press [N]; Tab`
   - Arrow keys accept a repeat count: `Press [N]; ArrowDown; 20` presses ArrowDown 20 times in one action. Use shortcuts first (Control+End etc.); only fall back to arrow-key repeats when shortcuts fail.
   - Backspace also accepts a repeat count: `Press [N]; Backspace; 9` presses Backspace 9 times. When deleting characters, use the cursor column indicator (`Col N`) in the observation to estimate how many characters remain before the line break. Code editors may auto-insert indentation on new lines, so the total characters to delete may exceed the visible text length — use `Press [N]; Backspace; count` with the appropriate count rather than pressing one at a time.
14. **Hidden controls** — some UI elements (edit buttons, delete icons, action menus, toolbars) only appear after hovering or right-clicking their parent element. If you cannot find a button or control described in the task, try `Hover` on the target element first. If that does not reveal the control, try `RightClick` on the target element — it may open a context menu with the needed action.
15. **`not-visible` elements** — elements tagged `not-visible` (e.g. `[5]<button not-visible>▲ />`) exist in the DOM but are currently invisible. Do NOT click or type on them while `not-visible`. To make them visible, first try `Hover [N]` on the element. If it becomes visible in the next observation (tag gone), proceed to interact. If it remains `not-visible` after hover, try other approaches (scroll, click a related toggle, etc.).
16. **Text selection** — use `SelectText [N]; <text>` to select a specific continuous text fragment inside element [N]. Prefer the smallest specific text element that contains the target text. If the desired continuous selection spans multiple inline text nodes/highlight spans, choose their smallest visible parent/ancestor document container and provide one continuous text snippet spanning the desired range. After selection, use `Input [N]; <replacement>` to replace the selected text in one step, or `Press [N]; Backspace` to delete it. For targeted edits described by the task, such as changing one word/string/value while leaving the rest unchanged, use `SelectText` + `Input`; do NOT `Clear` and rewrite the whole field/editor unless the task explicitly asks to replace all content. Check the `[Cursor: ...]` line in the observation to confirm the selection succeeded.
17. **Back navigation** — if the task requires going back to a previous page, use `GoBack`. If `GoBack` has no effect (e.g. single-page app with no browser history), try clicking a navigation link or tab to reach the target page instead.
18. **Undo** — `Press [N]; Control+z` only works in text editing contexts (textarea, contenteditable). Do NOT use it to undo UI actions like clicks, toggles, or navigation — there is no general undo. If you made a wrong action, use `Reset` to restore the initial state. ** Do NOT use Reset to satisfy a task's required restored/original final state. Reset is only for unrecoverable errors; if you reset, continue the task from the restored start state instead of calling Done.**
19. **Scroll directions** — `up`/`down`/`left`/`right` scroll by one step; `top`/`bottom` jump instantly to the very start/end of the target. Use `top`/`bottom` to reach extreme positions in one action. For tasks that require triggering a **load-more / infinite-scroll** (e.g. "scroll down to load more items", "scroll to the bottom to reveal more content"), use `Scroll WINDOW; bottom` (or `Scroll [N]; bottom` for a specific container) — this reaches the end in a single action and reliably triggers the load-more event. Do NOT use repeated incremental `down` scrolls for this purpose.
20. **Required interaction path / content source** — when the task specifies that content must come from a particular UI path or source, follow that path and do NOT substitute a similar final state by typing or creating the content directly. For example, if the task says to select an emoji from the emoji panel, do not type an emoji into the text area; if it says to choose a topic/user from a suggestion list, do not manually type the completed topic/user. You may try reasonable ways to reveal or use the required control, but if the required path cannot be opened or used after reasonable attempts, call `Done` rather than bypassing it.
21. **Editing / adding content** — if the DOM shows dedicated controls for the operation (e.g. an "Add" button, a toolbar button, a "New Item" link), use them first. Only type directly into a text field when no dedicated control exists.
22. **Free-form text content** — when the task asks for free-form content without specifying exact text (e.g. *"type a message"*, *"type multi-line text"*, *"type long content"*), you choose the content. Keep it minimal — just enough to exercise the described shape:
   - Single-line: one short sentence (e.g. `Hello world`).
   - Multi-line: embed literal `\n` in the text value for line breaks (e.g. `line1\nline2\nline3`); 3–5 lines is typically enough. Do NOT use `Press [N]; Enter` to insert line breaks — the executor interprets `\n` as Enter automatically.
   - Adding a new line inside an existing editor: include a leading literal `\n` in the `Input` value (e.g. `\n  console.log(msg);`). Moving the cursor to a line and typing text without `\n` edits the current line; it does not create a new line.
   - Long / scroll-triggering content: start with **50 lines** of `\n`-separated text directly. Never dump hundreds of lines.
   - Very long character content (e.g. character-limit testing, overflow detection, 1000-character inputs): use a single block of approximately **1000 characters** of repeated text (e.g. `abcdefghij` repeated 100 times). Input it in one `Input` action — do not split across multiple actions.
   When the task specifies an exact string or character count, use that verbatim instead.
23. **Viewport visibility** — each observation starts with `Viewport: width=<w> height=<h>`. Each element has a `pos` (x, y) coordinate and `size`; use these to judge whether the element is currently within the viewport. When the task asks you to inspect or interact with an element and then return (e.g. "click item X to view details, then go back"), prefer elements that are fully visible in the viewport (`0 <= y` and `y + height <= viewport height`).
24. **Check if the described view is already visible** — when the task describes a target view that should be visible (e.g. "so that the list of all notes is visible", "so the full list is shown"), first check whether that view is already present in the current DOM before taking any action. If the described view is already visible — for instance, the app uses a side-by-side layout where the note list is always shown alongside the editor — call `Done` immediately.
25. **ClickAt for unindexed positions** — use `ClickAt; x=<px> y=<px>` only when you need to click a specific position and no element at that position appears in the observation list. Estimate the target coordinates from the `pos` and `size` of nearby indexed elements. Example: if the canvas occupies roughly (200, 100)–(1400, 900) based on surrounding element positions, `ClickAt; x=800 y=500` clicks the centre of the blank canvas area.

Your reply MUST be a single JSON object (no markdown fences):
{"thought": "Your brief thoughts", "action": "ONE Action in the format above"}

Then the user will provide:
Observation: {indexed DOM of the page}"""


TEST_ASSETS_TABLE_SHORT = """\
Available Test Assets (pick the path whose description matches your scenario — do NOT invent paths):

| File path | Description / test scenario |
|-----------|------------------------------|
| `./test_assets/test_image.png` | Valid PNG image — normal successful upload |
| `./test_assets/test_image.jpg` | Valid JPEG image — normal successful upload (alternate format) |
| `./test_assets/test_image.gif` | Valid GIF image — test GIF format support |
| `./test_assets/test_image.webp` | Valid WebP image — test WebP format support |
| `./test_assets/test_image.bmp` | BMP image — test unsupported/rejected image format |
| `./test_assets/test_vector.svg` | SVG vector file — test unsupported file type rejection |
| `./test_assets/test_document.pdf` | PDF document — test non-image file type rejection |
| `./test_assets/test_doc.doc` | Legacy Word document (.doc) — test non-image file type rejection |
| `./test_assets/test_doc.docx` | Word document (.docx) — test non-image file type rejection |
| `./test_assets/test_sheet.excel` | Legacy Excel spreadsheet (.excel) — test non-image file type rejection |
| `./test_assets/test_sheet.xlsx` | Excel spreadsheet (.xlsx) — test non-image file type rejection |
| `./test_assets/test_data.csv` | CSV file — test non-image file type rejection |
| `./test_assets/test_archive.zip` | ZIP archive — test non-image file type rejection |
| `./test_assets/test_binary.exe` | Executable file — test non-image file type rejection |
| `./test_assets/oversize_2mb.png` | PNG ~2 MB — use when the size limit is around 2 MB |
| `./test_assets/oversize_5mb.png` | PNG ~5 MB — use when the size limit is around 5 MB |
| `./test_assets/oversize_10mb.png` | PNG ~10 MB — test oversized image rejection |
| `./test_assets/oversize_50mb.png` | PNG ~50 MB — use by default when testing oversized / over-limit image upload |
| `./test_assets/oversize_5mb.pdf` | PDF ~5 MB — oversized non-image |
| `./test_assets/oversize_10mb.pdf` | PDF ~10 MB — oversized non-image |
| `./test_assets/oversize_25mb.pdf` | PDF ~25 MB — very large file rejection |
| `./test_assets/oversize_10mb.zip` | ZIP ~10 MB — oversized archive rejection |
| `./test_assets/new_avatar_green.png` | PNG avatar image (green) — avatar/profile picture upload scenarios |
| `./test_assets/new_avatar_red.png` | PNG avatar image (red) — second distinct avatar upload (e.g. changing avatar again) |
| `./test_assets/upload_image_1.png` | Generic upload image #1 — multi-image / gallery scenarios |
| `./test_assets/upload_image_2.png` | Generic upload image #2 — multi-image / gallery scenarios |
| `./test_assets/upload_image_3.png` | Generic upload image #3 — multi-image / gallery scenarios |
| `./test_assets/upload_image_4.png` | Generic upload image #4 — multi-image / gallery scenarios |
| `./test_assets/upload_image_5.png` | Generic upload image #5 — multi-image / gallery scenarios |
| `./test_assets/upload_image_6.png` | Generic upload image #6 — multi-image / gallery scenarios |
| `./test_assets/upload_image_7.png` | Generic upload image #7 — multi-image / gallery scenarios |
| `./test_assets/upload_image_8.png` | Generic upload image #8 — multi-image / gallery scenarios |
| `./test_assets/upload_image_9.png` | Generic upload image #9 — multi-image / gallery scenarios |"""


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class AgentTurn:
    iteration: int
    dom_text_chars: int
    response_text: str
    thought: str
    action_text: str
    parsed_action: dict | None
    executed: bool
    exec_ok: bool
    exec_err: str
    selector: str | None
    action_dict: dict | None = None  # exact action dict passed to execute_action (for replay)
    usage: dict = field(default_factory=dict)
    screenshot: str | None = None  # filename (relative to log_dir) of post-action screenshot


@dataclass
class AgentRunResult:
    status: str  # "DONE" | "MAX_ITER" | "ERROR" | "PARSE_FAIL"
    iterations: int
    trajectory: list[AgentTurn]
    final_selector: str | None  # selector of the last concrete action's target
    final_target_affordance_type: str | None  # inferred from last action verb
    error: str = ""
    total_usage: dict = field(default_factory=dict)
    final_screenshot: str | None = None  # filename (relative to log_dir) of post-loop screenshot used for postcondition judgement

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "iterations": self.iterations,
            "final_selector": self.final_selector,
            "final_target_affordance_type": self.final_target_affordance_type,
            "error": self.error,
            "total_usage": self.total_usage,
            "final_screenshot": self.final_screenshot,
            "trajectory": [
                {
                    "iteration": t.iteration,
                    "dom_text_chars": t.dom_text_chars,
                    "response_text": t.response_text,
                    "thought": t.thought,
                    "action_text": t.action_text,
                    "parsed_action": t.parsed_action,
                    "executed": t.executed,
                    "exec_ok": t.exec_ok,
                    "exec_err": t.exec_err,
                    "selector": t.selector,
                    "action_dict": t.action_dict,
                    "usage": t.usage,
                    "screenshot": t.screenshot,
                }
                for t in self.trajectory
            ],
        }


# ── Action parsing ────────────────────────────────────────────────────────────

_ACTION_VERBS = (
    "Click|Dismiss|DoubleClick|RightClick|LongPress|"
    "InputDate|Input|Append|Clear|Blur|Select|SelectText|Check|Uncheck|Press|Hover|"
    "Scroll|DragRange|Drag|ClickAt|Upload|Wait|Refresh|GoBack|Reset|Done"
)
_ACTION_LINE_RE = re.compile(
    rf"^\s*Action\s*:\s*(?P<verb>{_ACTION_VERBS})"
    r"(?:\s*\[\s*(?P<idx>\d+|WINDOW|window)\s*\]"
    r"|\s+(?P<bare_window>WINDOW|window))?"
    r"(?:\s*;\s*(?P<arg>.+?))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_THOUGHT_LINE_RE = re.compile(r"^\s*Thought\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


_CANONICAL_VERBS = {
    "click": "Click", "dismiss": "Dismiss",
    "doubleclick": "DoubleClick", "rightclick": "RightClick", "longpress": "LongPress",
    "inputdate": "InputDate", "input": "Input", "append": "Append", "clear": "Clear", "blur": "Blur",
    "select": "Select", "check": "Check", "uncheck": "Uncheck",
    "press": "Press", "hover": "Hover",
    "scroll": "Scroll", "dragrange": "DragRange", "drag": "Drag", "clickat": "ClickAt", "upload": "Upload",
    "wait": "Wait", "refresh": "Refresh", "goback": "GoBack", "reset": "Reset",
    "selecttext": "SelectText", "done": "Done",
}


def _parse_repeat_count(arg: str, *, max_repeat: int = 50) -> int | None:
    """Parse optional repeat count from action arg.

    Accepted form: "3".
    """
    text = (arg or "").strip()
    if not text:
        return None
    if not re.fullmatch(r"\d+", text):
        return None
    return max(1, min(max_repeat, int(text)))


def _parse_agent_response(text: str) -> tuple[str, str, dict | None]:
    """Return (thought, action_line, parsed_action_dict|None).

    Accepts JSON format: {"thought": "...", "action": "Click [17]"}
    Falls back to Thought:/Action: line format for robustness.
    """
    # ── Try JSON first ──
    try:
        # Strip markdown fences if present
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```\s*$", "", stripped)
        # Try direct parse; if that fails, find the first { and try each }
        # from right to left until one produces valid JSON (handles trailing garbage).
        obj = None
        try:
            obj = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            first = stripped.find("{")
            if first >= 0:
                for end in range(len(stripped) - 1, first, -1):
                    if stripped[end] == "}":
                        try:
                            obj = json.loads(stripped[first:end + 1])
                            break
                        except (json.JSONDecodeError, ValueError):
                            continue
        if isinstance(obj, dict) and "action" in obj:
            thought = str(obj.get("thought", ""))
            action_str = str(obj["action"]).strip()
            # For JSON path, extract verb + index for regex matching.
            # If ";" is present, only feed the part before ";" to regex
            # (arg will be extracted separately from the full action_str).
            # This avoids regex failure when ";" is followed immediately
            # by a newline (e.g. "Input [0]; \nconst x = 42;").
            if ";" in action_str:
                regex_input = action_str[:action_str.index(";")].strip()
            else:
                regex_input = action_str.split("\n")[0].strip()
            action_match = _ACTION_LINE_RE.search(f"Action: {regex_input}")
            if action_match:
                verb = _CANONICAL_VERBS.get(action_match.group("verb").lower())
                if verb is not None:
                    idx_raw = action_match.group("idx") or action_match.group("bare_window")
                    # Get arg from the full action_str (preserving newlines),
                    # not from regex which truncates at first newline.
                    semi_pos = action_str.find(";")
                    if semi_pos >= 0:
                        raw_arg = action_str[semi_pos + 1:]
                        # Treat one space after the semicolon as syntax sugar
                        # while preserving intentional leading/trailing input
                        # text, e.g. "Input [4];   KEY" -> "  KEY".
                        arg = raw_arg[1:] if raw_arg.startswith(" ") else raw_arg
                    else:
                        arg = (action_match.group("arg") or "").strip()
                    parsed: dict[str, Any] = {"verb": verb, "arg": arg}
                    if idx_raw is None:
                        parsed["index"] = None
                    elif str(idx_raw).upper() == "WINDOW":
                        parsed["index"] = "WINDOW"
                    else:
                        parsed["index"] = int(idx_raw)
                    if verb == "Click":
                        repeat = _parse_repeat_count(arg)
                        if repeat is not None:
                            parsed["repeat"] = repeat
                    return thought, action_str, parsed
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # Fallback: Thought:/Action: line format.
    thought_match = _THOUGHT_LINE_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    action_match = _ACTION_LINE_RE.search(text)
    if not action_match:
        return thought, "", None

    verb = _CANONICAL_VERBS.get(action_match.group("verb").lower())
    if verb is None:
        return thought, "", None
    idx_raw = action_match.group("idx")
    raw_action_line = action_match.group(0).split(":", 1)[1]
    action_line = raw_action_line.strip()
    if ";" in raw_action_line and verb in {"Input", "Append"}:
        raw_arg = raw_action_line[raw_action_line.index(";") + 1:]
        arg = raw_arg[1:] if raw_arg.startswith(" ") else raw_arg
    else:
        arg = (action_match.group("arg") or "").strip()

    parsed = {"verb": verb, "arg": arg}
    if idx_raw is None:
        parsed["index"] = None
    elif str(idx_raw).upper() == "WINDOW":
        parsed["index"] = "WINDOW"
    else:
        parsed["index"] = int(idx_raw)
    if verb == "Click":
        repeat = _parse_repeat_count(arg)
        if repeat is not None:
            parsed["repeat"] = repeat
    return thought, action_line, parsed


# ── Verb → executor action_dict mapping ───────────────────────────────────────

# Mapping between agent verbs and executor action `type` / affordance inference.
# Inferred affordance type is used only for executor's minor fallback paths
# (click text-match, dom monitor annotations); it does not affect DOM asserts.
_VERB_INFO = {
    "Click":       {"atype": "click",        "aff": "Button"},
    "Dismiss":     {"atype": "click",        "aff": "Button"},
    "DoubleClick": {"atype": "double_click", "aff": "Button"},
    "RightClick":  {"atype": "right_click",  "aff": "Button"},
    "LongPress":   {"atype": "long_press",   "aff": "Button"},
    "InputDate":   {"atype": "input_date",   "aff": "TextInput"},
    "Input":       {"atype": "input",        "aff": "TextInput"},
    "Append":      {"atype": "input",        "aff": "TextInput"},
    "Clear":       {"atype": "clear",        "aff": "TextInput"},
    "Blur":        {"atype": "blur",         "aff": "TextInput"},
    "Select":      {"atype": "select",       "aff": "Select"},
    "Check":       {"atype": "check",        "aff": "Checkbox"},
    "Uncheck":     {"atype": "uncheck",      "aff": "Checkbox"},
    "Press":       {"atype": "keypress",     "aff": "Button"},
    "Hover":       {"atype": "hover",        "aff": "Button"},
    "Scroll":      {"atype": "scroll",       "aff": "Button"},
    "Drag":        {"atype": "drag",         "aff": "Button"},
    "DragRange":   {"atype": "dragrange",    "aff": "Slider"},
    "ClickAt":     {"atype": "clickat",      "aff": "Button"},
    "Upload":      {"atype": "upload",       "aff": "Button"},
    "Wait":        {"atype": "wait",         "aff": "Button"},
    "Refresh":     {"atype": "refresh",      "aff": "Button"},
    "GoBack":      {"atype": "goback",       "aff": "Button"},
    "SelectText":  {"atype": "select_text",  "aff": "TextInput"},
}


def _verb_to_action_dict(verb: str, index: Any, arg: str) -> dict:
    """Translate a parsed agent action into executor.execute_action() format."""
    if verb not in _VERB_INFO:
        # "Done" (handled by loop) or unknown — return empty dict as a sentinel.
        return {"type": "_noop", "parameters": {}}
    info = _VERB_INFO[verb]
    atype = info["atype"]
    action: dict[str, Any] = {"type": atype, "parameters": {}}

    if verb == "Dismiss":
        action["parameters"]["click_position"] = "overlay_blank_area"
    elif verb == "InputDate":
        action["parameters"]["value"] = arg
    elif verb == "Input":
        action["parameters"]["value"] = arg
    elif verb == "Append":
        # executor's `input` handler uses keyboard.type which preserves content.
        action["parameters"]["value"] = arg
    elif verb == "Select":
        action["parameters"]["value"] = arg
    elif verb == "Press":
        parts = [p.strip() for p in (arg or "Enter").split(";")]
        action["parameters"]["key"] = parts[0] or "Enter"
        if len(parts) >= 2:
            try:
                action["parameters"]["times"] = max(1, int(parts[1]))
            except ValueError:
                pass
    elif verb == "LongPress":
        try:
            action["parameters"]["duration_ms"] = max(100, int((arg or "800").strip()))
        except ValueError:
            action["parameters"]["duration_ms"] = 800
    elif verb == "Scroll":
        direction_or_position = (arg or "down").strip().lower()
        if direction_or_position in ("top", "bottom", "left", "right"):
            action["parameters"]["position"] = direction_or_position
        else:
            action["parameters"]["direction"] = direction_or_position or "down"
            action["parameters"]["amount"] = 600
    elif verb == "Drag":
        # arg forms: "[M]" (target affordance index) OR "offset_x=<px>,offset_y=<px>"
        s = (arg or "").strip()
        if s.lower().startswith("offset"):
            for part in s.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip().lower()
                    try:
                        action["parameters"][k] = int(v.strip())
                    except ValueError:
                        pass
        elif s:
            action["parameters"]["target_index_raw"] = s
    elif verb == "DragRange":
        try:
            action["parameters"]["value"] = float((arg or "").strip())
        except ValueError:
            action["parameters"]["value_raw"] = (arg or "").strip()
    elif verb == "ClickAt":
        # arg form: "x=<px> y=<px>" or "x=<px>,y=<px>"
        for part in re.split(r"[,\s]+", (arg or "").strip()):
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip().lower()
                try:
                    action["parameters"][k] = int(v.strip())
                except ValueError:
                    pass
    elif verb == "Upload":
        files = [p.strip() for p in (arg or "").split("|") if p.strip()]
        action["parameters"]["files"] = files
    elif verb == "Wait":
        try:
            action["parameters"]["duration_ms"] = max(100, int((arg or "1000").strip()))
        except ValueError:
            action["parameters"]["duration_ms"] = 1000
    elif verb == "SelectText":
        action["parameters"]["text"] = arg
    elif verb == "Click":
        repeat = _parse_repeat_count(arg)
        if repeat is not None:
            action["parameters"]["times"] = repeat
    # DoubleClick / RightClick / Check / Uncheck / Clear / Blur / Hover / Refresh:
    # no extra params.
    return action


# ── LLM call with retry ───────────────────────────────────────────────────────


def _call_llm(client: OpenAI, messages: list[dict], model: str, max_tokens: int) -> tuple[str, dict]:
    """Stream-call the LLM. Retries on ANY exception AND on empty-text responses.

    Empty-text retries (Gemini occasionally returns 0-token completions, or
    burns the whole budget on reasoning without emitting visible text) bump
    max_tokens on the 2nd+ try to rescue truncation.
    """
    last_exc: Exception | None = None
    last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    attempts = [0] + _RETRY_DELAYS
    for attempt, delay in enumerate(attempts):
        if delay:
            time.sleep(delay)
        # Bump max_tokens on retries so reasoning-heavy models can still emit the Action.
        budget = max_tokens if attempt == 0 else max(max_tokens, 800 * (1 + attempt))
        try:
            chunks: list[str] = []
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            with client.chat.completions.create(
                model=model,
                max_tokens=budget,
                messages=messages,
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
                **config.reasoning_effort_kwargs(model),
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        chunks.append(delta)
                    if getattr(chunk, "usage", None):
                        u = chunk.usage
                        usage = {
                            "prompt_tokens":     getattr(u, "prompt_tokens", 0),
                            "completion_tokens": getattr(u, "completion_tokens", 0),
                            "total_tokens":      getattr(u, "total_tokens", 0),
                        }
            text = "".join(chunks).strip()
            last_usage = usage
            if text:
                return text, usage
            # Empty response: retry if we have attempts left
            if attempt < len(attempts) - 1:
                print(f"   ⚠  agent LLM empty response, retry in {attempts[attempt+1]}s "
                      f"(attempt {attempt+2}/{len(attempts)})")
                continue
            return text, usage  # final attempt — return empty (caller handles PARSE_FAIL)
        except Exception as e:
            last_exc = e
            if attempt < len(attempts) - 1:
                print(f"   ⚠  agent LLM error: {type(e).__name__}: {str(e)[:120]}; "
                      f"retry in {attempts[attempt+1]}s (attempt {attempt+2}/{len(attempts)})")
                continue
            break
    if last_exc:
        raise last_exc
    return "", last_usage


async def _call_llm_async(
    client: OpenAI,
    messages: list[dict],
    model: str,
    max_tokens: int,
    timeout_s: int | float | None,
) -> tuple[str, dict]:
    """Run the blocking streaming LLM call off the event loop with a hard cap."""
    call = asyncio.to_thread(_call_llm, client, messages, model, max_tokens)
    if timeout_s and timeout_s > 0:
        return await asyncio.wait_for(call, timeout=timeout_s)
    return await call


# ── Observation building ──────────────────────────────────────────────────────


async def _build_observation(page: Page) -> tuple[str, list[dict]]:
    """Return (indexed DOM text, interactive_elements list with index→xpath)."""
    try:
        viewport = await page.evaluate("""() => ({
            width: window.innerWidth,
            height: window.innerHeight
        })""")
        viewport_line = f"Viewport: width={int(viewport['width'])} height={int(viewport['height'])}"
    except Exception:
        viewport_line = "Viewport: width=unknown height=unknown"

    root_id, node_map = await _extract_dom_tree(page)
    interactive = _extract_interactive_elements(node_map)
    for item in interactive:
        node = node_map.get(str(item.get("node_id")))
        if isinstance(node, dict):
            item["bbox"] = node.get("boundingBox")
    _annotate_new_elements(page, interactive)
    meta_by_node_id = {it["node_id"]: it for it in interactive}
    dom_text = f"{viewport_line}\n" + _flat_tree_to_llm_text(root_id, node_map, meta_by_node_id)

    cursor_info = await _get_cursor_info(page, interactive)
    if cursor_info:
        dom_text += f"\n{cursor_info}"

    return dom_text, interactive


async def _get_cursor_info(page: Page, interactive: list[dict]) -> str:
    """Query focus and selection state, return a human-readable line."""
    info = await page.evaluate("""() => {
        const ae = document.activeElement;
        if (!ae || ae === document.body || ae === document.documentElement) return null;

        const sel = window.getSelection();
        const selectedText =
            (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA') && ae.selectionStart !== ae.selectionEnd
                ? (ae.value || '').substring(ae.selectionStart, ae.selectionEnd)
                : (sel && sel.rangeCount > 0 && !sel.isCollapsed ? sel.toString() : null);

        let cursorPos = null;
        if (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA') {
            if (ae.selectionStart != null) {
                cursorPos = { start: ae.selectionStart, end: ae.selectionEnd,
                              total: (ae.value || '').length };
            }
        } else if (ae.isContentEditable && sel && sel.rangeCount > 0 && !selectedText) {
            try {
                const walker = document.createTreeWalker(ae, NodeFilter.SHOW_TEXT);
                let charOffset = 0;
                let node;
                const focusNode = sel.focusNode;
                const focusOff = sel.focusOffset;
                if (focusNode === ae) {
                    // focusNode is the element itself — offset is child index
                    let idx = 0;
                    for (const child of ae.childNodes) {
                        if (idx >= focusOff) break;
                        charOffset += (child.textContent || '').length;
                        idx++;
                    }
                } else {
                    while (node = walker.nextNode()) {
                        if (node === focusNode) {
                            charOffset += focusOff;
                            break;
                        }
                        charOffset += node.textContent.length;
                    }
                }
                const totalLen = (ae.innerText || '').length;
                cursorPos = { start: charOffset, end: charOffset, total: totalLen };
            } catch (_) {}
        }

        const tagName = ae.tagName || ae.nodeName || "";
        const normalizedTag = (
            typeof tagName === "string"
                ? tagName
                : (typeof tagName.baseVal === "string" ? tagName.baseVal : String(tagName))
        ).toLowerCase();

        return {
            tag: normalizedTag,
            id: ae.id || null,
            selectedText: selectedText,
            cursorPos: cursorPos,
        };
    }""")
    if not info:
        return ""

    # Match active element to an interactive element's index
    focus_idx = None
    ae_id = info.get("id")
    if ae_id:
        for item in interactive:
            node_attrs = item.get("attributes") or {}
            if node_attrs.get("id") == ae_id:
                focus_idx = item["index"]
                break

    parts = []
    if focus_idx is not None:
        parts.append(f"element [{focus_idx}]")
    else:
        tag = info.get("tag", "?")
        ae_id_str = f" id={ae_id}" if ae_id else ""
        parts.append(f"<{tag}{ae_id_str}>")

    sel_text = info.get("selectedText")
    cursor_pos = info.get("cursorPos")
    if sel_text:
        parts.append(f'"{sel_text}" selected')
    elif cursor_pos:
        start = cursor_pos["start"]
        total = cursor_pos.get("total")
        if total is not None:
            parts.append(f"position {start}/{total}")
        else:
            parts.append(f"position {start}")

    return f"[Cursor: {', '.join(parts)}]"


async def _resolve_index_to_selector(
    page: Page, interactive: list[dict], index: int
) -> tuple[str | None, str | None]:
    """index → (unique XPath selector, xpath). Returns (None, xpath) if selector fails."""
    match = next((x for x in interactive if x["index"] == index), None)
    if match is None:
        return None, None
    xpath = match.get("xpath") or ""
    if xpath:
        xpath_selector = f"xpath={xpath}"
        ok, _count = await _validate_unique_selector(page, xpath_selector)
        if ok:
            return xpath_selector, xpath
    return None, xpath


def _center_click_action_from_index(interactive: list[dict], index: int) -> dict | None:
    """Build a coordinate click for elements that are observable but not selector-resolvable."""
    match = next((x for x in interactive if x.get("index") == index), None)
    if match is None:
        return None
    bbox = match.get("bbox") or {}
    try:
        x = round(float(bbox["x"]) + float(bbox["w"]) / 2)
        y = round(float(bbox["y"]) + float(bbox["h"]) / 2)
    except (KeyError, TypeError, ValueError):
        return None
    return {"type": "clickat", "parameters": {"x": x, "y": y}}


# ── Main entry ────────────────────────────────────────────────────────────────


def _make_task_description(transition: dict) -> str:
    """Flatten an ICG transition into a single-line TASK string for the agent."""
    return transition.get("agent_task", "")


async def run_agent_transition(
    page: Page,
    transition: dict,
    client: OpenAI,
    model: str | None = None,
    max_iter: int = 10,
    max_tokens_per_call: int = 4096,
    agent_timeout_s: int | float | None = None,
    action_timeout_s: int | float | None = None,
    include_assets_table: bool = True,
    log_dir: Path | str | None = None,
    save_screenshots: bool = True,
    before_refresh_cb: Any | None = None,
    after_refresh_cb: Any | None = None,
    reset_cb: Any | None = None,
) -> AgentRunResult:
    """Run the agent loop for a single ICG transition on `page`.

    Parameters:
      page:        Playwright page, already navigated to the starting state.
      transition:  an ICG transition dict; only `id` + `agent_task` are required.
      client:      OpenAI-compatible client.
      model:       override of config.MODEL_AGENT (defaults to that).
      max_iter:    hard cap on agent turns (default 10).
      include_assets_table: include the Available Test Assets table on turn 1.
    """
    model = model or config.MODEL_AGENT
    if agent_timeout_s is None:
        agent_timeout_s = getattr(config, "AGENT_CALL_TIMEOUT_S", 300)
    if action_timeout_s is None:
        action_timeout_s = getattr(config, "ACTION_TIMEOUT_S", 30)
    task_str = _make_task_description(transition)

    messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    trajectory: list[AgentTurn] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    final_selector: str | None = None
    final_target_aff: str | None = None

    status = "MAX_ITER"
    error = ""

    for it in range(1, max_iter + 1):
        # ── 1. Observe ──
        try:
            dom_text, interactive = await _build_observation(page)
        except Exception as e:
            status, error = "ERROR", f"observation failed: {e}"
            break

        # ── 2. Build user message ──
        if it == 1:
            assets_block = (TEST_ASSETS_TABLE_SHORT + "\n\n") if include_assets_table else ""
            user_content = f"TASK: {task_str}\n\n{assets_block}Observation:\n{dom_text}"
        else:
            user_content = f"Observation:\n{dom_text}"
        messages.append({"role": "user", "content": user_content})

        # ── 3. Ask LLM ──
        try:
            response_text, usage = await _call_llm_async(
                client, messages, model, max_tokens_per_call, agent_timeout_s
            )
        except asyncio.TimeoutError:
            status, error = "ERROR", f"LLM call timed out after {agent_timeout_s}s"
            break
        except Exception as e:
            status, error = "ERROR", f"LLM call failed: {e}"
            break
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

        # ── 4. Parse (retry up to 5 times on failure — same prompt each time) ──
        thought, action_text, parsed = _parse_agent_response(response_text)
        if parsed is None:
            parse_retries = 0
            retry_responses = [response_text]
            while parsed is None and parse_retries < 5:
                parse_retries += 1
                try:
                    # Bump tokens on each retry in case truncation caused the parse failure
                    retry_budget = max(max_tokens_per_call, 2048 * (1 + parse_retries))
                    response_text, retry_usage = await _call_llm_async(
                        client, messages, model, retry_budget, agent_timeout_s
                    )
                    retry_responses.append(response_text)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        usage[k] = usage.get(k, 0) + retry_usage.get(k, 0)
                    thought, action_text, parsed = _parse_agent_response(response_text)
                except asyncio.TimeoutError:
                    status, error = "ERROR", f"LLM parse-retry call timed out after {agent_timeout_s}s"
                    break
                except Exception:
                    continue
            if status == "ERROR":
                break

        # Only append assistant reply to history AFTER a successful parse.
        messages.append({"role": "assistant", "content": response_text})
        if parsed is None:
                trajectory.append(AgentTurn(
                    iteration=it, dom_text_chars=len(dom_text),
                    response_text=json.dumps(retry_responses, ensure_ascii=False),
                    thought=thought, action_text=action_text,
                    parsed_action=None, executed=False, exec_ok=False,
                    exec_err=f"parse_fail ({parse_retries} retries)",
                    selector=None, action_dict=None, usage=usage,
                    screenshot=None,
                ))
                status, error = "PARSE_FAIL", f"could not parse Action on turn {it} after {parse_retries} retries"
                break

        verb = parsed["verb"]

        # ── 5. Done? ──
        if verb == "Done":
            trajectory.append(AgentTurn(
                iteration=it, dom_text_chars=len(dom_text),
                response_text=response_text, thought=thought, action_text=action_text,
                parsed_action=parsed, executed=False, exec_ok=True,
                exec_err="", selector=None, usage=usage,
            ))
            status, error = "DONE", ""
            break

        # ── 5b. Reset? ──
        if verb == "Reset":
            reset_ok = False
            reset_err = "no reset callback"
            if reset_cb:
                try:
                    new_page = await reset_cb()
                    if new_page is not None:
                        page = new_page
                    reset_ok = True
                    reset_err = ""
                except Exception as e:
                    reset_err = f"Reset failed: {e}"
            trajectory.append(AgentTurn(
                iteration=it, dom_text_chars=len(dom_text),
                response_text=response_text, thought=thought, action_text=action_text,
                parsed_action=parsed, executed=True, exec_ok=reset_ok,
                exec_err=reset_err, selector=None, usage=usage,
            ))
            if not reset_ok:
                messages.append({
                    "role": "user",
                    "content": f"RESET ERROR: {reset_err}. Continue with the current state.",
                })
            continue

        # ── 6. Resolve index → selector ──
        selector: str | None = None
        fallback_action_dict: dict | None = None
        index = parsed["index"]
        needs_index = verb not in ("Dismiss", "Refresh", "GoBack", "Reset", "Wait", "ClickAt") and not (
            verb == "Scroll" and index == "WINDOW"
        )
        if needs_index:
            if not isinstance(index, int):
                trajectory.append(AgentTurn(
                    iteration=it, dom_text_chars=len(dom_text),
                    response_text=response_text, thought=thought, action_text=action_text,
                    parsed_action=parsed, executed=False, exec_ok=False,
                    exec_err=f"{verb} requires [index]", selector=None, usage=usage,
                ))
                # Feed the error back to the agent and continue
                messages.append({
                    "role": "user",
                    "content": f"ERROR: action `{verb}` requires a numeric index like `[3]`. Retry.",
                })
                continue
            selector, _xpath = await _resolve_index_to_selector(page, interactive, index)
            if selector is None:
                if verb == "Click":
                    fallback_action_dict = _center_click_action_from_index(interactive, index)
                if fallback_action_dict is not None:
                    selector = None
                else:
                    trajectory.append(AgentTurn(
                        iteration=it, dom_text_chars=len(dom_text),
                        response_text=response_text, thought=thought, action_text=action_text,
                        parsed_action=parsed, executed=False, exec_ok=False,
                        exec_err=f"index [{index}] did not resolve to a unique selector",
                        selector=None, usage=usage,
                    ))
                    messages.append({
                        "role": "user",
                        "content": (
                            f"ERROR: index [{index}] did not resolve to a unique element "
                            f"(perhaps stale). Re-read the latest observation and try again."
                        ),
                    })
                    continue

        # ── 7. Execute via shared executor ──
        action_dict = fallback_action_dict or _verb_to_action_dict(verb, index, parsed["arg"])
        if verb == "Click" and fallback_action_dict is not None:
            repeat = _parse_repeat_count(parsed["arg"])
            if repeat is not None:
                action_dict.setdefault("parameters", {})["times"] = repeat

        # Resolve Drag target index [M] → selector
        if verb == "Drag" and "target_index_raw" in action_dict.get("parameters", {}):
            raw = action_dict["parameters"].pop("target_index_raw")
            import re as _re
            _m = _re.fullmatch(r"\[(\d+)\]", raw.strip())
            if not _m:
                trajectory.append(AgentTurn(
                    iteration=it, dom_text_chars=len(dom_text),
                    response_text=response_text, thought=thought, action_text=action_text,
                    parsed_action=parsed, executed=False, exec_ok=False,
                    exec_err=f"Drag target must be [M] or offset_x=...,offset_y=... — got: {raw!r}",
                    selector=None, usage=usage,
                ))
                messages.append({"role": "user", "content":
                    f"ERROR: Drag target must be an index like `[3]` or `offset_x=0,offset_y=100`. Got: {raw!r}. Retry."})
                continue
            tgt_index = int(_m.group(1))
            tgt_selector, _ = await _resolve_index_to_selector(page, interactive, tgt_index)
            if not tgt_selector:
                trajectory.append(AgentTurn(
                    iteration=it, dom_text_chars=len(dom_text),
                    response_text=response_text, thought=thought, action_text=action_text,
                    parsed_action=parsed, executed=False, exec_ok=False,
                    exec_err=f"Drag target index [{tgt_index}] did not resolve to a unique selector",
                    selector=None, usage=usage,
                ))
                messages.append({"role": "user", "content":
                    f"ERROR: Drag target index [{tgt_index}] did not resolve to a unique element. "
                    f"Re-read the latest observation and retry with a valid index."})
                continue
            action_dict["parameters"]["target_selector"] = tgt_selector

        aff_dict = {
            "id": f"_agent_turn_{it}",
            "type": _VERB_INFO[verb]["aff"],
            "description": action_text,
        }
        if verb in ("Refresh", "GoBack") and before_refresh_cb:
            try:
                await before_refresh_cb()
            except Exception:
                pass
        try:
            action_call = execute_action(page, action_dict, aff_dict, selector)
            if action_timeout_s and action_timeout_s > 0:
                exec_ok, exec_err = await asyncio.wait_for(action_call, timeout=action_timeout_s)
            else:
                exec_ok, exec_err = await action_call
        except asyncio.TimeoutError:
            exec_ok, exec_err = False, f"action timed out after {action_timeout_s}s"
        except Exception as e:
            exec_ok, exec_err = False, f"execute_action raised: {e}"
        if verb in ("Refresh", "GoBack") and exec_ok and after_refresh_cb:
            try:
                await after_refresh_cb()
            except Exception:
                pass

        screenshot_path = None
        if save_screenshots and log_dir is not None:
            try:
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
                shot = log_path / f"{tid}_iter{it:02d}_{verb.lower()}.png"
                await page.screenshot(path=str(shot), full_page=False, timeout=3000)
                screenshot_path = str(shot.name)
            except Exception:
                screenshot_path = None

        trajectory.append(AgentTurn(
            iteration=it, dom_text_chars=len(dom_text),
            response_text=response_text, thought=thought, action_text=action_text,
            parsed_action=parsed, executed=True, exec_ok=bool(exec_ok),
            exec_err=exec_err or "", selector=selector,
            action_dict=action_dict, usage=usage,
            screenshot=screenshot_path,
        ))

        # Remember the most recent concrete action — useful for assertions that
        # reference "The target" (last acted-on element).
        if exec_ok and selector and verb not in ("Wait", "Scroll", "Refresh", "Hover"):
            final_selector = selector
            final_target_aff = _VERB_INFO[verb]["aff"]

        # One-shot stuck-loop guard: fires only on the penultimate iteration.
        # If the last 3 actions are identical (same verb + same index), the feature
        # is likely not implemented — force Done rather than burning the final turn.
        if it == max_iter - 1 and len(trajectory) >= 3:
            def _action_key(t):
                p = t.parsed_action or {}
                return (p.get("verb"), p.get("index"))
            last3 = [_action_key(t) for t in trajectory[-3:]]
            if len(set(last3)) == 1 and last3[0] != (None, None):
                status = "DONE"
                break

        # Feed execution outcome back to the agent so it can correct course.
        if not exec_ok:
            messages.append({
                "role": "user",
                "content": f"EXECUTION ERROR: {exec_err}. Retry with a different element or action.",
            })
        # On success, next turn's `user` message will carry the new observation.

    # ── Final post-loop screenshot for postcondition judgement ──
    # Honors transition["screenshot_mode"]: default is full-page; "viewport" → viewport-only.
    final_screenshot_path: str | None = None
    if log_dir is not None:
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            use_full_page = transition.get("screenshot_mode") != "viewport"
            tid = transition.get("id", "T?")
            final_shot = log_path / f"{tid}_final.png"
            await page.screenshot(path=str(final_shot), full_page=use_full_page, timeout=5000)
            final_screenshot_path = final_shot.name
        except Exception:
            final_screenshot_path = None

    result = AgentRunResult(
        status=status,
        iterations=len(trajectory),
        trajectory=trajectory,
        final_selector=final_selector,
        final_target_affordance_type=final_target_aff,
        error=error,
        total_usage=total_usage,
        final_screenshot=final_screenshot_path,
    )

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{transition.get('id', 'T?')}_agent_trajectory.json"
        payload = {
            "transition_id": transition.get("id"),
            "task": task_str,
            "model": model,
            "max_iter": max_iter,
            "full_messages": messages,
            **result.to_dict(),
        }
        (log_dir / fname).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return result


# ── CLI / standalone smoke test ───────────────────────────────────────────────


async def _smoke_test(html_path: str, agent_task: str) -> None:
    """Quick standalone run: load an HTML file, drive one agent_task via agent."""
    from playwright.async_api import async_playwright

    api_key, base_url = config.get_credentials()
    client = OpenAI(api_key=api_key, base_url=base_url)
    transition = {"id": "TSMOKE", "agent_task": agent_task}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=config.BROWSER_HEADLESS)
        context = await browser.new_context(
            viewport={"width": config.VIEWPORT_WIDTH, "height": config.VIEWPORT_HEIGHT}
        )
        page = await context.new_page()
        await page.goto(Path(html_path).absolute().as_uri(),
                        wait_until="load", timeout=config.PAGE_LOAD_TIMEOUT_MS)
        await page.wait_for_timeout(600)

        result = await run_agent_transition(page, transition, client, max_iter=8)

        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        await browser.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Agent-loop smoke test.")
    ap.add_argument("--html", required=True)
    ap.add_argument("--intent", required=True, help="ICG agent_task text")
    args = ap.parse_args()

    import asyncio
    asyncio.run(_smoke_test(args.html, args.intent))
