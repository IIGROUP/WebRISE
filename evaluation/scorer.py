"""
Scorer - Uses an MLLM (vision) to evaluate whether ICG postconditions hold after an action.

Both the before (pre) and after (post) screenshots are passed for temporal context.
Returns scored conditions, token usage, and a full API call log.
"""

import base64
import json
import re
import time
from openai import OpenAI

import config

_RETRY_DELAYS = [3, 8, 20]  # seconds between retries for transient 5xx errors


def _stream_chat(client: OpenAI, model: str, max_tokens: int, messages: list) -> tuple[str, dict]:
    """Call the chat API in streaming mode and return (full_text, usage_dict).

    Streaming avoids gateway timeouts on long vision requests.
    response_format is intentionally omitted — _extract_json_payload handles parsing.
    Usage is collected from the final stream chunk's usage field when available.
    """
    chunks = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        **config.reasoning_effort_kwargs(model),
    ) as stream:
        for chunk in stream:
            delta_obj = chunk.choices[0].delta if chunk.choices else None
            delta = getattr(delta_obj, "content", None)
            if delta:
                chunks.append(delta)
            if getattr(chunk, "usage", None):
                u = chunk.usage
                usage = {
                    "prompt_tokens":     getattr(u, "prompt_tokens",     0),
                    "completion_tokens": getattr(u, "completion_tokens", 0),
                    "total_tokens":      getattr(u, "total_tokens",      0),
                }
    return "".join(chunks), usage

SCORER_SYSTEM_PROMPT = """\
You are a strict UI test evaluator. You will compare two webpage screenshots
(Image 1 is the previous step, Image 2 is the current step) and determine
whether each of the following assertions is true of the current page.

You will receive:
  • Image 1 — screenshot of the page BEFORE the interaction
  • Image 2 — screenshot of the page AFTER the interaction
  • A short description of the action that was performed
  • A numbered list of assertions to verify

**Semantic equivalence — ALWAYS judge by meaning, not wording.** The assertion
and the page may use different terms for the same concept. Examples: "word count"
vs "character count", "publish" vs "post", "search bar" vs "filter input",
"delete" vs "remove", "avatar" vs "profile picture". If the page element serves
the same functional purpose as what the assertion describes, answer YES. Only
answer NO if the described element/feature is entirely absent or serves a clearly
different purpose.

Be strict on factual correctness, but lenient on terminology: answer YES if the
assertion is satisfied in substance, even if the page uses different labels or
phrasing.

Conditional assertions: if an assertion contains "if…then…" or "if…otherwise…"
branching logic, first determine which branch applies from what you observe,
then evaluate only that branch. A conditional assertion passes if the applicable
branch is satisfied — do not fail because an inapplicable branch was not checked.

Exception — viewport-clipped content: if an assertion says items have a certain
structure (e.g. "each with avatar, name, and timestamp") or requires a certain
count, first check whether the content is cut off at the edge of its container
or the viewport (the last visible item is partially clipped, or a scrollable
container does not show all content). If yes: evaluate only the FULLY VISIBLE
items — do not fail because a clipped or out-of-view item is missing a field.
The rendering is correct, the viewport just cannot show everything at once. Only
fail if the fully visible items clearly lack the required structure.

Exception — small numeric increments: if an assertion requires a count or
percentage to have increased by a small amount (e.g. +1 vote, a fraction of a
percent), check how the numbers are displayed. If numbers are abbreviated (e.g.
"1.2K") or percentages are rounded to whole numbers and both images round to the
same value (e.g. both show "28%"), the numeric change may be invisible — in this
case answer YES if the structural outcome is otherwise correct (e.g. the option
is now marked as voted/selected), or UNCERTAIN if the structural outcome cannot
be confirmed. Only answer NO if the numbers are displayed as full unabbreviated
integers and the expected increment is clearly absent.

Exception — filter/tag applied but result set is empty: if an assertion says that filtered or searched results are shown (e.g. "posts matching the selected tag appear", "items for the chosen category are displayed") and Image 2 shows an empty state or a no-results indicator, but the filter/tag IS visibly active (e.g. a selected chip, a highlighted tag, an active filter pill, a search query still present in the input), answer YES — the filtering action was correctly applied; the empty result is a valid outcome when no content matches the active filter, not a failure of the feature.

Exception — body/full-text search results: if an assertion checks search or
filter results that may match hidden, truncated, or non-preview body text, do
not answer UNCERTAIN merely because every visible result row does not display
the matching keyword. If the search query is visibly active, the visible result
set has narrowed or changed, and at least one visible or open result clearly
demonstrates a body-text match, answer YES unless
there is clear evidence that unrelated results are being shown.

Exception — image flip on ambiguous natural images: if an assertion about an
image editor requires a horizontal or vertical flip as part of a sequence of
edits, but the image content is a natural scene with no clear left/right or
up/down marker, do not answer UNCERTAIN solely because the flip is visually hard
to distinguish. If the other requested edits are clearly visible and there is no
contradictory evidence, answer YES.

Respond with a JSON object in exactly this format:
{
  "evaluations": [
    {"think": "<brief reasoning>", "result": "YES"},
    {"think": "<brief reasoning>", "result": "NO"},
    {"think": "<brief reasoning>", "result": "UNCERTAIN"}
  ]
}

One entry per assertion, in the same order as the input list.
"result" must be exactly one of: YES / NO / UNCERTAIN
Do not include any prose, markdown fences, or extra text before or after the JSON object.
"""

SCORER_USER_TEMPLATE = """\
Action performed between Image 1 and Image 2: {action_desc}

Assertions to verify on Image 2:
{conditions_numbered}
"""

PRECONDITION_SYSTEM_PROMPT = """\
You are a strict UI test evaluator. Your job is to determine whether an initial page screenshot
already satisfies a set of preconditions before any interaction is performed.

You will receive:
  • One screenshot of the currently opened page
  • A numbered list of preconditions to verify

Evaluate each precondition using ONLY the provided screenshot.
Be strict: only answer YES if the condition is clearly and unambiguously satisfied.

**Semantic equivalence — ALWAYS judge by meaning, not wording.** The condition
and the page may use different terms for the same concept. Examples: "word count"
vs "character count", "publish" vs "post", "search bar" vs "filter input",
"delete" vs "remove", "avatar" vs "profile picture". If the page element serves
the same functional purpose as what the condition describes, answer YES. Only
answer NO if the described element/feature is entirely absent or serves a clearly
different purpose.

Be strict on factual correctness, but lenient on terminology: answer YES if the
condition is satisfied in substance, even if the page uses different labels or
phrasing.

Conditional preconditions: if a condition contains "if…then…" or
"if…otherwise…" branching logic, first determine which branch applies from what
you observe, then evaluate only that branch. A conditional precondition passes
if the applicable branch is satisfied — do not fail because an inapplicable
branch was not checked.

Exception — semantic equivalence for text content: when a condition references
specific text labels (titles, topic names, option labels, post copy, product
names, etc.), judge by **semantic meaning**, not literal string match. If the
page displays content that conveys the same concept as the described text —
even with different wording — answer YES. Only fail on text content if the
meaning is clearly different or entirely absent.

Exception — viewport-clipped content: if a condition says items have a certain
structure (e.g. "each with avatar, name, and timestamp") or requires a certain
count, first check whether the content is cut off at the edge of its container
or the viewport (the last visible item is partially clipped, or a scrollable
container does not show all content). If yes: evaluate only the FULLY VISIBLE
items — do not fail because a clipped or out-of-view item is missing a field.
The rendering is correct, the viewport just cannot show everything at once. Only
fail if the fully visible items clearly lack the required structure.

Exception — small numeric values: if a condition requires a count or percentage
to show a small value, check how the numbers are displayed. If numbers are
abbreviated (e.g. "1.2K") or percentages are rounded to whole numbers, the exact
small value may be invisible — in this case answer YES if the structural outcome
is otherwise correct, or UNCERTAIN if the structural outcome cannot be
confirmed. Only answer NO if the numbers are displayed as full unabbreviated
integers and the expected value is clearly absent.

Exception — filter/tag active but result set is empty: if a condition says that
filtered or searched results are shown (e.g. "posts matching the selected tag
appear", "items for the chosen category are displayed") and the screenshot shows
an empty state or a no-results indicator, but the filter/tag IS visibly active
(e.g. a selected chip, a highlighted tag, an active filter pill, a search query
still present in the input), answer YES — the filtering state is correctly
applied; the empty result is a valid outcome when no content matches the active
filter, not a failure of the feature.

Exception — feature already open satisfies access condition: if a condition
says "a touchpoint / control / trigger to access [feature X] is present" or
"[feature X] is accessible", and the screenshot shows that feature X is ALREADY
fully open and visible (e.g. a panel is already expanded, a sidebar is already
displayed, a dialog is already open), answer YES — the feature is by definition
accessible; it does not matter whether a separate toggle button is also visible.
Do NOT answer NO simply because you cannot find a dedicated open/access button
when the target feature itself is already present on screen.

Respond with a JSON object in exactly this format:
{
  "evaluations": [
    {"think": "<brief reasoning>", "result": "YES"},
    {"think": "<brief reasoning>", "result": "NO"},
    {"think": "<brief reasoning>", "result": "UNCERTAIN"}
  ]
}

One entry per condition, in the same order as the input list.
"result" must be exactly one of: YES / NO / UNCERTAIN
Do not include any prose, markdown fences, or extra text before or after the JSON object.
"""

PRECONDITION_USER_TEMPLATE = """\
Preconditions to verify:
{conditions_numbered}
"""


def _b64_url(image_bytes: bytes) -> str:
    b64 = base64.standard_b64encode(image_bytes).decode()
    return f"data:image/png;base64,{b64}"


def _format_action(action: dict, affordance_type: str) -> str:
    value = action.get("value", "")
    return (
        f"{action['type']} '{value}' on {affordance_type}"
        if value
        else f"{action['type']} on {affordance_type}"
    )


async def score_postconditions(
    client: OpenAI,
    pre_screenshot: bytes,
    post_screenshot: bytes,
    conditions: list[str],
    action: dict,
    affordance_name: str,
    pre_screenshot_ref:  str = "[pre_screenshot]",
    post_screenshot_ref: str = "[post_screenshot]",
    action_desc: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Score each postcondition using before/after screenshots.

    If `action_desc` is provided, it is used verbatim as the action description
    in the prompt. Otherwise it is synthesized from `action` and
    `affordance_name`.

    Returns:
      (results, call_log)

      results:  list of { "condition": str, "verdict": str, "passed": bool, "think": str }
      call_log: { model, input_messages, response_text,
                  prompt_tokens, completion_tokens, total_tokens }
    """
    conditions_numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(conditions))
    if action_desc is None:
        action_desc = _format_action(action, affordance_name or "element")

    user_text = SCORER_USER_TEMPLATE.format(
        action_desc=action_desc,
        conditions_numbered=conditions_numbered,
    )

    messages = [
        {"role": "system", "content": SCORER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text",      "text": "Image 1 — page BEFORE the action:"},
                {"type": "image_url", "image_url": {"url": _b64_url(pre_screenshot),  "detail": "high"}},
                {"type": "text",      "text": "Image 2 — page AFTER the action:"},
                {"type": "image_url", "image_url": {"url": _b64_url(post_screenshot), "detail": "high"}},
                {"type": "text",      "text": user_text},
            ],
        },
    ]

    response_text = usage = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            print(f"   ⚠  scorer retry in {delay}s (attempt {attempt+1})...")
            time.sleep(delay)
        try:
            response_text, usage = _stream_chat(
                client, config.MODEL_SCORER, config.SCORER_MAX_TOKENS, messages
            )
            if response_text:
                break
            print(f"   ⚠  scorer returned empty content (attempt {attempt+1})")
        except Exception as e:
            if attempt == len(_RETRY_DELAYS):
                raise
            print(f"   ⚠  scorer error: {type(e).__name__}: {str(e)[:160]}")

    if not response_text:
        raise ValueError("Scorer API returned empty content after retries")

    # Build loggable input (replace base64 blobs with file references)
    loggable_messages = [
        {"role": "system", "content": SCORER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text",      "text": "Image 1 — page BEFORE the action:"},
                {"type": "image_url", "image_url": {"url": pre_screenshot_ref}},
                {"type": "text",      "text": "Image 2 — page AFTER the action:"},
                {"type": "image_url", "image_url": {"url": post_screenshot_ref}},
                {"type": "text",      "text": user_text},
            ],
        },
    ]

    call_log = {
        "model":          config.MODEL_SCORER,
        **config.reasoning_effort_kwargs(config.MODEL_SCORER),
        "input_messages": loggable_messages,
        "response_text":  response_text,
        **usage,
    }

    results = _parse_verdicts(response_text, conditions)
    return results, call_log


async def score_preconditions(
    client: OpenAI,
    screenshot: bytes,
    conditions: list[str],
    screenshot_ref: str = "[screenshot]",
) -> tuple[list[dict], dict]:
    """
    Score each initial precondition using a single screenshot.

    Returns:
      (results, call_log)

      results:  list of { "condition": str, "verdict": str, "passed": bool, "think": str }
      call_log: { model, input_messages, response_text,
                  prompt_tokens, completion_tokens, total_tokens }
    """
    conditions_numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(conditions))
    user_text = PRECONDITION_USER_TEMPLATE.format(
        conditions_numbered=conditions_numbered,
    )

    messages = [
        {"role": "system", "content": PRECONDITION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text",      "text": "Screenshot of the currently opened page:"},
                {"type": "image_url", "image_url": {"url": _b64_url(screenshot), "detail": "high"}},
                {"type": "text",      "text": user_text},
            ],
        },
    ]

    response_text = usage = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            print(f"   ⚠  scorer retry in {delay}s (attempt {attempt+1})...")
            time.sleep(delay)
        try:
            response_text, usage = _stream_chat(
                client, config.MODEL_SCORER, config.SCORER_MAX_TOKENS, messages
            )
            if response_text:
                break
            print(f"   ⚠  scorer returned empty content (attempt {attempt+1})")
        except Exception as e:
            if attempt == len(_RETRY_DELAYS):
                raise
            print(f"   ⚠  scorer error: {type(e).__name__}: {str(e)[:160]}")

    if not response_text:
        raise ValueError("Scorer API returned empty content after retries")

    loggable_messages = [
        {"role": "system", "content": PRECONDITION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text",      "text": "Screenshot of the currently opened page:"},
                {"type": "image_url", "image_url": {"url": screenshot_ref}},
                {"type": "text",      "text": user_text},
            ],
        },
    ]

    call_log = {
        "model":          config.MODEL_SCORER,
        **config.reasoning_effort_kwargs(config.MODEL_SCORER),
        "input_messages": loggable_messages,
        "response_text":  response_text,
        **usage,
    }

    results = _parse_verdicts(response_text, conditions)
    return results, call_log


def _parse_verdicts(text: str, conditions: list[str]) -> list[dict]:
    # Parse JSON response; fall back to UNCERTAIN on any error
    try:
        data = json.loads(text)
        evals = data.get("evaluations", [])
    except (json.JSONDecodeError, AttributeError):
        try:
            data = _extract_json_payload(text)
            evals = data.get("evaluations", [])
        except Exception:
            evals = []

    results = []
    for i, condition in enumerate(conditions):
        entry = evals[i] if i < len(evals) else {}
        raw = str(entry.get("result", "")).strip().upper()
        if raw.startswith("YES"):
            verdict = "YES"
        elif raw.startswith("NO"):
            verdict = "NO"
        else:
            verdict = "UNCERTAIN"
        results.append({
            "condition": condition,
            "verdict":   verdict,
            "passed":    verdict == "YES",
            "think":     entry.get("think", ""),
        })
    return results


def _extract_json_payload(text: str) -> dict:
    """Best-effort extraction for providers that ignore JSON-only instructions."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response text")

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if fenced:
        fenced_text = fenced.group(1).strip()
        try:
            data = json.loads(fenced_text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        text = fenced_text

    # Prefer JSON objects that explicitly contain the expected top-level key.
    starts = []
    for match in re.finditer(r'"evaluations"\s*:', text):
        start = text.rfind("{", 0, match.start())
        if start != -1:
            starts.append(start)
    starts.extend(m.start() for m in re.finditer(r"\{", text))

    seen = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        payload = _balanced_json_object_at(text, start)
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "evaluations" in data:
            return data

    raise ValueError("No JSON object found in response text")


def _balanced_json_object_at(text: str, start: int) -> str | None:
    """Return the balanced JSON-like object substring beginning at start."""
    if start < 0 or start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
