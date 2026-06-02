#!/usr/bin/env python3
"""Generate WebRISE interaction contracts from task requirements.

The pipeline derives a Test Data Contract, requirement-linked test items,
and an Interaction Contract Graph (ICG) without relying on a reference DOM.

Example:
  python icg_pipeline.py --input data_release/requirements_full.json \
    --data-dir generated_icg --filter D01_S01_T003

Intermediate files:
  <data-dir>/<task_id>/contract.json
  <data-dir>/<task_id>/test_items.json
  <data-dir>/<task_id>/icg.json
"""

from __future__ import annotations
import json, argparse, os, sys, re, time, uuid
from pathlib import Path
from datetime import datetime, timezone

# httpx/openai fails when SSL_CERT_FILE points to a missing file.
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

# Load a local .env file when present.
try:
    from dotenv import load_dotenv
    for _p in [
        Path(__file__).resolve().parent.parent / ".env",
        Path(".env"),
    ]:
        if _p.exists():
            load_dotenv(_p); break
except ImportError:
    pass

# ── Configuration ─────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("OPENAI_API_KEY", "").strip()
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
MODEL    = os.environ.get("WEBRISE_ICG_MODEL", "claude-opus-4-6")
MAX_TOKENS = 32000
INPUT_JSON       = "requirements_full.json"
DATA_DIR         = "./generated_icg"
API_CALL_LOG_DIR = "./api_calls"
DELAY = 5
MAX_CONSECUTIVE_FAILS = 3
TEST_TASK = "D01_S01_T003"


# ══════════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════════

# ── Call 1a: Contract only ──────────────────────────────────────────────────
SYSTEM_CALL1A_CONTRACT_V9 = """You are a frontend test specification designer.

Your task: given a requirement list, produce a minimal **test data contract** describing what the page must be functionally ready to do on first load.

No reference implementation is provided — derive everything from the requirements alone.

---

## Test Data Contract

The contract is the minimal set of **functional preconditions** that describe the page's ready state before any user interaction. It answers: "what must already be in place for the first test action to be possible?"

### Contract Rules

1. **Describe functional readiness, not UI structure.**
   - ✅ "The page has pre-loaded content that can be interacted with."
   - ✅ "The page starts in offline mode."
   - ✅ "A default conversation is open and messages are visible."
   - ❌ "Each message bubble displays an avatar, nickname, and timestamp." — UI structural detail.
   - ❌ "The feed renders 3 post cards." — specific count.
   - ❌ "The contact list is on the left side." — positional layout.

2. **For multi-page, multi-view, tabbed, wizard, or navigation-based apps, explicitly name the initial page/view/mode shown on first load.**
   - ✅ "The initial page on first load is the Discover page."
   - ✅ "The initial view on first load is the Inbox view."
   - ❌ "The Discover tab is at the top." — positional layout.
   Use functional page/view/route names only; do not describe physical placement. If the requirements imply multiple pages/views but do not specify a default, choose a reasonable initial page/view for the primary workflow and state it explicitly for the code model.

3. **No counts, no specific data values, no element-level structure, no positional descriptions, no asset sourcing.** The required initial page/view/mode name is allowed.

4. **Only describe what the requirements imply is necessary.** Do not invent initial conditions not derivable from the requirements, except for choosing the required initial page/view/mode when a multi-page app has no specified default.

5. **Do not expose any IMPLICIT requirements.** Cross-check every sentence against implicit requirements and remove any semantic overlap.

---

## Output Format

Return exactly one JSON object and nothing else:

{
  "test_data_contract": "string — functional preconditions describing page readiness"
}

No prose, no markdown fences, no extra text outside the JSON.
"""

# ── Call 1b: Test Items only ────────────────────────────────────────────────
SYSTEM_CALL1B_TESTITEMS_V9 = """You are a frontend test specification designer.

Your task: given a requirement list, produce a **test items list** covering every testable behavior in the requirements.

No reference implementation is provided — derive everything from the requirements alone.

---

## Test Items List

Generate one test item per distinct testable behavior. Every requirement ID (explicit and implicit) must appear in at least one test item's `req_ids`.

### Rules for test items

- **Implementation-neutral**: trigger and expected_result must describe user intent and semantic outcome, not specific control names, DOM structure, or positions.
- **Trigger = user intent**, not a click sequence. What is the user trying to accomplish? The trigger must describe a user **action** — something the user *does*. It must NOT be a pure observation or verification (e.g., "observe whether X is visible", "check that Y appears", "verify the count"). If what you want to verify is already the expected result of another trigger, fold the check into that item's `expected_result` instead of creating a separate observation-only trigger. The trigger must also NOT describe attempting to operate on a disabled or non-interactive control (e.g., "attempt to click the disabled button", "try to submit the locked form") — write the plain user action instead and let `expected_result` describe the blocked outcome.
- **Expected result = semantic outcome**. What does the user observe when the feature works correctly?
- **Combine tightly coupled behaviors** (same trigger, same artifact) into a single test item.
- **Split multi-case requirements**: if a requirement lists named cases (e.g., error type A vs. B), create one item per case.
- **Only primary requirements in `req_ids` — max 2 per item**: A requirement belongs in a test item's `req_ids` if and only if this test item's `expected_result` explicitly describes verifying that requirement as a **direct and primary outcome**. Do NOT add requirements that are merely ambient, incidental, or display-context side effects of the action. Hard rule: **at most 2 req_ids per test item**. If you believe 3 are inseparably observable in one action, 3 is the absolute ceiling. If verifying req A and req B require different user actions or produce separately observable outcomes, they belong in separate test items.
- **No unbound requirement checks**: If an item's `expected_result` explicitly verifies a requirement, that requirement ID MUST appear in the item's `req_ids`. If adding it would exceed the max-2 rule, split that verification into a separate test item instead of mentioning the requirement only in prose.
- **Do not invent behaviors** not described in the requirements. This rule is absolute. In particular:
  - **Failure / error / boundary scenarios are strictly requirement-scoped.** Only create test items for failure modes that are explicitly named in the requirements. Do NOT extrapolate additional failure modes you consider plausible. For example: if the requirements only mention "content must not be empty" and "character limit of 200", do NOT add test items for network errors, server failures, upload failures, rate limiting, or any other failure not named in the requirements. Each failure-scenario test item must trace directly to a requirement sentence that names that specific failure.
  - **Failure follow-up behavior must reuse a named failure.** If a test item checks what happens after failure/prevention, its trigger must use a specific failure/boundary case named in the requirements; never use a generic or unspecified failure trigger.
  - **Implicit requirements expand on explicitly stated behaviors — they do not introduce new scenarios.** An implicit requirement may add detail (e.g. "error feedback must be clear") to an explicit scenario, but cannot by itself justify a test item for a scenario the explicit requirements never mention.
- **Direction-neutral toggles**: When a requirement describes switching between two states (sort order, view mode, tab, theme toggle), write the trigger as switching FROM the current state TO the alternative — do NOT name which state is "default". Example: `"toggle the sort order so posts appear in a different sequence"` — NOT `"switch to hotness sort"`. The implementation may choose any default; the test verifies that switching produces an observable change, not which state is default.
- **Countdown-timer flows must not be split**: If a feature involves a countdown or expiry timer (e.g., an edit window, a send-code cooldown, a session timeout), do NOT split the trigger action (opening the timed mode) and the completion action (submitting/confirming) into separate test items. Combine the full open-and-submit flow into a single test item whose `trigger` covers the complete user intent from start to finish, so the entire flow executes in one agent run without interruption.
- **Do not create standalone "negative capability" test items**: If a requirement implies that operation X becomes unavailable after a state change (e.g. a recalled message cannot be edited, a deleted item cannot be restored), do NOT generate a separate test item whose sole purpose is "verify X is not available." Instead, fold that constraint into the `expected_result` of the test item that triggers the state change.
- **Selection-gated controls are one behavior, not two**: If a requirement says a control is unavailable with no selection and becomes available after selecting content (e.g. a comment trigger), create ONE test item whose trigger selects the content and invokes the control. Its expected result should mention both sides: no operation is available before selection, and after selection the operation becomes available / opens the intended mode. Do NOT create a separate item that asks the agent to try operating the disabled or absent control without selecting content.
- **Guard/prevention behaviors with a distinct trigger ARE valid separate test items**: If a requirement says an operation is blocked when a precondition is not met (e.g. submitting is prevented when the input is empty), and that block has its own distinct user trigger and a meaningfully observable outcome, create it as a separate test item. Its trigger is the user's action taken without satisfying the precondition; its expected_result describes the observable prevention. Do NOT describe the before/after precondition states in the trigger — just describe the user action.

---

## Output Format

Return exactly one JSON object and nothing else:

{
  "test_items": [
    {
      "item_id": "TI-1",
      "req_ids": ["full_req_id_1", "full_req_id_2"],
      "description": "Brief label for this test item",
      "trigger": "Implementation-neutral description of the user action",
      "expected_result": "Semantic outcome that indicates success"
    }
  ]
}

No prose, no markdown fences, no extra text outside the JSON.
"""

# ── Call 2: Contract check, ensuring implicit requirements are not exposed ───
SYSTEM_CHECK_CONTRACT_V1 = """You are a test data contract reviewer.

You perform FOUR checks on the given test data contract:

## Check 1: Implicit requirement exposure
Determine whether any sentence in the contract inadvertently exposes, describes, or hints at an implicit requirement.

An implicit requirement is "exposed" if any sentence in the contract directly describes the behavior the implicit requirement specifies.

You will also receive explicit requirements. Do NOT flag contract text that only restates or minimally supports an explicit requirement, even if it shares vocabulary with an implicit requirement. Flag only when the contract reveals the implicit requirement's extra behavioral outcome, lifecycle update, guard condition, or post-action state beyond what the explicit requirements already say.

Use **semantic matching**, not just literal overlap. For example: "the page must restore the editor text on refresh" and an implicit requirement saying "state is persisted and auto-restored across page refreshes" describe the same behavior — this counts as an exposure even if the wording is different.

## Check 2: Positional / layout descriptions
The contract must describe WHAT is present, not WHERE it appears. Flag any sentence that uses positional or layout qualifiers such as: "left panel", "right side", "top bar", "bottom section", "sidebar on the left", "header at the top", "footer area", etc. These are implementation details that constrain the code model's layout choices.

Naming the initial page/view/mode shown on first load is allowed and required for multi-page apps; this is not a positional or layout description as long as it does not mention physical placement.

## Check 3: Initial page/view clarity
If the contract describes a multi-page, multi-view, tabbed, wizard, or navigation-based app, it must explicitly state which page/view/mode is shown on first load. Flag contracts that mention multiple pages, views, tabs, routes, or navigation without naming the initial page/view/mode.

## Check 4: Language
The contract must be written entirely in English. Flag any Chinese characters or other non-English text.

You will receive:
1. **Explicit requirements** — behaviors the contract may minimally support
2. **Implicit requirements** — behaviors that must NOT appear in the contract unless already covered by explicit requirements
3. **Test data contract** — the text to evaluate

Respond with exactly one JSON object and nothing else:
{
  "verdict": "PASS",
  "issues": []
}
or:
{
  "verdict": "FAIL",
  "issues": ["<req_id or 'LAYOUT' or 'INITIAL_PAGE' or 'LANGUAGE'>: <brief description of the problem>", ...]
}

- "verdict": "PASS" if ALL checks pass; "FAIL" if any check finds issues.
- "issues": empty list for PASS; one entry per problem for FAIL. Use the req_id for implicit exposure issues, use "LAYOUT" for positional description issues, use "INITIAL_PAGE" for missing initial page/view/mode issues, use "LANGUAGE" for non-English text issues.
- No prose, no markdown fences, no extra text outside the JSON object.
"""

# ── Call 4: ICG JSON from test items ─────────────────────────────────────────
SYSTEM_CALL4_ICG_V9 = r"""You are a frontend interaction test case designer. Generate a **test specification** for a frontend task: `states` and `transitions`, each transition carrying a semantic `agent_task` and two lists of natural-language expected outcomes (`dom_assertions` + `postconditions`).

Your output is implementation-neutral: describe *what* the test checks and *why*, never *how* the HTML realizes it.

**Classify each expected outcome** into one of two lists based on HOW it is verifiable:
- **`dom_assertions`** — verifiable from DOM mutation events and element state: an element appearing or disappearing, its text content changing, or it becoming enabled/disabled, selected/deselected, or visible/hidden. Each item MUST be prefixed with `[CHANGE]` (transient/process-level) or `[AFTER]` (final-state after the action settles).
- **`postconditions`** — verifiable by comparing the before-action and after-action screenshots of the page. Use for outcomes the reader judges visually: a persistent visual change occurred, or the final visual state matches a description. No prefix tag.

## Inputs

You will receive:
1. **Test Items List** — the primary design guide. Each item has: `item_id`, `description`, `trigger` (user intent), `expected_result` (semantic success outcome).
2. **Test Data Contract** — functional preconditions for S0.

### How to use test items

- Add `"mapped_test_items": [{"id": "TI-n"}, ...]` to every transition to declare which test items this transition verifies. **Default to 1 test item per transition; 2 is normal only when the two expected outcomes share the same trigger, same artifact/page, and same final state. 3 is rare and only allowed when all three outcomes are inseparably produced by the exact same user action and are directly verifiable from the same final page state.**
- **No hitchhiking test items**: do NOT attach a test item merely because it is related to the same feature area, visible elsewhere in the app, or could be checked after extra navigation/filter changes. Every mapped test item MUST be directly caused by the transition's `agent_task`, and the final `to` state/screenshot/DOM must make that item's `expected_result` verifiable without doing additional actions.
- **One mapped test item needs one direct assertion**: for every ID in `mapped_test_items`, at least one `dom_assertion` or `postcondition` in this same transition must directly verify that item's `expected_result`. If a test item needs a different trigger, a different category/filter/page, or a different setup state, split it into another transition or map it to the transition that actually reaches the needed state.
- Each test item maps to **one or more transitions**. The item's `trigger` informs the `agent_task`; the `expected_result` informs the `postconditions` or `dom_assertions`.
- Embed the expected success signal naturally into the `agent_task` so the agent knows when it is done — phrase it as a goal description, not a formal assertion. Example: instead of *"Scroll up"*, write *"Scroll up to load older message history until older messages appear above the current ones."*
- Assertions are plain `{"assertion": "..."}` objects.
- Coverage requirement: **every `item_id` from the test items list must appear in at least one transition's `mapped_test_items`.**

---

## Component: States

States are replayable checkpoints in the interaction flow. The evaluator drives a task by replaying a sequence of `agent_task`s from `S0`, so each state must be a **stable** page condition that looks the same on every replay — independent of when the replay lands. States never self-change, fade, auto-dismiss, tick, or otherwise depend on timing. Transient UI (toast / snackbar / banner, loading spinner, "submitting…" / "saving…" indicator, countdown timer, animation keyframe, progress bar) is never a state — cover it as a `[CHANGE]` `dom_assertion` on the triggering transition.

```json
{"id": "S0", "description": "The initial feed page with the create-post touchpoint visible at the top."}
```

- **`id`** — `S{integer}` (e.g. `S0`, `S1`, `S12`). `S0` is always the initial page state. No other formats (`S1a`, `S_draft`, `A14_upload_state` are all invalid).
- **`description`** — one short natural-language sentence describing what is distinctive about this state (what the page looks like / what is open / what has just been accomplished). Implementation-neutral. Example: `"The post editor is open with the text area empty."`, `"The first coupon card shows a completed claimed state."`, `"The editor is open and the text area contains the 'imgfail' keyword."`.

- **Identity** — two states are the same **only** when every visible aspect is identical: same text in every field, same images/attachments present, same panels open/closed, same items in every list, same selection/highlight on every control. If ANY aspect differs, they are distinct states with distinct IDs and distinct descriptions. Define a new state only when the page is materially different from every existing state. A self-loop (`"from": "Sx", "to": "Sx"`) is valid only when the after-state is **observably identical** to the before-state. If the transition introduces new content (text typed, image attached, items added or removed), `to` is NOT the same as `from` — define a new state whose description captures what the page looks like now. Descriptions are **static page snapshots** — describe what is visible, not how it got there. Wrong: *"editor open with text restored from draft"*. Right: *"editor open with the text 'draft recovery test' in the text area"*.

- **State continuity — unchanged aspects MUST carry over into the target state's description.** A transition only mutates what its `agent_task` action mutates; everything else established by prior transitions persists. So the `to` state's `description` must still describe those persisting aspects. Do NOT collapse `to` back to an earlier state that lacks them just because the current action reverses one surface aspect. Concretely: if the `from` state is *"editor open with 'violation' text and the emoji panel expanded"*, and this transition's action closes the emoji panel, the `to` state is NOT an earlier *"editor open, empty"* state — it must be a state describing *"editor open with 'violation' text, emoji panel closed"*. Declare a new state if no existing state fits.

- **A running timer is never a state**: for any timer-driven flow (send-code cooldown, auto-save delay, throttle window), do NOT model the timer as `T_trigger → "timer active" state → T_wait → "timer ended" state` — a running timer is not replayable as a stable state, and a dummy wait-transition is invalid. Use a SINGLE transition whose `agent_task` triggers the timer, with `[CHANGE]` / `[AFTER]` `dom_assertions` covering both the transient behavior and the settled end state (e.g. *"[CHANGE] The send touchpoint becomes temporarily unavailable during the cooldown period."* + *"[AFTER] The send touchpoint is in an available (interactive) state."*). The `to` state is the stable post-timer state.

- **Integrity**: every declared state must be referenced by some transition's `from` or `to`, and every state ID referenced in a transition must be declared. No orphans, no dangling references.

---

## Component: Transitions

Each transition moves the page from the `from` state to the `to` state. `agent_task` describes the user operation that drives this move; `dom_assertions` / `postconditions` describe the observable effect of the operation.

Every state ID referenced anywhere in `transitions` must already be explicitly declared in `states`.

```json
{
  "id": "T1", "from": "S0", "to": "S1",
  "agent_task": "Publish a new post containing the text 'Hello world post'.",
  "mapped_test_items": [{"id": "TI-2"}],
  "preconditions": [
    {"assertion": "The feed lists post cards."}
  ],
  "dom_assertions": [
    {"assertion": "[CHANGE] A feedback message indicating the post was published successfully appears."},
    {"assertion": "[AFTER] The publishing touchpoint is in an available (non-loading) state."}
  ],
  "postconditions": [
    {"assertion": "The newly published post 'Hello world post' is visually rendered at the top of the feed as a post card showing avatar, username, and the submitted content."}
  ]
}
```

- **`id`** — unique transition identifier. Must be a strictly sequential integer series: `T1, T2, T3, …`. No letter suffixes (e.g. `T14b` is invalid — insert a proper `T15` and renumber subsequent transitions).
- **`from` / `to`** — state IDs; both must be declared in `states`.
- **`agent_task`** — natural-language goal-oriented instruction (see `agent_task` section).
- **`mapped_test_items`** — **required on every transition**. Array of one or more `{"id": "<TI-x>"}` entries — only test items whose expected outcomes are directly caused by and naturally verifiable from this single transition's action. Prefer 1-2 entries; avoid 3 unless inseparable; never use a transition as a catch-all for nearby but separately triggered checks. Every `item_id` from the input Test Items must appear in at least one transition's `mapped_test_items` across the ICG.
- **`preconditions`** — **required on T1, forbidden on all other transitions**. Assertions describing the initial page state on S0 that the test depends on.
- **`dom_assertions`** — atomic natural-language assertions verifiable from DOM mutation events + attribute/text state. Each item MUST start with `[CHANGE]` or `[AFTER]`. See DOM vs Post Classification below.
- **`postconditions`** — atomic natural-language assertions that require visual judgement on the final rendered screenshot. No prefix tag.
- At least one of `dom_assertions` / `postconditions` must be non-empty; omit the other list entirely if it has no entries (do not emit empty arrays).
- **`screenshot_mode`** — optional; `"viewport"` or omitted (default = full-page). See `screenshot_mode` section below.

### When to create a transition

**A transition must carry at least one meaningful assertion** — either about the stable post-state it reaches (an `[AFTER]` `dom_assertion` or a `postcondition`) or about a transient / process-level effect worth testing (a `[CHANGE]` `dom_assertion`).

- ✅ Meaningful — stable effects: a chip enters a selected state, a new post appears in the feed, an error state is shown on the editor, a panel opens and remains open, a text field now contains an emoji glyph the emoji-picker just inserted.
- ✅ Meaningful — transient effects worth testing: a success feedback message appears and then auto-dismisses, the submit touchpoint briefly enters a loading/disabled state during processing, a toast confirms the action.
- ❌ Not meaningful: the step has no observable system effect beyond value-echo. See Assertion Rule B11 — value-echo assertions are forbidden; if a transition has nothing else to assert, it should not exist.

**Typing / pick actions deserve their own transition only when the action itself has an observable system effect beyond value-echo** — e.g. typing `@` opens the mention picker (the picker appearance IS the observable effect), typing past a character limit triggers a counter state change, picking an emoji from the emoji picker inserts a glyph into the text area. If no such effect exists, broaden the `agent_task` to a larger user intent that does produce an observable effect and assert that effect on this same transition, or drop the step.

**Do NOT create a transition whose only meaningful assertion would be that a value was entered, a file was attached, or a field was cleared.** These are preparatory steps with no testable system effect on their own — fold them into the transition that produces the actual observable outcome. For example: typing `'violation'` and then publishing is ONE transition (`agent_task: "publish a post containing the word 'violation'"`) — the typing has no assertion point; the publish result (error feedback) does. Do not model the typing step as a standalone transition with a value-echo postcondition, and do not define an intermediate state for "editor contains 'violation'" unless another transition chain genuinely starts from that state.

**Do NOT create observation/self-check transitions.** If a test item merely says
the initial page is already in a mode, already has a permission level, or already
shows a ready-state indicator, cover that fact in
`T1.preconditions` and map that test item onto the first real actionable
transition that depends on the ready state. Do not create `S0 → S0` transitions
whose purpose is to examine, confirm, inspect, look for, or verify the ready
state.

**Scroll-position restoration: navigate-away and return MUST be one transition.**
After establishing a scrolled state, the navigate-to-detail and return-to-list steps must be combined into a single `agent_task` (a self-loop on the scrolled state). Do NOT split them: the "before" screenshot of a standalone T_return is the detail page, so the VLM has no reference to the scrolled list and cannot verify restoration.
- ❌ T_scroll (S0→S2) → T_navigate (S2→S3) → T_return (S3→S2): T_return's before frame is the detail page — "returned to the same scrolled position" is unverifiable.
- ✅ T_scroll (S0→S2): `agent_task` *"Scroll down the product listing until lower products are visible."* Then T_navigate+return (S2→S2 self-loop): `agent_task` *"Select a product from the currently visible scrolled position, navigate to its detail page, then return to the listing."* `screenshot_mode: "viewport"`, assertion: *"The product listing is still scrolled away from the top, showing the same lower-portion products as before navigating."*

### agent_task

A **self-contained, goal-oriented instruction** describing what the user wants to accomplish to drive this state transition. The agent executing this instruction has NO access to other information — the instruction must stand alone.

**Every `agent_task` MUST be an actionable user intent, NEVER a pure observation.** The agent executes the instruction and drives a state transition (including self-loops) by performing at least one real interaction. The following verbs as the primary or secondary purpose are **forbidden**: "Inspect", "Observe", "observe whether", "Check", "Verify", "Examine", "Confirm the layout of", "See if", "Notice", "Watch" — these imply looking, not acting. If what you want to verify is static properties of an already-reached state, put the check in the assertions of the transition that reached that state; do not create a separate transition for observation.
- ❌ *"Without selecting any text, observe whether the format toolbar is disabled."*
- ❌ *"Examine the document page to confirm it is in comment mode by looking for a comment-mode indicator, then select text to verify selection is possible."*
- ✅ Actionable intents: *"publish a new post…"*, *"navigate to the product detail page"*.

**Self-contained — no cross-transition references.** The `agent_task` may only describe an intent starting from the current transition's `from` state. It must NOT reference or rely on results / operations / artifacts of other transitions (e.g. *"using the draft saved earlier"*, *"after completing the previous upload"*, *"on the note from the prior step"*). If earlier work is needed, it is implicitly present in the current `from` state's description — refer to it by its persistent property (e.g. *"the draft note in the draft list"*), not by "previous transition".
- ❌ *"Delete the draft created in the earlier transition."*
- ✅ *"Delete the draft currently shown in the draft list."*

**Describe the user's intent, not an action sequence.** Different HTML implementations realize the same feature with different mechanics — different control labels, different click flows, different intermediate UI (some show a confirmation dialog for destructive actions, some don't). You do not know at spec-writing time which shape the implementation will take. Tell the agent what the user is trying to accomplish; the agent handles whatever clicks, confirms, or intermediate UI the specific implementation requires. **Do NOT specify pointer interaction types** such as "long-press", "click", "right-click", "double-click", or "hover" — even as parenthetical hints. A wrong hint actively misleads the agent into using the wrong mechanic. Scroll direction (e.g. "scroll to the top", "scroll down to load older messages") is permitted because it describes where to go, not how to click.
- ✅ *"Invoke the emoji selection panel on a message."*
- ✅ *"Scroll to the top of the message history to load older messages."*
- ❌ *"Long-press the first text message to open the emoji panel."*
- ❌ *"Invoke the emoji selection panel on a message (e.g. via long-press or context menu)."*  ← parenthetical hint locks in wrong mechanic


**What to include:**
1. **Inline concrete values by default.** For any text, label, tag, number, or keyword the agent must type or pick, write the exact value inline so assertions can reference it precisely: *"type the word `violation` into the post text area"*, *"type `Hello world post` into the editor"*, *"add a tag named `work`"*, *"type a 300-character string into the input"*. Never reference "the contract", "the spec", "the requirement", or any other artifact — the agent does not see them. This is the default choice whenever the assertion cares about *what* content appears.
2. **Describe input by shape ONLY when content doesn't matter.** When the transition tests a *form* property rather than specific content (e.g. multi-line rendering, overflow/scroll behavior, auto-save triggering, draft persistence of arbitrary text), describe the required shape and let the agent choose: *"type multi-line text"*, *"type content long enough to trigger scrolling"*, *"type a URL-shaped string"*. In this case, assertions MUST be content-agnostic (see Assertion Rule 12) — e.g. *"a new entry appears with the typed content"*, not *"the entry reads 'Hello world'"*. If in doubt between rule 1 and rule 2, prefer rule 1.
3. **When the value depends on HTML-specific mock data, phrase it as a rule grounded in what is on screen.** *"pick the first entry in the suggestion list"*, *"choose the first topic listed in the topic suggestion dropdown"* — not *"pick the entry that matches the contract's user mock"*.
4. **Avoid element occlusion on spatial canvases.** When the task adds multiple visual elements to a shared canvas or free-layout area (images, shapes, text blocks, layers), include a repositioning step to avoid occlusion — e.g. *"add a text element to the canvas, then drag it away from existing elements so all remain visible"*. The agent can see each element's position and size.
5. **Force overlap when asserting z-order / stacking — as a dedicated prior transition.** When a test item tests which layer is on top (bring-to-front, send-to-back, raise/lower), model the setup and the assertion as **two separate transitions**:
   - First transition: move the target elements until they physically overlap (stable intermediate state). This transition's `to` state describes the overlapping layout.
   - Second transition: from that overlapping state, perform the z-order change (bring to front / send to back). The postcondition asserts which element is visually on top / obscures the other.
   Do NOT combine overlap-positioning and z-order-change into the same `agent_task` — the stacking change must start from a confirmed overlapping state so the VLM can observe which element occludes the other. Example flow: T_overlap *"drag layer B onto layer A until they overlap"* → S_overlap; T_zorder (S_overlap → S_front) *"bring layer B to the front"*.
6. **Never instruct character-by-character or word-by-word typing.** Write the complete input value as a single intent: *"type `Hello world`"* — never *"type 'H', then 'e', then 'l'…"* or *"type one word at a time"*. If the feature reacts to each keystroke (live search, autocomplete, character counter), that intermediate DOM change is captured automatically during the agent's normal typing and can be verified with a `[CHANGE]` `dom_assertion` — there is no need to choreograph the input sequence in `agent_task`.
7. **Describe file attachments by the scenario, not by filename. For the valid case, just say so; for limit-triggering cases, spell out the contract's threshold or format restriction concretely.** When the test exercises a spec-imposed limit, resolve the limit at spec-writing time from the test items and contract and name it in the instruction — do not say *"oversized"*, *"exceeds the limit"*, *"a non-image document"*, or *"an unsupported format"*. Examples:
   - *"attach a valid image"* (accepted upload case — no constraint worth naming)
   - *"attach an image larger than 5MB"* (rejected-by-size case, when the contract's upload size limit is 5MB)
   - *"attach a PDF file"* (rejected-by-format case, when the contract only accepts image formats)

8. **Navigate to features whose location is unspecified.** When a test item describes an output (history list, saved drafts panel, notification log, activity feed, etc.) and neither the test items nor the contract specify *where* it is displayed, do NOT assume it is visible in the current view. The `agent_task` must include a step to actively navigate to or reveal that feature — e.g. *"after viewing the products, navigate to or open the browsing history section wherever it is accessible on the page"*. The postcondition then verifies the *content* of the feature, not its location.
9. **Creating a required state when none explicitly exists.** If the task requires a specific container/list state (e.g. an empty folder, a folder with no items) that is not explicitly guaranteed to exist in the initial page, do NOT assume one is already present. Instead, instruct the agent to produce it using available operations — for example: *"Navigate to a folder that contains no notes; if no empty folder is visible, first delete all notes inside an existing folder to make it empty, then navigate to that folder."* Do NOT leave the agent to search indefinitely for something that may not exist.

**What to avoid:**
- Do NOT write a purely observational `agent_task` — every `agent_task` must include at least one user interaction (click, type, scroll, drag, etc.). Tasks like *"Inspect the panel and verify its layout"* or *"Observe the current state of the editor"* have no user action and must NOT be modeled as transitions. Move such checks to assertions on the transition that established that state.
- When a transition involves a multi-option choice (conflict resolution, confirmation dialog with multiple buttons, mode selection), the `agent_task` must clearly specify which option to choose, and the assertions must match that choice. For example: *"resolve the conflict by keeping the local version"* — then the postcondition asserts the local content is displayed. Do not leave the choice ambiguous (e.g. *"resolve the conflict"*).
- Do NOT include verification claims — scoring lives in `dom_assertions` / `postconditions`, not in `agent_task`.

### Assertion Rules (apply to preconditions, dom_assertions, and postconditions)

All assertions must be implementation-neutral and describe *what* is observably true, not *how* the HTML realizes it.

#### A. Form

0. **Declarative post-action fact — no action triggers inside assertions.** An assertion describes what IS true in the settled post-action state. Do NOT embed the triggering action inside the assertion using `when <action>…`, `after clicking…`, `on hover…`, or similar. The `agent_task` already records what was done; the assertion records the resulting state.
   - Wrong: *"When the button is clicked, a tooltip appears."* / *"When hovering over the item, an edit icon is visible."*
   - Right: *"A tooltip is visible."* / *"An edit icon is visible on the item."*

1. **Atomic.** One fact per item. Split compound effects.
   - Wrong: *"The editor closes and the new post appears at the top."* — split into two items.

2. **Self-contained within the transition.** Understandable given only the current `agent_task`. Do not reference other transitions, the state graph, or earlier steps. You may reference the current action's direct outcome (e.g. *"the newly published post"*).
   - Wrong: *"A feedback message appears."* / *"The text entered in the previous step remains."*
   - Right: *"A feedback message indicating the post was rejected for content violation appears."*

3. **Semantic, not implementation.** Describe the functional outcome. Do NOT specify UI component types (list, grid, toast, modal, dropdown, panel, sidebar, tooltip), DOM structure (added/removed from list, node appears in container), visual details (color, border, shadow, position), or verbatim text labels from the contract. Refer to subjects by role/region/label (*"the publishing touchpoint"*, *"the text input area"*), not by color, CSS, or page position.
   - Wrong: *"A new card is appended to the list container."* / *"A red toast appears at the bottom."* / *"The chip is removed from the tags area."*
   - Right: *"The newly published post is visible in the feed."* / *"A feedback message indicating rejection appears."* / *"The previously selected topic is no longer in a selected state."*

3a. **Do NOT assume where associated content is displayed.** The same feature can be realized with associated content appearing inside the editor, above it, beside it, or in a dedicated panel — all are valid implementations. Write assertions about *what* the content IS or *what state* it is in, not about *which container* holds it.
   - Wrong: *"The selected topic text appears inside the text area."* / *"The chosen tag is inserted into the editor body."*
   - Right: *"The selected topic is visibly displayed in or near the editing area."* / *"The chosen tag is shown as associated with the current editing context."*

4. **Refer to list/table items by distinguishing label**, not ordinal position (name, title, ID, or visible content). Fall back to position only when no stable label exists. When the test item distinguishes multiple outcome types (e.g. *"sold out"* vs *"claim limit reached"*), name which one this assertion targets — do not collapse into generic *"an error appears."*

#### B. What you may assert

5. **Only values explicitly stated in the test items or contract, or deterministically introduced by this transition.** A value is "deterministically introduced" if the current `agent_task` supplies it (e.g. typed input of *"Alice"* → the field contains *"Alice"*). Do NOT assert inferred implementation defaults, unstated category labels, numeric deltas from unstated progressions, or per-item details the spec is silent on.

6. **No absolute data quantities.** Counts, list lengths, option counts, character limits, and delay values are implementation choices. Use direction/presence language (increased, decreased, prepended, appeared, disappeared) rather than exact numbers. For increments, use conditional form to absorb abbreviation (*"1.2K"*): *"The count has increased; if displayed as a full integer the value should reflect the increment, otherwise the absence of visible change is acceptable."*
   - Exception: a count the contract explicitly maps to a user action AND the count IS the outcome (e.g. *"case A → exactly 3 matching results"*).

7. **Conditional assertions for uncertain result sets.** For search, filter, tab-switch, or **tag/category selection** outcomes whose result depends on mock data, write both branches: *"If matching results are present, they are displayed; if none match, an empty-state or no-results message is shown."* Do not assert a specific non-empty outcome unless the contract guarantees matching data. This applies equally to tag/label/chip selection — selecting a tag filters the displayed list and may legitimately yield zero items if no content carries that tag; the empty state is a correct outcome, not a failure.

8. **Do NOT assert unspecified defaults.** No *"nothing is selected"*, *"no filter is applied"*, initial-selection claims unless the test items or contract say so — defaults may already be a selection (*"All"*, *"Newest"*).

9. **Do NOT assert disabled / non-interactive / unavailable state** unless explicitly described in the test items or contract. This includes behavioral-capability assertions such as *"the element is no longer selectable"*, *"the layer cannot be moved or dragged"*, *"the button is not clickable"* — these require actually attempting the interaction to verify and cannot be judged from a screenshot. If a feature produces a visual locked/disabled indicator (e.g. a lock icon, greyed appearance), assert that visual indicator instead.

10. **Do NOT assert the options inside a collapsed dropdown / picker / menu.** Entries aren't observable while collapsed — assert the opened-state structure or the outcome of picking instead. Do not design a transition whose sole purpose is to enumerate collapsed options.

11. **Do NOT write value-echo assertions.** After input/pick, do not assert the field now contains the value `agent_task` just supplied — that's mechanical, not a system effect. Assert a genuine side-effect (counter change, picker opens, suggestion surfaces, draft indicator appears) or drop the step.

12. **Content-agnostic assertions ONLY when agent_task used shape-based input (agent_task rule 2).** If `agent_task` specified concrete content (rule 1, the default), assertions may and should reference that specific content. If `agent_task` specified only shape ("multi-line text", "long enough to scroll"), assertions MUST be content-agnostic: *"the editor content has changed"*, *"a new entry with the typed content appears"*, *"the input reflects multi-line content"* — not *"the field contains 'Hello world post'"*.

13. **Do NOT invent transient states beyond the test items.** `[CHANGE]` assertions for loading spinners, progress bars, relocating indicators, etc. are valid only if the test items describe them. Otherwise use `[AFTER]` for the settled state only.

#### C. Choosing dom_assertions vs postconditions

14. **Postconditions must be directly judgeable from the before/after screenshots.** A VLM compares two frames; describe a visible property confirmable from those two images alone — no inferred state, no behavioral claims (*"cannot be clicked"*, *"does not respond"*). If evidence lives only in attributes/class tokens, use a `dom_assertion`. Feedback lifecycle: if a feedback element both appears AND auto-dismisses and both matter, write two atomic assertions (appearance + disappearance); otherwise only the appearance.

15. **Visual feedback → prefer DOM.** Highlights, focus effects, status indicators: put in `dom_assertions`. If persistence to the final frame is uncertain, tag `[CHANGE]` so any appearance in the timeline passes.

16. **Touchpoint availability assertions need an exposure step.** To assert a hover-gated / context-menu / collapsible control is available, the `agent_task` must first expose it (hover parent, right-click, expand panel).

### preconditions
- **Only on the first transition (T1)**. All other transitions MUST NOT have a `preconditions` field.
- **Strictly derive from the test data contract.** Assert only what the contract explicitly mentions — nothing more. Do NOT invent conditions the contract does not describe.
- **Static screenshot only.** Preconditions are checked from the initial S0 screenshot before any action and before DOM-event monitoring starts. They must be visual, static facts the scorer can verify without trying the UI.
- **Preconditions must only assert what is IMMEDIATELY and VISIBLY present on the page at first load, without requiring any user interaction.**Contract terms like "accessible", "available", or "supported" mean the feature is reachable via interaction — they do NOT mean the feature content must be displayed by default. For such features, the precondition should check for the TOUCHPOINT (button, icon, link) that triggers it, not for the content itself.
- **Do NOT put interactability/operability claims in preconditions.** Avoid assertions such as "items are interactive", "can be selected", "clickable", "available to open", "responds to click", "can be submitted", or "is enabled" unless the initial screenshot itself exposes an unambiguous enabled/disabled visual state that is explicitly required by the contract. For selectable/clickable data items, assert the visible prerequisite instead, and verify actual navigation/selection in the transition's `postconditions` or `dom_assertions`.
  - Wrong precondition: *"Each product in the list is interactive and can be selected."*
  - Right precondition: *"A scrollable list of product entries is visible."*
  - Then verify the interaction on the transition: *"The selected product's detail view is displayed."*
- **Do NOT assert contents of closed/collapsed controls.** If the contract says a picker, dropdown, or menu has multiple options, assert only that the control is *present* — not that specific options are available inside it. The options are only observable after the control is opened, which belongs in a transition, not a precondition. Wrong: *"Multiple playback speed options are available for selection."* Right: *"A playback speed control is visible on the page."*
- **Do NOT hard-assert uncertain contract language.** If the contract uses qualifiers such as "may", "might", "can", or "possibly" (e.g. "the list may already contain uploaded files"), do NOT turn that into a definite precondition. Either omit the assertion entirely, or write a conditional: *"If previously uploaded files exist, they are listed; otherwise the list area is empty or absent."*
- **Do NOT assert invisible data-volume conditions.** Assertions such as "enough messages to support pagination", "sufficient items to trigger infinite scroll", or "multiple pages of content exist" cannot be verified from a static screenshot. Omit them; rely on the contract to guarantee the necessary data and let the transition's `agent_task` scroll or load more to surface it.
- Preconditions follow all Assertion Rules above.

### dom_assertions vs postconditions

**Default to `postconditions`.** Most assertions belong here. Postconditions are judged from the final rendered screenshot, which is the visual ground truth. If the outcome can be described in plain visual terms from the final state (a new post appears in the feed, the editor is now open, a section has expanded, a completed state is shown, prior content is preserved), put it in `postconditions` — do not double-capture it in `dom_assertions`.

**Use `dom_assertions` only for two categories that `postconditions` cannot reliably verify:**

1. **Transient / process-level effects that do not persist to the final screenshot.** A toast appears and auto-dismisses, a control briefly becomes disabled during the action, an inline suggestion panel closes after the user picks from it, a flash of feedback fades. Visual screenshot judgment cannot verify these because the final screenshot no longer shows them.

2. **Persistent element-level state changes that DOM inspection reads more reliably than screenshots:**
   - **Selected / highlighted** state on chips, tabs, filter pills, topic tags.
   - **Enabled / disabled** state on controls.
   - **Text / label content change** on a specific element (e.g. a button's label flips from *"Claim"* to *"Claimed"*).
   - **Fine-grained content details that are hard to judge visually** — placeholder/empty rows in a diff view, line-number alignment, text content preservation after an action, presence or absence of specific text fragments. When the assertion depends on exact text or structural details that a screenshot may render ambiguously, prefer `[AFTER]` dom_assertion over postcondition.

   These are element-level attributes / classes / text state; DOM-level inspection is deterministic whereas screenshot judgment for subtle color / border shifts is not.

   **Toggle switches / checkboxes are an exception — do NOT assert the toggle's on/off state here.** The toggle's visual state usually lives on an internal indicator child (knob, track, check glyph), which DOM inspection of the outer control doesn't catch. The functionally meaningful outcome is what the toggle *controls* — the list filters, a section appears or disappears, a mode switches. Put that functional outcome in `postconditions` instead.

   **Text content inside a user-typable field (text area / input / contenteditable) is also an exception — do NOT put any content-of-field assertion in `dom_assertions`.** The final screenshot shows the field content directly, so visual judgment is the ground truth and DOM inspection has no advantage. Put such assertions in `postconditions` as plain visual statements (no `[CHANGE]` / `[AFTER]` prefix). Note Rule B11 still applies without exception: do not assert that typing/picking value X made the field contain X — that is value-echo and forbidden in BOTH lists. The legitimate case this exception is for is **persistence of prior content through a failed / rejected / cancelled action** (where the system's choice to keep the content IS the effect under test).
   - Wrong (in `dom_assertions`): *"[AFTER] The post text area still contains the 'violation' keyword after the failed publish attempt."*
   - Right (in `postconditions`): *"The post text area visibly still contains the 'violation' keyword after the failed publish attempt."*

**Tagging** — every `dom_assertion` text MUST begin with exactly one tag:
- `[CHANGE]` — transient / process-level effect (category 1 above). Passes whether the condition appears in the event timeline OR holds in the final state.
- `[AFTER]` — settled semantic state that persists but is visually unreliable (category 2 above). Passes only if the final state satisfies it.

**Scenario-specific rules:**

- **Download transitions** — write a `[CHANGE]` `dom_assertion` capturing the temporary download link; do NOT write postconditions. Example: *"[CHANGE] A temporary download link (an anchor with a download attribute or data URL) is created to trigger the file export."*

- **Refresh / full-reload transitions** — prefer `postconditions`; if a `dom_assertion` is genuinely needed under the two categories above, use `[AFTER]` only.

- **Debounce / throttle** — test only on transitions whose input produces a real observable outcome after the delay; phrase the assertion as the delayed outcome itself. Example: *"[CHANGE] After a brief delay following the input, a validation feedback message appears near the username field."*

- **File upload (drag-and-drop upload zone)** — The agent performs file upload via the browser's file-input API, which does **not** dispatch `dragenter` or `dragover` DOM events. Therefore, do NOT write a `[CHANGE]` dom_assertion about the upload zone entering an active-drag visual state (e.g. *"the drop zone becomes highlighted / outlined / changes border color while a file is being dragged over it"*). Those states are driven purely by drag events and are structurally unreachable when upload is performed programmatically. 

- **Failure / rejection feedback** — When an action fails or is rejected, write the failure assertion to cover the *presence of a failure signal* without mandating its specific UI form. Valid failure signals include: a feedback message, an error message, an inline error indicator, a highlighted or error-styled field, an error icon, or any other visible cue that the action was not completed. Write *"A failure indication or feedback message appears, signaling that [specific reason]"* — do NOT write *"An error toast appears"* or *"An inline error message appears below the field"*. Those forms over-constrain the implementation. The requirement is that the failure is communicated; HOW it is communicated is an implementation choice.
   - Wrong: *"[CHANGE] An error toast appears at the top of the page."* / *"[AFTER] An inline error text appears below the input field."*
   - Right: *"[CHANGE] A failure indication appears, signaling that the submission was rejected."* / *"[AFTER] A visual failure signal is present indicating the content exceeds the character limit."*

**Complementarity — do NOT duplicate the same fact across the two lists.**

- **Feedback messages** — pick one list based on the message's nature: transient / auto-dismissing → `dom_assertions`; persistent part of the final state → `postconditions`. Do not assert the same feedback in both.
- **Disabled / enabled state on a control** — always `dom_assertions` only. Do not write a postcondition describing the same control's visual grayed-out appearance.
- **Selected / highlighted state on an element** (chips, tabs, filter pills, topic tags) — always `dom_assertions` only. Do not write a postcondition about the same element's selected/highlighted appearance.

**Other rules:**
- **Follow every Assertion Rule above.**
- Do NOT invent a transition solely to test that a control is disabled / inactive / non-interactive — express that as an assertion on the preceding transition (only if grounded in the testitem).
- Do NOT create a transition whose `agent_task` instructs the agent to "attempt to" operate on a disabled or non-interactive element. Such tasks cause the agent to repeatedly try and fail, wasting iterations and corrupting the page state. If you need to verify a control is disabled, assert it as a `[AFTER]` dom_assertion on the transition that CAUSED the control to become disabled (e.g. after a permission change, assert the submit button is disabled).
- **Selection-gated trigger + empty-submit guard — both covered on one transition**: When a test item describes two sequential gates (e.g. a comment trigger that only appears after text is selected, AND a submit that is non-interactive when the input is empty), cover both on the SAME transition that selects text and invokes the trigger. Use `[CHANGE]` to assert the trigger appeared upon text selection — this implicitly proves the trigger was absent before selection, covering the pre-selection non-interactive state without a separate precondition or transition. Use `[AFTER]` to assert the submit control is non-interactive while the input area is empty — this covers the empty-submit guard. Do NOT write a separate "attempt to submit empty" transition; do NOT write `[AFTER] The trigger is in an available state` (the trigger was already invoked — that assertion is meaningless at that point).


### screenshot_mode (optional)

**Omit by default** — full-page capture. Set `"screenshot_mode": "viewport"` **only** when a postcondition depends on **scroll position** or **what is currently in view**, not on whether content exists somewhere on the page.

Signal phrases in the assertion text: *"visible in the viewport"*, *"scrolled to"*, *"currently shown"*, *"positioned at the user's current view"*, *"appears in the visible area"*. Do NOT use viewport mode for plain existence checks.

Placed at the transition level: `{"id": "Tx", ..., "screenshot_mode": "viewport", "postconditions": [...]}`.

**All assertions on a viewport transition MUST be viewport-dependent.** If the same action also needs full-page checks, split them into a separate transition with default (full-page) capture.

---

## General Rules

- **Shared prerequisite state**: If multiple independent features share a common prerequisite state (e.g., "editor is open"), that state only needs to be derived from S0 **once** — in the first chain that establishes it. All subsequent chains that require the same prerequisite should start directly from that already-established state (e.g., S1), not re-derive it from S0. The evaluator automatically replays the shortest known path to any reached state, so there is no need to repeat the derivation.
- **Transition design — every ICG must include BOTH of the following:**
  - **(a) Independent-feature transitions (fan-out):** If features A, B, C are independent (none requires the outcome of another as a precondition), create separate transitions that each start from the same `from` state — producing a fan: `S1→S2` (feature A), `S1→S3` (feature B), `S1→S4` (feature C). Do NOT chain them serially as `S1→S2`, `S2→S3`, `S3→S4` — that creates false dependencies where feature B is blocked if feature A fails. Serial chaining between transitions is only valid when the second transition genuinely requires the first transition's *outcome state* as its precondition (e.g., T_open_picker → T_pick_item: the picker must be open before picking). For each pair, ask: "does this transition require the prior transition's outcome?" If no, fan them out.
  - **(b) Feature-composition transitions (compound `agent_task`):** In addition to the independent tests above, add one or more transitions where a SINGLE `agent_task` combines 2–3 features that share an artifact in sequence. The purpose is to catch bugs where one feature clobbers another (e.g. inserting an emoji resets the selected tag; selecting a tag clears the text). Example `agent_task`: *"Compose a new post by first inserting the 🎉 emoji, then typing an @alice mention, and publish it."* — the matching postcondition: *"A new post is published containing the 🎉 emoji followed by the @alice mention, in that order."* Pick features that share an underlying artifact (text field, selection set, draft state) and are likely to interfere. Build one such compound transition per task where the feature set allows.
- **Do not invent UI controls beyond what the test items and contract describe.** Model the most direct interaction the test item implies — do not assume extra UI sugar that a reasonable HTML implementation might not include. Concrete case: when the test item says typing a trigger character (`@`, `#`, `/`, etc.) into a text field opens a picker/dropdown/autocomplete, write the `agent_task` as *"the user types `@` into the post text area to open the mention picker"* — NOT *"the user clicks the '@mention' trigger button"*. Only rely on a dedicated clickable control when the test item or contract explicitly names one.
- **Do not invent failure paths.** Only model error/failure transitions that are explicitly described in the test items or contract (e.g. a specific error trigger, a named failure case). Do not add failure transitions based on general UI conventions or assumptions about what might go wrong — if neither the test items nor the contract mention a failure scenario, do not test it.
- **Multi-user / collaborator assertions must be single-client testable.** The evaluator runs a single browser — it cannot verify cross-client view consistency (e.g. "all collaborators see the same document"). Do not assert outcomes that require a second client to observe. If the test items or contract describe a simulation control for collaborative features (e.g. a "Simulate Remote Edit" button), the `agent_task` must explicitly instruct the agent to use that specific control — do not write vague instructions like "simulate a remote collaborator's edit". If neither the test items nor the contract mention a simulation mechanism, do not design transitions that require remote/collaborator actions.
- **Label / tag — use a newly created label as the test target when the test items and contract do not specify existing-label content.** Existing labels are otherwise unreliable: mock data may have pre-assigned them in ways that make assignment actions no-ops or empty-state assertions trivially satisfied. Flow: create the new label first in an earlier transition, then build any subsequent operation on it (assign, filter by, delete, empty-state assertion) from that post-creation state. If the contract explicitly names pre-existing labels with their item mappings, testing against those directly is fine.
- **Test against data you create, not data you assume exists.** When the test items and contract do not explicitly state whether a list/panel/collection already contains items, do NOT assume it does — create the data yourself first, then assert against it. The creation and the assertion can be in the same transition's `agent_task` or split across transitions, as long as the creation action happens before the assertion. For example: if the contract does not say whether the draft list has saved drafts, the `agent_task` should include saving a draft before opening the draft list and asserting it contains that draft. Do not invent or assume pre-existing data that the test items and contract do not describe.
- **Empty-state tests must explicitly clear pre-existing content.** When a test item requires testing behavior with empty inputs (e.g. "compare two empty text areas", "submit a blank form", "search with no query"), and the contract does not guarantee the inputs start empty, the `agent_task` MUST include a step to clear the relevant fields before triggering the action. Do NOT write `agent_task` as "without entering any text, trigger X" — the page may have pre-loaded default content that the agent will not clear unless explicitly told to. Write instead: "clear both input areas so they contain no text, then trigger X."
- **Non-repeatability and toggle behavior**: if the contract specifies what happens on a second interaction (e.g. "clicking again reverts to initial state"), treat that as the authoritative description of the requirement. Assert the contract's specific outcome directly. Do not fall back to a generic "remains unchanged" assertion when the contract already tells you exactly what the second interaction produces. A toggle action that changes the control's state introduces a **new distinct state** — define it as a separate state node, not as a reuse of the pre-toggle state.
- **Cover every testable dimension of each test item.** When a test item describes a feature with multiple independently observable outcomes (e.g. a toggle has both an ON state and an OFF state; a sort order can go ascending and descending; a filter can be applied and then cleared), design transitions to exercise and assert EACH distinct observable outcome. Assigning only one direction of a bidirectional feature to mapped transitions is insufficient — every independently observable direction must appear in at least one transition whose assertions explicitly verify that direction. 

## Output Format

Return exactly one JSON object with two top-level fields: `states` and `transitions`.
The `transitions` array MUST contain no more than 35 transitions.

```json
{
  "states": [
    {"id": "S0", "description": "..."},
    {"id": "S1", "description": "..."}
  ],
  "transitions": [
    {
      "id": "T1", "from": "S0", "to": "S1",
      "agent_task": "...",
      "mapped_test_items": [{"id": "TI-1"}],
      "preconditions":  [{"assertion": "..."}],
      "dom_assertions": [{"assertion": "[CHANGE] ..."}],
      "postconditions": [{"assertion": "..."}]
    }
  ]
}
```

---

## Transition Count Budget

- Transitions must cover every base test item — do not skip any.
- Do not create extra transitions just to model intermediate states that have no assertion value. If two test items can share a transition (same trigger, compound `agent_task`), merge them.

---

## Final Self-Check

Before outputting the JSON, mechanically verify:
- Top-level JSON has exactly two fields: `states` and `transitions`.
- The `transitions` array contains no more than 35 transitions.
- Each transition has: `id`, `from`, `to`, `agent_task`, `mapped_test_items` (one or more entries), and at least one of `dom_assertions` / `postconditions` non-empty (omit the other entirely rather than emitting an empty array). T1 must carry a non-empty `preconditions` field; all other transitions must not have one.
- Every `item_id` from the input Test Items (`TI-*`) appears in at least one transition's `mapped_test_items`. Missing any item is a hard failure.
- Every state declared in `states` is referenced by at least one transition `from` or `to` — no orphans. Every state ID referenced in `transitions` has been explicitly declared.
- Every `dom_assertions` item text starts with exactly one of `[CHANGE]` / `[AFTER]` (one space after the closing bracket). Every `postconditions` item text has NO prefix tag.
- `screenshot_mode`, when present, has value exactly `"viewport"` — never `"full"` or any other value. Omit the field for the default full-page capture.

Output the JSON object only."""




# ══════════════════════════════════════════════════════════════════════════════
# Prompt Builders
# ══════════════════════════════════════════════════════════════════════════════

def _req_lines(reqs: list) -> str:
    lines = []
    for w in reqs:
        intent = w.get("intent_type", "?")
        fid    = w["full_id"]
        text   = w.get("content_en", "")
        lines.append(f"  [{intent}] {fid}: {text}")
    return "## Requirement List\n" + "\n".join(lines)

def build_call1a_prompt(reqs: list) -> str:
    """Call 1a: requirements -> contract only."""
    return _req_lines(reqs)

def build_call1a_retry_prompt(reqs: list, previous_contract: str, issues: list) -> str:
    """Call 1a retry: include reviewer feedback so the model can actually revise."""
    return (
        f"{_req_lines(reqs)}\n\n"
        f"## Previous Contract Rejected By Reviewer\n{previous_contract}\n\n"
        f"## Reviewer Issues To Fix\n{json.dumps(issues, ensure_ascii=False, indent=2)}\n\n"
        "Regenerate the test_data_contract JSON. Remove or generalize every phrase "
        "identified by the reviewer. Keep the required initial page/view/mode for "
        "multi-page apps, but do not expose implicit requirement outcomes."
    )

def build_call1b_prompt(reqs: list) -> str:
    """Call 1b: requirements -> test items only."""
    return _req_lines(reqs)


def build_call1b_retry_prompt(reqs: list, uncovered_req_ids: set, bloated_item_ids: list) -> str:
    """Retry Call 1b with concrete coverage feedback."""
    missing = [
        f"  {w['full_id']}: {w.get('content_en', '')}"
        for w in reqs
        if w.get("full_id") in uncovered_req_ids
    ]
    feedback = [
        "## Coverage Feedback From Previous Output",
        "The previous test_items JSON did not satisfy coverage validation.",
    ]
    if missing:
        feedback.extend([
            "",
            "### Missing requirement IDs",
            "Every ID below MUST appear in at least one test item's `req_ids`.",
            "If a broad parent requirement mentions the same content, that does NOT cover these IDs unless these exact IDs are also listed in `req_ids`.",
            "Create separate focused test items when the max-2-req_ids rule would otherwise hide a specific requirement.",
            "\n".join(missing),
        ])
    if bloated_item_ids:
        feedback.extend([
            "",
            "### Over-broad test items",
            f"These item IDs had too many req_ids and must be split or narrowed: {', '.join(bloated_item_ids)}",
        ])
    feedback.append("")
    feedback.append("Regenerate the complete JSON object from scratch.")
    return f"{_req_lines(reqs)}\n\n" + "\n".join(feedback)


def build_call2_prompt(washed_reqs: list, contract: str) -> str:
    """Call 2: check that the contract does not expose implicit requirements."""
    explicit = "\n".join(
        f"  {w['full_id']}: {w.get('content_en', '')}"
        for w in washed_reqs if w.get("intent_type") == "explicit"
    )
    implicit = "\n".join(
        f"  {w['full_id']}: {w.get('content_en', '')}"
        for w in washed_reqs if w.get("intent_type") != "explicit"
    )
    return (
        f"## Explicit Requirements (allowed context)\n"
        f"{explicit or '(none)'}\n\n"
        f"## Implicit Requirements (must NOT be exposed in contract)\n"
        f"{implicit or '(none)'}\n\n"
        f"## Test Data Contract\n{contract}"
    )


def build_call4_prompt(contract: str, test_items: list) -> str:
    """Call 4: test items -> ICG JSON."""
    all_item_ids = [ti["item_id"] for ti in test_items]
    items_json = json.dumps(test_items, ensure_ascii=False, indent=2)
    return (
        f"## Test Data Contract\n{contract}\n\n"
        f"## Test Items (use as primary design guide)\n{items_json}\n\n"
        f"## All Item IDs that must be covered (every item_id must appear in some transition's mapped_test_items)\n"
        f"{json.dumps(all_item_ids, ensure_ascii=False)}\n\n"
        "Output the test specification JSON (states + transitions only)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def call_api(system, user, client, model, max_tokens=MAX_TOKENS):
    t0 = time.monotonic()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    response = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=0.2,
    )
    usage = getattr(response, "usage", None)
    u = {
        "prompt_tokens":      getattr(usage, "prompt_tokens",      0) if usage else 0,
        "completion_tokens":  getattr(usage, "completion_tokens",  0) if usage else 0,
        "total_tokens":       getattr(usage, "total_tokens",       0) if usage else 0,
    }
    text = (response.choices[0].message.content or "").strip()
    dur  = int((time.monotonic() - t0) * 1000)
    return text, u, dur, msgs


def _clean_json_string(s):
    s = s.lstrip('﻿')
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
    stripped = s.strip()
    if stripped.startswith('{'):
        after_brace = stripped[1:].lstrip()
        if after_brace.startswith('“') or after_brace.startswith('”'):
            s = s.replace('“', '"').replace('”', '"')
    s = s.replace('‘', "'").replace('’', "'")
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_string and i + 1 < len(s):
            result.append(c); result.append(s[i + 1]); i += 2; continue
        if c == '"':
            in_string = not in_string; result.append(c); i += 1; continue
        if in_string:
            if   c == '\n': result.append('\\n')
            elif c == '\r': result.append('\\r')
            elif c == '\t': result.append('\\t')
            else:           result.append(c)
        else:
            result.append(c)
        i += 1
    return ''.join(result)


def _fix_unescaped_quotes(s):
    for _ in range(80):
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            pos = e.pos
            if pos >= len(s): raise
            if s[pos] == '"':
                s = s[:pos] + '\\' + s[pos:]
            else:
                seg = s[max(0, pos-30):pos]
                lq  = seg.rfind('"')
                if lq >= 0:
                    fp = max(0, pos-30) + lq
                    s  = s[:fp] + '\\' + s[fp:]
                else:
                    raise
    raise ValueError("fix_unescaped_quotes: too many iterations")


def _fix_one_extra_closer(s):
    """Repair common model typo: one extra } or ] outside strings."""
    positions = []
    in_string = False
    escape = False
    for i, c in enumerate(s):
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if not in_string and c in "}]":
            positions.append(i)

    for i in reversed(positions):
        attempt = s[:i] + s[i + 1:]
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    raise ValueError("fix_one_extra_closer: no repair found")


def parse_json(text):
    m = re.search(r"```(?:json)?\s*(\{.*?)\s*```", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        last = candidate.rfind('}')
        if last >= 0: candidate = candidate[:last + 1]
    else:
        # Prefer a JSON-looking { (followed by optional whitespace and a quoted key)
        m2 = re.search(r'\{(?=\s*")', text)
        first = m2.start() if m2 else text.find('{')
        last = text.rfind('}')
        if first >= 0 and last > first: candidate = text[first:last + 1]
        else: raise ValueError("No JSON found in response")

    try: return json.loads(candidate)
    except json.JSONDecodeError: pass

    cleaned = _clean_json_string(candidate)
    cleaned_no_trailing = re.sub(r',\s*([}\]])', r'\1', cleaned)
    for attempt in [cleaned, cleaned_no_trailing]:
        try: return json.loads(attempt)
        except json.JSONDecodeError: pass

    try: return _fix_unescaped_quotes(cleaned_no_trailing)
    except (json.JSONDecodeError, ValueError): pass

    try: return _fix_one_extra_closer(cleaned_no_trailing)
    except ValueError: pass

    try:
        json.loads(cleaned_no_trailing)
    except json.JSONDecodeError as e:
        print(f"\n[JSON PARSE] final failure: {e}")
        if hasattr(e, 'pos'):
            pos = e.pos
            s, end = max(0, pos-50), min(len(cleaned_no_trailing), pos+50)
            print(f"[JSON PARSE] error context: ...{cleaned_no_trailing[s:pos]}<<<HERE>>>{cleaned_no_trailing[pos:end]}...")
        raise ValueError(f"JSON parse failed: {e}")
    raise ValueError("JSON parse failed: unknown error")


def build_cascade_coverage(icg, all_items: list, washed_reqs: list) -> dict:
    """Cascade coverage from transitions to test items to requirements."""
    req_map = {w["full_id"]: w for w in washed_reqs}

    # item_id -> list[transition_id]
    item_covered_by: dict[str, list[str]] = {ti["item_id"]: [] for ti in all_items}
    for tr in icg.get("transitions", []):
        tid = tr.get("id", "?")
        for mi in tr.get("mapped_test_items", []) or []:
            if not isinstance(mi, dict): continue
            iid = mi.get("id")
            if iid and iid in item_covered_by:
                item_covered_by[iid].append(tid)

    # test_items_coverage
    ti_cov = []
    for ti in all_items:
        iid = ti["item_id"]
        ti_cov.append({
            "item_id":                iid,
            "description":            ti.get("description", ""),
            "req_ids":                ti.get("req_ids", []),
            "covered_by_transitions": item_covered_by.get(iid, []),
        })

    # requirements_coverage — cascade through test_items
    req_by_items: dict[str, list[str]] = {}
    req_by_trans: dict[str, list[str]] = {}
    for entry in ti_cov:
        iid  = entry["item_id"]
        tids = entry["covered_by_transitions"]
        for rid in entry["req_ids"]:
            req_by_items.setdefault(rid, [])
            req_by_trans.setdefault(rid, [])
            if iid not in req_by_items[rid]:
                req_by_items[rid].append(iid)
            for tid in tids:
                if tid not in req_by_trans[rid]:
                    req_by_trans[rid].append(tid)

    req_cov = []
    for fid, w in sorted(req_map.items()):
        req_cov.append({
            "req_id":                 fid,
            "intent":                 w.get("intent_type", ""),
            "content_en":             w.get("content_en", ""),
            "covered_by_items":       req_by_items.get(fid, []),
            "covered_by_transitions": req_by_trans.get(fid, []),
        })

    return {"test_items_coverage": ti_cov, "requirements_coverage": req_cov}


def validate_coverage(icg, all_items: list):
    """Validate test-item coverage and basic ICG structure."""
    issues = []

    # Test-item coverage.
    all_item_ids = {ti["item_id"] for ti in all_items}
    covered_ids: set[str] = set()
    for tr in icg.get("transitions", []):
        for mi in tr.get("mapped_test_items", []) or []:
            if isinstance(mi, dict) and mi.get("id"):
                covered_ids.add(mi["id"])
    uncov = all_item_ids - covered_ids
    if uncov: issues.append(f"Uncovered test items: {sorted(uncov)}")
    unknown = covered_ids - all_item_ids
    if unknown: issues.append(f"Unknown item IDs in mapped_test_items: {sorted(unknown)}")

    # ICG structure.
    for f in ["states", "transitions"]:
        if not icg.get(f): issues.append(f"Missing: {f}")
    state_defined = {s.get("id") or s.get("state_id", "?") for s in icg.get("states", []) if isinstance(s, dict)}
    state_used = set()
    for tr in icg.get("transitions", []):
        if tr.get("from"): state_used.add(tr["from"])
        if tr.get("to"):   state_used.add(tr["to"])
    unused = state_defined - state_used
    if unused: issues.append(f"Unused states: {sorted(unused)}")
    prefix_re = re.compile(r"^\s*\[(?:CHANGE|AFTER)\]")
    for tr in icg.get("transitions", []):
        tid = tr.get("id", "?")
        if not tr.get("agent_task"):        issues.append(f"T{tid} missing agent_task")
        mti = tr.get("mapped_test_items") or []
        if not mti: issues.append(f"T{tid} missing mapped_test_items")
        doms  = tr.get("dom_assertions") or []
        posts = tr.get("postconditions") or []
        if not doms and not posts: issues.append(f"T{tid} has neither dom_assertions nor postconditions")
        for idx, item in enumerate(doms):
            text = item.get("assertion", "") if isinstance(item, dict) else ""
            if not prefix_re.match(text):
                issues.append(f"T{tid}.dom_assertions[{idx}] missing [CHANGE]/[AFTER] prefix")
        for idx, item in enumerate(posts):
            text = item.get("assertion", "") if isinstance(item, dict) else ""
            if prefix_re.match(text):
                issues.append(f"T{tid}.postconditions[{idx}] must not carry [CHANGE]/[AFTER] prefix")
    return issues


def save_call_log(api_call_log_dir, task_id, step, model, msgs, raw, usage, dur, parsed, status):
    api_call_log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    e = {
        "call_id": str(uuid.uuid4()), "timestamp": now.isoformat(),
        "task_id": task_id, "step": step, "model": model,
        "status": status, "duration_ms": dur,
        "prompt_tokens":     usage.get("prompt_tokens",     0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens":      usage.get("total_tokens",      0),
        "input_messages": msgs, "response_text": raw, "parsed_output": parsed,
    }
    ts = now.strftime("%Y%m%d_%H%M%S_%f")
    fn = f"{ts}_{task_id}_{step}_{status}.json"
    (api_call_log_dir / fn).write_text(json.dumps(e, ensure_ascii=False, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Core Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_task(task, data_dir: Path, api_call_log_dir: Path,
                 client, model: str, force=False, force_all=False, force_contract=False,
                 force_items=False):
    task_id  = task["full_id"]
    task_dir = data_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    contract_cache = task_dir / "contract.json"
    items_cache    = task_dir / "test_items.json"
    icg_out        = task_dir / "icg.json"

    # Clear caches requested by CLI flags.
    if force_all:
        for f in [contract_cache, items_cache, icg_out]:
            if f.exists(): f.unlink()
    elif force_contract:
        for f in [contract_cache, items_cache]:
            if f.exists(): f.unlink()
    elif force_items:
        for f in [items_cache, icg_out]:
            if f.exists(): f.unlink()

    wr = task["washed_requirements"]
    implicit_reqs = [w for w in wr if w.get("intent_type") != "explicit"]

    if icg_out.exists() and not force:
        return {"task": task["folder_name"], "status": "skipped_exists"}

    contract         = None
    base_test_items  = []

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1a: Generate the Test Data Contract.
    # ══════════════════════════════════════════════════════════════════════
    if contract_cache.exists():
        cached_contract = json.loads(contract_cache.read_text(encoding="utf-8"))
        contract = cached_contract.get("test_data_contract", "")
        print("[contract-cache] ", end="", flush=True)
    else:
        c1a_user = build_call1a_prompt(wr)
        for att in range(1, 4):
            raw1a, u1a, d1a, m1a = "", {}, 0, []
            try:
                raw1a, u1a, d1a, m1a = call_api(SYSTEM_CALL1A_CONTRACT_V9, c1a_user, client, model)
                cr1a = parse_json(raw1a)
                contract = cr1a.get("test_data_contract", "").strip()
                if not contract:
                    raise ValueError("empty test_data_contract")
                save_call_log(api_call_log_dir, task_id, "call1a", model, m1a, raw1a, u1a, d1a, cr1a, "success")
                break
            except Exception as e:
                print(f"\n[WARN] Call1a att={att} {type(e).__name__}: {str(e)[:80]}")
                save_call_log(api_call_log_dir, task_id, "call1a", model, m1a, raw1a, u1a, d1a, None, "error")
                if att < 3: time.sleep(3)
                else: return {"task": task["folder_name"], "status": "error", "msg": f"Call1a: {str(e)[:100]}"}

        # Call 2: Check that the contract does not expose implicit requirements.
        if implicit_reqs:
            step1_5_passed = False
            for att in range(1, 4):
                raw2, u2, d2, m2 = "", {}, 0, []
                try:
                    c2_user = build_call2_prompt(wr, contract)
                    raw2, u2, d2, m2 = call_api(SYSTEM_CHECK_CONTRACT_V1, c2_user, client, model, max_tokens=1000)
                    cr2 = parse_json(raw2)
                    verdict = cr2.get("verdict", "FAIL")
                    issues2 = cr2.get("issues", [])
                    save_call_log(api_call_log_dir, task_id, "call2", model, m2, raw2, u2, d2, cr2,
                                  "pass" if verdict == "PASS" else "fail")
                    if verdict == "PASS":
                        step1_5_passed = True
                        break
                    print(f"\n[RETRY] Call2 att={att} contract issues: {issues2}")
                    raw1ar, u1ar, d1ar, m1ar = "", {}, 0, []
                    try:
                        retry_user = build_call1a_retry_prompt(wr, contract, issues2)
                        raw1ar, u1ar, d1ar, m1ar = call_api(SYSTEM_CALL1A_CONTRACT_V9, retry_user, client, model)
                        cr1ar = parse_json(raw1ar)
                        contract = cr1ar.get("test_data_contract", "")
                        save_call_log(api_call_log_dir, task_id, "call1a_retry", model, m1ar, raw1ar, u1ar, d1ar, cr1ar, "success")
                    except Exception as e2:
                        print(f"\n[WARN] Call1a retry att={att} {type(e2).__name__}: {str(e2)[:60]}")
                    if att < 3: time.sleep(3)
                except Exception as e:
                    print(f"\n[WARN] Call2 att={att} {type(e).__name__}: {str(e)[:80]}")
                    save_call_log(api_call_log_dir, task_id, "call2", model, m2, raw2, u2, d2, None, "error")
                    step1_5_passed = True
                    break
            if not step1_5_passed:
                return {"task": task["folder_name"], "status": "failed_contract_check",
                        "msg": "Call2: contract still exposes implicit reqs after 3 attempts"}

        contract_cache.write_text(
            json.dumps({"task_id": task_id, "test_data_contract": contract},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1b: Generate test items.
    # ══════════════════════════════════════════════════════════════════════
    if items_cache.exists():
        cached = json.loads(items_cache.read_text(encoding="utf-8"))
        base_test_items   = cached.get("test_items",        [])
        print("[items-cache] ", end="", flush=True)
    else:
        c1b_user = build_call1b_prompt(wr)
        all_req_ids = {w["full_id"] for w in wr}
        retry_uncov_reqs = set()
        retry_bloated = []
        for att in range(1, 4):
            raw1b, u1b, d1b, m1b = "", {}, 0, []
            try:
                c1b_attempt_user = (
                    build_call1b_retry_prompt(wr, retry_uncov_reqs, retry_bloated)
                    if (retry_uncov_reqs or retry_bloated)
                    else c1b_user
                )
                raw1b, u1b, d1b, m1b = call_api(SYSTEM_CALL1B_TESTITEMS_V9, c1b_attempt_user, client, model)
                cr1b = parse_json(raw1b)
                base_test_items = cr1b.get("test_items", [])
                covered_reqs = {rid for ti in base_test_items for rid in ti.get("req_ids", []) or []}
                uncov_reqs = all_req_ids - covered_reqs
                bloated = [ti["item_id"] for ti in base_test_items if len(ti.get("req_ids", [])) > 3]
                if (uncov_reqs or bloated) and att < 3:
                    retry_uncov_reqs = uncov_reqs
                    retry_bloated = bloated
                    if uncov_reqs:
                        print(f"\n[RETRY] Call1b att={att} uncovered reqs: {sorted(uncov_reqs)}")
                    if bloated:
                        print(f"\n[RETRY] Call1b att={att} TIs with >3 req_ids: {bloated}")
                    save_call_log(api_call_log_dir, task_id, "call1b", model, m1b, raw1b, u1b, d1b, cr1b, "coverage_incomplete")
                    time.sleep(3); continue
                status_tag = "success" if (not uncov_reqs and not bloated) else "coverage_incomplete"
                save_call_log(api_call_log_dir, task_id, "call1b", model, m1b, raw1b, u1b, d1b, cr1b, status_tag)
                if uncov_reqs:
                    return {"task": task["folder_name"], "status": "failed_req_coverage",
                            "msg": f"Call1b: uncovered reqs after 3 attempts: {sorted(uncov_reqs)}"}
                if bloated:
                    print(f"\n[WARN] Call1b: TIs still have >3 req_ids after 3 attempts: {bloated} — proceeding")
                break
            except Exception as e:
                print(f"\n[WARN] Call1b att={att} {type(e).__name__}: {str(e)[:80]}")
                save_call_log(api_call_log_dir, task_id, "call1b", model, m1b, raw1b, u1b, d1b, None, "error")
                if att < 3: time.sleep(3)
                else: return {"task": task["folder_name"], "status": "error", "msg": f"Call1b: {str(e)[:100]}"}

        items_data = {
            "task_id":            task_id,
            "task_name":          task.get("task_name", ""),
            "test_data_contract": contract,
            "test_items":         base_test_items,
        }
        items_cache.write_text(json.dumps(items_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Generate the ICG JSON.
    # ══════════════════════════════════════════════════════════════════════
    all_items = base_test_items

    def _fill_icg_meta(raw_icg):
        # Normalize: ensure states and transitions use "id" field (not "state_id" etc.)
        for s in raw_icg.get("states", []):
            if isinstance(s, dict) and "id" not in s:
                s["id"] = s.get("state_id", s.get("name", "S?"))
        for tr in raw_icg.get("transitions", []):
            if isinstance(tr, dict) and "id" not in tr:
                tr["id"] = tr.get("transition_id", tr.get("name", "T?"))

        explicit_content = " ".join(
            w.get("content_en", "")
            for w in wr if w.get("intent_type") == "explicit"
        )
        scenario_en  = task.get("scenario", "")
        task_name_en = task.get("task_name", "")
        input_text = (
            f"Please implement a {task_name_en} web page "
            f"in a {scenario_en} scenario. "
            f"{explicit_content}\n\nTest Data Contract: {contract}"
        )
        cov = build_cascade_coverage(raw_icg, all_items, wr)
        return {
            "task_id":             task_id,
            "task_name":           task.get("task_name", ""),
            "domain":              task.get("domain", ""),
            "scenario":            task.get("scenario", ""),
            "test_data_contract":  contract,
            "input_text":          input_text,
            "test_items_coverage":   cov["test_items_coverage"],
            "requirements_coverage": cov["requirements_coverage"],
            "states":      raw_icg.get("states",      []),
            "transitions": raw_icg.get("transitions", []),
        }

    icg  = None
    c4_user = build_call4_prompt(contract, base_test_items)
    for att in range(1, 4):
        raw4, u4, d4, m4 = "", {}, 0, []
        try:
            raw4, u4, d4, m4 = call_api(SYSTEM_CALL4_ICG_V9, c4_user, client, model, max_tokens=32000)
            icg    = _fill_icg_meta(parse_json(raw4))
            issues = validate_coverage(icg, all_items)
            if issues and att < 3:
                print(f"\n[RETRY] Call4 att={att} coverage: {issues}")
                save_call_log(api_call_log_dir, task_id, "call4", model, m4, raw4, u4, d4, icg, "coverage_incomplete")
                time.sleep(3); continue
            save_call_log(api_call_log_dir, task_id, "call4", model, m4, raw4, u4, d4, icg,
                          "success" if not issues else "coverage_incomplete")
            break
        except Exception as e:
            print(f"\n[WARN] Call4 att={att} {type(e).__name__}: {str(e)[:80]}")
            save_call_log(api_call_log_dir, task_id, "call4", model, m4, raw4, u4, d4, None, "error")
            if att < 3: time.sleep(3)
            else: return {"task": task["folder_name"], "status": "error", "msg": f"Call4: {str(e)[:100]}"}

    if icg is None:
        return {"task": task["folder_name"], "status": "error", "msg": "Call4: no output"}

    icg_out.write_text(json.dumps(icg, ensure_ascii=False, indent=2), encoding="utf-8")

    final_issues  = validate_coverage(icg, all_items)
    ti_cov        = icg.get("test_items_coverage",   [])
    req_cov       = icg.get("requirements_coverage", [])
    n_ti_covered  = sum(1 for ti in ti_cov  if ti.get("covered_by_transitions"))
    n_req_covered = sum(1 for rc in req_cov if rc.get("covered_by_transitions"))
    return {
        "task":          task["folder_name"],
        "status":        "success",
        "n_s":           len(icg.get("states",      [])),
        "n_t":           len(icg.get("transitions", [])),
        "n_ti_covered":  n_ti_covered,
        "n_ti_total":    len(all_items),
        "n_req_covered": n_req_covered,
        "n_req_total":   len(wr),
        "issues":        final_issues,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Task Loader
# ══════════════════════════════════════════════════════════════════════════════

def _iter_domains(data):
    """Support {"domains": [...]}, a single domain object, or a domain list."""
    if isinstance(data, list):
        return data
    if "domains" in data:
        return data["domains"]
    if "scenarios" in data:
        return [data]
    return []

def load_tasks(json_path: str, data_dir: Path) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "tasks" in data and "domains" not in data and "scenarios" not in data:
        tasks = []
        for t in data["tasks"]:
            fid = str(t.get("full_task_id") or t["task_id"])
            tasks.append({
                "full_id":             fid,
                "folder_name":         fid,
                "sf":                  "",
                "domain":              "",
                "scenario":            "",
                "task_name":           t.get("task_name", fid),
                "washed_requirements": t.get("requirements", []),
                "has_contract": (data_dir / fid / "contract.json").exists(),
                "has_items":   (data_dir / fid / "test_items.json").exists(),
                "has_icg":     (data_dir / fid / "icg.json").exists(),
            })
        return tasks

    tasks = []
    for d in _iter_domains(data):
        did = d["domain_id"]
        de  = d.get("domain", "")
        for s in d["scenarios"]:
            sid = s["scenario_id"]
            se  = s.get("scenario", "")
            for t in s["tasks"]:
                tid  = int(t["task_id"])
                name = t.get("task_name", str(t["task_id"]))
                fid  = t.get("full_task_id") or f"D{did:02d}_S{sid:02d}_T{tid:03d}"
                sf   = "_".join(fid.split("_")[:2])
                tasks.append({
                    "full_id":            fid,
                    "folder_name":        fid,
                    "sf":                 sf,
                    "domain":             de,
                    "scenario":           se,
                    "task_name":          name,
                    "washed_requirements": t.get("requirements", []),
                    "has_contract": (data_dir / fid / "contract.json").exists(),
                    "has_items":   (data_dir / fid / "test_items.json").exists(),
                    "has_icg":     (data_dir / fid / "icg.json").exists(),
                })
    return tasks


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="WebRISE ICG generation pipeline")
    p.add_argument("-i",  "--input",           default=INPUT_JSON)
    p.add_argument("--data-dir",               default=DATA_DIR)
    p.add_argument("--api-call-log-dir",       default=API_CALL_LOG_DIR)
    p.add_argument("--model",                  default=MODEL)
    p.add_argument("--api-key",                default=None)
    p.add_argument("--base-url",               default=None)
    p.add_argument("--test",     action="store_true")
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--filter",   default=None)
    p.add_argument("--list-scenarios", action="store_true")
    p.add_argument("--delay",    type=int, default=DELAY)
    p.add_argument("--force",    action="store_true",
                   help="Regenerate Call 4 while keeping contract and test item caches")
    p.add_argument("--force-all", action="store_true",
                   help="Regenerate all calls and remove all caches")
    p.add_argument("--force-contract", action="store_true",
                   help="Regenerate Call 1a+1b by deleting contract and test item caches")
    p.add_argument("--force-items", action="store_true",
                   help="Regenerate Call 1b+Call 4 while keeping the contract cache")
    a = p.parse_args()
    if a.force_all or a.force_items: a.force = True

    dd  = Path(a.data_dir)
    acd = Path(a.api_call_log_dir)
    tasks = load_tasks(a.input, dd)
    if a.test: a.filter = TEST_TASK

    if a.list_scenarios:
        sc = {}
        for t in tasks:
            s = sc.setdefault(t["sf"], {"name": t["scenario"], "n": 0, "contract": 0, "items": 0, "icg": 0, "w": 0})
            s["n"] += 1
            if t["has_contract"]: s["contract"] += 1
            if t["has_items"]:    s["items"]    += 1
            if t["has_icg"]:      s["icg"]      += 1
            if t["washed_requirements"]: s["w"] += 1
        for sid in sorted(sc):
            s = sc[sid]
            print(f"  {sid:<12} {s['name']:>20} n={s['n']:>3} w={s['w']:>3} "
                  f"contract={s['contract']:>3} items={s['items']:>3} icg={s['icg']:>3} "
                  f"todo={max(0, s['w'] - s['icg']):>3}")
        return

    if a.filter:
        tasks = [t for t in tasks if a.filter in t["folder_name"] or a.filter in t["full_id"]]
    ready   = [t for t in tasks if t["washed_requirements"] or (t["has_contract"] and t["has_items"])]
    pending = ready if a.force else [t for t in ready if not t["has_icg"]]
    if not pending: print("All done."); return

    print(f"Ready: {len(ready)}, existing ICG: {len(ready) - len(pending)}, pending: {len(pending)}")

    if a.dry_run:
        for t in pending:
            if t["has_items"]:       tag = "[items]"
            elif t["has_contract"]:  tag = "[ctrt ]"
            else:                    tag = "[ new ]"
            print(f"  {tag} {t['folder_name']}  ({len(t['washed_requirements'])} reqs)")
        return

    ak = a.api_key or API_KEY or os.environ.get("OPENAI_API_KEY", "")
    bu = a.base_url or BASE_URL or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not ak:
        print("❌ No API Key. Set OPENAI_API_KEY or use --api-key or edit API_KEY in script.")
        sys.exit(1)

    from openai import OpenAI
    cl = OpenAI(api_key=ak, base_url=bu)
    try:
        cl.models.list()
        print("✅ API connection ok")
    except Exception as e:
        print(f"❌ API connection failed: {e}\n   BASE_URL: {bu}")
        sys.exit(1)

    print(f"\nStarting {len(pending)} task(s) (model={a.model})...\n")
    ok = fa = cf = 0

    for i, t in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}] {t['folder_name']}...", end=" ", flush=True)
        r = process_task(t, dd, acd, cl, a.model, force=a.force, force_all=a.force_all,
                         force_contract=a.force_contract, force_items=a.force_items)

        if r["status"] == "success":
            iss = f" ⚠{len(r['issues'])}" if r.get("issues") else ""
            print(f"✅ ({r['n_s']}S {r['n_t']}T "
                  f"TI={r['n_ti_covered']}/{r['n_ti_total']} R={r['n_req_covered']}/{r['n_req_total']}"
                  f"{iss})")
            if r.get("issues"):
                for issue in r["issues"]: print(f"      ⚠  {issue}")
            ok += 1; cf = 0
        elif r["status"] == "skipped_exists":
            print("⏭️")
        else:
            print(f"❌ {r.get('msg', '')[:80]}")
            fa += 1; cf += 1

        if cf >= MAX_CONSECUTIVE_FAILS:
            print(f"\n⛔ {cf} consecutive failures; stopping"); break
        if i < len(pending):
            time.sleep(a.delay)

    print(f"\n✅ {ok}  ❌ {fa}")
    if fa > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
