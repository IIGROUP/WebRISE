"""
DOM assertion scorer - Uses an LLM to judge transient/process assertions from
structured DOM event evidence collected by MutationObserver.

Input evidence is text-only JSON (no screenshots). The scorer evaluates each
natural-language DOM assertion against the provided event timeline.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

_RETRY_DELAYS = [3, 8, 20]

import config


def _stream_chat(client: OpenAI, model: str, max_tokens: int, messages: list) -> tuple[str, dict]:
    """Call the chat API in streaming mode and return (full_text, usage_dict)."""
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

DOM_SCORER_SYSTEM_PROMPT = """\
You are a strict UI test evaluator. Your job is to determine whether specific DOM assertions
are satisfied based on structured DOM event evidence collected during a web interaction.

You will receive:
  • The action that was performed
  • A numbered list of DOM assertions to verify, each optionally prefixed with [CHANGE] or [AFTER]
  • A JSON evidence object containing:
    - initial_snapshot (page_text)
    - events (each with added_nodes, removed_nodes, changed_attributes, snapshot)
    - final_snapshot (page_text)

Assertion prefix semantics — both tags require searching the ENTIRE timeline
(initial_snapshot, every event's added_nodes / removed_nodes / changed_attributes /
snapshot.page_text, and final_snapshot):
  - [CHANGE] — the assertion describes something that happened at any point. If the
    described condition appears ANYWHERE in the timeline — even transiently, even if it
    later reverts, even if it was already true initially — the verdict is YES.
  - [AFTER] — the assertion describes the final state. Search the full timeline for
    evidence, but the described condition must hold in the FINAL state to pass. Use
    final_snapshot.page_text AND the added_nodes / changed_attributes entry for the
    matching element to determine the final state. If the final state satisfies the
    assertion, the verdict is YES regardless of what the initial state was.

Field meanings:
  - t_ms: milliseconds since monitoring started
  - mutation_types: the high-level mutation categories observed in that batch
  - added_nodes / removed_nodes: summarized nodes that appeared or disappeared in that batch.
    Each summary includes the node's tag/id/role/class/text plus rich state fields
    (disabled, pointer_events, value, aria_selected, aria_expanded, aria_pressed,
    aria_checked) and `descendant_signatures` (an optional `{"<tag.class>": count}` map of
    inner descendants) so you can recognize inner structure like skeleton placeholders,
    card templates, etc. even when the added node itself is only a generic wrapper (e.g.
    an empty `div.product-grid` whose children are the actual skeleton cards).
  - changed_attributes: summarized attribute or text-related changes observed in that batch.
    Each entry has { attribute, old_value, new_value, node } where `node` carries the same
    rich state fields listed above for the element that was mutated.
  - initial_snapshot.page_text / final_snapshot.page_text: page text at the start
    and end of the observation window (intermediate events do not carry page_text).
    Text prefixed with `[not-visible]` exists in the DOM but is not visually visible
    because the element or an ancestor is hidden, opacity-hidden, or has no visible
    box. Do NOT treat `[not-visible]` text as evidence that the text or
    control is visibly present in the UI. Also do NOT use `[not-visible]` text as
    contradictory evidence that a hidden status/control is active in the visible UI;
    a hidden "Dangling" or "Closed" template label does not negate visible active
    state evidence elsewhere.
  - initial_snapshot.interactive_elements / final_snapshot.interactive_elements: a list of
    all visible interactive elements (buttons, inputs, selects, textareas, links, etc.)
    at the start and end of the observation window. Each entry has tag, id, class, text,
    visible, disabled, pointer_events, value, min, max, selected_text, aria_label, aria_selected, aria_expanded,
    aria_pressed, aria_checked. Use these to verify [AFTER] assertions about element state
    (e.g. "the publish button is interactive") even when no MutationObserver event was
    recorded for that element — the snapshot captures its current state directly.

No single target element is pre-resolved. Locate the element an assertion describes by
matching on node.tag / node.role / node.class / node.text / node.id inside added_nodes,
removed_nodes, or changed_attributes[].node, AND in the interactive_elements lists.
If the element cannot be located anywhere in the evidence, prefer UNCERTAIN over NO —
EXCEPT when the assertion claims the element is disabled / non-interactive / not present:
if the element is absent from final_snapshot.interactive_elements, that absence IS
evidence of non-interactivity, so answer YES.

Additional guidance:
  - **Debounce / delayed update behavior**: the executor may apply an entire text input in
    one `Input` action, so the evidence often does not contain per-keystroke intermediate
    values. For assertions that say filtering, saving, or requests are debounced / delayed
    until typing pauses, answer YES when the timeline shows a single delayed update/save
    after input settles and does not show repeated intermediate updates for each typed
    character. Do NOT require explicit keystroke-level events to prove debounce.
  - **Disabled / non-interactive state**: when an assertion says an element "is disabled" or
    "cannot be clicked/interacted with", use this priority order on the node:
    (1) `pointer_events === "none"` — browser-enforced; the element truly cannot receive clicks
        regardless of element type. This is the most authoritative signal.
    (2) `disabled: true` — valid for native form elements (button, input, select), or a
        changed_attributes entry with attribute="disabled" and truthy new_value, or
        attribute="aria-disabled" with new_value="true".
    (3) `class` contains a token like "disabled", "inactive", "closed" — CSS-based disabling
        used by many custom components that are not native form elements.
    Answer YES if ANY of the above is true. Do NOT require `disabled: true` when the element
    is a non-form element (div, span, li, custom component) — such elements never have
    `disabled: true` regardless of their interactive state.
  - **Selected / highlighted / active state**: when an assertion says an element "is selected",
    "is highlighted", or "is active", use this priority order on the node:
    (1) `aria_selected`, `aria_pressed`, or `aria_checked` equals `true`, or a
        changed_attributes entry with the corresponding aria-* attribute and new_value="true"
        — the most reliable semantic signal for selection state.
    (2) `class` contains a token like "selected", "active", "highlighted", "current", "checked"
        — valid secondary evidence when aria attributes are absent.
    Answer YES if ANY of the above is true.
  - **Semantic equivalence — judge by meaning, not wording.** The assertion and the
    implementation may use different terms for the same concept. Examples: "word count"
    vs "character count", "publish" vs "post", "delete" vs "remove", "avatar" vs
    "profile picture". If the DOM element serves the same functional purpose as what the
    assertion describes, answer YES. Only answer NO if the described element/feature is
    entirely absent or serves a clearly different purpose.
  - Be strict on factual correctness, but lenient on terminology.
  - Answer NO if the evidence clearly contradicts the assertion.
  - Answer UNCERTAIN if the evidence is incomplete or ambiguous.

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

DOM_SCORER_USER_TEMPLATE = """\
Action performed: {action_desc}

DOM assertions to verify:
{conditions_numbered}

DOM evidence JSON:
{evidence_json}
"""


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def _format_action(action: dict[str, Any], affordance_type: str) -> str:
    value = action.get("value", "")
    return (
        f"{action['type']} '{value}' on {affordance_type}"
        if value
        else f"{action['type']} on {affordance_type}"
    )


_EVIDENCE_CHAR_BUDGET = 200_000   # Keep DOM scorer calls within request-size limits.


def _truncate_str(s: Any, limit: int) -> Any:
    if isinstance(s, str) and len(s) > limit:
        return s[:limit] + f"…[truncated {len(s) - limit} chars]"
    return s


def _json_len(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False))


def _cap_list(items: Any, limit: int, label: str) -> Any:
    if not isinstance(items, list) or len(items) <= limit:
        return items
    if limit <= 0:
        return [{f"_omitted_{label}": len(items)}]
    head = max(1, limit // 2)
    tail = max(0, limit - head)
    capped = items[:head]
    capped.append({f"_omitted_{label}": len(items) - limit})
    if tail:
        capped.extend(items[-tail:])
    return capped


_KEYWORD_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "then", "when",
    "after", "before", "should", "shows", "show", "visible", "appears",
    "displayed", "user", "action", "page", "button", "field", "control",
    "state", "status", "value", "text", "item", "list", "row", "column",
    "change", "changed", "update", "updated", "selected", "click",
    "clicked", "enter", "entered", "input", "final", "initial", "same",
}


def _condition_keywords(conditions: list[str] | None) -> set[str]:
    text = " ".join(str(c) for c in (conditions or [])).lower()
    tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text))
    quoted = set(q.lower() for q in re.findall(r"['\"]([^'\"]{2,})['\"]", text))
    return {t for t in tokens | quoted if t not in _KEYWORD_STOPWORDS}


def _relevance_score(obj: Any, keywords: set[str]) -> int:
    if not keywords:
        return 0
    text = json.dumps(obj, ensure_ascii=False).lower()
    return sum(1 for kw in keywords if kw and kw in text)


def _cap_relevant_list(items: Any, limit: int, label: str,
                       keywords: set[str] | None = None) -> Any:
    if not isinstance(items, list) or len(items) <= limit:
        return items
    keywords = keywords or set()
    if not keywords:
        return _cap_list(items, limit, label)

    selected: set[int] = set()
    ranked = sorted(
        (( _relevance_score(item, keywords), i) for i, item in enumerate(items)),
        reverse=True,
    )
    relevant_budget = max(1, limit // 2)
    for score, i in ranked:
        if score <= 0 or len(selected) >= relevant_budget:
            break
        selected.add(i)

    edge_budget = max(0, limit - len(selected))
    head = edge_budget // 2
    tail = edge_budget - head
    selected.update(range(min(head, len(items))))
    if tail:
        selected.update(range(max(0, len(items) - tail), len(items)))

    if len(selected) > limit:
        selected = set(sorted(selected)[:limit])

    capped = [items[i] for i in sorted(selected)]
    capped.append({f"_omitted_{label}": len(items) - len(selected)})
    return capped


def _compact_node(node: Any, text_limit: int) -> Any:
    if not isinstance(node, dict):
        return _truncate_str(node, text_limit)
    keep = (
        "tag", "id", "role", "class", "text", "visible", "disabled",
        "pointer_events", "value", "min", "max", "selected_text",
        "aria_label", "aria_selected", "aria_expanded", "aria_pressed",
        "aria_checked", "descendant_signatures",
    )
    out = {}
    for key in keep:
        if key in node and node[key] not in (None, "", [], {}):
            out[key] = _truncate_str(node[key], text_limit)
    return out


def _compact_changed_attr(item: Any, text_limit: int) -> Any:
    if not isinstance(item, dict):
        return _truncate_str(item, text_limit)
    return {
        k: v for k, v in {
            "attribute": _truncate_str(item.get("attribute"), text_limit),
            "old_value": _truncate_str(item.get("old_value"), text_limit),
            "new_value": _truncate_str(item.get("new_value"), text_limit),
            "node": _compact_node(item.get("node") or {}, text_limit),
        }.items()
        if v not in (None, "", [], {})
    }


def _compact_snapshot(snapshot: Any, page_text_limit: int,
                      interactive_limit: int, text_limit: int,
                      keywords: set[str] | None = None) -> Any:
    if not isinstance(snapshot, dict):
        return {}
    out = {}
    if "page_text" in snapshot:
        out["page_text"] = _truncate_str(snapshot.get("page_text"), page_text_limit)
    if "target" in snapshot:
        out["target"] = _compact_node(snapshot.get("target") or {}, text_limit)
    if "interactive_elements" in snapshot:
        elems = [_compact_node(e, text_limit) for e in snapshot.get("interactive_elements") or []]
        out["interactive_elements"] = _cap_relevant_list(
            elems, interactive_limit, "interactive_elements", keywords
        )
    return out


def _compact_event(event: Any, item_limit: int, page_text_limit: int,
                   interactive_limit: int, text_limit: int,
                   keywords: set[str] | None = None) -> Any:
    if not isinstance(event, dict):
        return {}
    return {
        "t_ms": event.get("t_ms"),
        "kind": event.get("kind"),
        "mutation_types": _cap_list(event.get("mutation_types") or [], 10, "mutation_types"),
        "added_nodes": _cap_relevant_list(
            [_compact_node(n, text_limit) for n in event.get("added_nodes") or []],
            item_limit,
            "added_nodes",
            keywords,
        ),
        "removed_nodes": _cap_relevant_list(
            [_compact_node(n, text_limit) for n in event.get("removed_nodes") or []],
            item_limit,
            "removed_nodes",
            keywords,
        ),
        "changed_attributes": _cap_relevant_list(
            [_compact_changed_attr(n, text_limit) for n in event.get("changed_attributes") or []],
            item_limit,
            "changed_attributes",
            keywords,
        ),
        "snapshot": _compact_snapshot(
            event.get("snapshot") or {},
            page_text_limit,
            interactive_limit,
            text_limit,
            keywords,
        ),
    }


def _force_compact_evidence(evidence: dict[str, Any], *, event_limit: int,
                            item_limit: int, interactive_limit: int,
                            page_text_limit: int, text_limit: int,
                            keywords: set[str] | None = None) -> dict[str, Any]:
    events = evidence.get("events") or []
    capped_events = _cap_relevant_list(events, event_limit, "events", keywords)
    compact_events = [
        e if isinstance(e, dict) and any(k.startswith("_omitted_") for k in e)
        else _compact_event(e, item_limit, page_text_limit, interactive_limit, text_limit, keywords)
        for e in capped_events
    ]
    return {
        "initial_snapshot": _compact_snapshot(
            evidence.get("initial_snapshot") or {},
            page_text_limit,
            interactive_limit,
            text_limit,
            keywords,
        ),
        "events": compact_events,
        "final_snapshot": _compact_snapshot(
            evidence.get("final_snapshot") or {},
            page_text_limit,
            interactive_limit,
            text_limit,
            keywords,
        ),
        "observed_elements": _cap_relevant_list(
            [_compact_node(e, text_limit) for e in evidence.get("observed_elements") or []],
            interactive_limit,
            "observed_elements",
            keywords,
        ),
        "_truncation_note": (
            "Evidence was compacted to fit the DOM scorer budget, prioritizing "
            "events/nodes that match assertion keywords; "
            f"original_events={len(events)}"
        ),
    }


def _truncate_evidence(evidence: dict[str, Any], budget: int = _EVIDENCE_CHAR_BUDGET,
                       conditions: list[str] | None = None) -> dict[str, Any]:
    """Progressively shrink evidence JSON until it fits within `budget` characters.

    Passes applied in order; each pass stops early if the budget is already met.
    Pass 1 - truncate all page_text fields to 2 000 chars.
    Pass 2 - truncate all node .text fields to 500 chars.
    Pass 3 - truncate any remaining string value > 200 chars to 200 chars.
    Pass 4 - keep only the last 30 events (most recent = most diagnostic).
    Pass 5+ — if the evidence is still too large, compact arrays while
    preserving entries that match assertion keywords, plus initial/final edges.
    """
    import copy
    keywords = _condition_keywords(conditions)
    orig_len = _json_len(evidence)
    if orig_len <= budget:
        return evidence

    ev = copy.deepcopy(evidence)

    # Pass 1: page_text → 2 000 chars
    def _clip_page_text(obj: Any) -> None:
        if isinstance(obj, dict):
            if "page_text" in obj:
                obj["page_text"] = _truncate_str(obj["page_text"], 2000)
            for v in obj.values():
                _clip_page_text(v)
        elif isinstance(obj, list):
            for item in obj:
                _clip_page_text(item)

    _clip_page_text(ev)
    if _json_len(ev) <= budget:
        print(f"[dom_scorer] evidence truncated (pass 1): {orig_len:,} → {_json_len(ev):,} chars")
        return ev

    # Pass 2: node .text → 500 chars
    for event in ev.get("events", []):
        for key in ("added_nodes", "removed_nodes"):
            for node in event.get(key, []):
                if isinstance(node, dict) and "text" in node:
                    node["text"] = _truncate_str(node["text"], 500)

    if _json_len(ev) <= budget:
        print(f"[dom_scorer] evidence truncated (pass 2): {orig_len:,} → {_json_len(ev):,} chars")
        return ev

    # Pass 3: any remaining string > 200 chars
    def _clip_all(obj: Any, limit: int = 200) -> Any:
        if isinstance(obj, str):
            return _truncate_str(obj, limit)
        if isinstance(obj, dict):
            return {k: _clip_all(v, limit) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clip_all(item, limit) for item in obj]
        return obj

    ev = _clip_all(ev)
    if _json_len(ev) <= budget:
        print(f"[dom_scorer] evidence truncated (pass 3): {orig_len:,} → {_json_len(ev):,} chars")
        return ev

    # Pass 4: keep last 30 events
    if len(ev.get("events", [])) > 30:
        ev["events"] = ev["events"][-30:]
        ev["_truncation_note"] = f"Events capped to last 30 (original: {len(evidence.get('events', []))})"

    new_len = _json_len(ev)
    if new_len <= budget:
        print(f"[dom_scorer] evidence truncated (pass 4): {orig_len:,} → {new_len:,} chars")
        return ev

    # Pass 5: hard budget enforcement. The earlier passes shorten long strings,
    # but pages with thousands of short mutation entries can still blow up the
    # request. Progressively cap list cardinality too.
    for i, params in enumerate((
        (30, 30, 100, 1500, 160),
        (24, 20, 80, 1200, 120),
        (16, 12, 60, 900, 100),
        (10, 8, 40, 600, 80),
        (6, 5, 25, 400, 60),
    ), start=5):
        candidate = _force_compact_evidence(
            ev,
            event_limit=params[0],
            item_limit=params[1],
            interactive_limit=params[2],
            page_text_limit=params[3],
            text_limit=params[4],
            keywords=keywords,
        )
        cand_len = _json_len(candidate)
        if cand_len <= budget:
            print(f"[dom_scorer] evidence truncated (pass {i}): {orig_len:,} → {cand_len:,} chars")
            return candidate

    ev = _force_compact_evidence(
        ev,
        event_limit=3,
        item_limit=3,
        interactive_limit=15,
        page_text_limit=250,
        text_limit=50,
        keywords=keywords,
    )
    new_len = _json_len(ev)
    print(f"[dom_scorer] evidence truncated (final hard cap): {orig_len:,} → {new_len:,} chars")
    return ev


def _strip_raw_mutations(dom_log: dict[str, Any]) -> dict[str, Any]:
    def clean_event(event: dict[str, Any]) -> dict[str, Any]:
        cleaned = {
            "t_ms": event.get("t_ms"),
            "kind": event.get("kind"),
            "mutation_types": event.get("mutation_types") or [],
            "added_nodes": event.get("added_nodes") or [],
            "removed_nodes": event.get("removed_nodes") or [],
            "changed_attributes": event.get("changed_attributes") or [],
            "snapshot": event.get("snapshot"),
        }
        if event.get("target_transition") is not None:
            cleaned["target_transition"] = event.get("target_transition")
        return cleaned

    result = {
        "initial_snapshot": dom_log.get("initial_snapshot") or {},
        "events": [clean_event(e) for e in (dom_log.get("events") or []) if isinstance(e, dict)],
        "final_snapshot": dom_log.get("final_snapshot") or {},
    }

    initial_target = (dom_log.get("initial_snapshot") or {}).get("target") or {}
    final_target = (dom_log.get("final_snapshot") or {}).get("target") or {}
    if initial_target or final_target:
        changed = {k: initial_target.get(k) != final_target.get(k)
                   for k in set(initial_target) | set(final_target)}
        result["final_target_transition"] = {
            "before": initial_target,
            "after": final_target,
            "changed": changed,
        }

    observed_elements = dom_log.get("observed_elements")
    if observed_elements:
        result["observed_elements"] = observed_elements
    return result


async def score_dom_assertions(
    client: OpenAI,
    dom_log: dict[str, Any],
    conditions: list[str],
    action: dict[str, Any],
    affordance_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Score natural-language DOM assertions using structured DOM event evidence.

    Returns:
      (results, call_log)

      results:  list of { "condition": str, "verdict": str, "passed": bool, "think": str }
      call_log: { model, input_messages, response_text,
                  prompt_tokens, completion_tokens, total_tokens }
    """
    conditions_numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(conditions))
    action_desc = _format_action(action, affordance_name or "element")
    evidence = _strip_raw_mutations(dom_log)
    evidence = _truncate_evidence(evidence, conditions=conditions)
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)

    user_text = DOM_SCORER_USER_TEMPLATE.format(
        action_desc=action_desc,
        conditions_numbered=conditions_numbered,
        evidence_json=evidence_json,
    )

    messages = [
        {"role": "system", "content": DOM_SCORER_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    response_text = usage = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            print(f"   ⚠  dom_scorer retry in {delay}s (attempt {attempt+1})...")
            time.sleep(delay)
        try:
            response_text, usage = _stream_chat(
                client, config.MODEL_SCORER, config.SCORER_MAX_TOKENS, messages
            )
            if response_text:
                break
            # Empty response is retryable.
            print(f"   ⚠  dom_scorer returned empty content (attempt {attempt+1})")
        except Exception as e:
            if attempt == len(_RETRY_DELAYS):
                raise
            print(f"   ⚠  dom_scorer error: {type(e).__name__}: {str(e)[:160]}")

    if not response_text:
        raise ValueError("DOM scorer API returned empty content after retries")

    call_log = {
        "model": config.MODEL_SCORER,
        **config.reasoning_effort_kwargs(config.MODEL_SCORER),
        "input_messages": messages,
        "response_text": response_text,
        "evidence": evidence,
        **usage,
    }

    results = _parse_verdicts(response_text, conditions)
    return results, call_log


def _parse_verdicts(text: str, conditions: list[str]) -> list[dict[str, Any]]:
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

        think = _normalize_text(entry.get("think", ""))
        passed = verdict == "YES"
        results.append(
            {
                "condition": condition,
                "verdict": verdict,
                "passed": passed,
                "think": think,
            }
        )
    return results


def _extract_json_payload(text: str) -> dict[str, Any]:
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

    starts: list[int] = []
    for match in re.finditer(r'"evaluations"\s*:', text):
        start = text.rfind("{", 0, match.start())
        if start != -1:
            starts.append(start)
    starts.extend(m.start() for m in re.finditer(r"\{", text))

    seen: set[int] = set()
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

    raise ValueError("No JSON object found in response")


def _balanced_json_object_at(text: str, start: int) -> str | None:
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
