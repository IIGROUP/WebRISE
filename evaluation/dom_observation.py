"""
Indexed DOM observation utilities for the contract-guided agent.

Flow:
  1. Use browser-use `dom_tree/index.js` to extract interactive elements with indices.
  2. Convert DOM tree into page-agent-like readable text:
       [index]<tag attrs>text />
     with indentation for hierarchy.
  3. The agent chooses an index in its action; agent_executor resolves it to a selector.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from playwright.async_api import Page

_DOM_TREE_JS_CACHE: str | None = None
_PAGE_OBSERVATION_STATE: dict[int, dict[str, Any]] = {}


def _short_text(text: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)] + "..."


def _signature_of_item(item: dict[str, Any]) -> str:
    xpath = str(item.get("xpath") or "").strip()
    if xpath:
        return f"xpath:{xpath}"

    tag = str(item.get("tag") or "")
    attrs = item.get("attrs", {}) or {}
    key_attrs = []
    for key in ("id", "name", "role", "type", "aria-label", "placeholder", "value"):
        val = attrs.get(key)
        if val is not None and str(val).strip():
            key_attrs.append(f"{key}={str(val).strip()}")
    attrs_sig = "|".join(key_attrs)
    text = str(item.get("text") or "").strip()
    return f"{tag}|{attrs_sig}|{text}"


def _load_dom_tree_js() -> str:
    global _DOM_TREE_JS_CACHE
    if _DOM_TREE_JS_CACHE is not None:
        return _DOM_TREE_JS_CACHE

    here = Path(__file__).resolve()
    dom_tree_path = (
        here.parent
        / "browser-use"
        / "browser_use"
        / "dom"
        / "dom_tree"
        / "index.js"
    )

    if not dom_tree_path.exists():
        raise FileNotFoundError(
            "Cannot find browser-use dom_tree/index.js at: "
            f"{dom_tree_path}. "
            "Please place browser-use under evaluation/browser-use."
        )

    js = dom_tree_path.read_text(encoding="utf-8")
    # page-agent's index.js uses ES module `export default` syntax which is
    # incompatible with page.evaluate(). Strip the prefix so the remaining
    # expression is a plain arrow-function that page.evaluate() can call.
    js = js.replace("export default ", "", 1)
    _DOM_TREE_JS_CACHE = js
    return _DOM_TREE_JS_CACHE


def _attrs_for_prompt(attrs: dict[str, Any], text: str) -> str:
    keep_keys = [
        "title",
        "type",
        "checked",
        "name",
        "role",
        "value",
        "placeholder",
        "data-date-format",
        "alt",
        "aria-label",
        "aria-expanded",
        "aria-selected",
        "aria-current",
        "aria-disabled",
        "data-state",
        "aria-checked",
        "id",
        "for",
        "href",
        "disabled",
        "target",
        "aria-haspopup",
        "aria-controls",
        "aria-owns",
        "contenteditable",
        "readonly",
        "draggable",
        "min",
        "max",
        "step",
    ]

    attrs_to_include: dict[str, str] = {}
    for key in keep_keys:
        value = attrs.get(key)
        if value is None:
            continue
        sval = str(value).strip()
        if sval:
            attrs_to_include[key] = sval

    # Add data-* attributes (up to 2)
    data_keys = sorted([k for k in attrs.keys() if k.startswith("data-")])[:5]
    for key in data_keys:
        sval = str(attrs.get(key, "")).strip()
        if sval:
            attrs_to_include[key] = sval

    raw_class = (attrs.get("class", "") or "").strip()
    if raw_class:
        attrs_to_include["class"] = raw_class

    # De-dup long duplicate values
    keys = list(attrs_to_include.keys())
    if len(keys) > 1:
        seen_values: dict[str, str] = {}
        keys_to_remove: set[str] = set()
        for key in keys:
            value = attrs_to_include[key]
            if len(value) <= 5:
                continue
            if value in seen_values:
                keys_to_remove.add(key)
            else:
                seen_values[value] = key
        for key in keys_to_remove:
            attrs_to_include.pop(key, None)

    if attrs_to_include.get("role") == attrs.get("tagName", ""):
        attrs_to_include.pop("role", None)

    # Remove attrs that duplicate visible text
    text_l = (text or "").strip().lower()
    if text_l:
        for key in ["aria-label", "placeholder", "title"]:
            val = attrs_to_include.get(key, "").strip().lower()
            if val and val == text_l:
                attrs_to_include.pop(key, None)

    if not attrs_to_include:
        return ""

    return " ".join(f"{k}={v}" for k, v in attrs_to_include.items())


def _is_scrollable_node(node: dict[str, Any]) -> bool:
    extra = node.get("extra") or {}
    return bool(isinstance(extra, dict) and extra.get("scrollable"))


def _scroll_status_for_prompt(node: dict[str, Any]) -> str:
    extra = node.get("extra") or {}
    if not isinstance(extra, dict) or not extra.get("scrollable"):
        return ""
    scroll_data = extra.get("scrollData") or {}
    if not isinstance(scroll_data, dict):
        return " scrollable"

    def rounded(name: str) -> int | None:
        try:
            return int(round(float(scroll_data.get(name, 0))))
        except (TypeError, ValueError):
            return None

    parts = ["scrollable"]
    top = rounded("top")
    bottom = rounded("bottom")
    left = rounded("left")
    right = rounded("right")
    if top is not None:
        parts.append(f"scroll-top={top}")
    if bottom is not None:
        parts.append(f"scroll-bottom={bottom}")
    if left is not None and right is not None and (left > 0 or right > 0):
        parts.append(f"scroll-left={left}")
        parts.append(f"scroll-right={right}")
    return " " + " ".join(parts)


def _range_thumb_for_prompt(tag: str, attrs: dict[str, Any], bbox: dict[str, Any] | None) -> str:
    if tag.lower() != "input":
        return ""
    if str(attrs.get("type", "")).lower() != "range":
        return ""
    if not bbox:
        return ""

    def num(name: str, fallback: float) -> float:
        try:
            return float(attrs.get(name, fallback))
        except (TypeError, ValueError):
            return fallback

    min_v = num("min", 0.0)
    max_v = num("max", 100.0)
    value = num("value", min_v)
    if max_v == min_v:
        return ""
    ratio = max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))
    x = int(round(float(bbox["x"]) + float(bbox["w"]) * ratio))
    y = int(round(float(bbox["y"]) + float(bbox["h"]) / 2))
    return f" thumb-pos={x},{y}"


def _collect_text_for_node(
    root_id: str,
    node_map: dict[str, Any],
    max_chars: int = 120,
    *,
    scroll_visible_only: bool = False,
) -> str:
    visited: set[str] = set()

    def walk(node_id: str, is_root: bool) -> str:
        node_id = str(node_id)
        if node_id in visited:
            return ""
        visited.add(node_id)

        node = node_map.get(node_id)
        if not isinstance(node, dict):
            return ""

        if node.get("type") == "TEXT_NODE":
            if not node.get("isVisible", True):
                return ""
            if scroll_visible_only and not node.get("isVisibleInScrollable", True):
                return ""
            return str(node.get("text", "")).strip()

        # Do not mix text from nested independently interactive elements.
        if not is_root and node.get("highlightIndex") is not None:
            return ""

        parts: list[str] = []
        for child_id in node.get("children", []):
            txt = walk(str(child_id), is_root=False)
            if txt:
                parts.append(txt)
        return " ".join(parts)

    text = walk(str(root_id), is_root=True)
    text = re.sub(r"\s+", " ", text).strip()
    return _short_text(text, max_chars)


async def _extract_dom_tree(page: Page) -> tuple[str, dict[str, Any]]:
    js_code = _load_dom_tree_js()
    args = {
        "doHighlightElements": False,
        "focusHighlightIndex": -1,
        "viewportExpansion": -1,
        "debugMode": False,
        # page-agent version requires these; default to empty lists
        "interactiveBlacklist": [],
        "interactiveWhitelist": [],
        "highlightOpacity": 0.1,
        "highlightLabelOpacity": 0.5,
    }
    raw = await page.evaluate(js_code, args)
    root_id = str(raw.get("rootId", ""))
    node_map = raw.get("map", {}) or {}
    return root_id, node_map


def _extract_interactive_elements(node_map: dict[str, Any]) -> list[dict[str, Any]]:
    interactive: list[dict[str, Any]] = []
    for node_id, node in node_map.items():
        if not isinstance(node, dict):
            continue
        idx = node.get("highlightIndex")
        if idx is None:
            continue
        attrs = node.get("attributes", {}) or {}
        scroll_visible_only = _is_scrollable_node(node)
        interactive.append(
            {
                "index": int(idx),
                "node_id": str(node_id),
                "tag": node.get("tagName", ""),
                "xpath": node.get("xpath", ""),
                "attrs": attrs,
                "text": _collect_text_for_node(
                    str(node_id),
                    node_map,
                    max_chars=99999,
                    scroll_visible_only=scroll_visible_only,
                ),
            }
        )
    interactive.sort(key=lambda x: x["index"])
    return interactive


def _annotate_new_elements(page: Page, interactive_elements: list[dict[str, Any]]) -> None:
    page_key = id(page)
    current_url = page.url

    entry = _PAGE_OBSERVATION_STATE.get(page_key, {"url": current_url, "signatures": set()})
    prev_url = entry.get("url", "")
    prev_signatures: set[str] = set(entry.get("signatures", set()))

    current_signatures: set[str] = set()
    for item in interactive_elements:
        sig = _signature_of_item(item)
        item["_sig"] = sig
        current_signatures.add(sig)

        # Keep page-agent-like behavior: first seen in this page context is "new".
        if prev_url != current_url:
            item["is_new"] = True
        else:
            item["is_new"] = sig not in prev_signatures

    _PAGE_OBSERVATION_STATE[page_key] = {
        "url": current_url,
        "signatures": current_signatures,
    }


def _flat_tree_to_llm_text(
    root_id: str,
    node_map: dict[str, Any],
    interactive_meta_by_node_id: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = []

    def has_highlight_ancestor(path: list[dict[str, Any]]) -> bool:
        for n in path:
            if isinstance(n, dict) and n.get("highlightIndex") is not None:
                return True
        return False

    def has_scrollable_ancestor(path: list[dict[str, Any]]) -> bool:
        return any(isinstance(n, dict) and _is_scrollable_node(n) for n in path)

    def walk(node_id: str, depth: int, ancestors: list[dict[str, Any]]) -> None:
        node = node_map.get(str(node_id))
        if not isinstance(node, dict):
            return

        ntype = node.get("type")

        if ntype == "TEXT_NODE":
            text = str(node.get("text", "") or "").strip()
            if not text:
                return
            if not node.get("isVisible", True):
                return
            if has_scrollable_ancestor(ancestors) and not node.get("isVisibleInScrollable", True):
                return
            if has_highlight_ancestor(ancestors):
                return
            lines.append(f"{'    ' * depth}{text}")
            return

        tag = node.get("tagName", "") or "element"
        attrs = node.get("attributes", {}) or {}
        idx = node.get("highlightIndex")
        is_scrollable = _is_scrollable_node(node)
        text = _collect_text_for_node(
            str(node_id),
            node_map,
            max_chars=99999,
            scroll_visible_only=is_scrollable,
        )

        next_depth = depth
        if idx is not None:
            meta = interactive_meta_by_node_id.get(str(node_id), {})
            is_new = bool(meta.get("is_new"))
            not_visible = not node.get("isVisible", True)
            indicator = f"*[{idx}]" if is_new else f"[{idx}]"
            attr_str = _attrs_for_prompt(attrs, text)
            nv_tag = " not-visible" if not_visible else ""
            bbox = node.get("boundingBox")
            pos_tag = f" pos={bbox['x']},{bbox['y']} size={bbox['w']}x{bbox['h']}" if bbox else ""
            scroll_tag = _scroll_status_for_prompt(node)
            range_thumb_tag = _range_thumb_for_prompt(tag, attrs, bbox)
            line = f"{'    ' * depth}{indicator}<{tag}{nv_tag}{pos_tag}{scroll_tag}"
            if range_thumb_tag:
                line += range_thumb_tag
            if attr_str:
                line += f" {attr_str}"
            if text:
                if not attr_str:
                    line += " "
                line += f">{text}"
            else:
                if not attr_str:
                    line += " "
            # For <select> elements, append the list of available option labels so the
            # agent knows which values are valid (option elements are invisible when the
            # dropdown is closed and their text would otherwise be absent from the output).
            if tag.lower() == "select":
                def _option_text(opt_node_id: str) -> str:
                    """Collect text from an <option> node ignoring visibility."""
                    opt = node_map.get(str(opt_node_id))
                    if not isinstance(opt, dict):
                        return ""
                    parts: list[str] = []
                    for gcid in opt.get("children", []) or []:
                        gc = node_map.get(str(gcid))
                        if isinstance(gc, dict) and gc.get("type") == "TEXT_NODE":
                            t = str(gc.get("text", "")).strip()
                            if t:
                                parts.append(t)
                    return " ".join(parts).strip()

                option_labels: list[str] = []
                for cid in node.get("children", []) or []:
                    child = node_map.get(str(cid))
                    if not isinstance(child, dict):
                        continue
                    if child.get("tagName", "").lower() != "option":
                        continue
                    label = _option_text(str(cid))
                    if label:
                        option_labels.append(label)
                if option_labels:
                    line += f" options=[{', '.join(option_labels)}]"
            if is_scrollable:
                line += " [offscreen scroll content omitted]"
            line += " />"
            lines.append(line)
            next_depth += 1

        children = node.get("children", []) or []
        for child_id in children:
            walk(str(child_id), next_depth, ancestors + [node])

    if root_id:
        walk(root_id, 0, [])

    if not lines:
        return ""

    return "\n".join(lines)



async def _validate_unique_selector(page: Page, selector: str) -> tuple[bool, int]:
    try:
        count = await page.locator(selector).count()
        if count != 1:
            return False, count
        el = await page.query_selector(selector)
        if el is None:
            return False, 0
        return True, 1
    except Exception:
        return False, 0
