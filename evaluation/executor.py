"""
Executor — Playwright-based action runner for ICG transitions.

All action parameters go in action["parameters"]. action["value"] is NOT used.

╔══════════════╦══════════════════════════════════════════════════════════════╗
║ action type  ║ parameters                                                 ║
╠══════════════╬══════════════════════════════════════════════════════════════╣
║ click        ║ text: str (optional, match button/tab by visible text)     ║
║              ║ click_position: "overlay_blank_area" — click outside the    ║
║              ║   modal/dialog content to dismiss an overlay backdrop       ║
║ double_click ║ (none)                                                     ║
║ right_click  ║ (none)                                                     ║
║ long_press   ║ duration_ms: int (default 800)                             ║
║ hover        ║ (none)                                                     ║
║ input        ║ value: str (text to type)                                  ║
║ input_date   ║ value: str (ISO date, YYYY-MM-DD, for input[type=date])    ║
║ clear        ║ (none)                                                     ║
║ blur         ║ (none)                                                     ║
║ select       ║ value: str (option label text)                             ║
║ check        ║ (none)                                                     ║
║ uncheck      ║ (none)                                                     ║
║ keypress     ║ key: str ("Enter"/"Escape"/"Control+z"/"ArrowDown"/...)    ║
║ scroll       ║ direction: "down"|"up", amount: int (px), target: str      ║
║              ║   OR position: "top"|"bottom" (scroll to absolute edge)    ║
║ drag         ║ target_selector: str (resolved from [M]) OR offset_x/offset_y: int ║
║ dragrange    ║ value: number (target value for a range slider)            ║
║ clickat      ║ x: int, y: int (absolute page coordinates)                 ║
║ upload       ║ files: list[str] (file paths)                              ║
║ wait         ║ duration_ms: int (milliseconds to wait)                    ║
║ refresh      ║ timeout_ms: int (default 30000), settle_ms: int (500)     ║
╠══════════════╬══════════════════════════════════════════════════════════════╣
║ ALL types    ║ then_wait_ms: int (post-action async settle, optional)     ║
║              ║ times: int (repeat count, optional, default 1)             ║
║              ║ settle_ms: int (extra settle, optional)                    ║
╚══════════════╩══════════════════════════════════════════════════════════════╝
"""

from pathlib import Path

import config
from playwright.async_api import Page


async def execute_action(page: Page, action: dict, affordance: dict, selector: str | None) -> tuple[bool, str]:
    params = action.get("parameters", {})
    times = max(1, int(params.get("times", action.get("times", 1))))

    for i in range(times):
        ok, err = await _execute_once(page, action, affordance, selector)
        if not ok:
            return (False, f"[repeat {i+1}/{times}] {err}") if times > 1 else (False, err)

    then_wait = params.get("then_wait_ms")
    if then_wait:
        await page.wait_for_timeout(int(then_wait))

    return True, ""


async def _execute_once(page, action, affordance, selector):
    atype = action["type"].lower()
    params = action.get("parameters", {})
    aff_type = affordance.get("type", "").lower()

    # ── No-selector actions ────────────────────────────────────────────
    if atype == "wait":
        await page.wait_for_timeout(int(params.get("duration_ms", 1000)))
        return True, ""

    if atype == "refresh":
        try:
            await page.reload(timeout=params.get("timeout_ms", 30000))
            await page.wait_for_load_state("networkidle", timeout=params.get("timeout_ms", 30000))
            await page.wait_for_timeout(params.get("settle_ms", 500))
            return True, ""
        except Exception as e:
            return False, f"Refresh failed: {e}"

    if atype == "goback":
        try:
            await page.evaluate("history.back()")
            await page.wait_for_timeout(params.get("settle_ms", 1000))
            if page.url == "about:blank":
                return False, "GoBack failed: no browser history available. Use Reset to restore the page, then try a different approach (e.g. click a navigation link)."
            return True, ""
        except Exception as e:
            return False, f"GoBack failed: {e}"

    if atype == "scroll":
        return await _do_scroll(page, params, selector)

    if atype == "keypress":
        return await _do_keypress(page, params, selector)

    if atype == "clickat":
        return await _do_click_at(page, params)

    # ── Selector-optional: click on overlay blank area ─────────────────
    # No element locator required. Uses JS to detect the modal dialog and
    # clicks a viewport corner outside it.
    if atype == "click" and params.get("click_position") == "overlay_blank_area":
        return await _click_overlay_blank_area(page)

    # ── Selector-required actions ──────────────────────────────────────
    if selector is None:
        desc = affordance.get("description", affordance.get("id", "?"))
        return False, f"Affordance '{desc}' not located on page."

    HANDLERS = {
        "click":        lambda: _do_click(page, selector, params, aff_type),
        "double_click": lambda: _do_double_click(page, selector),
        "right_click":  lambda: _do_right_click(page, selector),
        "long_press":   lambda: _do_long_press(page, selector, params),
        "hover":        lambda: _do_hover(page, selector),
        "input":        lambda: _do_input(page, selector, params),
        "type":         lambda: _do_input(page, selector, params),
        "input_date":   lambda: _do_input_date(page, selector, params),
        "clear":        lambda: _do_clear(page, selector),
        "blur":         lambda: _do_blur(page, selector),
        "select":       lambda: _do_select(page, selector, params),
        "select_text":  lambda: _do_select_text(page, selector, params),
        "check":        lambda: _do_check(page, selector, True),
        "uncheck":      lambda: _do_check(page, selector, False),
        "drag":         lambda: _do_drag(page, selector, params),
        "dragrange":    lambda: _do_drag_range(page, selector, params),
        "upload":       lambda: _do_upload(page, selector, params),
    }

    handler = HANDLERS.get(atype)
    if not handler:
        return False, f"Unknown action type: '{atype}'"

    try:
        ok, err = await handler()
        settle = params.get("settle_ms")
        if ok and settle:
            await page.wait_for_timeout(int(settle))
        return ok, err
    except Exception as exc:
        return False, str(exc)


# ── click ──────────────────────────────────────────────────────────────────────

_CONFIRM_KEYWORDS = {
    # affirmative / proceed
    "confirm", "yes", "ok", "okay", "sure", "proceed", "continue",
    "accept", "agree", "approve", "allow", "got it", "understood",
    # destructive
    "delete", "remove", "discard", "submit", "save",
}

async def _auto_confirm_dialog(page) -> bool:
    """After a click, silently dismiss any confirmation dialog that appeared.

    Looks for a visible dialog/modal and clicks the first button whose text
    matches a confirmation keyword. Returns True if a dialog was dismissed.
    This handles cases where the implementation added an unexpected confirmation
    step for destructive actions — the ICG does not model this dialog, so we
    transparently close it before the post-screenshot is taken.
    """
    await page.wait_for_timeout(300)
    try:
        for container in [
            'dialog[open]',
            '[role="dialog"]', '[role="alertdialog"]',
            '.modal-overlay', '.modal', '.dialog', '.popup', '.confirm',
            '.overlay', '.alert-box', '.confirm-box', '.confirm-dialog',
        ]:
            el = await page.query_selector(container)
            if not el or not await el.is_visible():
                continue
            buttons = await el.query_selector_all("button, [role='button']")
            for btn in buttons:
                try:
                    text = (await btn.inner_text()).strip().lower()
                    if any(k in text for k in _CONFIRM_KEYWORDS):
                        await btn.click()
                        await page.wait_for_timeout(300)
                        print(f"   ◇  DOM modal confirm clicked (keyword-match): {text!r} in {container}")
                        return True
                except Exception:
                    continue
        # Fallback: search visible confirm-keyword buttons anywhere on the page
        buttons = await page.query_selector_all("button, [role='button']")
        for btn in buttons:
            try:
                if not await btn.is_visible():
                    continue
                text = (await btn.inner_text()).strip().lower()
                if any(k in text for k in _CONFIRM_KEYWORDS):
                    await btn.click()
                    await page.wait_for_timeout(300)
                    print(f"   ◇  DOM page-wide confirm clicked (keyword-match): {text!r}")
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


async def _do_click(page, selector, params, aff_type):
    auto_confirm = params.get("auto_confirm", False)

    # Use the validated selector from the locator first
    el = await page.query_selector(selector)
    if el:
        await el.click()
        if auto_confirm:
            await _auto_confirm_dialog(page)        # Tier 1: scan DOM modals
        await page.wait_for_timeout(config.ACTION_SETTLE_MS)
        return True, ""

    # Fallback: text-based matching for button/tab when validated selector fails
    text = params.get("text", "")
    if text and aff_type in ("tab", "button"):
        for sel in [
            f"button:has-text('{text}')", f"[role='tab']:has-text('{text}')",
            f"a:has-text('{text}')", f"li:has-text('{text}')",
        ]:
            try:
                el = await page.query_selector(sel)
                if el:
                    await el.click()
                    if auto_confirm:
                        await _auto_confirm_dialog(page)
                    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
                    return True, ""
            except Exception:
                continue

    return False, f"'{selector}' not found for click"


async def _click_overlay_blank_area(page):
    """Dismiss a modal by clicking outside its content area in the viewport.

    No element locator is required. Strategy:
    - Detect the modal dialog via JS using common selectors (best-effort safety).
    - Click a viewport corner (inset 10px) that is outside the detected dialog.
    - If no dialog is detected, click the top-left corner.
    """
    # Detect dialog bounding box via common selectors (best-effort; safety net).
    dialog_box = await page.evaluate("""() => {
        const selectors = [
            '[role="dialog"]', '.modal-content', '.modal-dialog',
            '.dialog', '.popup', '.lightbox-content',
        ];
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el) {
                const r = el.getBoundingClientRect();
                if (r.width > 0 && r.height > 0) {
                    return {x: r.left, y: r.top, width: r.width, height: r.height};
                }
            }
        }
        return null;
    }""")

    vw = await page.evaluate("window.innerWidth")
    vh = await page.evaluate("window.innerHeight")

    def _outside_dialog(px, py):
        if not dialog_box:
            return True
        dx, dy = dialog_box["x"], dialog_box["y"]
        dw, dh = dialog_box["width"], dialog_box["height"]
        return not (dx <= px <= dx + dw and dy <= py <= dy + dh)

    # Try four viewport corners inset by 10px
    candidates = [
        (10,      10),
        (10,      vh - 10),
        (vw - 10, 10),
        (vw - 10, vh - 10),
    ]
    for px, py in candidates:
        if _outside_dialog(px, py):
            # Stash the chosen coord on the page so callers (eval_gt writer)
            # can read it for operation_record. Attribute is harmless for
            # normal eval — nothing reads it there.
            try: page._last_overlay_click_coord = [px, py]
            except Exception: pass
            await page.mouse.click(px, py)
            await page.wait_for_timeout(config.ACTION_SETTLE_MS)
            return True, ""

    # Last resort: absolute top-left
    try: page._last_overlay_click_coord = [10, 10]
    except Exception: pass
    await page.mouse.click(10, 10)
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


async def _do_double_click(page, selector):
    el = await page.query_selector(selector)
    if not el:
        return False, f"'{selector}' not found for double_click"
    await el.dblclick()
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


async def _do_right_click(page, selector):
    el = await page.query_selector(selector)
    if not el:
        return False, f"'{selector}' not found for right_click"
    await el.click(button="right")
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


async def _do_long_press(page, selector, params):
    el = await page.query_selector(selector)
    if not el:
        return False, f"'{selector}' not found for long_press"
    box = await el.bounding_box()
    if not box:
        return False, "No bounding box for long_press"
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2
    await page.mouse.move(cx, cy)
    await page.mouse.down()
    await page.wait_for_timeout(int(params.get("duration_ms", 800)))
    await page.mouse.up()
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


# ── hover ──────────────────────────────────────────────────────────────────────

async def _do_hover(page, selector):
    el = await page.query_selector(selector)
    if not el:
        return False, f"'{selector}' not found for hover"
    await el.hover()
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


# ── input / clear / blur ──────────────────────────────────────────────────────

async def _do_input(page, selector, params):
    el = await page.query_selector(selector)
    if not el:
        return False, f"Input '{selector}' not found"
    value = params.get("value", "")
    if value:
        is_date_input = await el.evaluate(
            "node => node instanceof HTMLInputElement && (node.type || '').toLowerCase() === 'date'"
        )
        if is_date_input:
            return await _do_input_date(page, selector, params)
        is_number_input = await el.evaluate(
            "node => node instanceof HTMLInputElement && (node.type || '').toLowerCase() === 'number'"
        )
        if is_number_input:
            return await _do_input_number(page, selector, params)

    # If there is already an active non-empty selection (e.g. left by a prior
    # SelectText action), preserve it so that keyboard.type() replaces it.
    # Calling el.focus() or setSelectionRange() would collapse the selection
    # and cause the typed text to be appended rather than replacing the
    # selected content.
    # NOTE: textarea/input selections are tracked via selectionStart/End, not
    # window.getSelection(), so both must be checked.
    has_selection = await page.evaluate("""() => {
        const s = window.getSelection();
        if (s && s.toString().length > 0) return true;
        const a = document.activeElement;
        return !!(a && (a.tagName === 'TEXTAREA' || a.tagName === 'INPUT')
                  && a.selectionStart !== a.selectionEnd);
    }""")
    if not has_selection:
        # No prior selection — focus only; leave the caret wherever the previous
        # action left it so that agents can position the cursor with arrow keys /
        # Enter before calling Input to insert at a specific location.
        await el.focus()
    if value:
        # For contenteditable elements, keyboard.type() opens an IME composition
        # context; when it hits a non-BMP character (emoji) it calls CDP
        # insertText internally, which commits/discards the in-progress
        # composition and loses previously typed text.  Use
        # page.keyboard.insertText() instead — it sends the whole string as one
        # atomic CDP insertText command, fires beforeinput+input events (so
        # real-time character counters still work), and handles emoji correctly.
        # For textarea/input, keep keyboard.type() so keydown/keypress events
        # are dispatched (needed for keystroke-based validators).
        is_contenteditable = await page.evaluate("""() => {
            // Check activeElement first.
            const a = document.activeElement;
            if (a && a.isContentEditable) return true;
            // el.focus() on a non-focusable child (e.g. a span inside a
            // contenteditable) moves focus to BODY while preserving the
            // selection range — so also check the selection container.
            const sel = window.getSelection();
            if (sel && sel.rangeCount > 0) {
                let node = sel.getRangeAt(0).startContainer;
                while (node) {
                    if (node.nodeType === 1 && node.isContentEditable) return true;
                    node = node.parentElement;
                }
            }
            return false;
        }""")
        if is_contenteditable:
            # Chromium's execCommand('insertText') has a boundary bug: when the
            # cursor or selection starts at offset 0 of an inline element's first
            # text node (e.g. a styled <span>), it inserts the text BEFORE the
            # span in the parent rather than inside it. This applies to both
            # non-collapsed selections and collapsed cursors positioned at the
            # very start of an inline element.
            # Use the Range API for all contenteditable inserts: deleteContents()
            # removes the selection (no-op for collapsed cursors), and insertNode()
            # places the new text node at the exact Range position — always inside
            # the span, regardless of offset. Fall back to execCommand only when
            # there is no Range (no cursor in the document).
            await page.evaluate("""(v) => {
                var sel = window.getSelection();
                if (sel && sel.rangeCount > 0) {
                    var range = sel.getRangeAt(0);
                    range.deleteContents();
                    var textNode = document.createTextNode(v);
                    range.insertNode(textNode);
                    range.setStartAfter(textNode);
                    range.collapse(true);
                    sel.removeAllRanges();
                    sel.addRange(range);
                    var ce = textNode.parentNode;
                    while (ce && !ce.isContentEditable) ce = ce.parentElement;
                    if (ce) ce.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertText', data: v}));
                } else {
                    document.execCommand('insertText', false, v);
                }
            }""", value)
        else:
            await page.keyboard.type(value, delay=10)
    elif has_selection:
        # Empty Input with an active selection = delete the selection.
        await page.keyboard.press("Delete")
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


async def _do_input_date(page, selector, params):
    el = await page.query_selector(selector)
    if not el:
        return False, f"InputDate '{selector}' not found"
    value = str(params.get("value", "")).strip()
    if not value:
        return False, "InputDate requires a value in YYYY-MM-DD format"

    ok = await el.evaluate("""(node, value) => {
        if (!(node instanceof HTMLInputElement)) {
            return { ok: false, reason: 'target is not an input element' };
        }
        if ((node.type || '').toLowerCase() !== 'date') {
            return { ok: false, reason: `target input type is ${node.type || 'unknown'}, not date` };
        }
        if (!/^\\d{4}-\\d{2}-\\d{2}$/.test(value)) {
            return { ok: false, reason: 'date value must use YYYY-MM-DD format' };
        }
        node.focus();
        node.value = value;
        node.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertReplacementText', data: value }));
        node.dispatchEvent(new Event('change', { bubbles: true }));
        node.blur();
        return { ok: node.value === value, value: node.value };
    }""", value)
    if not ok.get("ok"):
        reason = ok.get("reason") or f"browser rejected date value; current value={ok.get('value')!r}"
        return False, f"InputDate failed: {reason}"
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


async def _do_input_number(page, selector, params):
    el = await page.query_selector(selector)
    if not el:
        return False, f"Input number '{selector}' not found"
    value = str(params.get("value", "")).strip()
    ok = await el.evaluate("""(node, value) => {
        if (!(node instanceof HTMLInputElement)) {
            return { ok: false, reason: 'target is not an input element' };
        }
        if ((node.type || '').toLowerCase() !== 'number') {
            return { ok: false, reason: `target input type is ${node.type || 'unknown'}, not number` };
        }
        node.focus();
        node.value = value;
        node.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertReplacementText', data: value }));
        node.dispatchEvent(new Event('change', { bubbles: true }));
        return { ok: node.value === value, value: node.value };
    }""", value)
    if not ok.get("ok"):
        reason = ok.get("reason") or f"browser rejected number value; current value={ok.get('value')!r}"
        return False, f"Input number failed: {reason}"
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


async def _do_clear(page, selector):
    el = await page.query_selector(selector)
    if not el:
        return False, f"'{selector}' not found for clear"
    await el.fill("")
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


async def _do_blur(page, selector):
    el = await page.query_selector(selector)
    if not el:
        return False, f"'{selector}' not found for blur"
    await el.evaluate("el => el.blur()")
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


# ── select ─────────────────────────────────────────────────────────────────────

async def _do_select(page, selector, params):
    el = await page.query_selector(selector)
    if not el:
        return False, f"Select '{selector}' not found"
    value = params.get("value", "")
    slug1 = value.lower().replace(": ", "-").replace(" ", "-")
    slug2 = value.lower().replace(":", "").replace(" ", "-").replace("--", "-")
    for strategy in [
        lambda: el.select_option(label=value, timeout=3000),
        lambda: el.select_option(label=value.lower(), timeout=3000),
        lambda: el.select_option(value=value, timeout=3000),
        lambda: el.select_option(value=slug1, timeout=3000),
        lambda: el.select_option(value=slug2, timeout=3000),
    ]:
        try:
            await strategy()
            await page.wait_for_timeout(config.ACTION_SETTLE_MS)
            return True, ""
        except Exception:
            continue
    return False, f"Could not select '{value}' in '{selector}'"


# ── select_text ───────────────────────────────────────────────────────────────

async def _do_select_text(page, selector, params):
    el = await page.query_selector(selector)
    if not el:
        return False, f"SelectText target '{selector}' not found"
    text = params.get("text", "")
    if not text:
        return False, "SelectText: no text specified"
    found = await el.evaluate(r"""(el, searchText) => {
        el.focus();
        // For textarea/input, use value + setSelectionRange
        if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
            const val = el.value || '';
            let idx = val.indexOf(searchText);
            let matchLen = searchText.length;
            // Exact match failed and searchText crosses a newline — retry with
            // flexible indentation: treat each \n in searchText as \n followed
            // by optional spaces/tabs (auto-indent may have been inserted).
            if (idx === -1 && searchText.includes('\n')) {
                const pattern = searchText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
                                          .replace(/\n/g, '\\n[ \\t]*');
                const m = val.match(new RegExp(pattern));
                if (m) { idx = m.index; matchLen = m[0].length; }
            }
            if (idx === -1) return false;
            el.setSelectionRange(idx, idx + matchLen);
            document.dispatchEvent(new Event('selectionchange'));
            return true;
        }
        // For contenteditable / other elements, use TreeWalker + Range API.
        // A previous comment/highlight can split one visible sentence across
        // multiple text nodes, so match against the concatenated text and map
        // the result back to start/end nodes.
        const textNodes = [];
        const walker = document.createTreeWalker(
            el,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode(node) {
                    if (!node.textContent) return NodeFilter.FILTER_REJECT;
                    const parent = node.parentElement;
                    if (!parent) return NodeFilter.FILTER_REJECT;
                    const style = window.getComputedStyle(parent);
                    if (style.display === 'none' || style.visibility === 'hidden') {
                        return NodeFilter.FILTER_REJECT;
                    }
                    return NodeFilter.FILTER_ACCEPT;
                }
            }
        );
        let node;
        let fullText = '';
        while (node = walker.nextNode()) {
            const start = fullText.length;
            const text = node.textContent || '';
            fullText += text;
            textNodes.push({ node, start, end: fullText.length });
        }

        let idx = fullText.indexOf(searchText);
        if (idx === -1) {
            // Retry NBSP-insensitive matching without changing offsets.
            idx = fullText.replace(/\u00a0/g, ' ').indexOf(searchText.replace(/\u00a0/g, ' '));
        }
        if (idx === -1) return false;

        const endIdx = idx + searchText.length;
        // Boundary handling matters: if a match begins exactly at the start of
        // a text node, it is also equal to the previous node's end offset in
        // the concatenated string. Choose the following node for range starts,
        // otherwise the Range can begin between block elements and wrapping the
        // selection may accidentally wrap an entire <p>.
        const startInfo = textNodes.find(item => idx >= item.start && idx < item.end);
        const endInfo = textNodes.find(item => endIdx > item.start && endIdx <= item.end);
        if (!startInfo || !endInfo) return false;

        const range = document.createRange();
        range.setStart(startInfo.node, idx - startInfo.start);
        range.setEnd(endInfo.node, endIdx - endInfo.start);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        const rect = range.getBoundingClientRect();
        const evtInit = {
            bubbles: true, cancelable: true,
            clientX: rect.left + rect.width / 2,
            clientY: rect.top + rect.height / 2
        };
        el.dispatchEvent(new MouseEvent('mousedown', evtInit));
        el.dispatchEvent(new MouseEvent('mouseup', evtInit));
        document.dispatchEvent(new Event('selectionchange'));
        return true;
    }""", text)
    if not found:
        return False, f"Text '{text}' not found in '{selector}'"
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


# ── check / uncheck ───────────────────────────────────────────────────────────

async def _do_check(page, selector, checked):
    el = await page.query_selector(selector)
    if not el:
        return False, f"Checkbox '{selector}' not found"
    try:
        await (el.check() if checked else el.uncheck())
    except Exception:
        # Fallback for custom checkbox components (non-native or missing role="checkbox"):
        # read current checked state and click only if a toggle is needed.
        is_checked = await el.evaluate(
            "el => el.checked !== undefined ? el.checked : el.getAttribute('aria-checked') === 'true'"
        )
        if bool(is_checked) != checked:
            await el.click()
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


# ── keypress ───────────────────────────────────────────────────────────────────

async def _do_keypress(page, params, selector):
    key = params.get("key", "Enter")
    try:
        # contenteditable divs don't reliably handle Control+End/Home via
        # keyboard events — Playwright's el.press() refocuses the element
        # which resets the cursor, and the key combo may only jump within
        # the current block node. Use the Selection API directly instead.
        if selector and key in ("Control+End", "Control+Home"):
            el = await page.query_selector(selector)
            if el and await el.evaluate("el => el.isContentEditable"):
                to_end = key == "Control+End"
                await el.evaluate("""(el, toEnd) => {
                    el.focus();
                    const sel = window.getSelection();
                    sel.selectAllChildren(el);
                    toEnd ? sel.collapseToEnd() : sel.collapseToStart();
                }""", to_end)
                await page.wait_for_timeout(config.ACTION_SETTLE_MS)
                return True, ""
        # If there is an active selection, skip el.press() — Playwright's
        # el.press() refocuses the element which collapses window.getSelection()
        # ranges set by SelectText. Use page.keyboard.press() to preserve the
        # selection so that Backspace/Delete operate on the selected range.
        has_sel = await page.evaluate("""() => {
            const s = window.getSelection();
            if (s && s.toString().length > 0) return true;
            const a = document.activeElement;
            return !!(a && (a.tagName === 'TEXTAREA' || a.tagName === 'INPUT')
                      && a.selectionStart !== a.selectionEnd);
        }""")
        # Press exactly once — repetition is handled by execute_action's outer loop.
        if selector and not has_sel:
            el = await page.query_selector(selector)
            if el:
                await el.press(key)
            else:
                await page.keyboard.press(key)
        else:
            await page.keyboard.press(key)
        await page.wait_for_timeout(config.ACTION_SETTLE_MS)
        return True, ""
    except Exception as e:
        return False, f"Keypress '{key}': {e}"


# ── scroll ─────────────────────────────────────────────────────────────────────

async def _do_scroll(page, params, selector):
    position = params.get("position")  # "top" | "bottom" | "left" | "right" | None
    direction = params.get("direction", "down")  # "down" | "up" | "left" | "right"
    amount = int(params.get("amount", 600))
    try:
        if position in ("top", "bottom", "left", "right"):
            if selector:
                el = await page.query_selector(selector)
                if not el:
                    return False, f"Scroll target '{selector}' not found"
                if position == "top":
                    await el.evaluate("el => el.scrollTop = 0")
                elif position == "bottom":
                    await el.evaluate("el => el.scrollTop = el.scrollHeight")
                elif position == "left":
                    await el.evaluate("el => el.scrollLeft = 0")
                else:  # right
                    await el.evaluate("el => el.scrollLeft = el.scrollWidth")
            else:
                if position == "top":
                    await page.evaluate("window.scrollTo(0, 0)")
                elif position == "bottom":
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                elif position == "left":
                    await page.evaluate("window.scrollTo(0, window.scrollY)")
                else:  # right
                    await page.evaluate("window.scrollTo(document.body.scrollWidth, window.scrollY)")
        else:
            horizontal = direction in ("left", "right")
            delta = amount if direction in ("down", "right") else -amount
            if selector:
                el = await page.query_selector(selector)
                if not el:
                    return False, f"Scroll target '{selector}' not found"
                if horizontal:
                    await el.evaluate(f"el => el.scrollBy({delta}, 0)")
                else:
                    await el.evaluate(f"el => el.scrollBy(0, {delta})")
            else:
                if horizontal:
                    await page.evaluate(f"window.scrollBy({delta}, 0)")
                else:
                    await page.evaluate(f"window.scrollBy(0, {delta})")
        await page.wait_for_timeout(config.ACTION_SETTLE_MS)
        return True, ""
    except Exception as e:
        return False, str(e)


# ── clickat ────────────────────────────────────────────────────────────────────

async def _do_click_at(page, params):
    x = int(params.get("x", 0))
    y = int(params.get("y", 0))
    await page.mouse.click(x, y)
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


# ── drag ───────────────────────────────────────────────────────────────────────

async def _do_drag(page, selector, params):
    src = await page.query_selector(selector)
    if not src:
        return False, f"Drag source '{selector}' not found"
    sb = await src.bounding_box()
    if not sb:
        return False, "No bounding box for drag source"
    sx = sb["x"] + sb["width"] / 2
    sy = sb["y"] + sb["height"] / 2

    ts = params.get("target_selector")
    if ts:
        tgt = await page.query_selector(ts)
        if not tgt:
            return False, f"Drag target '{ts}' not found"
        tb = await tgt.bounding_box()
        if not tb:
            return False, "No bounding box for drag target"
        tx = tb["x"] + tb["width"] / 2
        ty = tb["y"] + tb["height"] / 2
    else:
        tx = sx + int(params.get("offset_x", 100))
        ty = sy + int(params.get("offset_y", 0))

    await page.mouse.move(sx, sy)
    await page.mouse.down()
    await page.mouse.move(tx, ty, steps=10)
    await page.mouse.up()
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


# ── dragrange ─────────────────────────────────────────────────────────────────

async def _do_drag_range(page, selector, params):
    el = await page.query_selector(selector)
    if not el:
        return False, f"DragRange target '{selector}' not found"
    box = await el.bounding_box()
    if not box:
        return False, "No bounding box for DragRange target"

    try:
        info = await el.evaluate(
            """node => {
                if (!(node instanceof HTMLInputElement) || (node.type || '').toLowerCase() !== 'range') {
                    return { ok: false, reason: 'target is not a range slider' };
                }
                const num = (value, fallback) => {
                    const n = Number(value);
                    return Number.isFinite(n) ? n : fallback;
                };
                const min = num(node.min, 0);
                const max = num(node.max, 100);
                const value = num(node.value, min);
                return { ok: true, min, max, value };
            }"""
        )
    except Exception as e:
        return False, f"DragRange target inspection failed: {e}"

    if not info or not info.get("ok"):
        return False, info.get("reason", "target is not a range slider") if isinstance(info, dict) else "target is not a range slider"

    try:
        target = float(params.get("value"))
    except (TypeError, ValueError):
        raw = params.get("value_raw", params.get("value"))
        return False, f"DragRange requires a numeric target value; got {raw!r}"

    min_v = float(info["min"])
    max_v = float(info["max"])
    cur_v = float(info["value"])
    if max_v == min_v:
        return False, "Range slider has identical min and max"
    target = max(min_v, min(max_v, target))

    def x_for(value):
        ratio = (value - min_v) / (max_v - min_v)
        # Leave a tiny inset so the mouse lands inside the control even when
        # the target is exactly min/max.
        inset = min(8, box["width"] / 2)
        usable = max(1, box["width"] - inset * 2)
        return box["x"] + inset + ratio * usable

    y = box["y"] + box["height"] / 2
    sx = x_for(cur_v)
    tx = x_for(target)

    await page.mouse.move(sx, y)
    await page.mouse.down()
    await page.mouse.move(tx, y, steps=12)
    await page.mouse.up()
    await page.wait_for_timeout(config.ACTION_SETTLE_MS)
    return True, ""


# ── upload ─────────────────────────────────────────────────────────────────────

async def _do_upload(page, selector, params):
    el = await page.query_selector(selector)
    if not el:
        return False, f"Upload target '{selector}' not found"

    files = params.get("files", [])
    if isinstance(files, str):
        files = [files]
    if not files:
        return False, "No file path in parameters.files"
    files = [_resolve_upload_file_path(f) for f in files]

    try:
        is_file_input = await el.evaluate(
            """node => (
                node instanceof HTMLInputElement &&
                (node.type || '').toLowerCase() === 'file'
            )"""
        )
    except Exception as e:
        return False, f"Upload target inspection failed: {e}"

    try:
        if is_file_input:
            await el.set_input_files(files)
        else:
            # Support upload affordances that point to a visible trigger button.
            async with page.expect_file_chooser(timeout=3000) as fc_info:
                await el.click()
            file_chooser = await fc_info.value
            await file_chooser.set_files(files)

        await page.wait_for_timeout(config.ACTION_SETTLE_MS)
        return True, ""
    except Exception as e:
        return False, f"Upload failed: {e}"


def _resolve_upload_file_path(path: str) -> str:
    raw = str(path)
    p = Path(raw).expanduser()
    if p.exists() or p.is_absolute():
        return str(p)

    eval_relative = Path(__file__).resolve().parent / p
    if eval_relative.exists():
        return str(eval_relative)

    return raw
