#!/usr/bin/env python3
"""
build_video_inputs.py — Build Video-modality inputs from agent eval operation records.

For each task:
  1. Find the best transition chain from S0 in icg.json, ranked by explicit
     requirement coverage first, total requirement coverage second, and chain
     length third
  2. Locate the most recent agent_eval run that has trajectory files for
     every transition in the chain
  3. Launch Playwright with video recording against the same HTML used by the
     source agent_eval run
  4. Replay the stored action sequences (selector + action_dict) for each
     transition in chain order — no LLM calls, no re-running the agent
  5. Save Video/<task_id>_chain.webm or .mp4
  6. Save Video/coverage.json  (chain + covered explicit requirement IDs)
  7. Save Video/input.json     (video-only model prompt paired with the recorded video)

Usage:
  python build_video_inputs.py                       # all tasks under the data root
  python build_video_inputs.py D04_S23_T298          # specific task(s)
  python build_video_inputs.py --input-only          # only write Video/input.json
  python build_video_inputs.py --dry-run             # print chain without recording
  python build_video_inputs.py --force               # overwrite existing video
  python build_video_inputs.py --settle-ms 800       # wait between transitions (ms)
  python build_video_inputs.py --extract-fps 1       # also extract frames from videos
  python build_video_inputs.py --html-source gt      # replay against the configured GT HTML root
  python build_video_inputs.py --html-source seed    # replay against HTML inside each seed dir
  python build_video_inputs.py --passed-only         # only use transitions that passed in the source eval run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parents[2]
SEED_ROOT    = Path(os.environ.get("SEED_ROOT", RELEASE_ROOT / "data_release"))
_DEFAULT_GT_HTML_ROOT = SEED_ROOT
GT_HTML_ROOT = Path(os.environ.get("GT_HTML_ROOT", _DEFAULT_GT_HTML_ROOT))
REQ_JSON     = Path(os.environ.get("REQ_JSON", RELEASE_ROOT / "data_release" / "requirements_full.json"))

# Reuse executor from code_release/evaluation/
EVAL_DIR = SCRIPT_DIR.parents[1] / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

import config as cfg
from executor import execute_action

VIDEO_SYSTEM_PROMPT = (
    "You are an expert in writing front-end HTML web code. Your generated web "
    "page must faithfully replicate the interactive behavior and layout shown "
    "in the input video."
)

PRE_ACTION_PAUSE_MS = 250
POST_ACTION_PAUSE_MS = 200
DEFAULT_WAIT_MS = 600
CURSOR_MOVE_STEPS = 15
CURSOR_SETTLE_MS = 100
SELECT_DROPDOWN_DWELL_MS = 700
TYPING_DELAY_MS = 55
UPLOAD_PICKER_DWELL_MS = 500
CONFIRM_KEYWORDS = (
    "confirm", "yes", "ok", "okay", "sure", "proceed", "continue",
    "accept", "agree", "approve", "allow", "got it", "understood",
    "delete", "remove", "discard", "submit", "save",
)

CURSOR_INIT_SCRIPT = r"""
(() => {
  if (window.__wi_overlay_installed) return;
  window.__wi_overlay_installed = true;

  // Use a non-div element so recorded absolute XPath selectors like
  // html/body/div[2]/... keep pointing at the same app/modal nodes.
  const cursor = document.createElement('span');
  cursor.id = '__wi_cursor';
  cursor.style.cssText = `
    position: fixed; left: -100px; top: -100px;
    width: 20px; height: 20px; border-radius: 50%;
    background: rgba(59,130,246,0.55);
    border: 2px solid rgba(37,99,235,0.95);
    pointer-events: none; z-index: 2147483647;
    transform: translate(-50%, -50%);
    box-shadow: 0 0 12px rgba(37,99,235,0.45);
    transition: transform 60ms linear, background 120ms;
  `;
  const attach = () => {
    const parent = document.body || document.documentElement;
    if (parent && !document.getElementById('__wi_cursor')) parent.appendChild(cursor);
  };
  document.addEventListener('DOMContentLoaded', attach);
  attach();

  document.addEventListener('mousemove', (e) => {
    cursor.style.left = e.clientX + 'px';
    cursor.style.top  = e.clientY + 'px';
  }, true);

  const flashClick = () => {
    cursor.style.background = 'rgba(239,68,68,0.75)';
    cursor.style.transform  = 'translate(-50%, -50%) scale(0.7)';
    setTimeout(() => {
      cursor.style.background = 'rgba(59,130,246,0.55)';
      cursor.style.transform  = 'translate(-50%, -50%) scale(1)';
    }, 220);
  };
  document.addEventListener('mousedown', flashClick, true);

  window.__wi_show_select_dropdown_for_el = (sel, highlightLabel) => {
    if (!sel || sel.tagName !== 'SELECT') return false;
    const prev = document.getElementById('__wi_select_panel');
    if (prev) prev.remove();

    const rect = sel.getBoundingClientRect();
    const opts = Array.from(sel.options);
    const panel = document.createElement('div');
    panel.id = '__wi_select_panel';
    panel.style.cssText = `
      position: fixed;
      left: ${rect.left}px;
      top: ${rect.bottom + 2}px;
      min-width: ${rect.width}px;
      background: #fff;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.18);
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, sans-serif;
      color: #0f172a;
      padding: 4px 0;
      max-height: 280px;
      overflow: auto;
      z-index: 2147483646;
    `;

    const styleEl = document.createElement('style');
    styleEl.textContent = `
      #__wi_select_panel .__wi_opt { padding: 6px 12px; white-space: nowrap; cursor: pointer; pointer-events: auto; }
      #__wi_select_panel .__wi_opt:hover {
        background: rgba(37,99,235,0.12); color: #1d4ed8; font-weight: 600;
      }
    `;
    panel.appendChild(styleEl);

    opts.forEach(opt => {
      const item = document.createElement('div');
      item.className = '__wi_opt';
      item.textContent = opt.text;
      if (opt.text === highlightLabel || opt.value === highlightLabel) {
        item.id = '__wi_active_opt';
        item.addEventListener('click', () => {
          sel.value = opt.value;
          sel.dispatchEvent(new Event('input',  { bubbles: true }));
          sel.dispatchEvent(new Event('change', { bubbles: true }));
          item.style.background = 'rgba(37,99,235,0.28)';
          item.style.color = '#1d4ed8';
          setTimeout(() => panel.remove(), 160);
        });
      }
      panel.appendChild(item);
    });

    document.body.appendChild(panel);
    return !!document.getElementById('__wi_active_opt');
  };

  window.__wi_hide_select_dropdown = () => {
    const p = document.getElementById('__wi_select_panel');
    if (p) p.remove();
  };
})();
"""

# ── Chain selection ───────────────────────────────────────────────────────────

def best_coverage_chain(transitions: list[dict], start: str = "S0", icg: dict | None = None) -> list[dict]:
    """Return the best transition path starting from `start`.

    DFS may revisit states, including self-loop transitions such as S1 -> S1,
    but each transition id can appear at most once in a path. Paths are ranked
    by covered explicit requirements first, then by total covered requirements,
    then by chain length.
    """
    adj: dict[str, list[dict]] = defaultdict(list)
    for t in transitions:
        adj[t["from"]].append(t)

    best: list[dict] = []
    best_score = (-1, -1, -1)

    def score(path: list[dict]) -> tuple[int, int, int]:
        if not icg:
            return (0, 0, len(path))
        return (
            len(covered_req_ids(icg, path, intent="explicit")),
            len(covered_req_ids(icg, path, intent=None)),
            len(path),
        )

    def dfs(state: str, path: list[dict], used: set[str]) -> None:
        nonlocal best, best_score
        cur_score = score(path)
        if cur_score > best_score:
            best = path[:]
            best_score = cur_score
        for t in adj[state]:
            tid = t.get("id") or t.get("transition_id")
            if tid in used:
                continue
            used.add(tid)
            path.append(t)
            dfs(t["to"], path, used)
            path.pop()
            used.remove(tid)

    dfs(start, [], set())
    return best


# ── Eval-run finder ──────────────────────────────────────────────────────────

def find_best_run(task_dir: Path, chain_ids: list[str]) -> Path | None:
    """
    Return the most recent agent_eval run directory that contains a clean
    operation_record.json for every transition in chain_ids, falling back to
    raw agent_logs/<tid>_agent_trajectory.json when no operation record exists.
    """
    eval_root = task_dir / "agent_eval"
    if not eval_root.exists():
        return None

    runs = sorted(eval_root.iterdir(), reverse=True)  # newest first (timestamp sort)
    for run in runs:
        if not run.is_dir():
            continue
        record = load_operation_record(run)
        if record:
            transitions = record.get("transitions", {})
            if all(tid in transitions for tid in chain_ids):
                return run
        logs_dir = run / "agent_logs"
        if all((logs_dir / f"{tid}_agent_trajectory.json").exists() for tid in chain_ids):
            return run
    return None


def find_latest_run_with_report(task_dir: Path) -> Path | None:
    """Return the newest agent_eval run that has report.json."""
    eval_root = task_dir / "agent_eval"
    if not eval_root.exists():
        return None
    for run in sorted(eval_root.iterdir(), reverse=True):
        if run.is_dir() and (run / "report.json").exists():
            return run
    return None


def passed_transition_ids(run_dir: Path) -> set[str] | None:
    """Return transition ids marked PASS in report.json, or None if unavailable."""
    path = run_dir / "report.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    results = data.get("transition_results")
    passed: set[str] = set()
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            tid = item.get("transition_id") or item.get("id")
            if tid and str(item.get("status", "")).upper() == "PASS":
                passed.add(str(tid))
    elif isinstance(results, dict):
        for tid, item in results.items():
            if isinstance(item, dict) and str(item.get("status", "")).upper() == "PASS":
                passed.add(str(tid))
    return passed


def load_operation_record(run_dir: Path) -> dict | None:
    path = run_dir / "operation_record.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _trajectory_steps_after_last_reset(run_dir: Path, tid: str) -> list[dict] | None:
    path = run_dir / "agent_logs" / f"{tid}_agent_trajectory.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    trajectory = data.get("trajectory", [])
    last_reset = -1
    for i, step in enumerate(trajectory):
        action_text = (step.get("action_text") or "").strip().lower()
        parsed_verb = ((step.get("parsed_action") or {}).get("verb") or "").strip().lower()
        atype = ((step.get("action_dict") or {}).get("type") or "").strip().lower()
        if action_text == "reset" or parsed_verb == "reset" or atype == "reset":
            last_reset = i
    if last_reset < 0:
        return None

    steps = []
    for step in trajectory[last_reset + 1:]:
        atype = ((step.get("action_dict") or {}).get("type") or "").strip().lower()
        if atype in ("done", "reset") or not atype or not step.get("exec_ok"):
            continue
        steps.append(step)
    return steps


def load_trajectory(run_dir: Path, tid: str) -> list[dict]:
    """Load replayable action steps for transition `tid` from a run directory."""
    post_reset_steps = _trajectory_steps_after_last_reset(run_dir, tid)
    if post_reset_steps is not None:
        return post_reset_steps

    record = load_operation_record(run_dir)
    if record:
        transitions = record.get("transitions") or {}
        if tid not in transitions:
            transitions = {}
        tr_record = transitions.get(tid) or {}
        steps = []
        for step in tr_record.get("steps", []):
            atype = (step.get("action_dict") or {}).get("type", "")
            if atype in ("done", "reset") or not atype:
                continue
            steps.append(step)
        if tr_record:
            return steps

    path = run_dir / "agent_logs" / f"{tid}_agent_trajectory.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = []
    for step in data.get("trajectory", []):
        atype = ((step.get("action_dict") or {}).get("type") or "").strip().lower()
        # Skip Done/Reset — these are terminal/reset signals, not real actions
        if atype in ("done", "reset") or not atype or not step.get("exec_ok"):
            continue
        steps.append(step)
    return steps


# ── Requirements helpers ──────────────────────────────────────────────────────

def load_req_map() -> dict[str, dict]:
    if not REQ_JSON.exists():
        return {}
    data = json.loads(REQ_JSON.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for dom in data.get("domains", []):
        did = dom.get("domain_id")
        for sc in dom.get("scenarios", []):
            sid = sc.get("scenario_id")
            for t in sc.get("tasks", []):
                fid = t.get("full_task_id")
                if not fid:
                    tid_int = t.get("task_id")
                    if None in (did, sid, tid_int):
                        continue
                    fid = f"D{int(did):02d}_S{int(sid):02d}_T{int(tid_int):03d}"
                out[fid] = {
                    "task_name": t.get("task_name") or "",
                    "scenario":  sc.get("scenario") or "",
                    "reqs":      t.get("requirements", []),
                }
    return out


def _humanize(s: str) -> str:
    return (s or "").replace("_", " ").strip()


def covered_req_ids(icg: dict, chain: list[dict], intent: str | None = "explicit") -> set[str]:
    chain_ids = {t["id"] for t in chain}
    covered: set[str] = set()
    for entry in icg.get("requirements_coverage", []):
        if intent is not None and entry.get("intent") != intent:
            continue
        req_id = entry.get("req_id") or entry.get("washed_req_id", "")
        if set(entry.get("covered_by_transitions", [])) & chain_ids:
            covered.add(req_id)
    return covered


def requirement_ids(icg: dict, intent: str | None = "explicit") -> set[str]:
    ids: set[str] = set()
    for entry in icg.get("requirements_coverage", []):
        if intent is not None and entry.get("intent") != intent:
            continue
        req_id = entry.get("req_id") or entry.get("washed_req_id", "")
        if req_id:
            ids.add(req_id)
    return ids


def build_input_json(task_id: str, icg: dict, req_entry: dict) -> dict:
    task_name_human = _humanize(req_entry.get("task_name") or icg.get("task_name", ""))
    scenario_human  = _humanize(req_entry.get("scenario") or icg.get("scenario", ""))

    first  = f"Please implement a {task_name_human} webpage in a {scenario_human} scenario."
    video_only_user_prompt = (
        f"{first}\n\n"
        "Implement the interaction flow shown in the video, and complete any "
        "remaining required behavior not covered by the video."
    )

    return {
        "task_id":       task_id,
        "task_name":     icg.get("task_name", task_id),
        "system_prompt": VIDEO_SYSTEM_PROMPT,
        "video_only_user_prompt": video_only_user_prompt,
    }


def save_video_input_json(task_id: str, task_dir: Path, icg: dict, req_map: dict) -> Path:
    video_dir = task_dir / "Video"
    video_dir.mkdir(parents=True, exist_ok=True)
    payload = build_input_json(task_id, icg, req_map.get(task_id, {}))
    out_path = video_dir / "input.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


# ── HTML finder ───────────────────────────────────────────────────────────────

def find_gt_html(task_id: str, gt_html_root: Path) -> Path | None:
    for html in gt_html_root.rglob(f"{task_id}_*/*.html"):
        return html
    return None


def find_eval_html(run_dir: Path) -> Path | None:
    report = run_dir / "report.json"
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            html = Path(data.get("html_path") or "")
            if html.exists():
                return html
        except Exception:
            pass

    log = run_dir / "eval.log"
    if log.exists():
        import re

        text = log.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"^html:\s*(.+)$", text, re.M) or re.search(r"^\s*HTML\s*:\s*(.+)$", text, re.M)
        if match:
            html = Path(match.group(1).strip())
            if html.exists():
                return html
    return None


def find_seed_html(task_dir: Path) -> Path | None:
    """Find the task HTML copied into a seeddata task directory."""
    htmls = sorted(task_dir.glob("*.html"))
    if htmls:
        return htmls[0]

    ignored = {"agent_eval", "agent_eval_defect", "Video", "MD", "Sketch", "Text", "Image"}
    for html in sorted(task_dir.rglob("*.html")):
        try:
            rel = html.relative_to(task_dir)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in ignored:
            continue
        return html
    return None


# ── Visual replay helpers ─────────────────────────────────────────────────────

def _step_parts(step: dict) -> tuple[str, dict, str | None]:
    """Normalize operation_record and trajectory step shapes."""
    action_dict = step.get("action_dict") or {}
    atype = (action_dict.get("type") or step.get("type") or "").lower()
    params = action_dict.get("parameters")
    if params is None:
        params = step.get("params") or {}
    return atype, dict(params or {}), step.get("selector")


async def _move_cursor_to(page, selector: str | None) -> None:
    if not selector:
        return
    try:
        el = await page.query_selector(selector)
        if not el:
            return
        await el.scroll_into_view_if_needed(timeout=3000)
        box = await el.bounding_box()
        if not box:
            return
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        await page.mouse.move(cx, cy, steps=CURSOR_MOVE_STEPS)
        await page.wait_for_timeout(CURSOR_SETTLE_MS)
    except Exception:
        pass


async def _visual_select(page, selector: str, value: str) -> bool:
    if not selector or not value:
        return False
    try:
        el = await page.query_selector(selector)
        if not el:
            return False
        is_native = await el.evaluate("el => !!el && el.tagName === 'SELECT'")
        if not is_native:
            return False
        had_target = await el.evaluate(
            "(el, value) => window.__wi_show_select_dropdown_for_el(el, value)",
            value,
        )
        if not had_target:
            await page.evaluate("() => window.__wi_hide_select_dropdown && window.__wi_hide_select_dropdown()")
            return False
        await page.wait_for_timeout(220)
        await _move_cursor_to(page, "#__wi_active_opt")
        try:
            await page.click("#__wi_active_opt")
        except Exception:
            await page.evaluate("() => window.__wi_hide_select_dropdown && window.__wi_hide_select_dropdown()")
            return False
        await page.wait_for_timeout(SELECT_DROPDOWN_DWELL_MS)
        return True
    except Exception:
        try:
            await page.evaluate("() => window.__wi_hide_select_dropdown && window.__wi_hide_select_dropdown()")
        except Exception:
            pass
        return False


def _resolve_upload_paths(params: dict) -> dict:
    files = params.get("files") or []
    if isinstance(files, str):
        files = [files]
    if not files:
        return params
    rewritten = []
    for f in files:
        p = Path(f)
        abs_p = p if p.is_absolute() else (EVAL_DIR / p)
        try:
            abs_p = abs_p.resolve()
        except Exception:
            pass
        try:
            rewritten.append(os.path.relpath(abs_p))
        except ValueError:
            rewritten.append(str(abs_p))
    out = dict(params)
    out["files"] = rewritten
    return out


async def _find_confirm_button_selector(page) -> str | None:
    js = """
    (kws) => {
      const containers = ['dialog[open]', '[role="dialog"]', '[role="alertdialog"]',
        '.modal-overlay', '.modal', '.dialog', '.popup', '.confirm',
        '.overlay', '.alert-box', '.confirm-box', '.confirm-dialog'];
      const isVisible = (el) => {
        const s = window.getComputedStyle(el);
        if (s.display === 'none' || s.visibility === 'hidden' || s.opacity === '0') return false;
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      };
      const matchKw = (text) => kws.some(k => text.includes(k));
      const cssEsc = (s) => CSS && CSS.escape ? CSS.escape(s) : s.replace(/[^a-zA-Z0-9_-]/g, c => '\\\\' + c);
      const buildSelector = (el) => {
        if (el.id) return '#' + cssEsc(el.id);
        const parts = [];
        let cur = el;
        while (cur && cur !== document.body && cur.parentElement) {
          const parent = cur.parentElement;
          const idx = Array.from(parent.children).indexOf(cur) + 1;
          parts.unshift(cur.tagName.toLowerCase() + ':nth-child(' + idx + ')');
          cur = parent;
        }
        return 'body > ' + parts.join(' > ');
      };
      for (const sel of containers) {
        const c = document.querySelector(sel);
        if (!c || !isVisible(c)) continue;
        const buttons = c.querySelectorAll('button, [role="button"]');
        for (const b of buttons) {
          if (!isVisible(b)) continue;
          const t = (b.innerText || b.textContent || '').trim().toLowerCase();
          if (matchKw(t)) return buildSelector(b);
        }
      }
      const all = document.querySelectorAll('button, [role="button"]');
      for (const b of all) {
        if (!isVisible(b)) continue;
        const t = (b.innerText || b.textContent || '').trim().toLowerCase();
        if (matchKw(t)) return buildSelector(b);
      }
      return null;
    }
    """
    try:
        return await page.evaluate(js, list(CONFIRM_KEYWORDS))
    except Exception:
        return None


async def _click_confirm_button_visibly(page) -> None:
    sel = None
    elapsed = 0
    while elapsed < 1200:
        sel = await _find_confirm_button_selector(page)
        if sel:
            break
        await page.wait_for_timeout(150)
        elapsed += 150
    if not sel:
        return
    await _move_cursor_to(page, sel)
    try:
        await page.click(sel)
    except Exception as e:
        print(f"         [WARN] auto-confirm click failed on {sel!r}: {e}")


async def _click_overlay_blank_area_visibly(page) -> bool:
    try:
        coord = await page.evaluate("""() => {
            const selectors = [
                '[role="dialog"]', '.modal-content', '.modal-dialog',
                '.dialog', '.popup', '.lightbox-content'
            ];
            let dialog = null;
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                if (!el) continue;
                const r = el.getBoundingClientRect();
                if (r.width > 0 && r.height > 0) {
                    dialog = {x: r.left, y: r.top, width: r.width, height: r.height};
                    break;
                }
            }
            const vw = window.innerWidth;
            const vh = window.innerHeight;
            const outside = (px, py) => !dialog ||
                !(dialog.x <= px && px <= dialog.x + dialog.width &&
                  dialog.y <= py && py <= dialog.y + dialog.height);
            const candidates = [[10,10], [10, vh - 10], [vw - 10, 10], [vw - 10, vh - 10]];
            for (const c of candidates) if (outside(c[0], c[1])) return c;
            return [10, 10];
        }""")
        px, py = int(coord[0]), int(coord[1])
        await page.mouse.move(px, py, steps=CURSOR_MOVE_STEPS)
        await page.wait_for_timeout(CURSOR_SETTLE_MS)
        await page.mouse.click(px, py)
        return True
    except Exception:
        return False


async def _type_visibly(page, selector: str, params: dict) -> bool:
    el = await page.query_selector(selector)
    if not el:
        return False
    value = params.get("value", "")
    try:
        has_selection = await page.evaluate("""() => {
            const s = window.getSelection();
            if (s && s.toString().length > 0) return true;
            const a = document.activeElement;
            return !!(a && (a.tagName === 'TEXTAREA' || a.tagName === 'INPUT')
                      && a.selectionStart !== a.selectionEnd);
        }""")
        if not has_selection:
            await el.focus()
        if value:
            delay = TYPING_DELAY_MS
            if len(value) > 300:
                delay = 5
            elif len(value) > 120:
                delay = 15
            is_contenteditable = await page.evaluate("""() => {
                const a = document.activeElement;
                if (a && a.isContentEditable) return true;
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
                for ch in value:
                    await page.evaluate(
                        "(v) => document.execCommand('insertText', false, v)",
                        ch,
                    )
                    await page.wait_for_timeout(delay)
            else:
                await page.keyboard.type(value, delay=delay)
        elif has_selection:
            await page.keyboard.press("Delete")
        await page.wait_for_timeout(300)
        return True
    except Exception:
        return False


async def _execute_visual_step(page, step: dict) -> None:
    atype, params, selector = _step_parts(step)

    if selector and atype not in ("refresh", "wait"):
        await _move_cursor_to(page, selector)

    if atype in ("input", "type") and selector:
        if await _type_visibly(page, selector, params):
            return

    if atype == "upload":
        params = _resolve_upload_paths(params)
        await page.wait_for_timeout(UPLOAD_PICKER_DWELL_MS)

    if atype == "select" and selector:
        if await _visual_select(page, selector, params.get("value", "")):
            return

    if atype == "clickat":
        try:
            px, py = int(params.get("x")), int(params.get("y"))
            await page.mouse.move(px, py, steps=CURSOR_MOVE_STEPS)
            await page.wait_for_timeout(CURSOR_SETTLE_MS)
            await page.mouse.click(px, py)
            return
        except Exception:
            pass

    if atype == "click" and params.get("click_position") == "overlay_blank_area":
        if await _click_overlay_blank_area_visibly(page):
            return

    if atype == "click" and params.get("auto_confirm"):
        action_no_confirm = {
            "type": atype,
            "parameters": {k: v for k, v in params.items() if k != "auto_confirm"},
        }
        ok, err = await execute_action(page, action_no_confirm, {}, selector)
        if not ok:
            print(f"         [WARN] {step.get('action_text', atype)} → {err}")
            return
        await _click_confirm_button_visibly(page)
        return

    action = {"type": atype, "parameters": params}
    ok, err = await execute_action(page, action, {}, selector)
    if not ok:
        print(f"         [WARN] {step.get('action_text', atype)} → {err}")

    if atype == "refresh":
        try:
            vw = await page.evaluate("window.innerWidth")
            vh = await page.evaluate("window.innerHeight")
            await page.mouse.move(vw / 2, vh / 2, steps=1)
        except Exception:
            pass


async def _wait_for_images_and_dwell(page) -> None:
    try:
        await page.evaluate("""
            async () => {
                const imgs = Array.from(document.querySelectorAll('img'));
                await Promise.all(imgs.map(img => {
                    if (img.complete && img.naturalWidth > 0) return null;
                    return new Promise(res => {
                        img.addEventListener('load', res, {once: true});
                        img.addEventListener('error', res, {once: true});
                        setTimeout(res, 3000);
                    });
                }));
            }
        """)
    except Exception:
        pass
    await page.wait_for_timeout(1500)


async def _wait_replay_transients(page) -> None:
    """Wait out body-level transient UI before replaying recorded selectors.

    Operation records often use absolute XPath selectors. Toasts and temporary
    visual panels are not durable page states, but they can shift body child
    indexes while they exist, making selectors such as html/body/div[2]/...
    point at the wrong element during video replay.
    """
    try:
        has_toast = await page.evaluate("() => !!document.querySelector('.toast')")
        if has_toast:
            await page.wait_for_function(
                "() => !document.querySelector('.toast')",
                timeout=3500,
            )
    except Exception:
        pass
    try:
        await page.evaluate(
            "() => window.__wi_hide_select_dropdown && window.__wi_hide_select_dropdown()"
        )
    except Exception:
        pass


def _extract_frames(video_path: Path, fps: float, frames_dir: Path) -> None:
    if fps <= 0:
        return
    if not shutil.which("ffmpeg"):
        print("      [WARN] ffmpeg not found; skipping frame extraction")
        return
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"fps={fps}",
                str(frames_dir / "frame_%04d.png"),
            ],
            check=True,
            capture_output=True,
        )
        print(f"      frames → {frames_dir}")
    except subprocess.CalledProcessError as e:
        print(f"      [WARN] frame extraction failed: {e}")


# ── Playwright replay ─────────────────────────────────────────────────────────

async def replay_and_record(
    html_path: Path,
    chain: list[dict],
    run_dir: Path,
    video_dir: Path,
    task_id: str,
    settle_ms: int = 800,
) -> Path | None:
    """
    Replay stored trajectories for each transition in `chain` while recording.
    Actions are replayed using their stored CSS selector and action_dict,
    so no LLM calls are made.
    """
    from playwright.async_api import async_playwright

    video_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.webm", "*.mp4"):
        for old in video_dir.glob(pattern):
            old.unlink()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": cfg.VIEWPORT_WIDTH, "height": cfg.VIEWPORT_HEIGHT},
            record_video_dir=str(video_dir),
            record_video_size={"width": cfg.VIEWPORT_WIDTH, "height": cfg.VIEWPORT_HEIGHT},
        )
        await context.add_init_script(CURSOR_INIT_SCRIPT)
        page = await context.new_page()
        await page.goto(html_path.resolve().as_uri(), timeout=cfg.PAGE_LOAD_TIMEOUT_MS)
        try:
            await page.wait_for_load_state("networkidle", timeout=cfg.PAGE_LOAD_TIMEOUT_MS)
        except Exception:
            pass
        await page.wait_for_timeout(1000)  # initial settle

        for tr in chain:
            tid = tr["id"]
            steps = load_trajectory(run_dir, tid)
            print(f"      {tid}: replaying {len(steps)} action(s)")

            await page.wait_for_timeout(PRE_ACTION_PAUSE_MS)
            for step in steps:
                await _wait_replay_transients(page)
                await _execute_visual_step(page, step)
                await page.wait_for_timeout(POST_ACTION_PAUSE_MS)

            # Pause between transitions so state is visible in the video
            await page.wait_for_timeout(settle_ms)

        await _wait_for_images_and_dwell(page)

        # Retrieve video path before closing
        video = page.video
        raw_path = await video.path() if video else None

        await context.close()
        await browser.close()

    # Rename auto-generated UUID file to predictable name, then convert to MP4
    if raw_path:
        raw_path = Path(raw_path)
        webm_path = video_dir / f"{task_id}_chain.webm"
        mp4_path  = video_dir / f"{task_id}_chain.mp4"
        if raw_path.exists():
            raw_path.rename(webm_path)
            if not shutil.which("ffmpeg"):
                print(f"      📹  saved → {webm_path.name}  (ffmpeg not found; kept webm)")
                return webm_path
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(webm_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        str(mp4_path),
                    ],
                    check=True,
                    capture_output=True,
                )
                webm_path.unlink()
                print(f"      📹  saved → {mp4_path.name}")
                return mp4_path
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                # ffmpeg not installed or failed — keep webm as fallback
                print(f"      📹  saved → {webm_path.name}  (mp4 convert failed: {e})")
                return webm_path
    return None


# ── Per-task orchestration ────────────────────────────────────────────────────

async def process_task(
    task_id: str,
    req_map: dict,
    seed_root: Path,
    gt_html_root: Path,
    html_source: str = "eval",
    passed_only: bool = False,
    dry_run: bool = False,
    settle_ms: int = 800,
    force: bool = False,
    extract_fps: float = 0,
    input_only: bool = False,
) -> tuple[bool, str]:
    task_dir  = seed_root / task_id
    icg_path  = task_dir / "icg.json"
    video_dir = task_dir / "Video"

    if not icg_path.exists():
        return False, "missing icg.json"

    icg   = json.loads(icg_path.read_text(encoding="utf-8"))

    if input_only:
        out_path = save_video_input_json(task_id, task_dir, icg, req_map)
        return True, f"input written -> {out_path}"

    chain = best_coverage_chain(icg.get("transitions", []), icg=icg)
    if not chain:
        return False, "no transitions in ICG"

    chain_ids = [t["id"] for t in chain]
    explicit_covered = covered_req_ids(icg, chain, intent="explicit")
    explicit_total = requirement_ids(icg, intent="explicit")
    all_covered = covered_req_ids(icg, chain, intent=None)
    all_total = requirement_ids(icg, intent=None)
    explicit_ratio = (len(explicit_covered) / len(explicit_total)) if explicit_total else 1.0
    print(f"  chain: {' → '.join(chain_ids)}")
    print(
        f"  score: display_reqs={len(explicit_covered)}/{len(explicit_total)} "
        f"({explicit_ratio:.1%}) total_reqs={len(all_covered)}/{len(all_total)} len={len(chain_ids)}"
    )

    if passed_only:
        run_dir = find_latest_run_with_report(task_dir)
        if not run_dir:
            return False, "no eval run with report.json for --passed-only"
        print(f"  run:   {run_dir.name}")
        passed_ids = passed_transition_ids(run_dir)
        if passed_ids is None:
            return False, f"--passed-only requested but report.json has no transition status data in {run_dir}"
        chain = best_coverage_chain(
            [t for t in icg.get("transitions", []) if t.get("id") in passed_ids],
            icg=icg,
        )
        if not chain:
            return False, "no passed transitions available for video chain"
        chain_ids = [t["id"] for t in chain]
        print(f"  pass:  using passed-only chain: {' → '.join(chain_ids)}")
        replay_run = find_best_run(task_dir, chain_ids)
        if not replay_run:
            return False, f"no eval run with trajectories for passed-only chain {chain_ids}"
        if replay_run != run_dir:
            print(f"  run:   replay trajectories from {replay_run.name}")
            run_dir = replay_run
    else:
        run_dir = find_best_run(task_dir, chain_ids)
        if not run_dir:
            return False, f"no eval run with trajectories for chain {chain_ids}"
        print(f"  run:   {run_dir.name}")

    if html_source == "gt":
        html_path = find_gt_html(task_id, gt_html_root)
        if not html_path:
            return False, f"GT HTML not found under {gt_html_root}"
    elif html_source == "seed":
        html_path = find_seed_html(task_dir)
        if not html_path:
            return False, f"seed HTML not found in {task_dir}"
    else:
        html_path = find_eval_html(run_dir)
        if not html_path:
            return False, f"source eval HTML not found in {run_dir}"
    print(f"  html:  {html_path}")

    if dry_run:
        return True, f"dry-run OK  run={run_dir.name}"

    already_done = (video_dir / f"{task_id}_chain.mp4").exists() or \
                   (video_dir / f"{task_id}_chain.webm").exists()
    if not force and already_done:
        input_path = save_video_input_json(task_id, task_dir, icg, req_map)
        if extract_fps > 0:
            existing_video = (
                video_dir / f"{task_id}_chain.mp4"
                if (video_dir / f"{task_id}_chain.mp4").exists()
                else video_dir / f"{task_id}_chain.webm"
            )
            _extract_frames(existing_video, extract_fps, video_dir / "frames")
        return True, f"skipped (already recorded); input written -> {input_path}"

    # Replay trajectories with video recording
    video_path = await replay_and_record(
        html_path=html_path,
        chain=chain,
        run_dir=run_dir,
        video_dir=video_dir,
        task_id=task_id,
        settle_ms=settle_ms,
    )
    if video_path and extract_fps > 0:
        _extract_frames(video_path, extract_fps, video_dir / "frames")

    # Save coverage.json
    covered_ids = covered_req_ids(icg, chain, intent="explicit")
    total_explicit_ids = requirement_ids(icg, intent="explicit")
    total_req_ids = requirement_ids(icg, intent=None)
    display_ratio = (len(covered_ids) / len(total_explicit_ids)) if total_explicit_ids else 1.0
    total_ratio = (len(covered_req_ids(icg, chain, intent=None)) / len(total_req_ids)) if total_req_ids else 1.0
    coverage = {
        "task_id":   task_id,
        "selection_strategy": "dfs_from_S0_rank_by_explicit_req_coverage_then_total_req_coverage_then_chain_length",
        "chain":     chain_ids,
        "source_run": run_dir.name,
        "html_source": html_source,
        "html_path": str(html_path),
        "passed_only": passed_only,
        "display_req_coverage": {
            "covered": len(covered_ids),
            "total": len(total_explicit_ids),
            "ratio": display_ratio,
            "text": f"{len(covered_ids)}/{len(total_explicit_ids)}",
        },
        "total_req_coverage": {
            "covered": len(covered_req_ids(icg, chain, intent=None)),
            "total": len(total_req_ids),
            "ratio": total_ratio,
            "text": f"{len(covered_req_ids(icg, chain, intent=None))}/{len(total_req_ids)}",
        },
        "score": {
            "explicit_req_count": len(covered_ids),
            "explicit_req_total": len(total_explicit_ids),
            "total_req_count": len(covered_req_ids(icg, chain, intent=None)),
            "total_req_total": len(total_req_ids),
            "chain_length": len(chain_ids),
        },
        "covered_requirements": {
            "explicit": sorted(covered_ids),
        },
    }
    (video_dir / "coverage.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    save_video_input_json(task_id, task_dir, icg, req_map)

    return True, f"done  chain={chain_ids}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build Video-modality inputs from existing agent eval trajectories."
    )
    ap.add_argument("task_ids", nargs="*",
                    help="Task IDs (default: all tasks under the data root)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show chain + run without recording")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing video")
    ap.add_argument("--settle-ms", type=int, default=800,
                    help="Extra pause between transitions in ms (default: 800)")
    ap.add_argument("--extract-fps", type=float, default=0,
                    help="If > 0, extract frames from each recorded video at this fps")
    ap.add_argument("--seed-root", default=str(SEED_ROOT),
                    help="Data root containing task folders")
    ap.add_argument("--gt-html-root", default=str(GT_HTML_ROOT),
                    help="GT HTML root")
    ap.add_argument("--html-source", choices=("eval", "gt", "seed"), default="eval",
                    help="HTML to replay against: eval = source agent_eval HTML (default), gt = configured GT root, seed = HTML inside each task dir")
    ap.add_argument("--passed-only", action="store_true",
                    help="Build the video chain only from transitions marked PASS in the source eval report")
    ap.add_argument("--input-only", action="store_true",
                    help="Only write Video/input.json; do not select/replay/record a video")
    args = ap.parse_args()

    seed_root = Path(args.seed_root)
    gt_html_root = Path(args.gt_html_root)

    if args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = sorted(p.name for p in seed_root.iterdir()
                          if p.is_dir() and (p / "icg.json").exists())

    req_map = load_req_map()

    ok = fail = 0
    for task_id in task_ids:
        print(f"\n[{task_id}]")
        try:
            success, msg = asyncio.run(process_task(
                task_id=task_id,
                req_map=req_map,
                seed_root=seed_root,
                gt_html_root=gt_html_root,
                html_source=args.html_source,
                passed_only=args.passed_only,
                dry_run=args.dry_run,
                settle_ms=args.settle_ms,
                force=args.force,
                extract_fps=args.extract_fps,
                input_only=args.input_only,
            ))
        except Exception as e:
            success = False
            msg = f"error: {type(e).__name__}: {e}"
        if success:
            print(f"  ✅  {msg}")
            ok += 1
        else:
            print(f"  ❌  {msg}")
            fail += 1

    print(f"\nTotal: {ok + fail}    OK: {ok}    Fail: {fail}")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
