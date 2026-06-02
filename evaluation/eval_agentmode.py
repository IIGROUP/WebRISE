#!/usr/bin/env python3
"""
WebRISE - Agent-mode Evaluation Script
======================================
Evaluates LLM-generated HTML pages against an ICG spec (states, transitions,
agent goals, DOM assertions, and visual postconditions).

This script drives each transition via the WebVoyager-style agent loop in
`agent_executor.py`. There is
no separate locator / selector cache / setup_actions phase; the agent observes
the DOM text, picks an action, and executes it until the intent is DONE or the
iteration budget is exhausted.

Pipeline per transition:
  1. Navigate to `tr.from` (replay agent trajectories of prior passing chains).
  2. Capture pre_ss (honours `tr.screenshot_mode`).
  3. Start page-level DOM monitor (target_selector=None, observed=None).
  4. Run the agent loop (`run_agent_transition`) until DONE / MAX_ITER / ERROR.
  5. Finish DOM monitor with a settle wait (default 2000 ms).
  6. Score `dom_assertions` from the event log (via `score_dom_assertions`).
  7. Score `postconditions` from pre/post screenshots (via `score_postconditions`).
  8. Attach requirement tags, record result, update state-path cache.

ICG input shape:
  {
    "task_id": ...,
    "states": [...],
    "transitions": [
      {"id": "T1", "from": "S0", "to": "S1",
       "agent_task": "User invokes the publishing touchpoint.",
       "preconditions": [...],
       "dom_assertions": [{"assertion": "[CHANGE] ...", "primary_req_id": "..."}],
       "postconditions": [{"assertion": "...", "primary_req_id": "..."}],
       "screenshot_mode": "viewport"
      },
      ...
    ]
  }

Usage:
  python eval_agentmode.py --html page.html --icg icg.json
  python eval_agentmode.py --html page.html --icg icg.json --output ./results
  python eval_agentmode.py --html page.html --icg icg.json --max-iter 12
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Force UTF-8 console output on Windows.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from openai import OpenAI
from playwright.async_api import async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

import config as cfg
from executor import execute_action
from agent_executor import run_agent_transition
from scorer import score_postconditions, score_preconditions
from dom_scorer import score_dom_assertions
from dom_assert import (
    start_dom_monitor,
    finish_dom_monitor,
    summarize_dom_log,
    capture_snapshot,
    synthesize_evidence,
)
from metrics import compute_3d_metrics, compute_req_metrics, compute_ti_metrics


def format_agent_metrics_summary(metrics: dict, req_metrics: dict | None = None) -> str:
    """Agent-mode summary for state, transition, and requirement metrics."""
    s = metrics["state_reachability"]
    t = metrics["transition_validity"]

    lines = [
        "┌───────────────────────────────────────────────────────┐",
        f"│  S%   State Reachability      {s['score']:6.1%}  ({s['reached']}/{s['total']})      │",
        f"│  T%   Transition Validity     {t['score']:6.1%}  ({t['passed']}/{t['total']})      │",
    ]
    if req_metrics:
        re = req_metrics["R_explicit"]
        ri = req_metrics["R_implicit"]
        ro = req_metrics["R_overall"]
        lines += [
            "│───────────────────────────────────────────────────────│",
            f"│  Re%  Explicit Req Rate       {re['score']:6.1%}  ({re['passed_count']}/{re['total']})      │",
            f"│  Ri%  Implicit Req Rate       {ri['score']:6.1%}  ({ri['passed_count']}/{ri['total']})      │",
            f"│  R%   Overall Req Rate        {ro['score']:6.1%}  ({ro['passed_count']}/{ro['total']})      │",
        ]
    lines.append("└───────────────────────────────────────────────────────┘")
    return "\n".join(lines)

# ── Agent-mode tuning knobs ───────────────────────────────────────────────────
# Settle wait after the agent loop finishes, before closing the DOM monitor.
# Long enough to catch toast auto-dismiss and other transient process evidence.
DEFAULT_DOM_SETTLE_MS = 8000
# Hard cap on agent turns per transition.
DEFAULT_MAX_ITER = 15
# Hard caps for one agent LLM call and one browser action.
DEFAULT_AGENT_TIMEOUT_S = getattr(cfg, "AGENT_CALL_TIMEOUT_S", 300)
DEFAULT_ACTION_TIMEOUT_S = getattr(cfg, "ACTION_TIMEOUT_S", 30)
# Hard cap for page bootstrap (new context, navigation, subresource waits,
# and the initial screenshot). Pages with a JS main-thread deadlock are recorded
# as successful zero-score quality results instead of being retried forever.
DEFAULT_PAGE_INIT_TIMEOUT_S = int(os.environ.get("WEB_EVAL_PAGE_INIT_TIMEOUT_S", "60"))

if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

# ── ICG v1 Assertion Helpers ──────────────────────────────────────────────────

def _assertion_text(item: dict) -> str:
    """Extract the assertion string from an assertion object."""
    return item.get("assertion", "")


def _assertion_texts(items: list[dict]) -> list[str]:
    """Extract assertion strings from a list of assertion objects."""
    return [_assertion_text(it) for it in items]


def _attach_req_tags(results: list[dict], icg_items: list[dict]) -> None:
    """Copy requirement tags from ICG assertion objects onto eval result dicts."""
    for i, result in enumerate(results):
        if i < len(icg_items):
            if icg_items[i].get("primary_req_id"):
                result["primary_req_id"] = icg_items[i]["primary_req_id"]
            if icg_items[i].get("primary_req_id_invalid"):
                result["primary_req_id_invalid"] = True
        else:
            result["primary_req_id"] = None


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_icg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and isinstance(raw.get("icg"), dict):
        core = dict(raw["icg"])
        for key in ("task_id", "task_name", "domain", "scenario", "input_text"):
            if key in raw and key not in core:
                core[key] = raw[key]
        raw = core

    if isinstance(raw, dict):
        raw.setdefault("task_id", Path(path).stem)
        raw.setdefault("task_name", raw["task_id"])
        raw.setdefault("domain", "")
        raw.setdefault("scenario", "")
        raw.setdefault("input_text", "")
    return raw




def _status_icon(status: str) -> str:
    return {"PASS": "✓", "FAIL": "✗", "BLOCKED": "⊗", "SKIPPED": "⊘"}.get(status, "?")


# ── Main evaluation loop ──────────────────────────────────────────────────────

async def evaluate(
    html_path: str,
    icg_path:  str,
    output_dir: str,
    api_key:   str,
    base_url:  str,
    max_iter:  int = DEFAULT_MAX_ITER,
    dom_settle_ms: int = DEFAULT_DOM_SETTLE_MS,
    agent_timeout_s: int = DEFAULT_AGENT_TIMEOUT_S,
    action_timeout_s: int = DEFAULT_ACTION_TIMEOUT_S,
    run_id: str | None = None,
    filter_transitions: set[str] | None = None,
    record_operations: bool = False,
    preconditions_only: bool = False,
) -> dict:
    icg = load_icg(icg_path)

    run_id  = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{icg['task_id']}_{run_id}"
    ss_dir  = run_dir / "screenshots"
    ss_dir.mkdir(parents=True, exist_ok=True)

    api_log_dir = run_dir / "calllog"
    api_log_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    print(f"\n{'═' * 62}")
    print(f"  WebRISE Evaluation")
    print(f"  Task    : {icg['task_id']} — {icg['task_name']}")
    print(f"  HTML    : {html_path}")
    print(f"  Timeout : agent={agent_timeout_s}s action={action_timeout_s}s")
    print(f"{'═' * 62}")

    import time as _time
    _eval_t0 = _time.time()
    api_call_log: list[dict] = []
    api_log_seq = 0

    report: dict = {
        "task_id":              icg["task_id"],
        "task_name":            icg["task_name"],
        "domain":               icg.get("domain", ""),
        "scenario":             icg.get("scenario", ""),
        "input_text":           icg.get("input_text", ""),
        "interaction_pattern":  icg.get("interaction_pattern", ""),
        "complexity_level":     icg.get("complexity_level", ""),
        "html_path":            str(Path(html_path).resolve()),
        "icg_path":             str(Path(icg_path).resolve()),
        "run_id":               run_id,
        "mode":                 "agent",
        "model_agent":          cfg.MODEL_AGENT,
        "model_scorer":         cfg.MODEL_SCORER,
        "reasoning_effort":     cfg.REASONING_EFFORT,
        "reasoning_effort_model_prefixes": list(cfg.REASONING_EFFORT_MODEL_PREFIXES),
        "max_iter":             max_iter,
        "dom_settle_ms":        dom_settle_ms,
        "agent_timeout_s":      agent_timeout_s,
        "action_timeout_s":     action_timeout_s,
        "record_operations":    record_operations,
        "preconditions_only":   preconditions_only,
        "transition_results":   [],
        "initial_state_check":  None,
        "metrics":              {},
        "req_metrics":          {},
    }

    report_path = run_dir / "report.json"
    operation_record_path = run_dir / "operation_record.json"

    operation_record: dict | None = None
    if record_operations:
        operation_record = {
            "task_id": icg["task_id"],
            "task_name": icg["task_name"],
            "html_path": str(Path(html_path).resolve()),
            "icg_path": str(Path(icg_path).resolve()),
            "run_id": run_id,
            "mode": "agent",
            "record_type": "passed_transition_replay_steps",
            "transitions": {},
            "state_paths": {},
        }

    _STRIP_KEYS = {
        "scorer_call_log", "dom_assertion_call_log",
        "dom_assertion_log", "dom_assertion_log_summary",
        "restore_method", "action_success",
        "agent_trajectory",
    }

    def _slim_transition(tr: dict) -> dict:
        slim = {k: v for k, v in tr.items() if k not in _STRIP_KEYS}
        for field in ("dom_assertion_results", "postcondition_results"):
            slim[field] = [
                {k: v for k, v in r.items() if k in ("condition", "verdict", "passed", "think")}
                for r in slim.get(field, [])
            ]
        return slim

    def _save_report() -> None:
        slim = {k: v for k, v in report.items()
                if k not in ("api_call_log",)}
        slim["transition_results"] = [_slim_transition(tr) for tr in report.get("transition_results", [])]
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2, default=str, ensure_ascii=False)

    def _save_operation_record() -> None:
        if operation_record is None:
            return
        with open(operation_record_path, "w", encoding="utf-8") as f:
            json.dump(operation_record, f, indent=2, default=str, ensure_ascii=False)

    def _persist_api_log_entry(entry: dict) -> None:
        nonlocal api_log_seq
        api_log_seq += 1
        api_call_log.append(entry)

        phase = entry.get("phase", "unknown")
        tr_id = entry.get("transition_id", "na")
        safe_parts = [f"{api_log_seq:04d}", tr_id, phase]
        fname = "_".join(safe_parts) + ".json"

        calllog_path = api_log_dir / fname
        with open(calllog_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, default=str, ensure_ascii=False)

    def _write_zero_quality_report(reason: str) -> dict:
        """Record an evaluable-but-broken page as a successful zero-score run.

        This is for candidate implementation failures such as a JS main-thread
        deadlock during page initialization. The evaluator completed its
        judgement, so the outer runner should not treat it as infrastructure
        failure or retry it forever.
        """
        initial_state = icg.get("initial_state_id", "S0")
        metrics_3d = compute_3d_metrics(
            icg=icg,
            transition_results=[],
            initial_state_reached=False,
        )
        req_metrics = compute_req_metrics(
            icg=icg,
            transition_results=[],
        )
        ti_metrics = compute_ti_metrics(icg=icg, transition_results=[])
        total_elapsed = round(_time.time() - _eval_t0, 1)

        zero_report = {
            **report,
            "status": "success",
            "quality_status": "page_init_failed",
            "zero_score_reason": reason,
            "metrics": metrics_3d,
            "req_metrics": req_metrics,
            "ti_metrics": ti_metrics or None,
            "initial_state_check": {
                "state_id": initial_state,
                "status": "FAIL",
                "conditions": [],
                "results": [],
                "fail_reason": reason,
            },
            "transition_results": [],
            "api_call_log_dir": str(api_log_dir),
            "elapsed_s": total_elapsed,
        }
        report_path.write_text(
            json.dumps(zero_report, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\n▶ Page initialization failed; recording zero-score quality result")
        print(f"  Reason: {reason}")
        print(f"\n{format_agent_metrics_summary(metrics_3d, req_metrics)}\n")
        print(f"  Time  — {total_elapsed}s ({total_elapsed/60:.1f}min)")
        print(f"\n  ✎  Report:   {report_path}")
        print(f"  📋  API logs: {api_log_dir}")
        print(f"  📷  Screenshots: {ss_dir}\n")
        return zero_report

    async with async_playwright() as pw:
        browser = await asyncio.wait_for(
            pw.chromium.launch(headless=cfg.BROWSER_HEADLESS),
            timeout=DEFAULT_PAGE_INIT_TIMEOUT_S,
        )
        file_url = f"file://{Path(html_path).resolve()}"

        def _consume_task_result(task: asyncio.Task) -> None:
            try:
                task.result()
            except Exception:
                pass

        async def _safe_close_context(ctx, *, timeout_s: float = 5.0) -> bool:
            if ctx is None:
                return True
            close_task = asyncio.create_task(ctx.close())
            try:
                done, _pending = await asyncio.wait({close_task}, timeout=timeout_s)
                if close_task not in done:
                    close_task.add_done_callback(_consume_task_result)
                    print(f"  ⚠  context.close() exceeded {timeout_s:g}s; abandoning stale context")
                    return False
                await close_task
                return True
            except Exception as exc:
                print(f"  ⚠  context.close() did not finish cleanly: {type(exc).__name__}: {str(exc)[:120]}")
                return False

        async def _safe_close_browser(br, *, timeout_s: float = 5.0) -> bool:
            if br is None:
                return True
            close_task = asyncio.create_task(br.close())
            try:
                done, _pending = await asyncio.wait({close_task}, timeout=timeout_s)
                if close_task not in done:
                    close_task.add_done_callback(_consume_task_result)
                    print(f"  ⚠  browser.close() exceeded {timeout_s:g}s; abandoning stale browser")
                    return False
                await close_task
                return True
            except Exception as exc:
                print(f"  ⚠  browser.close() did not finish cleanly: {type(exc).__name__}: {str(exc)[:120]}")
                return False

        async def _restart_browser() -> None:
            nonlocal browser
            await _safe_close_browser(browser)
            browser = await asyncio.wait_for(
                pw.chromium.launch(headless=cfg.BROWSER_HEADLESS),
                timeout=DEFAULT_PAGE_INIT_TIMEOUT_S,
            )

        _EVENT_TRACKER_SCRIPT = """
        (() => {
            const tracked = new Set(['click','mousedown','mouseup','dblclick','contextmenu','pointerdown','pointerup']);
            const orig = EventTarget.prototype.addEventListener;
            EventTarget.prototype.addEventListener = function(type, ...args) {
                if (tracked.has(type) && this instanceof HTMLElement) {
                    if (!this.__trackedEvents) this.__trackedEvents = new Set();
                    this.__trackedEvents.add(type);
                }
                return orig.call(this, type, ...args);
            };
        })();
        """

        _DETERMINISTIC_RANDOM_SCRIPT = """
        (() => {
            if (window.__WEB_INTER_BENCH_DETERMINISTIC_RANDOM__) return;
            window.__WEB_INTER_BENCH_DETERMINISTIC_RANDOM__ = true;
            let seed = 123456789;
            Math.random = function() {
                seed = (1664525 * seed + 1013904223) >>> 0;
                return seed / 4294967296;
            };
        })();
        """

        def _looks_like_closed_browser(exc: BaseException) -> bool:
            msg = str(exc)
            return (
                "Target page, context or browser has been closed" in msg
                or "Connection closed" in msg
                or "Target closed" in msg
                or "Browser has been closed" in msg
            )

        async def _create_fresh_page_once():
            ctx = None
            try:
                ctx = await browser.new_context(
                    viewport={"width": cfg.VIEWPORT_WIDTH, "height": cfg.VIEWPORT_HEIGHT},
                    accept_downloads=True,
                )
                # Keep model-generated pages deterministic across fresh-context replay.
                # Some pages generate element ids with Math.random(); if those ids change
                # after reset, replayed selectors like "#el-..." become invalid.
                if os.environ.get("EVAL_DETERMINISTIC_RANDOM", "1").lower() not in {"0", "false", "no"}:
                    await ctx.add_init_script(_DETERMINISTIC_RANDOM_SCRIPT)
                await ctx.add_init_script(_EVENT_TRACKER_SCRIPT)
                pg = await ctx.new_page()
                # Eval only needs the DOM to be usable. Waiting for the full
                # `load` event can hang on model pages with broken subresources,
                # so prefer DOMContentLoaded and fall back to a committed
                # document plus readiness check.
                try:
                    await pg.goto(
                        file_url,
                        wait_until="domcontentloaded",
                        timeout=cfg.PAGE_LOAD_TIMEOUT_MS,
                    )
                except PlaywrightTimeoutError:
                    try:
                        await pg.goto(
                            file_url,
                            wait_until="commit",
                            timeout=cfg.PAGE_LOAD_TIMEOUT_MS,
                        )
                    except PlaywrightTimeoutError:
                        pass
                    try:
                        ready = await pg.evaluate("() => document.readyState")
                    except Exception:
                        ready = "unknown"
                    if ready not in ("interactive", "complete"):
                        raise
                    print(f"  ⚠  page load timed out; DOM is '{ready}' — proceeding")
                # networkidle is best-effort: broken subresources will also prevent it.
                try:
                    await pg.wait_for_load_state("networkidle",
                                                 timeout=cfg.PAGE_LOAD_TIMEOUT_MS)
                except PlaywrightTimeoutError:
                    pass
                try:
                    await pg.wait_for_function(
                        """() => Array.from(document.images || []).every(img =>
                            img.complete && img.naturalWidth > 0
                        )""",
                        timeout=cfg.PAGE_LOAD_TIMEOUT_MS,
                    )
                except PlaywrightTimeoutError:
                    pass
                await pg.wait_for_timeout(cfg.ACTION_SETTLE_MS)
                return ctx, pg
            except BaseException:
                if ctx is not None:
                    await _safe_close_context(ctx)
                raise

        async def _open_fresh_page():
            nonlocal browser
            attempts = max(1, int(os.environ.get("EVAL_PAGE_OPEN_RETRIES", "3")))
            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    return await asyncio.wait_for(
                        _create_fresh_page_once(),
                        timeout=DEFAULT_PAGE_INIT_TIMEOUT_S,
                    )
                except (asyncio.TimeoutError, PlaywrightTimeoutError) as exc:
                    last_exc = exc
                    if attempt >= attempts:
                        raise
                    print(
                        f"  ⚠  fresh page open timed out "
                        f"(attempt {attempt}/{attempts}); retrying"
                    )
                    continue
                except Exception as exc:
                    last_exc = exc
                    if not _looks_like_closed_browser(exc):
                        raise
                    print(f"  ⚠  browser connection lost while opening page; relaunching Chromium")
                    await _restart_browser()
                    if attempt >= attempts:
                        raise
                    continue
            raise last_exc  # defensive; the loop should always return or raise.

        async def _safe_screenshot(pg, full_page=True, reset_scroll=True):
            """Screenshot helper. Optionally scrolls to top first so that
            sticky/fixed headers render at their natural position in
            full_page captures (Playwright's full_page mode otherwise renders
            sticky elements at the current scroll offset)."""
            if reset_scroll:
                try:
                    await pg.evaluate("window.scrollTo(0, 0)")
                    await pg.wait_for_timeout(100)
                except Exception:
                    pass
            return await pg.screenshot(full_page=full_page)

        async def _open_and_capture_initial():
            ctx, pg = await _open_fresh_page()
            ss = await _safe_screenshot(pg, full_page=True)
            return ctx, pg, ss

        try:
            context, page, init_ss = await asyncio.wait_for(
                _open_and_capture_initial(),
                timeout=DEFAULT_PAGE_INIT_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            reason = (
                f"Page initialization exceeded {DEFAULT_PAGE_INIT_TIMEOUT_S}s "
                "(navigation/load waits/initial screenshot). The page may have "
                "a blocking script or hung resource."
            )
            await _safe_close_browser(browser)
            return _write_zero_quality_report(reason)
        except PlaywrightTimeoutError as exc:
            reason = f"Page initialization failed: {str(exc)[:500]}"
            await _safe_close_browser(browser)
            return _write_zero_quality_report(reason)
        (ss_dir / "S0_initial.png").write_bytes(init_ss)

        initial_state = icg.get("initial_state_id", "S0")
        initial_state_reached = False
        current_state = initial_state
        reached_states: set[str] = set()
        state_path: dict[str, list[str]] = {initial_state: []}
        # tr_id -> replay steps from the passing agent trajectory.
        # If the agent used Reset inside a transition, only steps after the last
        # successful Reset are replayable. Some concrete actions may be marked
        # allow_failure=True because the original action failed at executor level
        # but the transition later passed after relying on its side effect.
        executed_turns: dict[str, list[dict]] = {}
        fatal_page_block_reason: str | None = None
        fatal_eval_error_reason: str | None = None
        if operation_record is not None:
            operation_record["initial_state"] = initial_state
            operation_record["state_paths"] = {initial_state: []}
            _save_operation_record()

        def _navigation_timeout_s(target_state: str) -> float:
            """Budget for reset + fresh page open + replay into a prior state."""
            replay_ids = state_path.get(target_state, [])
            per_action = (
                action_timeout_s
                if action_timeout_s and action_timeout_s > 0
                else DEFAULT_ACTION_TIMEOUT_S
            )
            return float(
                DEFAULT_PAGE_INIT_TIMEOUT_S
                + 10
                + max(1, len(replay_ids)) * (per_action + 5)
            )

        async def _navigate_to_with_watchdog(target_state: str) -> tuple[bool, bool]:
            """Run navigation/replay with a hard asyncio watchdog.

            `asyncio.wait_for()` is not sufficient for Playwright hangs because
            it waits for cancellation cleanup. Some bad HTML can wedge the
            browser/driver deeply enough that cancellation itself never returns.
            Here we stop waiting once the budget expires, cancel in the
            background, and let the caller record the transition immediately.

            Returns (replay_ok, abandoned). If abandoned is True, the underlying
            Playwright task may still be unwinding, so the current eval process
            must not keep using the same browser/page state.
            """
            timeout_s = _navigation_timeout_s(target_state)
            nav_task = asyncio.create_task(_navigate_to(target_state))
            done, _pending = await asyncio.wait({nav_task}, timeout=timeout_s)
            if nav_task in done:
                return await nav_task, False
            nav_task.cancel()
            nav_task.add_done_callback(_consume_task_result)
            return False, True

        def _classify_agent_block(agent_result) -> str:
            """Classify non-DONE agent exits without conflating page quality
            failures with evaluator/API failures."""
            status = str(getattr(agent_result, "status", "") or "").upper()
            err = str(getattr(agent_result, "error", "") or "")
            low = err.lower()
            trajectory = list(getattr(agent_result, "trajectory", []) or [])
            last_turn = trajectory[-1] if trajectory else None
            last_exec_err = str(getattr(last_turn, "exec_err", "") or "").lower() if last_turn else ""
            last_action_timed_out = (
                "timed out" in last_exec_err
                or "timeout" in last_exec_err
            )
            page_closed_after_action = (
                low.startswith("observation failed:")
                and (
                    "target page, context or browser has been closed" in low
                    or "target closed" in low
                    or "connection closed" in low
                    or "browser has been closed" in low
                )
            )
            if last_action_timed_out and page_closed_after_action:
                return "page_runtime_hang"
            if status == "MAX_ITER":
                return "agent_max_iter"
            if status == "PARSE_FAIL":
                return "agent_parse_fail"
            if status == "ERROR":
                if any(marker in low for marker in ("llm", "api", "badrequest", "authentication")):
                    return "agent_llm_error"
                if low.startswith("observation failed:"):
                    return "agent_observation_error"
            return f"agent_{status.lower() or 'error'}"

        def _record_state_path(state_id: str, path_ids: list[str]) -> None:
            if state_id == initial_state:
                return
            prev = state_path.get(state_id)
            if prev is None or len(path_ids) < len(prev):
                state_path[state_id] = list(path_ids)

        def _turn_verb(turn) -> str:
            parsed = getattr(turn, "parsed_action", None) or {}
            return str(parsed.get("verb") or "").lower()

        def _extract_replay_steps(trajectory: list) -> tuple[list[dict], int | None, bool]:
            """Return concrete replay steps from the last successful Reset onward.

            A PASS transition can include a failed concrete action that still had
            a page side effect, such as a repeated click that opens a modal and
            then times out. Keep those steps and tolerate their failure during
            replay so the side effect can be recreated.
            """
            last_reset_idx: int | None = None
            for i, turn in enumerate(trajectory):
                if (
                    getattr(turn, "executed", False)
                    and getattr(turn, "exec_ok", False)
                    and _turn_verb(turn) == "reset"
                ):
                    last_reset_idx = i

            start = 0 if last_reset_idx is None else last_reset_idx + 1
            steps: list[dict] = []
            has_tolerated_failure = False
            for turn in trajectory[start:]:
                action_dict = getattr(turn, "action_dict", None)
                if not getattr(turn, "executed", False) or not action_dict:
                    continue
                source_exec_ok = bool(getattr(turn, "exec_ok", False))
                if not source_exec_ok:
                    has_tolerated_failure = True
                steps.append({
                    "iteration": getattr(turn, "iteration", None),
                    "action_text": getattr(turn, "action_text", ""),
                    "action_dict": action_dict,
                    "selector": getattr(turn, "selector", None),
                    "source_exec_ok": source_exec_ok,
                    "allow_failure": not source_exec_ok,
                })
            return steps, last_reset_idx, has_tolerated_failure

        async def _wait_replay_transients(pg) -> None:
            """Replay uses recorded absolute selectors, so transient body-level
            UI like toasts can shift XPath positions. Transients are not stable
            states, so wait them out before replaying the next recorded action.
            """
            try:
                has_toast = await pg.evaluate("() => !!document.querySelector('.toast')")
                if has_toast:
                    await pg.wait_for_function(
                        "() => !document.querySelector('.toast')",
                        timeout=3000,
                    )
            except Exception:
                pass

        async def _replay_agent_turns(pg, turns: list[dict]) -> bool:
            """Re-execute the concrete action_dicts from a previously-passing
            agent trajectory. Unlike the fresh agent run, we trust the recorded
            selectors — no LLM calls, no re-planning."""
            replay_action_timeout = (
                action_timeout_s if action_timeout_s and action_timeout_s > 0
                else DEFAULT_ACTION_TIMEOUT_S
            )
            for r in turns:
                action = r.get("action_dict") or {}
                sel = r.get("selector")
                atype = action.get("type", "")
                if not atype:
                    continue
                aff_stub = {"type": "Element"}  # executor only uses `type` for click fallback
                try:
                    await _wait_replay_transients(pg)
                    ok, err = await asyncio.wait_for(
                        execute_action(pg, action, aff_stub, sel),
                        timeout=replay_action_timeout + 2,
                    )
                    if not ok:
                        if r.get("allow_failure"):
                            print(f"       replay tolerated failed {atype}: {err}")
                            await pg.wait_for_timeout(300)
                            continue
                        print(f"       replay failed on {atype}: {err}")
                        return False
                except asyncio.TimeoutError:
                    msg = f"replay action timed out after {replay_action_timeout}s"
                    if r.get("allow_failure"):
                        print(f"       replay tolerated timeout {atype}: {msg}")
                        try:
                            await pg.wait_for_timeout(300)
                        except Exception:
                            pass
                        continue
                    print(f"       replay timed out on {atype}: {msg}")
                    return False
                except Exception as e:
                    if r.get("allow_failure"):
                        print(f"       replay tolerated raised {atype}: {e}")
                        await pg.wait_for_timeout(300)
                        continue
                    print(f"       replay raised on {atype}: {e}")
                    return False
                await pg.wait_for_timeout(300)
            return True

        async def _navigate_to(target_state):
            nonlocal browser, context, page, current_state
            print(f"   ↺  Resetting to {initial_state}", end="")
            closed = await _safe_close_context(context)
            if not closed:
                print(" — relaunching browser", end="")
                await _restart_browser()
            context, page = await _open_fresh_page()
            current_state = initial_state
            replay_ids = state_path.get(target_state, [])
            if replay_ids:
                print(f" → replaying {len(replay_ids)} transition(s) to {target_state}")
            else:
                print()
            for rid in replay_ids:
                turns = executed_turns.get(rid, [])
                ok = await _replay_agent_turns(page, turns)
                if not ok:
                    current_state = None
                    return False
            await _wait_replay_transients(page)
            current_state = target_state
            return True

        # ── Preconditions (blocking: FAIL → S0 unreachable, all transitions SKIPPED) ──
        first_tr = icg["transitions"][0] if icg.get("transitions") else None
        raw_preconditions = (first_tr.get("preconditions") or []) if first_tr else []

        if raw_preconditions:
            precond_texts = _assertion_texts(raw_preconditions)
            init_ss_ref = str((ss_dir / "S0_initial.png").relative_to(run_dir))

            precond_results, precond_log = await score_preconditions(
                client=client,
                screenshot=init_ss,
                conditions=precond_texts,
                screenshot_ref=init_ss_ref,
            )
            _attach_req_tags(precond_results, raw_preconditions)
            _persist_api_log_entry({
                "phase": "initial_precondition",
                "transition_id": first_tr["id"],
                **precond_log,
            })

            precond_pass = all(r["passed"] for r in precond_results)
            report["initial_state_check"] = {
                "state_id": initial_state,
                "transition_id": first_tr["id"],
                "status": "PASS" if precond_pass else "FAIL",
                "screenshot": init_ss_ref,
                "conditions": precond_texts,
                "results": precond_results,
                "scorer_call_log": precond_log,
            }
            _save_report()

            if precond_pass:
                initial_state_reached = True
                reached_states.add(initial_state)

            print("\n▶ Checking initial state (S0)")
            for r in precond_results:
                icon = "✓" if r["passed"] else ("?" if r["verdict"] == "UNCERTAIN" else "✗")
                print(f"   {icon}  [{r['verdict']}]  {r['condition']}")
            print(f"   → {'PASS' if precond_pass else 'FAIL'}{'' if precond_pass else ' (S0 unreachable — all transitions will be SKIPPED)'}")
        else:
            initial_state_reached = True
            reached_states.add(initial_state)
            report["initial_state_check"] = {
                "state_id": initial_state,
                "transition_id": first_tr["id"] if first_tr else None,
                "status": "NOT_EVALUATED",
                "screenshot": str((ss_dir / "S0_initial.png").relative_to(run_dir)),
                "conditions": [],
                "results": [],
                "scorer_call_log": None,
            }

        if preconditions_only:
            await _safe_close_context(context)
            await _safe_close_browser(browser)

            report["mode"] = "preconditions_only"
            report["metrics"] = {
                "initial_state": {
                    "status": (report.get("initial_state_check") or {}).get("status"),
                    "passed": initial_state_reached,
                }
            }
            report["req_metrics"] = {}
            report["ti_metrics"] = {}
            report["elapsed_s"] = round(_time.time() - _eval_t0, 1)
            report_path.write_text(
                json.dumps(report, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            print("\n▶ Precondition-only mode — skipping transitions")
            print(f"  Time  — {report['elapsed_s']}s ({report['elapsed_s']/60:.1f}min)")
            print(f"\n  ✎  Report:   {report_path}")
            print(f"  📋  API logs: {api_log_dir}")
            print(f"  📷  Screenshots: {ss_dir}\n")
            return report

        # ── Transitions ──────────────────────────────────────────────────
        print("\n▶ Executing transitions")

        for tr in icg["transitions"]:
          # Skip transitions not in the filter set (if specified).
          if filter_transitions and tr["id"] not in filter_transitions:
              continue
          try:
            import time as _time
            _tr_t0 = _time.time()
            intent = tr.get("agent_task", "")
            _full_page = tr.get("screenshot_mode") != "viewport"
            _reset_scroll = _full_page

            t_result: dict = {
                "transition_id":         tr["id"],
                "from_state":            tr["from"],
                "to_state":              tr["to"],
                "agent_task":         intent,
                "screenshot_mode":       tr.get("screenshot_mode", "full"),
                "status":                None,
                "fail_reason":           None,
                "agent_status":          None,
                "agent_iterations":      0,
                "agent_trajectory":      [],
                "action_success":        False,
                "dom_assertion_results": [],
                "postcondition_results": [],
                "screenshots":           {},
                "error":                 None,
                "restore_method":        None,
                "dom_assertion_log":     None,
                "dom_assertion_log_summary": None,
                "dom_assertion_call_log": None,
                "scorer_call_log":       None,
            }

            print(f"\n   {tr['id']}  {tr['from']} → {tr['to']}")
            print(f"   Intent: {intent[:120]}")

            if fatal_eval_error_reason:
                t_result["status"] = "SKIPPED"
                t_result["fail_reason"] = "fatal_eval_error"
                t_result["error"] = (
                    "Skipped after prior fatal evaluator/API error: "
                    f"{fatal_eval_error_reason}"
                )
                print("   ⊘  SKIPPED — prior fatal evaluator/API error")
                report["transition_results"].append(t_result)
                _save_report()
                continue

            if fatal_page_block_reason:
                t_result["status"] = "SKIPPED"
                t_result["fail_reason"] = "state_unreachable"
                t_result["error"] = (
                    "Skipped after prior page_runtime_hang: "
                    f"{fatal_page_block_reason}"
                )
                print("   ⊘  SKIPPED — page/browser already unusable after prior runtime hang")
                report["transition_results"].append(t_result)
                _save_report()
                continue

            # ── Skip all transitions when S0 preconditions failed ──────────
            if not initial_state_reached:
                t_result["status"] = "SKIPPED"
                t_result["fail_reason"] = "state_unreachable"
                t_result["error"] = f"Initial state '{initial_state}' preconditions failed"
                print(f"   ⊘  SKIPPED (initial state preconditions failed)")
                report["transition_results"].append(t_result)
                _save_report()
                continue

            # ── Navigate to tr.from (replay agent trajectories of prior chains) ──
            if tr["from"] != current_state:
                if tr["from"] not in reached_states:
                    t_result["status"] = "SKIPPED"
                    t_result["fail_reason"] = "state_unreachable"
                    t_result["error"] = f"State '{tr['from']}' never reached"
                    print(f"   ⊘  SKIPPED (state '{tr['from']}' never reached)")
                    report["transition_results"].append(t_result)
                    _save_report()
                    continue
                t_result["restore_method"] = "fresh_context_replay"
                nav_timeout_s = _navigation_timeout_s(tr["from"])
                replay_ok, nav_abandoned = await _navigate_to_with_watchdog(tr["from"])
                if nav_abandoned:
                    t_result["status"] = "BLOCKED"
                    t_result["fail_reason"] = "page_runtime_hang"
                    hang_error = (
                        f"Navigation/replay to '{tr['from']}' exceeded "
                        f"{nav_timeout_s:.0f}s and the Playwright operation "
                        "did not cancel cleanly"
                    )
                    t_result["error"] = hang_error
                    fatal_page_block_reason = hang_error
                    current_state = None
                    print(
                        f"   ⊗  BLOCKED — navigation/replay to {tr['from']} "
                        f"timed out after {nav_timeout_s:.0f}s; abandoning stuck Playwright task"
                    )
                    context = None
                    page = None
                    report["transition_results"].append(t_result)
                    _save_report()
                    continue
                if not replay_ok:
                    t_result["status"] = "BLOCKED"
                    t_result["fail_reason"] = "replay_failed"
                    t_result["error"] = f"Replay to '{tr['from']}' failed"
                    print(f"   ⊗  BLOCKED — replay to {tr['from']} failed")
                    report["transition_results"].append(t_result)
                    _save_report()
                    continue
            else:
                t_result["restore_method"] = "none"

            # ── Pre-action screenshot ────────────────────────────────────────
            pre_path = ss_dir / f"{tr['id']}_pre.png"
            ss_ref = str(pre_path.relative_to(run_dir))
            pre_ss = await _safe_screenshot(page, full_page=_full_page, reset_scroll=_reset_scroll)
            pre_path.write_bytes(pre_ss)
            t_result["screenshots"]["pre"] = ss_ref

            # ── Start page-level DOM monitor (target=None, observed=None) ───
            raw_dom_assertions = tr.get("dom_assertions", [])
            raw_postconditions = tr.get("postconditions", [])
            monitor_started = False
            if raw_dom_assertions:
                await start_dom_monitor(page, target_selector=None, observed_selectors=None)
                monitor_started = True

            # ── Refresh callbacks: save DOM state before reload, re-inject after ──
            refresh_before_snap = None
            refresh_pre_dom_log = None
            refresh_occurred = False

            async def _before_refresh():
                nonlocal refresh_before_snap, refresh_pre_dom_log, refresh_occurred, monitor_started
                refresh_occurred = True
                if monitor_started:
                    refresh_before_snap = await capture_snapshot(page, None, None)
                    try:
                        refresh_pre_dom_log = await finish_dom_monitor(page, 0)
                    except Exception:
                        refresh_pre_dom_log = None
                    monitor_started = False

            async def _after_refresh():
                nonlocal monitor_started
                if raw_dom_assertions:
                    await start_dom_monitor(page, target_selector=None, observed_selectors=None)
                    monitor_started = True

            # ── Reset callback: replay to from_state, restart DOM monitor ──
            async def _reset_cb():
                nonlocal browser, context, page, monitor_started, refresh_occurred
                nonlocal refresh_before_snap, refresh_pre_dom_log
                if monitor_started:
                    try:
                        await finish_dom_monitor(page, 0)
                    except Exception:
                        pass
                    monitor_started = False
                refresh_before_snap = None
                refresh_pre_dom_log = None
                refresh_occurred = False
                closed = await _safe_close_context(context)
                if not closed:
                    await _restart_browser()
                context, page = await _open_fresh_page()
                replay_ids = state_path.get(tr["from"], [])
                for rid in replay_ids:
                    turns = executed_turns.get(rid, [])
                    ok = await _replay_agent_turns(page, turns)
                    if not ok:
                        raise RuntimeError(f"Replay to '{tr['from']}' failed during reset")
                if raw_dom_assertions:
                    await start_dom_monitor(page, target_selector=None, observed_selectors=None)
                    monitor_started = True
                print(f"   ↺  Reset: page restored to {tr['from']}")
                return page

            # ── Run agent loop for this transition ────────────────────────
            agent_log_dir = run_dir / "agent_logs"
            try:
                agent_result = await run_agent_transition(
                    page=page,
                    transition=tr,
                    client=client,
                    model=cfg.MODEL_AGENT,
                    max_iter=max_iter,
                    agent_timeout_s=agent_timeout_s,
                    action_timeout_s=action_timeout_s,
                    log_dir=agent_log_dir,
                    save_screenshots=True,
                    before_refresh_cb=_before_refresh,
                    after_refresh_cb=_after_refresh,
                    reset_cb=_reset_cb,
                )
            except Exception as e:
                # Flush monitor if it was open
                if monitor_started:
                    try:
                        await finish_dom_monitor(page, 0)
                    except Exception:
                        pass
                t_result["status"] = "ERROR"
                t_result["fail_reason"] = "agent_exception"
                t_result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"   ⊗  ERROR — agent crashed: {e}")
                report["transition_results"].append(t_result)
                _save_report()
                current_state = None
                continue

            t_result["agent_status"] = agent_result.status
            t_result["agent_iterations"] = agent_result.iterations
            t_result["agent_trajectory"] = [
                {"iter": t.iteration, "verb": (t.parsed_action or {}).get("verb"),
                 "selector": t.selector, "exec_ok": t.exec_ok,
                 "action_text": t.action_text[:120] if t.action_text else ""}
                for t in agent_result.trajectory
            ]
            _persist_api_log_entry({
                "phase": "agent", "transition_id": tr["id"],
                "status": agent_result.status, "iterations": agent_result.iterations,
                **agent_result.total_usage,
            })

            print(f"   ⏵  agent: {agent_result.status}  iters={agent_result.iterations}")
            for turn in agent_result.trajectory:
                mark = "✓" if turn.exec_ok else "✗"
                print(f"      [{turn.iteration}] {mark} {turn.action_text[:100]}")

            # Agent failure blocks scoring: the page may already be closed,
            # hung, or otherwise unreliable. Record it as BLOCKED before any
            # post-action settle/DOM-monitor call can turn it into an evaluator
            # infrastructure error.
            if agent_result.status != "DONE":
                t_result["status"] = "BLOCKED"
                t_result["fail_reason"] = _classify_agent_block(agent_result)
                t_result["error"] = agent_result.error or f"agent returned {agent_result.status}"
                print(f"   ⊗  BLOCKED — agent {agent_result.status} [{t_result['fail_reason']}]")
                # Do not run DOM-monitor settle or post screenshots here. For
                # page-runtime hangs the page may already be wedged; extra waits
                # turn a candidate HTML quality problem into evaluator noise.
                monitor_started = False
                closed = await _safe_close_context(context)
                context = None
                page = None
                if not closed:
                    try:
                        await _restart_browser()
                    except Exception as exc:
                        if t_result["fail_reason"] == "page_runtime_hang":
                            fatal_page_block_reason = (
                                "Browser/driver could not be recovered after "
                                f"{tr['id']} page_runtime_hang: "
                                f"{type(exc).__name__}: {str(exc)[:300]}"
                            )
                            t_result["browser_restart_after_blocked_error"] = fatal_page_block_reason
                        else:
                            raise
                if t_result["fail_reason"] == "agent_llm_error":
                    fatal_eval_error_reason = f"{tr['id']}: {t_result['error']}"
                    report["status"] = "failed"
                    report["fail_reason"] = "agent_llm_error"
                    report["error"] = fatal_eval_error_reason
                    print("   ⊘  Fatal eval/API error — skipping remaining transitions")
                report["transition_results"].append(t_result)
                _save_report()
                current_state = None
                continue

            # Settle before closing the monitor (so auto-dismissing toasts land in the log).
            if monitor_started:
                post_dom_log = await finish_dom_monitor(page, dom_settle_ms)
            else:
                post_dom_log = None

            if refresh_occurred and refresh_before_snap is not None:
                refresh_after_snap = await capture_snapshot(page, None, None)
                synth = synthesize_evidence(refresh_before_snap, refresh_after_snap, None)
                if refresh_pre_dom_log is not None:
                    synth["initial_snapshot"] = refresh_pre_dom_log.get("initial_snapshot", synth["initial_snapshot"])
                    synth["events"] = refresh_pre_dom_log.get("events", [])
                if post_dom_log is not None and post_dom_log.get("events"):
                    synth["events"] = synth.get("events", []) + post_dom_log["events"]
                dom_log = synth
                print(f"   ↻  Refresh detected — merged pre-refresh events ({len(refresh_pre_dom_log.get('events', []) if refresh_pre_dom_log else [])}) + synthesized final snapshot")
            elif post_dom_log is not None:
                dom_log = post_dom_log
            else:
                dom_log = None

            if dom_log is not None:
                t_result["dom_assertion_log"] = dom_log
                t_result["dom_assertion_log_summary"] = summarize_dom_log(dom_log)

            t_result["action_success"] = True

            # ── Post-action screenshot (right after DOM settle, before scoring) ──
            post_ss = await _safe_screenshot(page, full_page=_full_page, reset_scroll=_reset_scroll)
            post_path = ss_dir / f"{tr['id']}_post.png"
            post_path.write_bytes(post_ss)
            t_result["screenshots"]["post"] = str(post_path.relative_to(run_dir))

            # ── Score dom_assertions + postconditions in parallel ──
            dom_results: list[dict] = []
            cond_results: list[dict] = []

            async def _score_dom():
                if not (raw_dom_assertions and dom_log is not None):
                    return [], None
                dom_texts = _assertion_texts(raw_dom_assertions)
                action_stub = {"type": "compound_intent", "value": intent}
                results, log = await score_dom_assertions(
                    client=client,
                    dom_log=dom_log,
                    conditions=dom_texts,
                    action=action_stub,
                    affordance_name="agent-driven transition",
                )
                _attach_req_tags(results, raw_dom_assertions)
                return results, log

            async def _score_vlm():
                post_texts = _assertion_texts(raw_postconditions)
                if not post_texts:
                    return [], None
                results, log = await score_postconditions(
                    client=client,
                    pre_screenshot=pre_ss,
                    post_screenshot=post_ss,
                    conditions=post_texts,
                    action={},
                    affordance_name="",
                    action_desc=intent,
                    pre_screenshot_ref=ss_ref,
                    post_screenshot_ref=str(post_path.relative_to(run_dir)),
                )
                _attach_req_tags(results, raw_postconditions)
                return results, log

            (dom_results, dom_score_log), (cond_results, vlm_score_log) = await asyncio.gather(
                _score_dom(), _score_vlm()
            )

            if dom_score_log is not None:
                t_result["dom_assertion_call_log"] = dom_score_log
                _persist_api_log_entry({"phase": "dom_assert", "transition_id": tr["id"], **dom_score_log})
            t_result["dom_assertion_results"] = dom_results
            for r in dom_results:
                icon = "✓" if r["passed"] else ("?" if r.get("verdict") == "UNCERTAIN" else "✗")
                print(f"   {icon}  [DOM]  {r['condition']}")

            if vlm_score_log is not None:
                t_result["scorer_call_log"] = vlm_score_log
                _persist_api_log_entry({"phase": "score", "transition_id": tr["id"], **vlm_score_log})
            t_result["postcondition_results"] = cond_results

            dom_pass = all(r["passed"] for r in dom_results)
            visual_pass = all(r["passed"] for r in cond_results)
            all_pass = dom_pass and visual_pass
            t_result["status"] = "PASS" if all_pass else "FAIL"
            if not all_pass:
                if not dom_pass and not visual_pass:
                    t_result["fail_reason"] = "dom_and_postcondition_failed"
                elif not dom_pass:
                    t_result["fail_reason"] = "dom_assertion_failed"
                else:
                    t_result["fail_reason"] = "postcondition_failed"

            for r in cond_results:
                icon = "✓" if r["passed"] else ("?" if r["verdict"] == "UNCERTAIN" else "✗")
                print(f"   {icon}  [VLM]  {r['condition']}")
            print(f"   → {t_result['status']}")

            if t_result["status"] == "PASS":
                current_state = tr["to"]
                reached_states.add(current_state)
                _record_state_path(current_state, state_path.get(tr["from"], []) + [tr["id"]])
                # Record the passing trajectory for replay.
                replay_steps, replay_last_reset_idx, replay_has_tolerated_failure = _extract_replay_steps(
                    agent_result.trajectory
                )
                executed_turns[tr["id"]] = replay_steps
                t_result["replay_steps_count"] = len(replay_steps)
                t_result["replay_last_reset_iteration"] = (
                    None if replay_last_reset_idx is None
                    else agent_result.trajectory[replay_last_reset_idx].iteration
                )
                t_result["replay_has_tolerated_failure"] = replay_has_tolerated_failure
                if operation_record is not None:
                    operation_record["transitions"][tr["id"]] = {
                        "id": tr["id"],
                        "from": tr.get("from"),
                        "to": tr.get("to"),
                        "status": t_result["status"],
                        "agent_task": intent,
                        "steps": replay_steps,
                        "last_reset_iteration": t_result["replay_last_reset_iteration"],
                        "has_tolerated_failure": replay_has_tolerated_failure,
                    }
                    operation_record["state_paths"] = {
                        sid: list(path) for sid, path in state_path.items()
                    }
                    _save_operation_record()
            else:
                current_state = None

            t_result["elapsed_s"] = round(_time.time() - _tr_t0, 1)
            print(f"   ⏱  {t_result['elapsed_s']}s")
            report["transition_results"].append(t_result)
            _save_report()
          except Exception as exc:
            import traceback
            print(f"\n   ❌  Unexpected error in {tr['id']}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            t_result["status"] = "ERROR"
            t_result["fail_reason"] = "unexpected_exception"
            t_result["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            t_result["elapsed_s"] = round(_time.time() - _tr_t0, 1)
            report["transition_results"].append(t_result)
            _save_report()
            current_state = None

        await _safe_close_context(context)
        await _safe_close_browser(browser)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n▶ Computing metrics")

    metrics_3d = compute_3d_metrics(
        icg=icg,
        transition_results=report["transition_results"],
        initial_state_reached=initial_state_reached,
    )
    report["metrics"] = metrics_3d

    req_metrics = compute_req_metrics(
        icg=icg,
        transition_results=report["transition_results"],
    )
    report["req_metrics"] = req_metrics

    ti_metrics = compute_ti_metrics(
        icg=icg,
        transition_results=report["transition_results"],
    )
    if ti_metrics:
        report["ti_metrics"] = ti_metrics

    if "status" not in report:
        report["status"] = "success"
        report["fail_reason"] = None

    # ── Build slim report (strip heavy call logs from transitions) ─────────
    slim_transitions = [_slim_transition(tr) for tr in report["transition_results"]]

    slim_report = {
        "status":     report.get("status"),
        "fail_reason": report.get("fail_reason"),
        "error":      report.get("error"),
        "task_id":    report["task_id"],
        "task_name":  report["task_name"],
        "domain":     report["domain"],
        "scenario":   report["scenario"],
        "html_path":  report["html_path"],
        "icg_path":   report["icg_path"],
        "run_id":     report["run_id"],
        "mode":          report["mode"],
        "model_agent":   report["model_agent"],
        "model_scorer":  report["model_scorer"],
        "max_iter":      report["max_iter"],
        "dom_settle_ms": report["dom_settle_ms"],
        "agent_timeout_s": report.get("agent_timeout_s"),
        "action_timeout_s": report.get("action_timeout_s"),
        "metrics":    report["metrics"],
        "req_metrics": report["req_metrics"],
        "ti_metrics":  report.get("ti_metrics"),
        "initial_state_check": {
            k: v for k, v in (report.get("initial_state_check") or {}).items()
            if k in ("state_id", "status", "conditions", "results")
        } if report.get("initial_state_check") else None,
        "transition_results": slim_transitions,
        "api_call_log_dir": str(api_log_dir),
    }

    report_path.write_text(
        json.dumps(slim_report, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Print ─────────────────────────────────────────────────────────────
    print(f"\n{format_agent_metrics_summary(metrics_3d, req_metrics)}\n")

    print("  Transition breakdown:")
    for r in report["transition_results"]:
        reason = r.get("fail_reason")
        tag = f"  [{reason}]" if reason else ""
        print(f"   {_status_icon(r['status'])}  {r['transition_id']}  {r['status']}{tag}")

    print("\n  Requirement breakdown:")
    for r in req_metrics["per_req"]:
        icon = "✓" if r["passed"] else "✗"
        tag = "E" if r["intent"] == "explicit" else "I"
        n_p = sum(1 for a in r["assertions"] if a["passed"])
        print(f"   {icon}  [{tag}]  {r['washed_req_id']}  ({n_p}/{len(r['assertions'])} assertions)")

    total_elapsed = round(_time.time() - _eval_t0, 1)
    report["elapsed_s"] = total_elapsed

    print(f"  Time  — {total_elapsed}s ({total_elapsed/60:.1f}min)")

    print(f"\n  ✎  Report:   {report_path}")
    print(f"  📋  API logs: {api_log_dir}")
    print(f"  📷  Screenshots: {ss_dir}\n")

    return report


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WebRISE agent-mode evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python eval_agentmode.py --html page.html --icg icg.json
  python eval_agentmode.py --html page.html --icg icg.json --output ./results
  python eval_agentmode.py --html page.html --icg icg.json --max-iter 12 --dom-settle-ms 1500
""",
    )
    parser.add_argument("--html",           required=True)
    parser.add_argument("--icg",            required=True)
    parser.add_argument("--output",         default="./2026-04-results-agent")
    parser.add_argument("--api-key",        default=None)
    parser.add_argument("--base-url",       default=None)
    parser.add_argument("--max-iter",       type=int, default=DEFAULT_MAX_ITER,
                        help=f"Max agent turns per transition (default: {DEFAULT_MAX_ITER})")
    parser.add_argument("--dom-settle-ms",  type=int, default=DEFAULT_DOM_SETTLE_MS,
                        help=f"Wait ms after agent DONE before closing DOM monitor (default: {DEFAULT_DOM_SETTLE_MS})")
    parser.add_argument("--agent-timeout-s", type=int, default=DEFAULT_AGENT_TIMEOUT_S,
                        help=f"Timeout seconds for one agent LLM call (default: {DEFAULT_AGENT_TIMEOUT_S}; 0 disables)")
    parser.add_argument("--action-timeout-s", type=int, default=DEFAULT_ACTION_TIMEOUT_S,
                        help=f"Timeout seconds for one browser action (default: {DEFAULT_ACTION_TIMEOUT_S}; 0 disables)")
    parser.add_argument("--run-id",         default=None,
                        help="Pre-set run ID (timestamp string). If omitted, generated automatically.")
    parser.add_argument("--transitions",   default=None,
                        help="Comma-separated transition IDs to run (e.g. T18,T19). "
                             "Only transitions starting from S0 can be run standalone. "
                             "If omitted, all transitions are run.")
    parser.add_argument("--record-operations", action="store_true",
                        help="Write operation_record.json with clean replayable PASS-transition actions.")
    parser.add_argument("--preconditions-only", action="store_true",
                        help="Only evaluate initial-state preconditions, then write report.json and exit.")

    args = parser.parse_args()

    cfg_key, cfg_url = cfg.get_credentials()
    api_key  = args.api_key  or cfg_key
    base_url = args.base_url or cfg_url

    if not Path(args.html).exists():
        sys.exit(f"Error: HTML not found: {args.html}")
    if not Path(args.icg).exists():
        sys.exit(f"Error: ICG not found: {args.icg}")

    filter_tids = None
    if args.transitions:
        filter_tids = {t.strip() for t in args.transitions.split(",") if t.strip()}

    try:
        asyncio.run(evaluate(
            args.html, args.icg, args.output, api_key, base_url,
            max_iter=args.max_iter,
            dom_settle_ms=args.dom_settle_ms,
            agent_timeout_s=args.agent_timeout_s,
            action_timeout_s=args.action_timeout_s,
            run_id=args.run_id,
            filter_transitions=filter_tids,
            record_operations=args.record_operations,
            preconditions_only=args.preconditions_only,
        ))
    except Exception as exc:
        tb = traceback.format_exc()
        _write_failure_report(args, exc, tb)
        print(tb, file=sys.stderr)
        sys.exit(1)


def _write_failure_report(args, exc: Exception, tb: str) -> None:
    """Emit a report.json with zero metrics + fail_reason so downstream
    summaries still have something to read. Callers exit non-zero so the
    shell still marks the run as failed."""
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Best-effort ICG load so the failure report can populate denominators.
    icg_data: dict = {}
    try:
        icg_data = load_icg(args.icg)
    except Exception:
        pass

    # Reuse the run_dir evaluate() already created (latest <task_id>_<ts>)
    # if possible; otherwise spawn a new timestamped one.
    task_id = icg_data.get("task_id") or "UNKNOWN"
    candidates = sorted(output_path.glob(f"{task_id}_*"))
    if candidates:
        run_dir = candidates[-1]
    else:
        run_dir = output_path / f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Denominators from the ICG (0 if ICG failed to load).
    n_states      = len(icg_data.get("states", []) or [])
    n_transitions = len(icg_data.get("transitions", []) or [])
    try:
        req_metrics = compute_req_metrics(
            icg=icg_data,
            transition_results=[],
        )
    except Exception:
        req_metrics = {
            "per_req": [],
            "R_explicit": {"score": 0.0, "passed_count": 0, "total": 0},
            "R_implicit": {"score": 0.0, "passed_count": 0, "total": 0},
            "R_overall": {"score": 0.0, "passed_count": 0, "total": 0},
        }
    try:
        ti_metrics = compute_ti_metrics(icg=icg_data, transition_results=[])
    except Exception:
        ti_metrics = {}
    fail_report = {
        "task_id":           task_id,
        "task_name":         icg_data.get("task_name", ""),
        "domain":            icg_data.get("domain", ""),
        "scenario":          icg_data.get("scenario", ""),
        "html_path":         str(Path(args.html).resolve()),
        "icg_path":          str(Path(args.icg).resolve()),
        "run_id":            run_dir.name.split("_", 1)[-1] if "_" in run_dir.name else "",
        "mode":              "agent",
        "status":            "failed",
        "fail_reason":       f"{type(exc).__name__}: {str(exc)[:500]}",
        "traceback":         tb,
        "metrics": {
            "state_reachability": {
                "score": 0.0, "reached": 0, "total": n_states,
                "reached_ids": [], "unreached_ids": [s.get("id") for s in icg_data.get("states", [])],
            },
            "transition_validity": {
                "score": 0.0, "passed": 0, "total": n_transitions,
                "skipped": 0, "per_transition": {},
            },
        },
        "req_metrics":       req_metrics,
        "ti_metrics":        ti_metrics or None,
        "initial_state_check": None,
        "transition_results": [],
    }

    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(fail_report, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  ❌  Wrote failure report: {report_path}")
    print(f"     fail_reason: {fail_report['fail_reason']}")


if __name__ == "__main__":
    main()
