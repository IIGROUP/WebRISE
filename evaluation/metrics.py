"""
Metrics - WebRISE evaluation metrics.

Dimensions:
  S%   State Reachability     - how many states were reached?
  T%   Transition Validity     - how many transitions passed?
  Re%  Explicit Req Coverage   - explicit requirement coverage
  Ri%  Implicit Req Coverage   - implicit requirement coverage
  R%   Overall Req Coverage    - all requirement coverage
"""

from __future__ import annotations


def compute_3d_metrics(
    icg: dict,
    transition_results: list[dict],
    initial_state_reached: bool = True,
) -> dict:
    """Compute state reachability and transition validity."""

    reached = set()
    if initial_state_reached:
        reached.add(icg.get("initial_state_id", "S0"))
    for r in transition_results:
        if r["status"] == "PASS":
            reached.add(r["to_state"])
    s_total = len(icg["states"])

    t_passed = sum(1 for r in transition_results if r["status"] == "PASS")
    t_skipped = sum(1 for r in transition_results if r["status"] == "SKIPPED")
    t_denom = len(transition_results)

    return {
        "state_reachability": {
            "score": round(len(reached) / s_total, 4) if s_total else 1.0,
            "reached": len(reached),
            "total": s_total,
            "reached_ids": sorted(reached),
            "unreached_ids": [s["id"] for s in icg["states"] if s["id"] not in reached],
        },
        "transition_validity": {
            "score": round(t_passed / t_denom, 4) if t_denom else 0.0,
            "passed": t_passed,
            "total": len(icg["transitions"]),
            "skipped": t_skipped,
            "per_transition": {r["transition_id"]: r["status"] for r in transition_results},
        },
    }


def compute_ti_metrics(
    icg: dict,
    transition_results: list[dict],
) -> dict:
    """
    Compute test-item coverage (TI%).
    Returns {} if the ICG does not carry test_items_coverage.
    """
    ti_cov = icg.get("test_items_coverage")
    if not ti_cov:
        return {}

    tr_passed = {r["transition_id"] for r in transition_results if r["status"] == "PASS"}

    covered = []
    uncovered = []
    for ti_entry in ti_cov:
        iid   = ti_entry["item_id"]
        tids  = ti_entry.get("covered_by_transitions", [])
        # An item is covered if at least one of its transitions passed.
        if any(t in tr_passed for t in tids):
            covered.append(iid)
        else:
            uncovered.append(iid)

    total = len(ti_cov)
    return {
        "score":         round(len(covered) / total, 4) if total else 1.0,
        "covered":       len(covered),
        "total":         total,
        "covered_ids":   sorted(covered),
        "uncovered_ids": sorted(uncovered),
    }


def compute_req_metrics(
    icg: dict,
    transition_results: list[dict],
) -> dict:
    """
    Compute per-requirement coverage from the released ICG mapping.

    Primary path: assertions and postconditions tagged with `primary_req_id`
    are matched to their DOM/visual assertion verdicts. If a transition has no
    assertion-level tags, the metric falls back to the transition -> test item
    -> requirement mapping in `test_items_coverage` and `requirements_coverage`.
    """
    tr_map = {r["transition_id"]: r for r in transition_results}
    return _compute_req_metrics_from_icg(icg, tr_map)


def _compute_req_metrics_from_icg(icg: dict, tr_map: dict) -> dict:
    """Compute req coverage from assertion tags and ICG coverage mappings."""
    item_req_ids: dict[str, list[str]] = {}
    for ti in icg.get("test_items_coverage", []):
        item_req_ids[ti["item_id"]] = ti.get("req_ids", [])

    req_meta: dict[str, dict] = {}
    for rc in icg.get("requirements_coverage", []):
        req_meta[rc["req_id"]] = {
            "intent":     rc.get("intent", ""),
            "content_en": rc.get("content_en", ""),
        }

    req_results: dict[str, list[dict]] = {}

    def _add(rid: str, location: str, passed: bool) -> None:
        if rid not in req_results:
            req_results[rid] = []
        req_results[rid].append({"location": location, "passed": passed})

    def _tr_item_ids(tr_def: dict) -> list[str]:
        raw = tr_def.get("test_item_ids") or tr_def.get("mapped_test_items") or []
        out: list[str] = []
        for entry in raw:
            if isinstance(entry, str):
                out.append(entry)
            elif isinstance(entry, dict) and entry.get("id"):
                out.append(entry["id"])
        return out

    def _any_assertion_tagged(tr_def: dict) -> bool:
        for field in ("dom_assertions", "postconditions", "preconditions"):
            for a in tr_def.get(field, []):
                if a.get("primary_req_id") and not a.get("primary_req_id_invalid"):
                    return True
        return False

    for tr_def in icg.get("transitions", []):
        tid    = tr_def["id"]
        result = tr_map.get(tid)
        status = result.get("status", "") if result else ""
        tr_passed = (status == "PASS")
        blocked   = status in ("SKIPPED", "BLOCKED")

        if _any_assertion_tagged(tr_def):
            dom_items    = tr_def.get("dom_assertions",   []) or []
            dom_results  = (result.get("dom_assertion_results",  []) if result else []) or []
            for idx, a in enumerate(dom_items):
                rid = a.get("primary_req_id")
                if not rid or a.get("primary_req_id_invalid"):
                    continue
                if blocked:
                    assertion_passed = False
                elif idx < len(dom_results):
                    assertion_passed = dom_results[idx].get("passed", False)
                else:
                    assertion_passed = False
                _add(rid, f"{tid}.dom_assertions[{idx}]", assertion_passed)

            post_items   = tr_def.get("postconditions",  []) or []
            post_results = (result.get("postcondition_results", []) if result else []) or []
            for idx, a in enumerate(post_items):
                rid = a.get("primary_req_id")
                if not rid or a.get("primary_req_id_invalid"):
                    continue
                if blocked:
                    assertion_passed = False
                elif idx < len(post_results):
                    assertion_passed = post_results[idx].get("passed", False)
                else:
                    assertion_passed = False
                _add(rid, f"{tid}.postconditions[{idx}]", assertion_passed)

            for idx, a in enumerate(tr_def.get("preconditions", []) or []):
                rid = a.get("primary_req_id")
                if not rid or a.get("primary_req_id_invalid"):
                    continue
                _add(rid, f"{tid}.preconditions[{idx}]", tr_passed)

        else:
            for iid in _tr_item_ids(tr_def):
                for rid in item_req_ids.get(iid, []):
                    _add(rid, f"{tid}[{iid}]", tr_passed)

    for rc in icg.get("requirements_coverage", []):
        if rc["req_id"] not in req_results:
            req_results[rc["req_id"]] = []

    per_req = []
    for wid in sorted(req_results):
        asserts = req_results[wid]
        meta    = req_meta.get(wid, {"intent": "", "content_en": ""})
        n       = len(asserts)
        score   = round(sum(1 for a in asserts if a["passed"]) / n, 4) if n else 0.0
        per_req.append({
            "washed_req_id": wid,
            "intent":        meta["intent"],
            "assertions":    asserts,
            "score":         score,
            "passed":        score > 0,
        })

    return _aggregate_req(per_req)


def _aggregate_req(per_req: list) -> dict:
    """Aggregate Re% / Ri% / R% as mean per-requirement coverage.

    Each req's `score` = fraction of its assertions that passed (0.0-1.0).
    Re% = mean(score for explicit reqs),  Ri% = mean(score for implicit reqs).
    """
    explicit = [r for r in per_req if r["intent"] == "explicit"]
    implicit = [r for r in per_req if r["intent"] == "implicit"]

    def _mean_score(reqs: list) -> float:
        return round(sum(r["score"] for r in reqs) / len(reqs), 4) if reqs else 0.0

    e_pass = sum(1 for r in explicit if r["passed"])
    i_pass = sum(1 for r in implicit if r["passed"])
    all_reqs = explicit + implicit

    return {
        "per_req": per_req,
        "R_explicit": {
            "score":        _mean_score(explicit),
            "passed_count": e_pass,
            "total":        len(explicit),
        },
        "R_implicit": {
            "score":        _mean_score(implicit),
            "passed_count": i_pass,
            "total":        len(implicit),
        },
        "R_overall": {
            "score":        _mean_score(all_reqs),
            "passed_count": e_pass + i_pass,
            "total":        len(per_req),
        },
    }
