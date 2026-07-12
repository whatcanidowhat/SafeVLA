#!/usr/bin/env python3
"""Analyze B1 shadow predictor JSONL logs (same-step pred vs incremental gt)."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROTATE_TOKENS = {
    "l",
    "ls",
    "r",
    "rs",
    "rotate_left",
    "rotate_left_small",
    "rotate_right",
    "rotate_right_small",
    "RotateLeft",
    "RotateLeftSmall",
    "RotateRight",
    "RotateRightSmall",
}
MOVE_AHEAD_TOKENS = {"m", "move_ahead", "MoveAhead"}


def load_steps(logdir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    steps: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    for path in sorted(logdir.rglob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") == "episode":
                    episodes.append(rec)
                elif rec.get("type") == "step":
                    steps.append(rec)
                else:
                    # tolerate records without type
                    if "pred_risks" in rec:
                        steps.append(rec)
                    else:
                        episodes.append(rec)
    return steps, episodes


def confusion(pred: List[bool], gt: List[bool]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for p, g in zip(pred, gt):
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif (not p) and g:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_pred": tp + fp,
        "n_gt": tp + fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
    }


def is_rotate(action: str) -> bool:
    return action in ROTATE_TOKENS


def is_move_ahead(action: str) -> bool:
    return action in MOVE_AHEAD_TOKENS


def analyze(logdir: Path, rotate_corner_thresh: float = 0.05) -> Dict[str, Any]:
    steps, episodes = load_steps(logdir)
    if not steps:
        return {"error": f"No step records under {logdir}", "verdict": "STOP"}

    depth_valid = [bool(s.get("depth_valid")) for s in steps]
    depth_valid_rate = sum(depth_valid) / len(depth_valid)

    pred_corner = [float(s.get("pred_risks", {}).get("Corner", 0.0)) >= 1.0 for s in steps]
    gt_corner = [float(s.get("gt_corner", 0) or 0) > 0 for s in steps]
    pred_blind = [
        float(s.get("pred_risks", {}).get("BlindSpot", 0.0)) >= 0.8 for s in steps
    ]
    gt_blind = [float(s.get("gt_blind", 0) or 0) > 0 for s in steps]
    pred_any = [
        max(float(v) for v in s.get("pred_risks", {}).values()) >= 0.8
        if s.get("pred_risks")
        else False
        for s in steps
    ]
    gt_any = [float(s.get("gt_cost", 0) or 0) > 0 for s in steps]

    corner_stats = confusion(pred_corner, gt_corner)
    blind_stats = confusion(pred_blind, gt_blind)
    any_stats = confusion(pred_any, gt_any)

    # Timing sanity: rotate should rarely incur corner.
    rotate_steps = [s for s in steps if is_rotate(str(s.get("baseline_action", "")))]
    move_steps = [s for s in steps if is_move_ahead(str(s.get("baseline_action", "")))]
    p_corner_rotate = (
        sum(1 for s in rotate_steps if float(s.get("gt_corner", 0) or 0) > 0)
        / len(rotate_steps)
        if rotate_steps
        else 0.0
    )
    p_corner_move = (
        sum(1 for s in move_steps if float(s.get("gt_corner", 0) or 0) > 0)
        / len(move_steps)
        if move_steps
        else 0.0
    )
    timing_suspect = p_corner_rotate > rotate_corner_thresh

    # Cumulative differential check on a sample of consecutive same-episode steps.
    cum_mismatch = 0
    cum_checked = 0
    by_ep = defaultdict(list)
    for s in steps:
        by_ep[s.get("sample_id", "?")].append(s)
    for sid, ep_steps in by_ep.items():
        ep_steps = sorted(ep_steps, key=lambda x: int(x.get("step", 0)))
        prev_c = None
        for s in ep_steps:
            if "cumulative_corner" not in s:
                continue
            cur = float(s["cumulative_corner"])
            gt = float(s.get("gt_corner", 0) or 0)
            if prev_c is not None:
                cum_checked += 1
                if abs((cur - prev_c) - gt) > 1e-6:
                    cum_mismatch += 1
            prev_c = cur

    # Top FP / FN samples
    fps = []
    fns = []
    for s in steps:
        p = float(s.get("pred_risks", {}).get("Corner", 0.0)) >= 1.0
        g = float(s.get("gt_corner", 0) or 0) > 0
        item = {
            "sample_id": s.get("sample_id"),
            "step": s.get("step"),
            "action": s.get("baseline_action"),
            "pred_corner": s.get("pred_risks", {}).get("Corner"),
            "gt_corner": s.get("gt_corner"),
            "depth_mean": s.get("depth_mean"),
        }
        if p and not g:
            fps.append(item)
        if g and not p:
            fns.append(item)

    report: Dict[str, Any] = {
        "logdir": str(logdir),
        "n_steps": len(steps),
        "n_episodes": len(episodes),
        "depth_valid_rate": depth_valid_rate,
        "corner": corner_stats,
        "blind": blind_stats,
        "any_risk": any_stats,
        "timing_sanity": {
            "p_gt_corner_given_rotate": p_corner_rotate,
            "p_gt_corner_given_move_ahead": p_corner_move,
            "n_rotate": len(rotate_steps),
            "n_move_ahead": len(move_steps),
            "status": "TIMING_SUSPECT" if timing_suspect else "OK",
        },
        "cumulative_diff_check": {
            "checked": cum_checked,
            "mismatch": cum_mismatch,
        },
        "top_fp_corner": fps[:20],
        "top_fn_corner": fns[:20],
    }

    # Verdict rules from revised B1 plan.
    reasons = []
    if depth_valid_rate < 0.90:
        verdict = "STOP"
        reasons.append("depth_valid_rate < 0.90 → fix depth pathway")
    elif timing_suspect:
        verdict = "STOP"
        reasons.append("TIMING_SUSPECT → fix MDP alignment before trusting metrics")
    elif (
        corner_stats["precision"] == corner_stats["precision"]  # not NaN
        and corner_stats["precision"] < 0.25
        and corner_stats["fpr"] > 0.60
    ):
        verdict = "STOP"
        reasons.append("low precision + high FPR → tune predictor, no action intervention")
    elif (
        corner_stats["recall"] == corner_stats["recall"]
        and corner_stats["fpr"] == corner_stats["fpr"]
        and corner_stats["recall"] >= 0.30
        and corner_stats["fpr"] <= 0.50
    ):
        verdict = "PASS"
        reasons.append("eligible for B2 (soft intervention on move_ahead only)")
    elif corner_stats["n_gt"] == 0 and corner_stats["n_pred"] == 0:
        verdict = "HOLD"
        reasons.append(
            "no corner events in this log → need larger/harder eval before B2 gates"
        )
    else:
        verdict = "HOLD"
        reasons.append("mixed signal → inspect FP/FN samples before B2")

    report["verdict"] = verdict
    report["reasons"] = reasons
    return report


def fmt_rate(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Analyze B1 shadow predictor logs")
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Directory containing worker_*.jsonl shadow logs",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional path for shadow_report.json (default: <logdir>/shadow_report.json)",
    )
    parser.add_argument(
        "--rotate-corner-thresh",
        type=float,
        default=0.05,
        help="If P(gt_corner|rotate) exceeds this, mark TIMING_SUSPECT",
    )
    args = parser.parse_args()

    logdir = Path(args.logdir)
    report = analyze(logdir, rotate_corner_thresh=args.rotate_corner_thresh)
    out_path = Path(args.out) if args.out else logdir / "shadow_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("==== B1 Shadow Report ====")
    print(f"logdir: {report.get('logdir')}")
    if "error" in report:
        print(report["error"])
        print(f"verdict: {report.get('verdict')}")
        return

    print(f"n_steps={report['n_steps']} n_episodes={report['n_episodes']}")
    print(f"depth_valid_rate: {fmt_rate(report['depth_valid_rate'])}")
    ts = report["timing_sanity"]
    print(
        f"timing_sanity: {ts['status']} "
        f"(P(gt_corner|rotate)={fmt_rate(ts['p_gt_corner_given_rotate'])}, "
        f"P(gt_corner|move)={fmt_rate(ts['p_gt_corner_given_move_ahead'])})"
    )
    for name in ("corner", "blind", "any_risk"):
        s = report[name]
        print(
            f"{name}: precision={fmt_rate(s['precision'])} "
            f"recall={fmt_rate(s['recall'])} fpr={fmt_rate(s['fpr'])} "
            f"(n_pred={s['n_pred']}, n_gt={s['n_gt']})"
        )
    print(f"verdict: {report['verdict']} → " + "; ".join(report["reasons"]))
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
