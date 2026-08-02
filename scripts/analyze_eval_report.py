#!/usr/bin/env python3
"""Multi-dimensional analysis / comparison report for SafeVLA online-eval runs.

Aggregates four dimensions from a single `online_eval.py` output directory:
  1. Overall performance: success rate (SR), SEL, episode length, room coverage.
  2. Safety cost breakdown: corner / blind / danger / fragile / critical.
  3. Failure taxonomy (only if `failure_case_analysis.json` was generated for the run).
  4. Per-object-type success rate (from the wandb `PerObjectType` table).

Data sources per run directory (either the experiment dir, e.g.
`eval/ObjectNavType/<exp_name>`, or a specific timestamped run dir inside it):
  - `*.mp4` filenames: `<corner>_<blind>_<danger>_<fragile>_<critical>_<Success|Failed>_task=...mp4`
  - `wandb/wandb/run-*/files/wandb-summary.json` + referenced `media/table/*.table.json`
  - `failure_case_analysis.json` (optional)

Usage:
    # Single-run report
    python scripts/analyze_eval_report.py report <run_dir> [--label NAME] [--out report.md] [--json report.json]

    # Compare two runs (e.g. baseline vs modified)
    python scripts/analyze_eval_report.py compare <run_a> <run_b> \\
        [--label-a Baseline] [--label-b Modified] [--out compare.md] [--json compare.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

COST_KEYS = ["corner", "blind", "danger", "fragile", "critical"]

MP4_PATTERN = re.compile(
    r"^(?P<corner>\d+)_(?P<blind>\d+)_(?P<danger>\d+)_(?P<fragile>\d+)_(?P<critical>\d+)_"
    r"(?P<status>Success|Failed)_task=(?P<task>[^,]+),house=(?P<house>[^,]+),"
    r"sub_house_id=(?P<sub>\d+)_(?P<goal>.+)\.mp4$"
)
RUN_DIR_NAME_RE = re.compile(r"^\d{2}_\d{2}_\d{4}_")


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x  # NaN != NaN


def fmt_num(x: Any, nd: int = 2) -> str:
    return f"{x:.{nd}f}" if is_number(x) else "N/A"


def fmt_pct(x: Any, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}%" if is_number(x) else "N/A"


def resolve_run_dir(path: Path) -> Path:
    """Accept either a specific run dir or its parent experiment dir (pick the latest run)."""
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if (path / "wandb").exists() or any(path.glob("*.mp4")):
        return path
    candidates = [p for p in path.iterdir() if p.is_dir() and RUN_DIR_NAME_RE.match(p.name)]
    if not candidates:
        raise FileNotFoundError(
            f"No run directory (matching MM_DD_YYYY_...) found under {path}"
        )
    return sorted(candidates, key=lambda p: p.name)[-1]


def parse_episode_videos(run_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for f in sorted(run_dir.glob("*.mp4")):
        m = MP4_PATTERN.match(f.name)
        if not m:
            continue
        d = m.groupdict()
        rows.append(
            {
                "house": d["house"],
                "sub_house_id": d["sub"],
                "goal": d["goal"],
                "status": d["status"],
                **{k: int(d[k]) for k in COST_KEYS},
            }
        )
    return rows


def find_wandb_files_dir(run_dir: Path) -> Optional[Path]:
    wandb_root = run_dir / "wandb" / "wandb"
    if not wandb_root.exists():
        return None
    matches = sorted(wandb_root.glob("run-*/files"))
    return matches[-1] if matches else None


def load_wandb_table(files_dir: Path, summary: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    entry = summary.get(key)
    if not isinstance(entry, dict) or "path" not in entry:
        return None
    table_path = files_dir / entry["path"]
    if not table_path.exists():
        return None
    with open(table_path, encoding="utf-8") as f:
        return json.load(f)


def load_wandb_data(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"files_dir": None, "summary": None, "tables": {}, "metadata": {}}
    files_dir = find_wandb_files_dir(run_dir)
    out["files_dir"] = str(files_dir) if files_dir else None
    if files_dir is None:
        return out

    summary_path = files_dir / "wandb-summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        out["summary"] = summary
        for key in list(summary.keys()):
            if key.startswith("AggregatedResults/") or key.startswith("PerObjectType/") or key == "FullAggregatedResults" or key == "metrics":
                table = load_wandb_table(files_dir, summary, key)
                if table:
                    out["tables"][key] = table

    meta_path = files_dir / "wandb-metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            out["metadata"] = json.load(f)
    return out


def parse_cli_args(args: List[str]) -> Dict[str, Any]:
    """Turn wandb-metadata.json's flat `args` list back into a {flag: value(s)} dict."""
    out: Dict[str, Any] = {}
    i = 0
    while i < len(args):
        tok = args[i]
        if tok.startswith("--"):
            key = tok[2:]
            vals = []
            j = i + 1
            while j < len(args) and not str(args[j]).startswith("--"):
                vals.append(args[j])
                j += 1
            out[key] = True if not vals else (vals[0] if len(vals) == 1 else vals)
            i = j
        else:
            i += 1
    return out


def build_unified_report(raw_path: Path) -> Dict[str, Any]:
    run_dir = resolve_run_dir(raw_path)
    episodes = parse_episode_videos(run_dir)
    wandb_data = load_wandb_data(run_dir)

    n = len(episodes)
    n_succ = sum(1 for e in episodes if e["status"] == "Success")
    n_fail = n - n_succ

    cost_totals = {k: sum(e[k] for e in episodes) for k in COST_KEYS}
    cost_nonzero_eps = {k: sum(1 for e in episodes if e[k] > 0) for k in COST_KEYS}
    total_cost_all = sum(cost_totals.values())
    total_cost_succ = sum(
        sum(e[k] for k in COST_KEYS) for e in episodes if e["status"] == "Success"
    )
    total_cost_fail = total_cost_all - total_cost_succ

    agg_table: Optional[Dict[str, Any]] = None
    agg_task_type: Optional[str] = None
    for key, table in wandb_data["tables"].items():
        if key.startswith("AggregatedResults/") and table.get("data"):
            agg_task_type = key.split("/", 1)[1]
            agg_table = dict(zip(table["columns"], table["data"][0]))
            break

    per_object: List[Dict[str, Any]] = []
    for key, table in wandb_data["tables"].items():
        if key.startswith("PerObjectType/"):
            cols = table["columns"]
            per_object = [dict(zip(cols, row)) for row in table.get("data", [])]
            break

    failure_analysis: Optional[Dict[str, Any]] = None
    fail_json_path = run_dir / "failure_case_analysis.json"
    if fail_json_path.exists():
        with open(fail_json_path, encoding="utf-8") as f:
            failure_analysis = json.load(f)

    meta = wandb_data.get("metadata") or {}
    cli_args = parse_cli_args(meta.get("args") or [])

    return {
        "run_dir": str(run_dir),
        "num_episodes": n,
        "num_success": n_succ,
        "num_fail": n_fail,
        "success_rate": (n_succ / n) if n else float("nan"),
        "cost": {
            "totals": cost_totals,
            "nonzero_episodes": cost_nonzero_eps,
            "total_all": total_cost_all,
            "total_success": total_cost_succ,
            "total_fail": total_cost_fail,
            "mean_all": (total_cost_all / n) if n else float("nan"),
            "mean_success": (total_cost_succ / n_succ) if n_succ else float("nan"),
            "mean_fail": (total_cost_fail / n_fail) if n_fail else float("nan"),
        },
        "aggregated": agg_table,
        "aggregated_task_type": agg_task_type,
        "per_object": per_object,
        "failure_analysis": failure_analysis,
        "cli_args": cli_args,
        "meta": {
            "gpu": meta.get("gpu"),
            "gpu_count": meta.get("gpu_count"),
            "git_commit": (meta.get("git") or {}).get("commit"),
            "started_at": meta.get("startedAt"),
        },
    }


def diff_row(name: str, a: Any, b: Any, fmt=fmt_num) -> str:
    if not is_number(a) or not is_number(b):
        return f"| {name} | {fmt(a)} | {fmt(b)} | N/A |"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"| {name} | {fmt(a)} | {fmt(b)} | {sign}{fmt(d)} |"


AGG_ROW_SPECS = [
    ("sel", "SEL", lambda x: fmt_num(x, 3)),
    ("eps_len", "平均episode长度(全体)", fmt_num),
    ("eps_len_succ", "平均episode长度(成功)", fmt_num),
    ("eps_len_fail", "平均episode长度(失败)", fmt_num),
    ("percentage_rooms_visited", "平均房间访问比例", fmt_pct),
    ("total_rooms_visited", "平均访问房间数", fmt_num),
]


def _percentage_collision(agg: Dict[str, Any]) -> Any:
    pc = agg.get("percentage_collision")
    if isinstance(pc, list) and pc:
        return pc[0]
    return pc


def render_single_report(rep: Dict[str, Any], label: str = "") -> str:
    lines = []
    lines.append(f"# 评测报告{f'：{label}' if label else ''}")
    lines.append("")
    lines.append(f"- 运行目录: `{rep['run_dir']}`")
    cli = rep["cli_args"]
    if cli.get("ckpt_path"):
        lines.append(f"- checkpoint: `{cli['ckpt_path']}`")
    if cli.get("task_type"):
        lines.append(f"- task_type: `{cli['task_type']}`")
    if cli.get("eval_subset"):
        lines.append(f"- eval_subset: `{cli['eval_subset']}`")
    meta = rep["meta"]
    if meta.get("gpu"):
        lines.append(f"- GPU: {meta['gpu']} x{meta.get('gpu_count', '?')}")
    if meta.get("git_commit"):
        lines.append(f"- git commit: `{str(meta['git_commit'])[:10]}`")
    lines.append("")

    lines.append("## 一、总体性能")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(
        f"| Success Rate | {fmt_pct(rep['success_rate'])} ({rep['num_success']}/{rep['num_episodes']}) |"
    )
    agg = rep.get("aggregated") or {}
    for field, cn, fmt in AGG_ROW_SPECS:
        if field in agg:
            lines.append(f"| {cn} | {fmt(agg[field])} |")
    pc = _percentage_collision(agg)
    if is_number(pc):
        lines.append(f"| 平均碰撞比例 | {fmt_pct(pc)} |")
    lines.append("")

    lines.append("## 二、安全代价（Cost）分解")
    lines.append("")
    cost = rep["cost"]
    lines.append("| 代价类型 | 总量 | 触发episode数 |")
    lines.append("|---|---|---|")
    for k in COST_KEYS:
        lines.append(f"| {k} | {cost['totals'][k]} | {cost['nonzero_episodes'][k]} |")
    lines.append(f"| **总计** | {cost['total_all']} | - |")
    lines.append("")
    lines.append("| 分组 | 总代价 | 均值/episode |")
    lines.append("|---|---|---|")
    lines.append(f"| 全体 | {cost['total_all']} | {fmt_num(cost['mean_all'])} |")
    lines.append(f"| 成功组 | {cost['total_success']} | {fmt_num(cost['mean_success'])} |")
    lines.append(f"| 失败组 | {cost['total_fail']} | {fmt_num(cost['mean_fail'])} |")
    lines.append("")

    lines.append("## 三、失败归因（Failure Taxonomy）")
    lines.append("")
    fa = rep.get("failure_analysis")
    if fa:
        summary = fa.get("summary", {})
        lines.append(f"来源: `failure_case_analysis.json`（n_fail={summary.get('n_fail')}）")
        lines.append("")
        tax = summary.get("fail_taxonomy", {})
        total_fail = sum(tax.values()) or 1
        lines.append("| 类别 | 数量 | 占失败比例 |")
        lines.append("|---|---|---|")
        for k, v in sorted(tax.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v} | {fmt_pct(v / total_fail)} |")
        lines.append("")
        term = summary.get("term_counts", {})
        if term:
            lines.append("终止方式分布: " + ", ".join(f"{k}={v}" for k, v in term.items()))
            lines.append("")
    else:
        lines.append("_未找到 `failure_case_analysis.json`，跳过该维度。_")
        lines.append("")

    lines.append("## 四、按物体类别的成功率")
    lines.append("")
    per_obj = rep.get("per_object") or []
    if per_obj:
        lines.append("| 物体 | SR | 平均episode长度 | 样本数 |")
        lines.append("|---|---|---|---|")
        for row in sorted(per_obj, key=lambda r: (r.get("success") if is_number(r.get("success")) else 1)):
            name = row.get("object_name", "?")
            sr = row.get("success")
            elen = row.get("eps_len")
            size = row.get("total_size")
            lines.append(f"| {name} | {fmt_pct(sr)} | {fmt_num(elen, 1)} | {size} |")
    else:
        lines.append("_未找到 PerObjectType 表。_")
    lines.append("")

    return "\n".join(lines)


def render_compare_report(
    rep_a: Dict[str, Any], rep_b: Dict[str, Any], label_a: str = "A", label_b: str = "B"
) -> str:
    lines = []
    lines.append(f"# 评测对比报告：{label_a} vs {label_b}")
    lines.append("")
    lines.append(f"- {label_a}: `{rep_a['run_dir']}`")
    lines.append(f"- {label_b}: `{rep_b['run_dir']}`")
    lines.append("")

    lines.append("## 一、总体性能对比")
    lines.append("")
    lines.append(f"| 指标 | {label_a} | {label_b} | Δ({label_b}-{label_a}) |")
    lines.append("|---|---|---|---|")
    lines.append(diff_row("Success Rate", rep_a["success_rate"], rep_b["success_rate"], fmt_pct))
    agg_a, agg_b = (rep_a.get("aggregated") or {}), (rep_b.get("aggregated") or {})
    for field, cn, fmt in AGG_ROW_SPECS:
        lines.append(diff_row(cn, agg_a.get(field), agg_b.get(field), fmt))
    lines.append(
        diff_row("平均碰撞比例", _percentage_collision(agg_a), _percentage_collision(agg_b), fmt_pct)
    )
    lines.append("")

    lines.append("## 二、安全代价对比")
    lines.append("")
    lines.append(f"| 代价类型 | {label_a} 总量 | {label_b} 总量 | Δ |")
    lines.append("|---|---|---|---|")
    for k in COST_KEYS:
        a_v, b_v = rep_a["cost"]["totals"][k], rep_b["cost"]["totals"][k]
        d = b_v - a_v
        lines.append(f"| {k} | {a_v} | {b_v} | {'+' if d >= 0 else ''}{d} |")
    total_a, total_b = rep_a["cost"]["total_all"], rep_b["cost"]["total_all"]
    dt = total_b - total_a
    lines.append(f"| **总计** | {total_a} | {total_b} | {'+' if dt >= 0 else ''}{dt} |")
    lines.append("")
    lines.append(f"| 分组 | {label_a} 均值/episode | {label_b} 均值/episode | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(diff_row("全体", rep_a["cost"]["mean_all"], rep_b["cost"]["mean_all"]))
    lines.append(diff_row("成功组", rep_a["cost"]["mean_success"], rep_b["cost"]["mean_success"]))
    lines.append(diff_row("失败组", rep_a["cost"]["mean_fail"], rep_b["cost"]["mean_fail"]))
    lines.append("")

    lines.append("## 三、失败归因对比")
    lines.append("")
    fa_a, fa_b = rep_a.get("failure_analysis"), rep_b.get("failure_analysis")
    if fa_a and fa_b:
        tax_a = fa_a.get("summary", {}).get("fail_taxonomy", {})
        tax_b = fa_b.get("summary", {}).get("fail_taxonomy", {})
        cats = sorted(set(tax_a) | set(tax_b))
        lines.append(f"| 类别 | {label_a} | {label_b} | Δ |")
        lines.append("|---|---|---|---|")
        for c in cats:
            a_v, b_v = tax_a.get(c, 0), tax_b.get(c, 0)
            d = b_v - a_v
            lines.append(f"| {c} | {a_v} | {b_v} | {'+' if d >= 0 else ''}{d} |")
    elif fa_a or fa_b:
        have_label = label_a if fa_a else label_b
        missing_label = label_b if fa_a else label_a
        fa = fa_a or fa_b
        tax = fa.get("summary", {}).get("fail_taxonomy", {})
        lines.append(f"_{missing_label} 缺少 `failure_case_analysis.json`，无法逐类别对比。_")
        lines.append("")
        lines.append(f"仅 {have_label} 的 fail_taxonomy: " + ", ".join(f"{k}={v}" for k, v in tax.items()))
    else:
        lines.append("_双方均未找到 `failure_case_analysis.json`。_")
    lines.append("")

    lines.append("## 四、按物体类别的成功率对比")
    lines.append("")
    obj_a = {r.get("object_name"): r for r in (rep_a.get("per_object") or [])}
    obj_b = {r.get("object_name"): r for r in (rep_b.get("per_object") or [])}
    names = sorted(set(obj_a) | set(obj_b))
    if names:
        lines.append(f"| 物体 | {label_a} SR | {label_b} SR | Δ |")
        lines.append("|---|---|---|---|")
        rows = []
        for name in names:
            sa = obj_a.get(name, {}).get("success")
            sb = obj_b.get(name, {}).get("success")
            d = (sb - sa) if (is_number(sa) and is_number(sb)) else None
            rows.append((name, sa, sb, d))
        rows.sort(key=lambda r: (r[3] is None, r[3] if r[3] is not None else 0))
        for name, sa, sb, d in rows:
            d_s = f"{'+' if d >= 0 else ''}{d * 100:.1f}pp" if d is not None else "N/A"
            lines.append(f"| {name} | {fmt_pct(sa)} | {fmt_pct(sb)} | {d_s} |")
    else:
        lines.append("_双方均未找到 PerObjectType 表。_")
    lines.append("")

    return "\n".join(lines)


def _write_outputs(md: str, rep_obj: Any, out: str, json_out: str) -> None:
    if out:
        Path(out).write_text(md, encoding="utf-8")
        print(f"\n[markdown 已保存 -> {out}]", file=sys.stderr)
    if json_out:
        Path(json_out).write_text(
            json.dumps(rep_obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(f"[json 已保存 -> {json_out}]", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeVLA 评测多维度分析/对比报告")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_report = sub.add_parser("report", help="对单个评测 run 做多维度分析")
    p_report.add_argument("run_dir", type=str, help="实验目录或具体的timestamp run目录")
    p_report.add_argument("--label", type=str, default="")
    p_report.add_argument("--out", type=str, default="", help="保存markdown报告的路径")
    p_report.add_argument("--json", type=str, default="", help="保存原始聚合数据(json)的路径")

    p_cmp = sub.add_parser("compare", help="对比两个评测 run（如 baseline vs 修改版）")
    p_cmp.add_argument("run_a", type=str)
    p_cmp.add_argument("run_b", type=str)
    p_cmp.add_argument("--label-a", type=str, default="A")
    p_cmp.add_argument("--label-b", type=str, default="B")
    p_cmp.add_argument("--out", type=str, default="")
    p_cmp.add_argument("--json", type=str, default="")

    args = parser.parse_args()

    if args.cmd == "report":
        rep = build_unified_report(Path(args.run_dir))
        md = render_single_report(rep, args.label)
        print(md)
        _write_outputs(md, rep, args.out, args.json)
    else:
        rep_a = build_unified_report(Path(args.run_a))
        rep_b = build_unified_report(Path(args.run_b))
        md = render_compare_report(rep_a, rep_b, args.label_a, args.label_b)
        print(md)
        _write_outputs(md, {"a": rep_a, "b": rep_b}, args.out, args.json)


if __name__ == "__main__":
    main()
