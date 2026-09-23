"""窄/宽目标 ID 集合核对（纯重分析，不重跑模型）。

目的：验证 `vis_pix_*`（NumPixelsVisible，用 synset_to_object_ids=narrow）
与 success 判据（successful_if_done，用 broad_synset_to_object_ids）的目标 ID 集
是否一致，并统计 27 个失败 episode 中 narrow!=broad 的数量。

输入：
  - benchmark/objectnavtype_val.jsonl.gz（200 样本，含 narrow/broad ID 集）
  - wandb VideoTable 表（含每个 episode 的 vis_pix / eps_len / in_room / success）

输出：
  - narrow_broad_reanalysis.csv（逐失败案例一行）

运行：
  cd <run>/wandb/wandb/run-*/files
  python narrow_broad_reanalysis.py
"""
import gzip
import json
import re
import csv
import os

BENCH = "/nvme2/user/qyy/SafeVLA/benchmark/objectnavtype_val.jsonl.gz"
VIDEO_TABLE = "media/table/VideoTable/ObjectNavType_149_8c935d47dee2ab2e2441.table.json"


def flatten(d):
    out = set()
    for v in d.values():
        out.update(v)
    return out


def main():
    bench = []
    with gzip.open(BENCH, "rt") as f:
        for line in f:
            bench.append(json.loads(line))
    by_house = {str(r["house_index"]): r for r in bench}

    d = json.load(open(VIDEO_TABLE))
    cols = d["columns"]
    rows = d["data"]
    idx = {c: i for i, c in enumerate(cols)}
    fails = [r for r in rows if not r[idx["success"]]]

    out = []
    for r in fails:
        m = re.search(r"house=(\d+),sub_house_id=(\d+)", r[idx["video_path"]])
        house = m.group(1)
        sub = m.group(2)
        b = by_house.get(house)
        if b is None:
            continue
        s = flatten(b.get("synset_to_object_ids", {}))
        br = flatten(b.get("broad_synset_to_object_ids", {}))
        out.append({
            "sub_house_id": sub,
            "house": house,
            "synset": b["synsets"][0],
            "goal": r[idx["goal"]],
            "vis_pix_nav": r[idx["vis_pix_navigation"]],
            "vis_pix_manip": r[idx["vis_pix_manipulation"]],
            "eps_len": r[idx["eps_len"]],
            "in_room": r[idx["has_agent_been_in_room"]],
            "narrow==broad": s == br,
            "broad_extra": "|".join(sorted(br - s)),
        })

    with open("narrow_broad_reanalysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    nd = sum(1 for o in out if not o["narrow==broad"])
    nv0 = sum(1 for o in out if o["vis_pix_nav"] == 0 and not o["narrow==broad"])
    print(f"失败案例总数: {len(out)}  narrow!=broad: {nd}  "
          f"vis_nav=0 且 narrow!=broad: {nv0}")


if __name__ == "__main__":
    main()
