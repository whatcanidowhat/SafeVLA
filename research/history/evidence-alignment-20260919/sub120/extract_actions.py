"""从 W&B 视频恢复 sub120 的动作序列（ARTIFACT-RECOVERY-POC-002）。

原理：online_evaluator 每 step 生成一帧，`taken_action` 以黑色文本渲染在右侧 action 列表
（非执行动作灰色），见 visualization_utils.py:480 `fill="gray" if action != taken_action else "black"`。

action schema（本 run 无 ACTION_DICT / LONG_ACTION_NAME → 默认 ALL_STRETCH_ACTIONS）：
  nav 动作渲染顺序 = move_ahead, rotate_right, rotate_left, move_back,
                    done, sub_done, rotate_left_small, rotate_right_small
  对应短名           = m, r, l, b, end, sub_done, ls, rs

输出：sub120_actions.json（逐帧 action） + sub120_actions.csv
"""
import subprocess
import json
import csv
import os
import numpy as np
from PIL import Image
import io

VIDEO = "0_0_0_0_0_Failed_task=ObjectNavType,house=13419,sub_house_id=120_go-to-a-trash-can.mp4"
N_FRAMES = 600

# 8 个 nav 动作（渲染顺序固定），(短名, 长名)
NAV_ACTIONS = [
    ("m", "move_ahead"),
    ("r", "rotate_right"),
    ("l", "rotate_left"),
    ("b", "move_back"),
    ("end", "done"),
    ("sub_done", "sub_done"),
    ("ls", "rotate_left_small"),
    ("rs", "rotate_right_small"),
]

# 校准的 8 个动作文本中心 y（全帧坐标，来自帧 200/599 的文本行检测）
ACTION_Y = [36, 45, 54, 63, 72, 81, 90, 99]
# 文本区 x 范围（右对齐到 action_x≈908，文本向左延伸）
TEXT_X0, TEXT_X1 = 800, 910


def extract_frame(idx):
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", VIDEO,
         "-vf", "select=eq(n\\,%d)" % idx, "-vframes", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True,
    )
    return np.array(Image.open(io.BytesIO(r.stdout)).convert("RGB"))


def detect_black_action(frame):
    """返回黑色文本（taken action）对应的 y 中心，若无则 None。"""
    region = frame[30:105, TEXT_X0:TEXT_X1, :]
    # 黑色文本：三通道都 < 70
    black = (region[:, :, 0] < 70) & (region[:, :, 1] < 70) & (region[:, :, 2] < 70)
    # 每行（region 内）黑像素数
    rowsum = black.sum(axis=1)
    ys = np.where(rowsum > 3)[0]  # 至少 4 个黑像素
    if len(ys) == 0:
        return None
    # 聚类成连续段
    clusters = []
    cur = [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] <= 2:
            cur.append(y)
        else:
            clusters.append(int(np.mean(cur)))
            cur = [y]
    clusters.append(int(np.mean(cur)))
    # 每个 cluster 的 y 转成全帧坐标
    clusters_full = [c + 30 for c in clusters]
    return clusters_full


def map_y_to_action(black_ys):
    """把黑色文本 y 位置映射到唯一 action 索引。"""
    if not black_ys:
        return None
    # 取黑色像素最多的那个 y 簇（主簇），映射到最近的 action y
    # 简化：取所有簇的平均 y，映射到最近 action
    # 更稳：取「黑色像素最多」的簇
    best = None
    best_score = -1
    for y in black_ys:
        # 找最近的 action 中心
        dists = [abs(y - ay) for ay in ACTION_Y]
        if min(dists) <= 5:  # 必须在某 action 附近
            score = -min(dists)
            if score > best_score:
                best_score = score
                best = int(np.argmin(dists))
    return best


def main():
    seq = []
    ambiguous = []
    for idx in range(N_FRAMES):
        frame = extract_frame(idx)
        black_ys = detect_black_action(frame)
        a = map_y_to_action(black_ys)
        if a is None:
            # 无唯一黑动作：记录为歧义
            ambiguous.append((idx, black_ys))
            seq.append(None)
        else:
            seq.append(NAV_ACTIONS[a][0])

    # 输出
    out = {
        "video": VIDEO,
        "n_frames": N_FRAMES,
        "action_schema": [{"short": s, "long": l} for s, l in NAV_ACTIONS],
        "action_y_centers": ACTION_Y,
        "actions": seq,
        "n_recovered": sum(1 for a in seq if a is not None),
        "n_ambiguous": len(ambiguous),
        "ambiguous_frames": ambiguous[:20],
        "last_action": seq[-1],
    }
    with open("sub120_actions.json", "w") as f:
        json.dump(out, f, indent=2)
    with open("sub120_actions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "action"])
        for i, a in enumerate(seq):
            w.writerow([i, a])

    print("n_frames:", N_FRAMES)
    print("n_recovered (非 None):", out["n_recovered"])
    print("n_ambiguous:", len(ambiguous))
    print("last_action:", seq[-1])
    print("前 10 动作:", seq[:10])
    print("后 10 动作:", seq[-10:])
    # 动作分布
    from collections import Counter
    print("动作分布:", Counter(a for a in seq if a is not None))


if __name__ == "__main__":
    main()
