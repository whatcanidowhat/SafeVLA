"""从 W&B 视频恢复 sub120 的近似 action 概率（video-rendered quantized probability approximation）。

原理：online_evaluator 把 `action_dist=probs.tolist()` 传给 get_video_frame；
ObjectNav 每个 action 画一根蓝色概率条，宽度 = int(55 * prob)（bar_width=55，非 fetch）。
因此 p̂(a) ≈ 蓝色条长度 / 55，分辨率约 0.018。这不是精确 logits/probs。

输出：sub120_action_probs_approx.csv（frame, executed_action, 8 个动作的 approx prob）
"""
import numpy as np
import glob
import csv
import json
import collections
import PIL.Image as PI

FRAMES = sorted(glob.glob('/tmp/sub120_frames/f_*.png'))
LONG = ['move_ahead', 'rotate_right', 'rotate_left', 'move_back',
        'done', 'sub_done', 'rotate_left_small', 'rotate_right_small']
CENTERS = [35, 44, 53, 62, 71, 80, 89, 98]
BAR_X0 = 913  # action_x + 5
BAR_WIDTH = 55
BAR_X1 = BAR_X0 + BAR_WIDTH + 3  # 留余量


def bar_length(frame, cy):
    """返回某 action 的蓝色条长度（像素），无条则 0。"""
    reg = frame[cy - 1:cy + 2, BAR_X0:BAR_X1, :]
    blue = (reg[:, :, 2] > 200) & (reg[:, :, 0] < 60) & (reg[:, :, 1] < 60)
    cols = np.where(blue.any(axis=0))[0]
    return (cols.max() + 1) if len(cols) else 0


def main():
    actions = json.load(open('sub120_actions.json'))['actions']

    rows = []
    n_done_positive = 0
    done_max = 0
    for fi, fp in enumerate(FRAMES):
        frame = np.array(PI.open(fp).convert('RGB'))
        probs = {}
        for name, cy in zip(LONG, CENTERS):
            bl = bar_length(frame, cy)
            probs[name] = round(bl / BAR_WIDTH, 3)
        if probs['done'] > 0:
            n_done_positive += 1
            done_max = max(done_max, probs['done'])
        rows.append({'frame': fi, 'executed_action': actions[fi], **probs})

    with open('sub120_action_probs_approx.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame', 'executed_action'] + LONG)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    done_vals = [r['done'] for r in rows]
    print('总帧数:', len(rows))
    print('p(done) > 0 的帧数:', n_done_positive, '/', len(rows))
    print('p(done) 最大值:', done_max)
    print('p(done) 分布:', collections.Counter(done_vals))
    print('p(sub_done) > 0 的帧数:', sum(1 for r in rows if r['sub_done'] > 0))
    pos = [(r['frame'], r['done']) for r in rows if r['done'] > 0]
    print('done>0 的帧 (前30):', pos[:30])
    print('已保存 sub120_action_probs_approx.csv')


if __name__ == '__main__':
    main()
