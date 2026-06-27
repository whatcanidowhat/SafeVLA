"""
SafeVLA 小目标"表征-输出不对齐"诊断脚本
============================================

目的
----
验证 GRASP 论文的核心假设在 SafeVLA 上是否成立：
  "模型 Llama Decoder 的中间层已经编码了'目标物体是否足够近+可见'的信息，
   但这个信息在传递到动作分类头时被压制了——这正是小目标边界震荡的根因。"

用法
----
两种模式：

  模式 A  收集隐状态（在真实评测循环中调用）
    from online_evaluation.probe_small_object import HiddenStateCollector
    collector = HiddenStateCollector(policy)
    # 在每步 get_action_probs 前设置 pre-action 标签，get_action_probs 后记录隐状态：
    collector.record(hidden_states_list, label_dict)
    collector.save("probe_data.pt")

  模式 B  离线训练探针 + 画图（脚本直接运行）
    python probe_small_object.py --data probe_data.pt --out probe_result.png

数据结构（probe_data.pt）
------------------------
{
  "hidden_states": Tensor[N, L, D],   # N=样本数, L=层数, D=hidden_dim
  "labels": {
      "is_close_and_visible": Tensor[N],  # bool/int: successful_if_done() 真值
      "target_distance":      Tensor[N],  # float: dist_to_target_func() 米
      "action_was_done":      Tensor[N],  # bool: 该步模型选择了 done
  }
}
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# =============================================================================
# Part 1: 隐状态采集器（嵌入评测循环使用）
# =============================================================================

class DecoderLayerHook:
    """
    为 SafeVLA 的 TransformerDecoder.layers（nn.ModuleList of TransformerBlock）
    注册 forward hook，逐层捕获隐状态。

    使用方式:
        hook = DecoderLayerHook(policy.actor_critic.decoder)
        with hook:
            probs, action_list = policy.get_action_probs(frame, goal_spec)
        last_step_hidden = hook.get_last_step()   # shape: [n_layers, hidden_dim]
    """

    def __init__(self, decoder):
        # decoder 是 TransformerDecoder 实例
        self.layers = decoder.layers          # nn.ModuleList[TransformerBlock]
        self._handles: list = []
        self._captured: List[torch.Tensor] = []

    def __enter__(self):
        self._captured = []
        for layer in self.layers:
            # TransformerBlock.forward 返回 h: [B, T, D]
            handle = layer.register_forward_hook(self._hook_fn)
            self._handles.append(handle)
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook_fn(self, module, inp, out):
        # out: [B, T, D]，取最后一个时间步，squeeze batch dim（推理时 B=1）
        self._captured.append(out[0, -1, :].detach().cpu().float())

    def get_last_step(self) -> Optional[torch.Tensor]:
        """返回 [n_layers, hidden_dim]，若未捕获则 None。"""
        if not self._captured:
            return None
        return torch.stack(self._captured, dim=0)  # [L, D]


class HiddenStateCollector:
    """
    在评测循环中收集每步的隐状态 + 真值标签，
    最后 save() 为 .pt 文件供离线探针分析。

    接入方式（在 GRPOPredictiveAgent.act() 里，或 worker 中）:

        # 1. 初始化（每个 worker 进程一次）
        from online_evaluation.probe_small_object import HiddenStateCollector, DecoderLayerHook
        collector = HiddenStateCollector(max_samples=5000)

        # 2. 每步采集
        hook = DecoderLayerHook(policy.actor_critic.decoder)
        with hook:
            probs, action_list = policy.get_action_probs(frame, goal_spec)
        hs = hook.get_last_step()  # [L, D]

        # 3. 获取当前帧真值（必须在 task.step_with_action_str 前采集）
        label = {
            "is_close_and_visible": real_info.get("target_visible", False),
            "target_distance":      real_info.get("target_distance", float("inf")),
            "action_was_done":      chosen_action in ("end", "done", "Done"),
        }
        collector.record(hs, label)

        # 4. 定期保存
        if collector.n_samples % 500 == 0:
            collector.save("probe_data.pt")
    """

    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self._hs_buf: List[torch.Tensor] = []         # 每个: [L, D]
        self._labels_vis: List[int] = []
        self._labels_dist: List[float] = []
        self._labels_done: List[int] = []

    @property
    def n_samples(self) -> int:
        return len(self._hs_buf)

    def record(
        self,
        hidden_states: Optional[torch.Tensor],  # [L, D]
        label: dict,
    ):
        if hidden_states is None:
            return
        if self.n_samples >= self.max_samples:
            return
        self._hs_buf.append(hidden_states)
        self._labels_vis.append(int(bool(label.get("is_close_and_visible", False))))
        d = label.get("target_distance", float("inf"))
        self._labels_dist.append(float(d) if not np.isinf(d) else 10.0)
        self._labels_done.append(int(bool(label.get("action_was_done", False))))

    def save(self, path: str):
        if not self._hs_buf:
            print("[Probe] No data collected yet, skip save.")
            return
        data = {
            "hidden_states": torch.stack(self._hs_buf, dim=0),       # [N, L, D]
            "labels": {
                "is_close_and_visible": torch.tensor(self._labels_vis, dtype=torch.long),
                "target_distance":      torch.tensor(self._labels_dist, dtype=torch.float32),
                "action_was_done":      torch.tensor(self._labels_done, dtype=torch.long),
            },
        }
        torch.save(data, path)
        print(f"[Probe] Saved {self.n_samples} samples to {path}")
        print(f"  hidden_states: {data['hidden_states'].shape}  "
              f"(N={self.n_samples}, L={data['hidden_states'].shape[1]}, "
              f"D={data['hidden_states'].shape[2]})")
        vis_pos = sum(self._labels_vis)
        print(f"  is_close_and_visible: {vis_pos} positive / {self.n_samples} total "
              f"({100*vis_pos/self.n_samples:.1f}%)")


# =============================================================================
# Part 2: 线性探针训练 + 层-精度曲线（离线分析）
# =============================================================================

class LayerProbe(nn.Module):
    """轻量线性分类头，对每一层单独训练。"""

    def __init__(self, hidden_dim: int, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _load_probe_data(data_path: str):
    """Load one or more probe shards. Supports comma-separated paths and glob patterns."""
    paths = []
    for part in data_path.split(","):
        part = part.strip()
        if not part:
            continue
        matches = sorted(glob.glob(part))
        paths.extend(matches if matches else [part])

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"数据文件不存在: {missing[0]}")
    if not paths:
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    print(f"[Probe] Loading {len(paths)} shard(s):")
    loaded = []
    for path in paths:
        print(f"  - {path}")
        loaded.append(torch.load(path, map_location="cpu"))

    if len(loaded) == 1:
        return loaded[0]

    label_keys = loaded[0]["labels"].keys()
    return {
        "hidden_states": torch.cat([d["hidden_states"] for d in loaded], dim=0),
        "labels": {
            key: torch.cat([d["labels"][key] for d in loaded], dim=0)
            for key in label_keys
        },
    }


def train_probes(
    data_path: str,
    label_key: str = "is_close_and_visible",
    n_epochs: int = 20,
    lr: float = 1e-3,
    train_frac: float = 0.8,
    device: str = "cpu",
    out_path: str = "probe_result.png",
):
    """
    加载采集数据，对每一 Decoder 层训练独立线性探针，输出层-精度曲线。

    Parameters
    ----------
    data_path   : HiddenStateCollector.save() 产生的 .pt 文件路径
    label_key   : "is_close_and_visible" | "action_was_done"
    n_epochs    : 每个探针训练轮次
    lr          : AdamW 学习率
    train_frac  : 训练/测试切分比
    device      : "cpu" 或 "cuda:0"
    out_path    : 输出图片路径
    """
    data = _load_probe_data(data_path)
    hs = data["hidden_states"]          # [N, L, D]
    labels = data["labels"][label_key]  # [N]

    N, L, D = hs.shape
    n_classes = int(labels.max().item()) + 1
    print(f"  N={N} L={L} D={D} n_classes={n_classes} label={label_key}")
    class_counts = torch.bincount(labels, minlength=n_classes)
    class_ratio = class_counts.float() / max(1, int(class_counts.sum().item()))
    print(
        "  class distribution: "
        + ", ".join(
            f"class {i}={int(class_counts[i])} ({class_ratio[i].item()*100:.1f}%)"
            for i in range(n_classes)
        )
    )

    # 按层逐一训练探针
    layer_accs: List[float] = []
    layer_bal_accs: List[float] = []
    device_t = torch.device(device)

    for layer_idx in range(L):
        x = hs[:, layer_idx, :].to(device_t)  # [N, D]
        y = labels.to(device_t)

        dataset = TensorDataset(x, y)
        n_train = int(train_frac * N)
        n_val = N - n_train
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

        probe = LayerProbe(D, n_classes).to(device_t)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
        counts = torch.bincount(y, minlength=n_classes).float()
        class_weights = counts.sum() / (counts.clamp_min(1.0) * max(1, n_classes))
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device_t))

        best_acc = 0.0
        best_bal_acc = 0.0
        for _ in range(n_epochs):
            probe.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(probe(xb), yb)
                loss.backward()
                optimizer.step()

            probe.eval()
            correct = total = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = probe(xb).argmax(dim=1)
                    correct += (pred == yb).sum().item()
                    total += len(yb)
                    all_preds.append(pred.detach().cpu())
                    all_targets.append(yb.detach().cpu())
            acc = correct / total if total > 0 else 0.0
            if all_preds:
                pred_all = torch.cat(all_preds)
                target_all = torch.cat(all_targets)
                recalls = []
                for cls_idx in range(n_classes):
                    cls_mask = target_all == cls_idx
                    if cls_mask.any():
                        recalls.append(
                            (pred_all[cls_mask] == cls_idx).float().mean().item()
                        )
                bal_acc = float(np.mean(recalls)) if recalls else 0.0
            else:
                bal_acc = 0.0
            best_acc = max(best_acc, acc)
            best_bal_acc = max(best_bal_acc, bal_acc)

        layer_accs.append(best_acc)
        layer_bal_accs.append(best_bal_acc)
        if layer_idx % max(1, L // 8) == 0 or layer_idx == L - 1:
            print(
                f"  Layer {layer_idx:>3d}/{L-1}: "
                f"val_acc={best_acc*100:.1f}% "
                f"bal_acc={best_bal_acc*100:.1f}%"
            )

    # 与动作输出头的实际精度对比（用 action_was_done 作为 proxy）
    # ── 如果动作分类头无误分，则 action_was_done 与 is_close_and_visible 高度一致
    _plot_results(layer_accs, label_key, out_path, layer_bal_accs=layer_bal_accs)
    _print_summary(layer_accs, label_key, layer_bal_accs=layer_bal_accs)
    return layer_accs


def _print_summary(
    layer_accs: List[float],
    label_key: str,
    layer_bal_accs: Optional[List[float]] = None,
):
    # Balanced accuracy is the main diagnostic when close/visible positives are rare.
    arr = np.array(layer_bal_accs if layer_bal_accs is not None else layer_accs)
    raw_arr = np.array(layer_accs)
    peak_layer = int(np.argmax(arr))
    peak_acc = arr[peak_layer]
    final_acc = arr[-1]
    early_acc = arr[0] if arr.shape[0] > 0 else 0.0
    print("\n" + "="*60)
    print(f"探针诊断报告  label={label_key}")
    print("="*60)
    print(f"层数 L = {len(layer_accs)}")
    metric_name = "Balanced Acc" if layer_bal_accs is not None else "Accuracy"
    print(f"主指标 = {metric_name}")
    print(f"  第 0 层主指标    : {early_acc*100:.1f}%  (输入嵌入层)")
    print(f"  峰值层主指标     : {peak_acc*100:.1f}%  (Layer {peak_layer})")
    print(f"  最后层主指标     : {final_acc*100:.1f}%  (动作头前)")
    if layer_bal_accs is not None:
        print(
            f"  最后层 raw acc   : {raw_arr[-1]*100:.1f}%  "
            "(仅作参考，类别不平衡时会虚高)"
        )
    gap = peak_acc - final_acc
    print(f"  峰值 - 最后层    : {gap*100:+.1f}pp")
    print()
    if gap > 0.10:
        print("  ✅ 检测到明显的表征-输出不对齐！")
        print(f"     中间层已编码目标可见/距离信息(>{peak_acc*100:.0f}%)，")
        print(f"     但动作头前精度跌至 {final_acc*100:.0f}%。")
        print(f"     GRASP 类 VSV 干预有理论依据，建议进入阶段 2。")
    elif gap > 0.03:
        print("  ⚠️  存在轻度表征-输出不对齐，可进一步分析。")
    else:
        print("  ℹ️  未发现显著不对齐，失败根因可能在感知层而非动作头。")


def _plot_results(
    layer_accs: List[float],
    label_key: str,
    out_path: str,
    layer_bal_accs: Optional[List[float]] = None,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        L = len(layer_accs)
        xs = list(range(L))
        peak_idx = int(np.argmax(layer_accs))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, [a * 100 for a in layer_accs], "b-o", ms=4, label="Probe Accuracy")
        main_curve = layer_accs
        if layer_bal_accs is not None:
            ax.plot(
                xs,
                [a * 100 for a in layer_bal_accs],
                "m-s",
                ms=4,
                label="Balanced Accuracy",
            )
            main_curve = layer_bal_accs
            peak_idx = int(np.argmax(layer_bal_accs))
        ax.axhline(main_curve[-1] * 100, color="red", ls="--", lw=1.2,
                   label=f"Final layer ({main_curve[-1]*100:.1f}%)")
        ax.axvline(peak_idx, color="green", ls=":", lw=1.2,
                   label=f"Peak @ layer {peak_idx} ({main_curve[peak_idx]*100:.1f}%)")
        # 标注最后层精度（action head 的代理精度）
        ax.fill_between(xs[peak_idx:], main_curve[-1]*100, [a*100 for a in main_curve[peak_idx:]],
                        alpha=0.15, color="orange", label="Suppression zone")
        ax.set_xlabel("Decoder Layer Index")
        ax.set_ylabel("Probe Val Accuracy (%)")
        ax.set_title(f"SafeVLA Representation-Output Alignment\n(label: {label_key})")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_ylim(40, 105)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"[Probe] Plot saved to {out_path}")
    except ImportError:
        print("[Probe] matplotlib not available, skip plot.")


# =============================================================================
# Part 3: 快速接入说明（打印到 stdout）
# =============================================================================

INTEGRATION_GUIDE = """
============================================================
接入 SafeVLA 评测循环的最小改动（约 10 行）
============================================================

【文件】online_evaluation/grpo_predictive_inference.py

1. 在 GRPOPredictiveAgent.__init__ 中初始化采集器：
   
   from online_evaluation.probe_small_object import HiddenStateCollector, DecoderLayerHook
   self._probe_collector = HiddenStateCollector(max_samples=8000)
   # actor_critic 是 allenact_dino_transformer 里的模型；
   # 单 belief 配置下 self.decoder = LLAMATransformerDecoder(...)
   self._probe_hook = DecoderLayerHook(base_policy.actor_critic.decoder)

2. 在 act() 中，把 get_action_probs 包进 hook 上下文：

   with self._probe_hook:
       probs, action_list = self.policy.get_action_probs(frame, goal_spec)
   hs = self._probe_hook.get_last_step()   # [L, D]，可能 None

3. 在 worker 的 action 前设置 pre-action truth，update_after_execution() 中记录：

   label = {
       "is_close_and_visible": bool(task.successful_if_done(strict_success=False)),
       "target_distance":      float(task.dist_to_target_func()),
   }
   self.set_probe_label(label)

   # update_after_execution() 中消费 _last_hs + pre-action label
   self._probe_collector.record(
       getattr(self, "_last_hs", None), label
   )
   if self._probe_collector.n_samples % 1000 == 0:
       self._probe_collector.save("probe_data.pt")

   # 把 hs 从 act() 传过来只需在 act() 末尾加:
   self._last_hs = hs

【注意】
- DecoderLayerHook 需要能访问 decoder 的路径。
  SafeVLA 的路径是: policy.actor_critic.decoder
  （InferenceAgent → self.actor_critic → allenact_dino_transformer → self.decoder）
  如果路径不对，运行时会 AttributeError，根据报错调整即可。

- 采集约 2000~5000 个样本（多个 episode）就够做探针分析。
  推荐在 200 个 episode 评测完后再运行离线分析脚本。

【离线分析】
python online_evaluation/probe_small_object.py \\
    --data 'probe_data_worker*.pt' \\
    --label is_close_and_visible \\
    --out probe_result.png
============================================================
"""


# =============================================================================
# CLI 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SafeVLA 小目标线性探针分析")
    parser.add_argument("--data",   required=True,  help="probe_data.pt 路径")
    parser.add_argument("--label",  default="is_close_and_visible",
                        choices=["is_close_and_visible", "action_was_done"],
                        help="预测目标标签")
    parser.add_argument("--epochs", type=int,   default=20,     help="每层探针训练轮次")
    parser.add_argument("--lr",     type=float, default=1e-3,   help="AdamW 学习率")
    parser.add_argument("--device", default="cpu",               help="cuda:0 / cpu")
    parser.add_argument("--out",    default="probe_result.png",  help="输出图片路径")
    args = parser.parse_args()

    train_probes(
        data_path=args.data,
        label_key=args.label,
        n_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        out_path=args.out,
    )


if __name__ == "__main__":
    import sys
    if "--guide" in sys.argv:
        print(INTEGRATION_GUIDE)
    else:
        main()
