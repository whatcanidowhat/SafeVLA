import torch
import numpy as np
import copy
from collections import Counter, deque


# =============================================================================
# 1. 辅助模块：记忆与预测器 (必须包含在文件中以保证运行)
# =============================================================================

class SpatialMemory:
    """简易空间记忆模块：维护机器人周围 4 个方位的障碍物状态"""

    def __init__(self):
        self.occupancy = [False, False, False, False]  # 前, 右, 后, 左
        self.decay_counters = [0, 0, 0, 0]
        self.MEMORY_PERSISTENCE = 5

    def update(self, current_depth_risk, last_action_str):
        new_occupancy = self.occupancy[:]
        new_counters = self.decay_counters[:]

        # 简单的旋转逻辑模拟
        rotate_left_tokens = {"l", "ls", "rotate_left", "rotate_left_small"}
        rotate_right_tokens = {"r", "rs", "rotate_right", "rotate_right_small"}

        if last_action_str in rotate_left_tokens:
            new_occupancy = [self.occupancy[3], self.occupancy[0], self.occupancy[1], self.occupancy[2]]
            new_counters = [self.decay_counters[3], self.decay_counters[0], self.decay_counters[1],
                            self.decay_counters[2]]
        elif last_action_str in rotate_right_tokens:
            new_occupancy = [self.occupancy[1], self.occupancy[2], self.occupancy[3], self.occupancy[0]]
            new_counters = [self.decay_counters[1], self.decay_counters[2], self.decay_counters[3],
                            self.decay_counters[0]]

        # 衰减
        for i in range(4):
            if new_counters[i] > 0:
                new_counters[i] -= 1
            else:
                new_occupancy[i] = False

        # 写入当前观测
        if current_depth_risk:
            new_occupancy[0] = True
            new_counters[0] = self.MEMORY_PERSISTENCE

        self.occupancy = new_occupancy
        self.decay_counters = new_counters

    def check_risk(self, direction_idx):
        return self.occupancy[direction_idx]


class HeuristicSafetyPredictor:
    def __init__(self):
        self.SAFE_DISTANCE = 0.5
        self.CRITICAL_DISTANCE = 0.35
        self.memory = SpatialMemory()

    def update_state(self, real_info, last_action_str):
        """更新记忆状态"""
        front_blocked = False
        if "depth" in real_info:
            depth_map = real_info["depth"]
            if isinstance(depth_map, torch.Tensor):
                depth_map = depth_map.detach().cpu().numpy()
            h, w = depth_map.shape[:2]
            center_depth = np.mean(depth_map[h // 3:2 * h // 3, w // 3:2 * w // 3])
            if center_depth < self.CRITICAL_DISTANCE:
                front_blocked = True
        self.memory.update(front_blocked, last_action_str)

    def predict_heuristic(self, obs_history, action_str, current_obs_metadata=None):
        """
        Predict heuristic risks using only the internal spatial memory.

        The action strings in this repo are usually AllenAct/Stretch tokens like:
        - move ahead: `m`
        - move back: `b`
        - rotate left/right: `l/r` (and small variants `ls/rs`)
        and may also be long names if `LONG_ACTION_NAME=1`.
        """
        move_ahead_tokens = {"m", "move_ahead"}
        move_back_tokens = {"b", "move_back"}
        rotate_left_tokens = {"l", "ls", "rotate_left", "rotate_left_small"}
        rotate_right_tokens = {"r", "rs", "rotate_right", "rotate_right_small"}

        risk_scores = {"Corner": 0.0, "BlindSpot": 0.0, "Dangerous": 0.0}
        progress_score = 0.0
        feedback_msg = None

        # 1) Corner risk: moving forward into remembered obstacle in "front".
        if action_str in move_ahead_tokens:
            if self.memory.check_risk(0):  # Front
                risk_scores["Corner"] = 1.0
                feedback_msg = "Critical: obstacle remembered in front; don't move ahead."
            else:
                progress_score += 0.1

        # 2) Blind spot risk: moving backward into remembered obstacle in "back".
        elif action_str in move_back_tokens:
            if self.memory.check_risk(2):  # Back
                risk_scores["BlindSpot"] = 0.9
                feedback_msg = "Warning: obstacle remembered behind; don't move back."
            else:
                progress_score += 0.02

        # 3) Rotations are allowed (progress-free) so the agent can reorient.
        elif action_str in rotate_left_tokens or action_str in rotate_right_tokens:
            progress_score += 0.0

        # 4) Prefer navigation actions over manipulation actions when no other signal exists.
        else:
            progress_score -= 0.02

        return risk_scores, progress_score, feedback_msg


class R1PredictiveRewardSystem:
    def calculate_predicted_reward(self, risk_scores, progress_score, action_str):
        # 简单奖励逻辑: R = Progress - max(Risks)
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        return progress_score - max_risk * 2.0  # 惩罚权重加大

    def update_feedback(self, real_info):
        pass


# =============================================================================
# 2. 核心 Agent (修复了 act 逻辑和历史记录问题)
# =============================================================================

class GRPOPredictiveAgent:
    def __init__(self, base_policy, num_samples=8, max_refinement_steps=2, temperature=1.2):
        self.policy = base_policy
        self.G = num_samples
        self.max_refinement_steps = max_refinement_steps
        self.temperature = temperature

        self.predictor = HeuristicSafetyPredictor()
        self.reward_system = R1PredictiveRewardSystem()
        self.obs_history = []
        self.last_executed_action = "None"

        print(f"[GRPO] Initialized: N={self.G}, RefineSteps={self.max_refinement_steps}, Temp={self.temperature}")

    def _get_current_metadata(self, observation):
        """从当前观测中提取元数据，用于第一帧预测"""
        meta = {}
        # 尝试从 tensor 中提取深度均值 (假设 depth 是 tensor)
        if "depth" in observation:
            d = observation["depth"]
            if isinstance(d, torch.Tensor):
                d = d.cpu().numpy()
            # 假设 shape 是 (C, H, W) 或 (H, W, C)
            if len(d.shape) == 3:
                h, w = d.shape[1], d.shape[2] if d.shape[0] < d.shape[2] else (d.shape[0], d.shape[1])
                # 简单取个均值作为近似
                meta['depth_mean'] = float(np.mean(d))
        return meta

    def act(self, frame, goal_spec, prev_action=None):
        """
        One-shot GRPO-style selection:
        - compute base policy action distribution (probs)
        - sample N candidate actions from probs
        - score candidates with heuristic safety predictor + reward system
        - commit the selected action back into the base policy rollout state
        """
        with torch.no_grad():
            probs, action_list = self.policy.get_action_probs(frame, goal_spec)

            # Normalize + clamp for numerical stability.
            probs = probs.float()
            probs = probs.clamp(min=0)
            probs_sum = probs.sum()
            if probs_sum.item() <= 0:
                probs = torch.ones_like(probs) / probs.numel()
            else:
                probs = probs / probs_sum

            # Sample candidate actions.
            candidate_indices_t = torch.multinomial(
                probs, num_samples=self.G, replacement=True
            )
            candidate_indices = candidate_indices_t.cpu().numpy().astype(int).flatten()

            best_action_idx = int(candidate_indices[0])
            best_reward = -float("inf")

            for idx in candidate_indices:
                idx = int(idx)
                if idx < 0 or idx >= len(action_list):
                    continue
                action_str = action_list[idx]

                risk_scores, progress, _ = self.predictor.predict_heuristic(
                    self.obs_history, action_str
                )
                r = self.reward_system.calculate_predicted_reward(
                    risk_scores=risk_scores,
                    progress_score=progress,
                    action_str=action_str,
                )
                if r > best_reward:
                    best_reward = float(r)
                    best_action_idx = idx

            chosen_action_str = self.policy.commit_action_by_index(best_action_idx)
            self.last_executed_action = chosen_action_str
            return chosen_action_str, probs

    def update_after_execution(self, real_info):
        """每步执行完后调用，用真实深度/环境信号刷新记忆。"""
        self.reward_system.update_feedback(real_info)
        self.predictor.update_state(real_info, self.last_executed_action)

    def reset(self):
        self.predictor = HeuristicSafetyPredictor()
        self.reward_system = R1PredictiveRewardSystem()
        self.obs_history = []
        self.last_executed_action = "None"


# =============================================================================
# 3. 必须包含的 Proxy 类 (修复 Module Not Found 和 AttributeError 问题)
# =============================================================================

class GRPOAgentProxy:
    """
    OnlineEvaluator 需要调用的代理类。
    负责加载原始模型，并包裹 GRPOPredictiveAgent。
    """

    def __init__(self, base_agent, **kwargs):
        self.base_agent = base_agent

        # 提取参数，支持从外部传入温度等配置
        temp = kwargs.get("temperature", 1.2)

        print(f"[GRPO Proxy] Wrapping Agent... Temp={temp}")
        self.grpo_agent = GRPOPredictiveAgent(
            base_policy=self.base_agent,
            num_samples=8,
            max_refinement_steps=2,
            temperature=temp
        )

    @classmethod
    def build_agent(cls, **kwargs):
        # 1. 提取原始 Agent 类
        real_cls = kwargs.pop('__original_agent_class', None)

        if real_cls is None:
            # 如果没传，尝试从 input 字典里找（防止不同版本的调用差异）
            print("[GRPO Proxy] Warning: __original_agent_class missing, checking agent_input...")
            return None

            # 2. 加载原始模型 (SafeVLA / SPOC)
        if hasattr(real_cls, 'build_agent'):
            base_agent = real_cls.build_agent(**kwargs)
        else:
            base_agent = real_cls(**kwargs)

        # 3. 返回 Proxy 实例
        return cls(base_agent=base_agent, **kwargs)

    def act(self, *args, **kwargs):
        # 转发推理请求
        return self.grpo_agent.act(*args, **kwargs)

    def update_after_execution(self, *args, **kwargs):
        # 转发更新请求
        if hasattr(self.grpo_agent, 'update_after_execution'):
            self.grpo_agent.update_after_execution(*args, **kwargs)

    def __getattr__(self, name):
        # 转发其他属性 (如 .vocab, .model) 到原始 Agent
        return getattr(self.base_agent, name)