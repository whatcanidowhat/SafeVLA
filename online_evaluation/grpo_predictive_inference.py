import torch
import numpy as np
from collections import Counter


# =============================================================================
# 1. 辅助模块：均衡预测器 (物理 + 语义同等高压) / 刷榜版奖励曲线
#    CRITICAL_DISTANCE=0.25, 风险一票否决, 极简干预原则
# =============================================================================

# 与 utils/type_utils.THORActions 对齐的短 token (无 LONG_ACTION_NAME / ACTION_DICT 时即为它们)。
_MOVE_AHEAD_TOKENS = {"m", "move_ahead", "MoveAhead"}
_MOVE_BACK_TOKENS = {"b", "move_back", "MoveBack"}
_ROTATE_LEFT_TOKENS = {"l", "ls", "rotate_left", "rotate_left_small", "RotateLeft", "RotateLeftSmall"}
_ROTATE_RIGHT_TOKENS = {"r", "rs", "rotate_right", "rotate_right_small", "RotateRight", "RotateRightSmall"}
_PICKUP_TOKENS = {"p", "pickup", "Pickup"}


def _is_token(action_str, token_set):
    return action_str in token_set


class SpatialMemory:
    """极轻量短期记忆，防止在死角无限震荡 (4 方位: 前/右/后/左)。"""

    def __init__(self):
        self.occupancy = [False, False, False, False]
        self.decay_counters = [0, 0, 0, 0]
        self.MEMORY_PERSISTENCE = 3

    def update(self, current_depth_risk, last_action_str):
        new_occupancy = self.occupancy[:]
        new_counters = self.decay_counters[:]

        if _is_token(last_action_str, _ROTATE_LEFT_TOKENS):
            new_occupancy = [self.occupancy[3], self.occupancy[0], self.occupancy[1], self.occupancy[2]]
            new_counters = [self.decay_counters[3], self.decay_counters[0], self.decay_counters[1], self.decay_counters[2]]
        elif _is_token(last_action_str, _ROTATE_RIGHT_TOKENS):
            new_occupancy = [self.occupancy[1], self.occupancy[2], self.occupancy[3], self.occupancy[0]]
            new_counters = [self.decay_counters[1], self.decay_counters[2], self.decay_counters[3], self.decay_counters[0]]

        for i in range(4):
            if new_counters[i] > 0:
                new_counters[i] -= 1
            else:
                new_occupancy[i] = False

        if current_depth_risk:
            new_occupancy[0] = True
            new_counters[0] = self.MEMORY_PERSISTENCE

        self.occupancy = new_occupancy
        self.decay_counters = new_counters

    def check_risk(self, direction_idx):
        return self.occupancy[direction_idx]


def _extract_depth_mean(arr):
    """从任意 (H,W) / (H,W,C) / (C,H,W) 形状的深度阵列里取中央 1/3 区域均值。"""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if arr is None or not hasattr(arr, "shape") or arr.ndim < 2:
        return None
    if arr.ndim == 3 and arr.shape[0] < 5:  # (C, H, W)
        center = arr[:, arr.shape[1] // 3:2 * arr.shape[1] // 3, arr.shape[2] // 3:2 * arr.shape[2] // 3]
    else:                                    # (H, W) 或 (H, W, C)
        center = arr[arr.shape[0] // 3:2 * arr.shape[0] // 3, arr.shape[1] // 3:2 * arr.shape[1] // 3]
    try:
        return float(np.mean(center))
    except Exception:
        return None


class HeuristicSafetyPredictor:
    def __init__(self):
        self.CRITICAL_DISTANCE = 0.25  # 刷榜版: 对物理撞墙的高敏锐度
        self.memory = SpatialMemory()
        self.danger_keywords = ["stove", "burner", "knife", "fork", "fire"]
        self.fragile_keywords = ["vase", "glass", "laptop", "monitor", "television", "bottle", "bowl", "mug"]

    def update_state(self, real_info, last_action_str):
        """每步执行完后由 update_after_execution 调用; real_info 可能是 worker 发的原始
        ({'depth': raw_depth, 'closest_object_name': ...}) 也可能是 act() 内部的预处理 meta。"""
        front_blocked = False
        if isinstance(real_info, dict):
            if "depth_mean" in real_info:
                front_blocked = real_info["depth_mean"] < self.CRITICAL_DISTANCE
            elif "depth" in real_info:
                d_mean = _extract_depth_mean(real_info["depth"])
                if d_mean is not None:
                    front_blocked = d_mean < self.CRITICAL_DISTANCE
        self.memory.update(front_blocked, last_action_str)

    def predict_heuristic(self, obs_history, action_str, current_obs_metadata=None):
        risk_scores = {"Corner": 0.0, "BlindSpot": 0.0, "Dangerous": 0.0, "Fragile": 0.0}
        progress_score = 0.0
        feedback_msg = None

        if current_obs_metadata:
            depth_val = current_obs_metadata.get("depth_mean", 10.0)
        else:
            depth_val = obs_history[-1].get("depth_mean", 10.0) if obs_history else 10.0

        closest_obj = ""
        if current_obs_metadata and "closest_object" in current_obs_metadata:
            closest_obj = (current_obs_metadata["closest_object"] or "").lower()
        elif obs_history and "closest_object" in obs_history[-1]:
            closest_obj = (obs_history[-1]["closest_object"] or "").lower()

        is_move_ahead = _is_token(action_str, _MOVE_AHEAD_TOKENS)
        is_move_back = _is_token(action_str, _MOVE_BACK_TOKENS)
        is_pickup = _is_token(action_str, _PICKUP_TOKENS)

        is_danger = any(k in closest_obj for k in self.danger_keywords)
        is_fragile = any(k in closest_obj for k in self.fragile_keywords)

        # 1. 语义防线
        if (is_danger or is_fragile) and depth_val < 0.6:
            if is_move_ahead:
                if is_danger:
                    risk_scores["Dangerous"] = 1.0
                    feedback_msg = f"DANGER: {closest_obj} ahead."
                elif is_fragile:
                    risk_scores["Fragile"] = 0.9
                    feedback_msg = f"WARNING: Fragile {closest_obj} ahead."
            elif is_pickup:
                if is_danger:
                    risk_scores["Dangerous"] = 1.0

        # 2. 物理防撞防线 — 一票否决
        if is_move_ahead:
            if depth_val < self.CRITICAL_DISTANCE:
                risk_scores["Corner"] = 1.0
                if not feedback_msg:
                    feedback_msg = "Obstacle directly ahead."
            elif risk_scores["Dangerous"] == 0 and risk_scores["Fragile"] == 0:
                progress_score += 0.2  # 安全前进才有正向奖励

        # 3. 盲区防线 (后退方向 = 2)
        if is_move_back and self.memory.check_risk(2):
            risk_scores["BlindSpot"] = 0.8
            if not feedback_msg:
                feedback_msg = "Blind spot obstacle."

        return risk_scores, progress_score, feedback_msg


class R1PredictiveRewardSystem:
    def calculate_predicted_reward(self, risk_scores, progress_score, action_str):
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        # 风险一票否决: 任何 risk 都是负分; 安全旋转 = 0; 安全前进 > 0。
        return progress_score - (max_risk * 2.0)


# =============================================================================
# 2. 核心 Agent (单次前向 + Best-of-N 重排 + 记忆覆写)
# =============================================================================

class GRPOPredictiveAgent:
    def __init__(self, base_policy, num_samples=8, max_refinement_steps=2, temperature=1.2):
        # max_refinement_steps 仅为 API 兼容保留: 实际不做二次前向, 因为
        # base_policy.get_action_probs(...) 会推进 rollout_storage, 重复调用会污染状态。
        # 这恰好契合"极简干预原则": 不二次打扰模型。
        self.policy = base_policy
        self.G = num_samples
        self.max_refinement_steps = max_refinement_steps
        self.temperature = temperature

        self.predictor = HeuristicSafetyPredictor()
        self.reward_system = R1PredictiveRewardSystem()
        self.obs_history = []
        self.last_executed_action = "None"
        self._depth_warning_printed = False

        print(f"[GRPO] Initialized for SOTA: N={self.G}, Temp={self.temperature}", flush=True)

    def reset(self):
        self.predictor = HeuristicSafetyPredictor()
        self.reward_system = R1PredictiveRewardSystem()
        self.obs_history = []
        self.last_executed_action = "None"

    def _get_current_metadata(self, observation):
        """从当前帧 (worker 传给 get_action 的 dict) 抽取 depth_mean / closest_object,
        用于本步 best-of-N 候选评分阶段 (与 update_state 的全局记忆相互独立)。"""
        meta = {}
        for key in ("depth", "depth_sensor", "raw_depth_camera", "raw_navigation_depth"):
            if key in observation:
                d_mean = _extract_depth_mean(observation[key])
                if d_mean is not None:
                    meta["depth_mean"] = d_mean
                break
        if "closest_object_name" in observation:
            meta["closest_object"] = observation["closest_object_name"]
        if "depth_mean" not in meta and not self._depth_warning_printed:
            print("[GRPO][Warning] No depth in act-frame; relying on memory.", flush=True)
            self._depth_warning_printed = True
        return meta

    def act(self, frame, goal_spec, prev_action=None):
        """
        Best-of-N re-ranking with a single base-policy forward pass.

        - One call to base_policy.get_action_probs(frame, goal_spec) -> (probs, action_list)
        - Sample G candidate indices from temperature-scaled probs
        - Score candidates with HeuristicSafetyPredictor + R1 reward (一票否决)
        - Commit the argmax-reward index via base_policy.commit_action_by_index(best_idx)
        - Returns (chosen_action_str, probs) — matches the worker's
          `action, probs = agent.get_action(...)` contract.
        """
        current_meta = self._get_current_metadata(frame)

        with torch.no_grad():
            probs, action_list = self.policy.get_action_probs(frame, goal_spec)
            probs = probs.float()

            if self.temperature != 1.0:
                logits = torch.log(probs.clamp(min=1e-9)) / self.temperature
                probs_for_sampling = torch.softmax(logits, dim=-1)
            else:
                probs_for_sampling = probs.clamp(min=0.0)
                s = probs_for_sampling.sum()
                if s.item() <= 0:
                    probs_for_sampling = torch.ones_like(probs_for_sampling) / probs_for_sampling.numel()
                else:
                    probs_for_sampling = probs_for_sampling / s

            candidate_indices = (
                torch.multinomial(probs_for_sampling, self.G, replacement=True)
                .cpu()
                .numpy()
                .astype(int)
                .flatten()
            )

        rewards = []
        feedbacks = []
        for idx in candidate_indices:
            if idx < 0 or idx >= len(action_list):
                rewards.append(-float("inf"))
                continue
            action_str = action_list[idx]
            risks, progress, msg = self.predictor.predict_heuristic(
                self.obs_history, action_str, current_obs_metadata=current_meta
            )
            r = self.reward_system.calculate_predicted_reward(risks, progress, action_str)
            rewards.append(r)
            if msg:
                feedbacks.append(msg)

        rewards_arr = np.array(rewards, dtype=np.float64)
        best_idx_in_batch = int(np.argmax(rewards_arr))
        best_action_idx = int(candidate_indices[best_idx_in_batch])

        # 极简干预原则: 直接放行最佳候选, 绝不二次打扰模型 (rollout_storage 已被推进)。
        chosen_action_str = self.policy.commit_action_by_index(best_action_idx)
        self.last_executed_action = chosen_action_str

        # 当前帧 meta 进入历史; 真值 closest_object 等在 update_after_execution 里会再修正。
        self.obs_history.append(current_meta)
        if len(self.obs_history) > 5:
            self.obs_history.pop(0)

        # 占位以保留刷榜版日志钩子 (避免 lint 警告未使用变量)。
        _ = feedbacks and Counter(feedbacks).most_common(1)[0][0]

        return chosen_action_str, probs

    def update_after_execution(self, real_info):
        """worker 每步调用; real_info = {'depth': raw, 'closest_object_name': ...}"""
        self.predictor.update_state(real_info, self.last_executed_action)
        if (
            self.obs_history
            and isinstance(real_info, dict)
            and "closest_object_name" in real_info
        ):
            self.obs_history[-1]["closest_object"] = real_info["closest_object_name"]


# =============================================================================
# 3. 代理类 (实际生效的 Proxy 在 online_evaluator.py 中; 此处保留以便单独 import 测试)
# =============================================================================

class GRPOAgentProxy:
    def __init__(self, base_agent, **kwargs):
        self.base_agent = base_agent
        temp = kwargs.get("temperature", 1.2)
        self.grpo_agent = GRPOPredictiveAgent(
            base_policy=self.base_agent,
            num_samples=8,
            max_refinement_steps=2,
            temperature=temp,
        )

    @classmethod
    def build_agent(cls, **kwargs):
        real_cls = kwargs.pop("__original_agent_class", None)
        if real_cls is None:
            return None
        if hasattr(real_cls, "build_agent"):
            return cls(base_agent=real_cls.build_agent(**kwargs), **kwargs)
        return cls(base_agent=real_cls(**kwargs), **kwargs)

    def get_action(self, frame, goal_spec):
        return self.grpo_agent.act(frame, goal_spec)

    def act(self, *args, **kwargs):
        return self.grpo_agent.act(*args, **kwargs)

    def update_after_execution(self, real_info, *args, **kwargs):
        self.grpo_agent.update_after_execution(real_info)

    def reset(self):
        if hasattr(self.base_agent, "reset"):
            self.base_agent.reset()
        if self.grpo_agent is not None:
            self.grpo_agent.reset()

    def __getattr__(self, name):
        return getattr(self.base_agent, name)
