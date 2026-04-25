import torch
import numpy as np
import copy
from collections import Counter


# =============================================================================
# 1. 辅助模块：均衡预测器 (物理与语义同等高压)/极致刷榜版
# =============================================================================

class SpatialMemory:
    """极其轻量的短期记忆，防止在死角无限震荡"""

    def __init__(self):
        self.occupancy = [False, False, False, False]
        self.decay_counters = [0, 0, 0, 0]
        self.MEMORY_PERSISTENCE = 3

    def update(self, current_depth_risk, last_action_str):
        new_occupancy = self.occupancy[:]
        new_counters = self.decay_counters[:]

        if "RotateLeft" in last_action_str:
            new_occupancy = [self.occupancy[3], self.occupancy[0], self.occupancy[1], self.occupancy[2]]
            new_counters = [self.decay_counters[3], self.decay_counters[0], self.decay_counters[1],
                            self.decay_counters[2]]
        elif "RotateRight" in last_action_str:
            new_occupancy = [self.occupancy[1], self.occupancy[2], self.occupancy[3], self.occupancy[0]]
            new_counters = [self.decay_counters[1], self.decay_counters[2], self.decay_counters[3],
                            self.decay_counters[0]]

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


class HeuristicSafetyPredictor:
    def __init__(self):
        # 恢复对物理撞墙的敏锐度，防止乱撞
        self.CRITICAL_DISTANCE = 0.25
        self.memory = SpatialMemory()

        # 危险与易碎品词库
        self.danger_keywords = ["stove", "burner", "knife", "fork", "fire"]
        self.fragile_keywords = ["vase", "glass", "laptop", "monitor", "television", "bottle", "bowl", "mug"]

    def update_state(self, current_meta, last_action_str):
        front_blocked = current_meta.get('depth_mean', 10.0) < self.CRITICAL_DISTANCE
        self.memory.update(front_blocked, last_action_str)

    def predict_heuristic(self, obs_history, action_str, current_obs_metadata=None):
        risk_scores = {"Corner": 0.0, "BlindSpot": 0.0, "Dangerous": 0.0, "Fragile": 0.0}
        progress_score = 0.0
        feedback_msg = None

        if current_obs_metadata:
            depth_val = current_obs_metadata.get('depth_mean', 10.0)
        else:
            depth_val = obs_history[-1].get('depth_mean', 10.0) if obs_history else 10.0

        closest_obj = ""
        if current_obs_metadata and 'closest_object' in current_obs_metadata:
            closest_obj = current_obs_metadata['closest_object'].lower()
        elif obs_history and 'closest_object' in obs_history[-1]:
            closest_obj = obs_history[-1]['closest_object'].lower()

        # 1. 语义防线 (Semantic)
        is_danger = any(k in closest_obj for k in self.danger_keywords)
        is_fragile = any(k in closest_obj for k in self.fragile_keywords)

        if (is_danger or is_fragile) and depth_val < 0.6:
            if "MoveAhead" in action_str:
                if is_danger:
                    risk_scores["Dangerous"] = 1.0
                    feedback_msg = f"DANGER: {closest_obj} ahead."
                elif is_fragile:
                    risk_scores["Fragile"] = 0.9
                    feedback_msg = f"WARNING: Fragile {closest_obj} ahead."
            elif "Pickup" in action_str or "Interact" in action_str:
                if is_danger: risk_scores["Dangerous"] = 1.0

        # 2. 物理防撞防线 (Corner) - 绝对不能撞墙
        if action_str == "MoveAhead":
            if depth_val < self.CRITICAL_DISTANCE:
                risk_scores["Corner"] = 1.0  # 极高惩罚，一票否决
                if not feedback_msg: feedback_msg = "Obstacle directly ahead."
            elif risk_scores["Dangerous"] == 0 and risk_scores["Fragile"] == 0:
                # 只有前面既没墙，又没危险品，前进才有正向奖励
                progress_score += 0.2

                # 3. 盲区防线 (BlindSpot)
        target_dir = -1
        if "MoveLeft" in action_str:
            target_dir = 3
        elif "MoveRight" in action_str:
            target_dir = 1
        elif "MoveBack" in action_str:
            target_dir = 2

        if target_dir != -1 and self.memory.check_risk(target_dir):
            risk_scores["BlindSpot"] = 0.8
            if not feedback_msg: feedback_msg = "Blind spot obstacle."

        return risk_scores, progress_score, feedback_msg


class R1PredictiveRewardSystem:
    def calculate_predicted_reward(self, risk_scores, progress_score, action_str):
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        # 风险一票否决制：只要有风险，动作就是负分。安全转头为0分。安全前进为正分。
        return progress_score - (max_risk * 2.0)


# =============================================================================
# 2. 核心 Agent (Best-of-N Re-ranking 机制)
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
        self._depth_warning_printed = False

        print(f"[GRPO] Initialized for SOTA: N={self.G}, Temp={self.temperature}", flush=True)

    def _get_current_metadata(self, observation):
        meta = {}
        depth_keys = ["depth", "depth_sensor", "raw_depth_camera", "raw_navigation_depth"]
        found_depth = False

        for key in depth_keys:
            if key in observation:
                d = observation[key]
                if isinstance(d, torch.Tensor): d = d.cpu().numpy()
                if len(d.shape) >= 2:
                    if len(d.shape) == 3 and d.shape[0] < 5:
                        center_d = d[:, d.shape[1] // 3:2 * d.shape[1] // 3, d.shape[2] // 3:2 * d.shape[2] // 3]
                    else:
                        center_d = d[d.shape[0] // 3:2 * d.shape[0] // 3, d.shape[1] // 3:2 * d.shape[1] // 3]
                    meta['depth_mean'] = float(np.mean(center_d))
                found_depth = True
                break

        if "closest_object_name" in observation:
            meta["closest_object"] = observation["closest_object_name"]

        if not found_depth and not self._depth_warning_printed:
            print(f"\n[Warning] No depth sensor found!\n", flush=True)
            self._depth_warning_printed = True

        return meta

    def act(self, observation, prev_action=None):
        target_key = "text"
        if "natural_language_instruction" in observation:
            target_key = "natural_language_instruction"
        elif "goal" in observation:
            target_key = "goal"

        original_instruction = observation.get(target_key, "")
        current_meta = self._get_current_metadata(observation)
        current_obs = copy.copy(observation)

        for step in range(self.max_refinement_steps + 1):
            with torch.no_grad():
                logits = self.policy.get_logits(current_obs)
                logits = logits / self.temperature
                probs = torch.softmax(logits, dim=-1)

            # 采样 8 个动作
            candidate_indices = torch.multinomial(probs, self.G, replacement=True)
            candidate_indices = candidate_indices.cpu().numpy().flatten()

            rewards = []
            feedbacks = []

            for idx in candidate_indices:
                action_str = self.policy.vocab.index2word(idx)
                risks, progress, msg = self.predictor.predict_heuristic(
                    self.obs_history, action_str, current_obs_metadata=current_meta
                )
                r = self.reward_system.calculate_predicted_reward(risks, progress, action_str)
                rewards.append(r)
                if msg: feedbacks.append(msg)

            rewards = np.array(rewards)
            best_idx_in_batch = np.argmax(rewards)
            best_reward = rewards[best_idx_in_batch]
            best_action_idx = candidate_indices[best_idx_in_batch]

            # 【核心逻辑修改】
            # 极简干预原则：只要 best_reward >= -0.5 (说明至少找到了一个安全的旋转动作或前进动作)
            # 就直接执行它，绝不修改 Prompt 打扰模型！
            if step < self.max_refinement_steps and best_reward < -0.5 and feedbacks:
                most_common_feedback = Counter(feedbacks).most_common(1)[0][0]
                # 只有走投无路时，才用 Prompt 强行引导
                new_instruction = f"{original_instruction} (Note: {most_common_feedback})"
                current_obs[target_key] = new_instruction
                continue
            else:
                # 选出最安全的动作，直接放行
                self.last_executed_action = self.policy.vocab.index2word(best_action_idx)
                if target_key in observation:
                    observation[target_key] = original_instruction

                self.update_after_execution(current_meta)
                return best_action_idx

    def update_after_execution(self, current_meta):
        self.predictor.update_state(current_meta, self.last_executed_action)
        self.obs_history.append(current_meta)
        if len(self.obs_history) > 5:
            self.obs_history.pop(0)


# =============================================================================
# 3. 代理类 (保持不变)
# =============================================================================

class GRPOAgentProxy:
    def __init__(self, base_agent, **kwargs):
        self.base_agent = base_agent
        temp = kwargs.get("temperature", 1.2)
        self.grpo_agent = GRPOPredictiveAgent(
            base_policy=self.base_agent,
            num_samples=8,
            max_refinement_steps=2,
            temperature=temp
        )

    @classmethod
    def build_agent(cls, **kwargs):
        real_cls = kwargs.pop('__original_agent_class', None)
        if real_cls is None: return None
        if hasattr(real_cls, 'build_agent'):
            return cls(base_agent=real_cls.build_agent(**kwargs), **kwargs)
        else:
            return cls(base_agent=real_cls(**kwargs), **kwargs)

    def act(self, *args, **kwargs):
        return self.grpo_agent.act(*args, **kwargs)

    def update_after_execution(self, real_info, *args, **kwargs):
        if hasattr(self.grpo_agent, 'obs_history') and len(self.grpo_agent.obs_history) > 0:
            if "closest_object_name" in real_info:
                self.grpo_agent.obs_history[-1]["closest_object"] = real_info["closest_object_name"]

    def __getattr__(self, name):
        return getattr(self.base_agent, name)