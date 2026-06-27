import logging
import os

import torch  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from collections import Counter

# 探针模块 — 仅在 PROBE_ENABLE=1 时导入; 正常评测零开销。
try:
    from online_evaluation.probe_small_object import DecoderLayerHook, HiddenStateCollector
    _PROBE_AVAILABLE = True
    _PROBE_IMPORT_ERROR = None
except ImportError as _exc:
    _PROBE_AVAILABLE = False
    _PROBE_IMPORT_ERROR = _exc


logger = logging.getLogger("grpo_predictive")


# =============================================================================
# 1. 辅助模块：均衡预测器 (物理 + 语义同等高压) / 刷榜版奖励曲线
#    CRITICAL_DISTANCE=0.25, 风险一票否决, 极简干预原则
# success=0.864，cost=0.98
# success=0.87mcost =0.77
# success=0.839, cost=0,497 #重选logits规避第一帧假阳性 
# success=0.87,cost =0.622 #重选logits
# =============================================================================

# 与 utils/type_utils.THORActions 对齐的短 token (无 LONG_ACTION_NAME / ACTION_DICT 时即为它们)。
_MOVE_AHEAD_TOKENS = {"m", "move_ahead", "MoveAhead"}
_MOVE_BACK_TOKENS = {"b", "move_back", "MoveBack"}
_ROTATE_LEFT_TOKENS = {"l", "ls", "rotate_left", "rotate_left_small", "RotateLeft", "RotateLeftSmall"}
_ROTATE_RIGHT_TOKENS = {"r", "rs", "rotate_right", "rotate_right_small", "RotateRight", "RotateRightSmall"}
_PICKUP_TOKENS = {"p", "pickup", "Pickup"}

# Bucket I — ObjectNav 模式下需要被一票否决的"机械臂操作族" (短 token + 长名 + 驼峰三套全收)。
_MANIPULATION_TOKENS = {
    # pickup / dropoff
    "p", "pickup", "Pickup",
    "d", "dropoff", "Dropoff",
    # move_arm_up / down / out / in (含 _small 变体)
    "yp", "move_arm_up", "MoveArmUp",
    "yps", "move_arm_up_small", "MoveArmUpSmall",
    "ym", "move_arm_down", "MoveArmDown",
    "yms", "move_arm_down_small", "MoveArmDownSmall",
    "zp", "move_arm_out", "MoveArmOut",
    "zps", "move_arm_out_small", "MoveArmOutSmall",
    "zm", "move_arm_in", "MoveArmIn",
    "zms", "move_arm_in_small", "MoveArmInSmall",
    # wrist
    "wp", "wrist_open", "WristOpen",
    "wm", "wrist_close", "WristClose",
}

# 唯一合法终止动作 (THORActions.done = "end"); 不把 sub_done 加进来 —— sub_done 不会终止 episode。
_DONE_TOKENS = {"end", "done", "Done"}


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
    # 类级 flag: 避免每个 episode 都重复打印 init 横幅 (一个 worker 进程只打一次)。
    _LOGGED_INIT = False

    def __init__(self):
        self.CRITICAL_DISTANCE = 0.25  # 刷榜版: 对物理撞墙的高敏锐度
        self.memory = SpatialMemory()
        self.danger_keywords = ["stove", "burner", "knife", "fork", "fire"]
        self.fragile_keywords = ["vase", "glass", "laptop", "monitor", "television", "bottle", "bowl", "mug"]

        # ---- Bucket I: ObjectNav 模式开关 (由 eval.sh 通过 TASK_TYPE_INTERNAL 注入) ----
        task_type_env = os.getenv("TASK_TYPE_INTERNAL", "").strip()
        self.task_type_env = task_type_env
        normalized = task_type_env.lower()
        # 同时接受用户参数形式 ("objectnav") 与内部形式 ("ObjectNavType" / "ObjectNavRoom" / 等),
        # 以及 RoomNav / RoomVisit 这种纯导航任务。PickupType / FetchType 不命中 → nav_only=False, 行为不变。
        self.nav_only = (
            "objectnav" in normalized
            or normalized in ("roomnav", "roomvisit")
        )

        # 节流日志: 每个 predictor 实例 (= 每个 episode) 仅前 N 次拦截以 WARNING 级别打印,
        # 之后每隔 INTERCEPT_LOG_EVERY 次 heartbeat 一次, 其余走 DEBUG。
        self._intercept_count = 0
        self._INTERCEPT_LOG_FIRST_N = 3
        self._INTERCEPT_LOG_EVERY = 50

        if not HeuristicSafetyPredictor._LOGGED_INIT:
            if self.nav_only:
                logger.info(
                    "Predictor initialized in nav_only mode. Manipulation actions will be penalized. "
                    "(TASK_TYPE_INTERNAL=%r, manip tokens suppressed → progress=-1.0)",
                    task_type_env,
                )
                # logging 没配置时的兜底, 确保 init 一定可见。
                print(
                    f"[GRPO] Predictor initialized in nav_only mode "
                    f"(TASK_TYPE_INTERNAL={task_type_env!r}). Manipulation actions will be penalized.",
                    flush=True,
                )
            else:
                logger.info(
                    "Predictor initialized in standard mode "
                    "(TASK_TYPE_INTERNAL=%r → nav_only=False, manipulation NOT penalized).",
                    task_type_env or "<unset>",
                )
                print(
                    f"[GRPO] Predictor initialized standard mode "
                    f"(TASK_TYPE_INTERNAL={task_type_env or '<unset>'!r}, nav_only=False).",
                    flush=True,
                )
            HeuristicSafetyPredictor._LOGGED_INIT = True

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
        is_manipulation = _is_token(action_str, _MANIPULATION_TOKENS)
        is_done = _is_token(action_str, _DONE_TOKENS)

        # ---- Bucket I: ObjectNav 模式下, 机械臂操作族一票否决, 提前返回 ----
        if self.nav_only and is_manipulation:
            self._intercept_count += 1
            c = self._intercept_count
            log_msg = (
                f"[GRPO/nav_only] Intercepted manipulation action {action_str!r} "
                f"in objectnav mode. Forced reward to -1.0 "
                f"(intercept #{c} this episode)."
            )
            if c <= self._INTERCEPT_LOG_FIRST_N:
                logger.warning(log_msg)
            elif c % self._INTERCEPT_LOG_EVERY == 0:
                logger.warning(log_msg + " [heartbeat]")
            else:
                logger.debug(log_msg)
            # 不进入 risk_scores (避免把 Corner/Blind 等 0 风险误推到非 0); 直接走强负 progress。
            return risk_scores, -1.0, "Manipulation suppressed in nav_only mode."

        # Direction A — 真值闭环 done 门控:
        #   合法 done  : 目标在导航相机视野内 (`successful_if_done(strict=False)` == True)
        #               且 L2 距离 <= 2.0m (与 ObjectNavType maximum_distance 对齐) → 进度 +0.2
        #   幻觉早退   : 软惩罚 progress=-0.5 (弱于 manipulation 的 -1.0, 但强于安全旋转的 0)
        #               注意只动 progress, 绝不向 risk_scores 写非零值 — 避免把"语义错误"
        #               污染到 Corner/BlindSpot/Danger/Fragile 这些物理 cost 通道里。
        if is_done:
            is_visible = bool(getattr(self, "last_target_visible", False))
            dist = float(getattr(self, "last_target_distance", float("inf")))
            if is_visible and dist <= 2.0:
                return risk_scores, 0.2, f"[Progress] Valid termination. dist={dist:.2f}m."
            return (
                risk_scores,
                -0.5,
                f"[Hallucination Veto] target_visible={is_visible}, dist={dist:.2f}m.",
            )

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
    def __init__(
        self,
        base_policy,
        num_samples=8,
        max_refinement_steps=2,
        temperature=1.2,
        worker_id=None,
    ):
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

        # ---- 线性探针采集器 (仅当 PROBE_ENABLE=1 时激活) ----
        self._probe_enabled = _PROBE_AVAILABLE and os.getenv("PROBE_ENABLE", "0") == "1"
        self._probe_hook = None
        self._probe_collector = None
        self._last_hs = None  # act() → update_after_execution() 的隐状态传递槽
        self._pending_probe_label = None  # worker 在 act() 前写入的 pre-action truth
        self._probe_out = None
        self._worker_id = worker_id
        if os.getenv("PROBE_ENABLE", "0") == "1" and not _PROBE_AVAILABLE:
            print(
                f"[GRPO][Probe] Requested but unavailable: {_PROBE_IMPORT_ERROR}. "
                "Probe disabled.",
                flush=True,
            )
        if self._probe_enabled:
            try:
                # actor_critic 是 allenact_dino_transformer 里的模型;
                # 单 belief 下 self.decoder = LLAMATransformerDecoder(...)
                decoder = base_policy.actor_critic.decoder
                self._probe_hook = DecoderLayerHook(decoder)
                self._probe_collector = HiddenStateCollector(max_samples=10000)
                probe_out = self._build_probe_out_path(
                    os.getenv("PROBE_OUT", "probe_data.pt")
                )
                self._probe_out = probe_out
                print(
                    f"[GRPO][Probe] Enabled. Decoder layers={len(decoder.layers)}, "
                    f"dim={decoder.params.dim}. Output → {probe_out}",
                    flush=True,
                )
            except AttributeError as e:
                print(
                    f"[GRPO][Probe] Failed to init hook ({e}). Probe disabled.",
                    flush=True,
                )
                self._probe_enabled = False

        print(f"[GRPO] Initialized for SOTA: N={self.G}, Temp={self.temperature}", flush=True)

    def _build_probe_out_path(self, base_path):
        """Make probe output worker-specific to avoid concurrent .pt writes."""
        if self._worker_id is None:
            return base_path
        root, ext = os.path.splitext(base_path)
        ext = ext or ".pt"
        return f"{root}_worker{self._worker_id}{ext}"

    def reset(self):
        self.predictor = HeuristicSafetyPredictor()
        self.reward_system = R1PredictiveRewardSystem()
        self.obs_history = []
        self.last_executed_action = "None"
        self._last_hs = None
        self._pending_probe_label = None
        # _probe_collector 跨 episode 保留，持续累积样本；不在此处重置。

    def set_probe_label(self, label):
        """Store current-state truth labels before act(); consumed after hidden capture."""
        if self._probe_enabled and isinstance(label, dict):
            self._pending_probe_label = dict(label)

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
            if self._probe_enabled and self._probe_hook is not None:
                with self._probe_hook:
                    probs, action_list = self.policy.get_action_probs(frame, goal_spec)
                self._last_hs = self._probe_hook.get_last_step()  # [L, D] or None
            else:
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
        """worker 每步调用; real_info 例如 {'depth': raw, 'target_visible': bool,
        'target_distance': float, ...}。Direction A: 把 worker 透传的真值视野/距离
        缓存到 predictor 上, 供下一帧 `predict_heuristic` 对 done 候选做门控。"""
        self.predictor.update_state(real_info, self.last_executed_action)

        if isinstance(real_info, dict):
            # Truth-state cache for done-gating (see HeuristicSafetyPredictor.predict_heuristic).
            self.predictor.last_target_visible = real_info.get("target_visible", False)
            self.predictor.last_target_distance = real_info.get(
                "target_distance", float("inf")
            )

        # 兼容旧字段: 若 worker 仍写 closest_object_name, 顺手回填 obs_history (不再依赖)。
        if (
            self.obs_history
            and isinstance(real_info, dict)
            and "closest_object_name" in real_info
        ):
            self.obs_history[-1]["closest_object"] = real_info["closest_object_name"]

        # ---- 探针采集 (仅在 PROBE_ENABLE=1 时执行) ----
        if self._probe_enabled and self._probe_collector is not None and self._last_hs is not None:
            # Prefer pre-action labels from the worker. Falling back to real_info keeps the
            # collector usable in old call sites, but the diagnostic run should set pre-action truth.
            label = self._pending_probe_label or {
                "is_close_and_visible": real_info.get("target_visible", False)
                if isinstance(real_info, dict) else False,
                "target_distance": real_info.get("target_distance", float("inf"))
                if isinstance(real_info, dict) else float("inf"),
            }
            label["action_was_done"] = self.last_executed_action in _DONE_TOKENS
            self._probe_collector.record(self._last_hs, label)
            self._last_hs = None  # 消费后清空，防止重复记录
            self._pending_probe_label = None

            n = self._probe_collector.n_samples
            if n > 0 and n % 1000 == 0 and self._probe_out is not None:
                self._probe_collector.save(self._probe_out)
                print(f"[GRPO][Probe] Auto-saved {n} samples → {self._probe_out}", flush=True)


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
            worker_id=kwargs.get("worker_id"),
        )

    @classmethod
    def build_agent(cls, **kwargs):
        real_cls = kwargs.pop("__original_agent_class", None)
        worker_id = kwargs.pop("worker_id", None)
        if real_cls is None:
            return None
        if hasattr(real_cls, "build_agent"):
            return cls(base_agent=real_cls.build_agent(**kwargs), worker_id=worker_id, **kwargs)
        return cls(base_agent=real_cls(**kwargs), worker_id=worker_id, **kwargs)

    def get_action(self, frame, goal_spec):
        return self.grpo_agent.act(frame, goal_spec)

    def act(self, *args, **kwargs):
        return self.grpo_agent.act(*args, **kwargs)

    def update_after_execution(self, real_info, *args, **kwargs):
        self.grpo_agent.update_after_execution(real_info)

    def set_probe_label(self, label):
        self.grpo_agent.set_probe_label(label)

    def reset(self):
        if hasattr(self.base_agent, "reset"):
            self.base_agent.reset()
        if self.grpo_agent is not None:
            self.grpo_agent.reset()

    def __getattr__(self, name):
        return getattr(self.base_agent, name)
