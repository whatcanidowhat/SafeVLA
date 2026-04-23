import numpy as np


# =============================================================================
# 1. 新增 SpatialMemory 类 (必须包含在文件中)
# =============================================================================
class SpatialMemory:
    """
    简易空间记忆模块：维护机器人周围 4 个方位的障碍物状态
    方位索引: 0:前(Front), 1:右(Right), 2:后(Back), 3:左(Left)
    """

    def __init__(self):
        # 初始化：假设四周都是空的 (False = 无障碍, True = 有障碍)
        self.occupancy = [False, False, False, False]
        # 记忆衰减计时器（防止永久记住一个不存在的障碍物）
        self.decay_counters = [0, 0, 0, 0]
        self.MEMORY_PERSISTENCE = 5  # 记忆保持 5 步

    def update(self, current_depth_risk, last_action_str):
        """
        根据动作更新方位，并根据当前观测刷新前方状态
        """
        # 1. 自身运动导致的状态流转 (Ego-motion Update)
        new_occupancy = self.occupancy[:]
        new_counters = self.decay_counters[:]

        if "RotateLeft" in last_action_str:
            # 左转90度：Old Left (3) -> New Front (0)
            new_occupancy = [self.occupancy[3], self.occupancy[0], self.occupancy[1], self.occupancy[2]]
            new_counters = [self.decay_counters[3], self.decay_counters[0], self.decay_counters[1],
                            self.decay_counters[2]]

        elif "RotateRight" in last_action_str:
            # 右转90度：Old Right (1) -> New Front (0)
            new_occupancy = [self.occupancy[1], self.occupancy[2], self.occupancy[3], self.occupancy[0]]
            new_counters = [self.decay_counters[1], self.decay_counters[2], self.decay_counters[3],
                            self.decay_counters[0]]

        # 2. 记忆衰减 (每过一步，倒计时减1)
        for i in range(4):
            if new_counters[i] > 0:
                new_counters[i] -= 1
            else:
                new_occupancy[i] = False  # 遗忘

        # 3. 写入当前观测 (Current Observation Overwrite)
        # 如果当前深度图显示前方有障碍，则强行写入 Front(0)
        if current_depth_risk:
            new_occupancy[0] = True
            new_counters[0] = self.MEMORY_PERSISTENCE  # 重置倒计时

        self.occupancy = new_occupancy
        self.decay_counters = new_counters

    def check_risk(self, direction_idx):
        return self.occupancy[direction_idx]


# =============================================================================
# 2. 核心预测器类 (已适配 GRPO 接口)
# =============================================================================
class HeuristicSafetyPredictor:
    def __init__(self):
        # 阈值参数
        self.SAFE_DISTANCE = 0.5
        self.CRITICAL_DISTANCE = 0.35

        # [改动] 实例化记忆模块
        self.memory = SpatialMemory()

    def update_state(self, real_info, last_action_str):
        """
        [改动] 新增此方法，供 GRPO Agent 在 update_after_execution 中调用
        """
        # 1. 简单的深度判断：前方是否有障碍？
        front_blocked = False
        if "depth" in real_info:
            depth_map = real_info["depth"]
            # 处理可能的 Tensor 格式
            if hasattr(depth_map, 'cpu'):
                depth_map = depth_map.cpu().numpy()

            # 提取中心深度
            if len(depth_map.shape) >= 2:
                h, w = depth_map.shape[:2]
                center_depth = np.mean(depth_map[h // 3:2 * h // 3, w // 3:2 * w // 3])
                if center_depth < self.CRITICAL_DISTANCE:
                    front_blocked = True

        # 2. 更新记忆
        self.memory.update(front_blocked, last_action_str)

    def predict_heuristic(self, obs_history, action_str, current_obs_metadata=None):
        """
        [改动] 增加了 current_obs_metadata 参数，兼容第一帧预测

        返回:
        - risk_scores: dict
        - progress_score: float
        - feedback_msg: str (关键：用于生成 Token Prior 的自然语言建议)
        """
        risk_scores = {
            "Corner": 0.0, "BlindSpot": 0.0, "Fragile": 0.0,
            "Critical": 0.0, "Dangerous": 0.0
        }
        progress_score = 0.0
        feedback_msg = None

        # --- 数据提取 (健壮性处理) ---
        # 优先使用 current_obs_metadata (当前帧)，如果为空再查历史
        if current_obs_metadata:
            depth_val = current_obs_metadata.get('depth_mean', 10.0)
            closest_obj = current_obs_metadata.get('closest_object', "").lower()
        elif obs_history:
            depth_val = obs_history[-1].get('depth_mean', 10.0)
            closest_obj = obs_history[-1].get('closest_object', "").lower()
        else:
            # 没有任何数据时的默认值
            depth_val = 10.0
            closest_obj = ""

        # =========================================================
        # 1. 撞墙/死角检测 (Corner)
        # =========================================================
        if action_str == "MoveAhead":
            if depth_val < self.CRITICAL_DISTANCE:
                risk_scores["Corner"] = 1.0
                feedback_msg = "Critical Warning: Obstacle directly ahead. Moving forward will crash. You must Rotate Left or Right."
            elif depth_val < self.SAFE_DISTANCE:
                risk_scores["Corner"] = 0.6
                feedback_msg = "Warning: Too close to obstacle. Moving forward is risky. Try rotating."
            else:
                progress_score += 0.1

        # =========================================================
        # 2. 盲区检测 (BlindSpot) - [改动] 实装记忆查表逻辑
        # =========================================================
        target_dir = -1
        if "MoveLeft" in action_str:
            target_dir = 3  # Left
        elif "MoveRight" in action_str:
            target_dir = 1  # Right
        elif "MoveBack" in action_str:
            target_dir = 2  # Back

        # 如果动作涉及方向移动，且该方向在记忆中是“有障碍”的
        if target_dir != -1 and self.memory.check_risk(target_dir):
            risk_scores["BlindSpot"] = 0.9  # 重罚

            # 生成特定的语义反馈
            dir_name = ["Front", "Right", "Back", "Left"][target_dir]
            feedback_msg = f"Warning: You previously saw an obstacle to your {dir_name}. Moving there is unsafe. Try Rotating."

        # 严禁无理由后退 (通常没有后视眼，BlindSpot 风险极高)
        if "MoveBack" in action_str:
            risk_scores["BlindSpot"] = max(risk_scores["BlindSpot"], 0.8)
            if not feedback_msg:
                feedback_msg = "Warning: Moving backward into blind spot is unsafe. Please Rotate to check behind first."

        # =========================================================
        # 3. 危险品检测 (Dangerous)
        # =========================================================
        dangerous_keywords = ["stove", "fire", "burner", "knife"]
        if any(k in closest_obj for k in dangerous_keywords):
            if depth_val < 0.5:
                risk_scores["Dangerous"] = 0.9
                feedback_msg = f"Danger: Too close to {closest_obj}. Do not interact. Move away immediately."

        return risk_scores, progress_score, feedback_msg