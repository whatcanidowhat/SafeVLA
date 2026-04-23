class R1PredictiveRewardSystem:
    def __init__(self):
        self.w_result = 1.0
        self.w_process = 0.5
        self.w_punish = 2.0
        self.dynamic_factor = 1.0  # 动态调整因子

    def calculate_predicted_reward(self, risk_scores, progress_score, action_str):
        """
        基于【预测值】计算奖励 -> 用于 GRPO 选动作
        """
        # 1. 惩罚 (Punishment): 只要预测有高风险就惩罚
        punishment = 0.0
        for risk_type, prob in risk_scores.items():
            if prob > 0.5:  # 阈值
                punishment += prob * 5.0  # 概率越高惩罚越重

        # 2. 结果 (Result): 预测的进度
        result_reward = progress_score * 5.0

        # 3. 过程 (Process): 鼓励安全意识
        process_reward = 0.0
        # 逻辑：如果动作为旋转（旨在消除盲区），且没有BlindSpot风险，给予奖励
        if "Rotate" in action_str:
            process_reward += 0.5

        # R1 公式
        total_reward = (self.w_result * result_reward) + \
                       (self.w_process * process_reward) - \
                       (self.w_punish * punishment * self.dynamic_factor)
        return total_reward

    def update_feedback(self, real_outcome_info):
        """
        根据【真实环境】的反馈调整下一轮的权重
        这是 "推理阶段动态调整" 的核心
        """
        # 如果真实环境发生了碰撞 (Collision)
        if real_outcome_info.get("collided", False):
            # 下一轮推理时，大幅增加惩罚权重，变得极其保守
            self.dynamic_factor += 0.5
            self.w_process += 0.2  # 更加鼓励“思考/观察”过程
            print(f"检测到真实碰撞！调整保守系数为: {self.dynamic_factor}")

        # 如果成功推进
        elif real_outcome_info.get("reward", 0) > 0:
            # 适当放松，逐步恢复自信
            self.dynamic_factor = max(1.0, self.dynamic_factor - 0.1)