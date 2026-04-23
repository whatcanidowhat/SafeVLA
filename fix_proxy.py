import os
import re

target_path = 'online_evaluation/online_evaluator.py'

# 新的、包含 build_agent 的完整代理类代码
new_proxy_code = r'''class GRPOAgentProxy:
    """GRPO 代理类：拦截动作请求并转发给预测推理器"""
    def __init__(self, base_agent, **kwargs):
        self.base_agent = base_agent
        
        # 初始化 GRPO Agent
        if GRPOPredictiveAgent and self.base_agent:
            print(f"[GRPO Proxy] Wrapping Agent with GRPO Inference (N=8)...")
            self.grpo_agent = GRPOPredictiveAgent(base_policy=self.base_agent, num_samples=8)
        else:
            print("[GRPO Proxy] Warning: GRPO Agent not loaded, falling back to base policy.")
            self.grpo_agent = None

    @classmethod
    def build_agent(cls, **kwargs):
        # 1. 提取原始 Agent 类
        real_cls = kwargs.pop('__original_agent_class', None)
        
        if real_cls is None:
            print("[GRPO Proxy] Error: __original_agent_class missing in build_agent kwargs")
            return None

        # 2. 使用原始类的 build_agent 方法创建实例
        if hasattr(real_cls, 'build_agent'):
            base_agent = real_cls.build_agent(**kwargs)
        else:
            base_agent = real_cls(**kwargs)

        # 3. 返回包装后的 Proxy 实例
        return cls(base_agent=base_agent)

    def act(self, *args, **kwargs):
        if self.grpo_agent:
            return self.grpo_agent.act(*args, **kwargs)
        return self.base_agent.act(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base_agent, name)
'''

with open(target_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 尝试定位并替换旧的 GRPOAgentProxy 类
# 我们匹配从 "class GRPOAgentProxy" 开始，直到 "return getattr" 结束的块
# 为了稳妥，我们使用正则替换，或者简单的字符串替换
# 这里采用更稳健的 block 替换逻辑

start_marker = "class GRPOAgentProxy:"
end_marker = "return getattr(self.base_agent, name)"

# 简单的基于标记的替换逻辑
if start_marker in content and end_marker in content:
    start_idx = content.find(start_marker)
    # 找到 end_marker 后，还需要找到这一行的结尾
    end_idx_start = content.find(end_marker, start_idx)
    
    if end_idx_start != -1:
        # 找到 end_marker 这一行的换行符
        end_idx = content.find('\n', end_idx_start)
        if end_idx == -1: end_idx = len(content)
        
        # 替换旧代码块
        new_content = content[:start_idx] + new_proxy_code + content[end_idx:]
        
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ 成功更新 GRPOAgentProxy 类！(build_agent 已添加)")
    else:
        print("⚠️ 无法定位旧代码结束位置，请检查文件。")
else:
    print("❌ 找不到 GRPOAgentProxy 类，请确认 online_evaluator.py 是否被正确还原。")
