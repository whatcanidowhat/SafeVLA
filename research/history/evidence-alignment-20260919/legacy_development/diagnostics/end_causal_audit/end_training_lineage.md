# Expert → IL → SafeRL end 学习静态审计

## 因果链与证据等级

1. Expert trajectory generation → Expert end rule
   - 【已证实事实】本 repo 的 scripts/download_training_data.py 只下载预生成 astar/fifteen 数据。
   - 【Evidence Gap】未找到 ObjectNav expert trajectory 生成器，因而无法证明 expert 何时生成 end、是否使用 evaluation 同一 stop-success rule、是否含 oracle/自动终止或特权信息。

2. IL action labels
   - 【已证实事实】training/offline/chores_dataset.py 使用 actions = sensors["last_action_str"][1:]，prev-action 输入来自 last_action_str[:-1]；末帧因“没有对应动作”被丢弃。
   - 【已证实事实】动作表为 20 类；动态解析 end_idx = action_names.index("end") = 4，诊断代码没有硬编码 4。
   - 【Evidence Gap】当前机器未找到实际 IL 训练数据或 checkpoint 的原始 IL 命令，无法统计真实 action frequency、end 比例/位置/类别条件分布，也无法证明 last-step oversampling 或 redundancy reduction 是否启用。
   - 【已证实事实】代码默认 init_prob_sample_last_steps=0、final_prob_sample_last_steps=0、reduce_action_redundancy=False；“代码支持”不等于“checkpoint 训练时启用”。

3. IL supervision
   - 【已证实事实】EarlyFusionCnnTransformer.actor = nn.Linear(512, 20)；CrossEntropyLoss(ignore_index=-1)；loss 为逐有效 token 的多类交叉熵。
   - 【已证实事实】end 是 20 分类动作之一，不是独立 Stop Head；正确术语为 policy_end_prob = Pπ(a=end|s)，不是 success probability。
   - 【已证实事实】actor 输入包含视觉-文本融合表征、prev action、time encoding，并经 decoder 输出 actor representation。

4. IL → SafeRL initialization
   - 【已证实事实】在线模型支持 prev_checkpoint，通过 load_pl_ckpt_allenact 加载 IL；另支持 prev_rl_checkpoint。
   - 【Evidence Gap】当前 checkpoint 未保存本次训练实际使用的 IL path/hash，不能建立完整、可复验的 IL checkpoint 血缘。

5. SafeRL objective
   - 【已证实事实】ObjectNav reward config：step=0、successful end=+10、failed end=0、horizon=0、shaping=0、failed-action penalty=-0.00。
   - 【已证实事实】SafePPO clipped surrogate 使用 A_combined=(A_reward-λ A_cost)/(1+λ)，对 ratio 与 clipped ratio 取最小 surrogate 后取负；λ 来自 Lagrange 更新。
   - 【已证实事实】Safety cost 是 corner+danger+blind+fragile+critical 的 step 累加；不等于 collision probability。
   - 【合理推断】失败 end 与无回报移动的即时 reward 同为 0，而继续移动可能累积 cost；目标具有改变“继续探索 vs 结束”相对偏好的可能。
   - 【待验证假设 H-SAFEQUIT】SafeRL 训练因此系统性提高非法状态的 policy_end_prob。
   - 【暂不允许结论】“SafeVLA 因为害怕危险而提前退出”。
