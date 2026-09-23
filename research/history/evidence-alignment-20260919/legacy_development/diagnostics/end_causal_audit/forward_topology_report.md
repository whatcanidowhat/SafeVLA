# Policy Forward Topology Audit

结论：PASS（单 episode runtime）。

- 【已证实事实】8 个 environment decision step 对应 8 个 top-level policy decision call。
- 【已证实事实】每个 top-level call 内 actor decoder、reward critic decoder、cost critic decoder 各恰好 1 次；总数均为 8。
- 【已证实事实】actor branch 的 actor.linear 每 step 恰好 1 次并产生 20-D logits；logger 没有调用模型。
- 【已证实事实】SafeDinoLLAMATxNavActorCriticSeparate.forward 顺序执行三套完整网络分支。actor distribution 来自根 actor；reward value 来自 critic_tsfm；cost value 来自 c_critic_tsfm。
- 正式 Actor Probe 只能使用根分支 actor_critic.decoder 的输出；reward/cost critic hidden state 不得混入。
- 每个 hook event 已记录 env_step_id / policy_decision_call_id / module_forward_call_id / branch / module_name。
