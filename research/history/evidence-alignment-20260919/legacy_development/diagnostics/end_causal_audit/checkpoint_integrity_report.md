# Checkpoint Parameter Integrity Audit

结论：PASS。

- 【已证实事实】checkpoint 为 AllenAct model_state_dict，SHA256 05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301。
- 【已证实事实】checkpoint 与当前评估模型各含 417 个 tensor；加载结果 missing_keys=[]、unexpected_keys=[]、shape mismatch=0。
- 【已证实事实】actor.linear.weight 形状为 [20,512]，actor.linear.bias 为 [20]，关键 actor / visual / decoder / reward critic / cost critic 参数均存在并实际加载。
- 【已证实事实】训练状态记录 total_steps=12,551,992，三个阶段步数为 202,404、800,688、11,548,900，和等于 total_steps。
- 【Evidence Gap】checkpoint 文件不保存 IL checkpoint 路径、SafeRL 完整启动命令或 cost limit；不能仅从代码默认值反推该 checkpoint 的训练超参数。
