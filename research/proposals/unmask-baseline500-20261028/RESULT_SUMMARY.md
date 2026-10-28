# Baseline500 correction and online readiness

原标准导航评测有效配置是 **max_steps500 / K-V序列容量500 / Episode horizon600**。本轮只修正实验fixture、当前设计和验证证据，原BaseConfigParams、模型工厂、Actor/Critic生产源码、cache/counter/rollover及官方任务口径逐字节保持。模型类默认1000仍是被标准评测覆盖的类默认值；不是本次Baseline。

38项CPU检查PASS；其中500步rollover与原forward比较，实际缓存值/计数器保持原行为。新增只读断言在完整checkpoint加载后、root mask设置前检查Actor/Reward Critic/Cost Critic及每层真实K/V序列维500，错误即停止，不修改模型使断言通过。

已通过原评测入口构造完整CPU策略，**417/417** checkpoint完整加载，未覆盖params.max_steps。实际三分支max_steps与所有K/V序列维均500；初始sampler维可为0，符合原num_train_processes=0的构造与延迟分配。该完整策略核验**0次policy forward、CUDA未初始化**。另按要求仅重新生成一次现有的小组件500重放（4次warmup+6次独立边界输入），不扩大logits研究，不以其判断SR/Cost。

独立环境的Torch2.4.1/Torchvision0.19.1/Torchaudio2.4.1、NumPy2.1.2、jsonschema4.25.1已对齐；策略和评测模块CPU导入、完整策略加载PASS。Pandas目标2.2.3与旧服务器2.3.3的差异及pip声明冲突保留，须独立PI审查，不宣称完整环境等价。Pandas2.3.3本机源码编译失败、初次NLTK路径失败与后续修正均有日志，未修改原native环境。

选定200 TaskSpec引用的外部资产无缺失；4039个引用assetId对应盘点中15237个文件、1115494780字节已哈希。房屋val与annotations哈希和旧服务器核对一致。单个外部文件是目标冻结身份，不冒称全部与旧服务器逐文件比对。完整大索引留服务器；内置资源、渲染设备和功能仍需审查/核验。静态Unity二进制已记录；无模拟器或GPU启动。

正式计划为同200有序TaskSpecs、seed123、worker1、原随机采样/增强、同一权重与GPU，**MASK_ON200→MASK_OFF200**，两臂Actor Reset OFF、max_steps/cache均500、horizon600，共400次Episode starts。报告ΔSR与ΔCost(OFF−ON)、五类Cost及Episode级五分项配对差异、失败提前END、horizon耗尽、房间覆盖/是否进入目标房间和重复/往返行为。探索指标只作代理，不做未经证实的因果诊断。

**正式SR/Safety Cost尚未测量，ΔSR/ΔCost均不可判断。** 本轮0新Claim、0 GPU作业、0在线Episode。本地控制缓存为旧终止周期BLOCKED/next_actor PI；新服务器ls-remote超时，目标端实时状态标REMOTE_UNVERIFIED，没有网络重试或领取。下一步是独立PI审查正确的共享SHA、500配置、目标环境/渲染和本轮logger决策，完成新批准CI→唯一新Claim→Claim CI以及实际网络门禁后执行。不得用旧Claim、旧waiver或旧OFF代替。详见DESIGN.md与PI_REQUEST.json。
