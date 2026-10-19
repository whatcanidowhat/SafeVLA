# SafeVLA PR-2：静态审核交付

状态：开发与 CPU 验证完成，交付 PI 静态审核。未 Push、未创建 PR、未合并。

- 基线：`f260e7e0c55489d4ae1695577d15c95d702f00d7`
- 本地候选：`4d2b52e3992d5b2330d1e7ee0e7a6e3c824f47cf`
- 分支：`codex/agent-readonly-pr2`
- 完整补丁：`SafeVLA-PR2-readonly-tools.patch`

## 实现与范围

| 文件 | 改动 |
| --- | --- |
| scripts/research_loop_status.py | 真实远端 ref → 同一 SHA 的状态与两份设计 → 再查 ref；身份、授权、期限、角色、Claim 绑定检查；支持只读本地 committed snapshot。 |
| scripts/research_loop_ci_wait.py | 绑定 SHA、branch、workflow ID/name/path、event、当前 attempt；有界轮询，最终一次 JSON 和可区分退出码。 |
| scripts/research_loop_readonly.py | 固定 GitHub 仓库/HTTPS origin，GET-only、TLS 验证、禁止跳转、响应与文件 hash 验证、超时边界、统一四字段输出。 |
| scripts/test_research_loop_readonly.py | 33 项离线测试及五任务合成测量入口；不访问真实 GitHub。 |
| scripts/validate_research_loop.py | 仅增加上述四个精确文件白名单条目。 |
| scripts/test_research_loop.py | fixture 显式 UTF-8/LF，修复本地 Windows 换行差异；25 项测试语义未改变。 |
| .github/workflows/research-loop-validate.yml | 增加离线只读工具测试步骤；原有测试、完整历史检查、权限和超时不变。 |
| research/HANDOFF_PROTOCOL.md | 追加只读工具白名单与“不授予执行权”的说明，原状态机和门禁不变。 |
| .codex/skills/safevla-research-loop/SKILL.md | 加入只读工具与参数契约链接，保持 PR-1 条件读取和原 Stop Conditions。 |
| research/agent_workflow/READONLY_TOOLS.md | 参数、退出码、故障语义、边界、测试及测量方法。 |

所有结果强制 `EVIDENCE.CAN_EXECUTE=false`。状态观察成功为 OBSERVED；CI 成功为 SUCCESS。两者 exit 0 都不是 Claim/实验执行许可。现有 CODEX/OPERATOR 权限、Claim 流程、真实状态和实验设计未修改。

状态查询是紧凑 state/design observer，不是完整历史、终态证据、运行来源或 runtime gate 审计。输出显式说明范围。完整历史、Claim CI、实际网络 witness、资源与来源校验仍须独立执行，不能以状态工具 exit 0 替代。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 新增离线测试（Windows，jsonschema 4.25.1） | 33 tests，OK，0.476 秒 |
| 原有控制回归（Windows） | 25 tests，OK，102.315 秒；该次本机默认 jsonschema 为 4.17.3 |
| 新增离线测试（隔离 Linux，现有 jsonschema 4.25.1） | 33 tests，OK，0.107 秒 |
| 原有控制回归（隔离 Linux，同上） | 25 tests，OK，7.232 秒 |
| 候选对基线 snapshot/transition 校验 | CONTROL_VALIDATION_PASS |
| 完整历史校验（最终候选，隔离 Linux） | CONTROL_HISTORY_PASS commits=162 |
| Skill 格式与 git diff --check | PASS |
| 验证器 AST 对比 | 除 audit_tree 的精确白名单 literal set 外完全相同 |
| 保护文件 SHA-256 对比 | LOOP_STATE、两份 NEXT_EXPERIMENT、CODEX/OPERATOR Claim 脚本字节不变 |
| 当前 P2-004 本地状态查询 | REMOTE_UNVERIFIED；status=BLOCKED；历史 Claim 不可复用；CAN_EXECUTE=false |

Windows 上先前失败定位为两项环境差异：fixture 文件及临时 clone 的 CRLF、超过 260 字符的归档路径。修复 fixture 写出后，仅为测试进程设置 `core.autocrlf=false`；未修改全局 Git 配置。工作树 snapshot 使用 Windows 扩展路径前缀复验通过，Linux 隔离副本再次验证。未放宽状态行解析或文件检查。

最终 Linux 测试目录：`/tmp/safevla-agent-pr2.6ktA4k/final`，由本地 Git bundle 创建，未读取或修改现有服务器控制 worktree。这里的 Claim 回归仅针对临时合成 Git fixtures，不领取真实 P2-004 Claim。

## 异常场景证据映射

测试完整名称与结果保存在 `PR2-readonly-tests.txt` 和 Linux 日志中。

| 场景 | 断言/测试 |
| --- | --- |
| BLOCKED、PI_REVIEW、CODEX/OPERATOR approved/running 等全部状态 | test_all_states_never_grant_execution：全部 CAN_EXECUTE=false |
| 状态/角色错误、设计 identity 冲突、缺失证据 | test_conflicts_and_missing_evidence、test_missing_state_or_network_failure：UNKNOWN |
| 过期或未来批准、控制工作树/预算冲突 | test_expired_approval、test_future_approval_and_scope_conflicts：BLOCKED/UNKNOWN |
| 旧 Claim、已运行 Claim、批准携带旧 Claim | test_claim_is_not_reusable、test_approval_cannot_contain_old_claim：不恢复、不授予执行许可 |
| 错误 SHA、event、workflow、branch 或 run ID | test_identity_errors：UNKNOWN |
| 查询前后远端 HEAD 改变 | status/CI 对应 movement 测试：UNKNOWN |
| CI 重跑、attempt 竞态、新 run 替换 | 三项 rerun/attempt/new_run 测试：未知或不一致即停止；只允许显式当前 attempt |
| 多个 run、缺失 run、分页不完整 | test_no_run_or_multiple_runs、test_incomplete_pagination：UNKNOWN/NOT_FOUND |
| failure/cancelled/timed_out/neutral/skipped | test_terminal_outcomes：只有 success 可报告 SUCCESS |
| HTTP 权限错误、404、限流、5xx | test_http_errors_sanitized_and_not_retried：明确分类、无重试、无敏感错误回显 |
| TLS/代理/网络失败、DNS 停滞 | test_tls_proxy_timeout_fail_closed、test_stalled_dns_is_bounded：停止并有界返回 |
| TLS 证书验证与跳转 | test_tls_context_verifies_certificates、test_redirect_refused：启用验证、拒绝跳转 |
| 重复查询 | state/CI repeated 测试：重新读取、不变更状态、不创建 Claim |
| 单 JSON、退出码和 2 KiB 上限 | CLI/output 测试：一次输出；过大输出改为 UNKNOWN |
| 白名单与 1 MiB 上限 | test_exact_whitelist_does_not_allow_other_scripts：拒绝其他脚本及超大文件；既有秘密扫描回归继续通过 |

## 五个合成控制任务与效果

原始结果：`PR2-synthetic-measurements.json`。输入是 fake GitHub，sleep 是虚拟时钟。

| 任务 | 最终结果 | 轮询轮数 | 模型化工具调用：旧 → 新 | 模型化等待交互：旧 → 新 | 错误授予执行许可 |
| --- | --- | ---: | --- | --- | --- |
| 立即成功 | SUCCESS | 1 | 1 → 1 | 0 → 0 | 0 |
| 排队后成功 | SUCCESS | 3 | 3 → 1 | 2 → 0 | 0 |
| 运行失败 | FAILURE | 2 | 2 → 1 | 1 → 0 | 0 |
| 取消 | CANCELLED | 1 | 1 → 1 | 0 → 0 | 0 |
| 等待超时 | TIMED_OUT | 3 | 3 → 1 | 2 → 0 | 0 |

上述“旧”基线明确假设每轮查询由 LLM 发起一次工具调用；没有把这个模型化计数称为真实会话测量。

- 真实 LLM 工具调用数：**NOT_MEASURED**。
- 实际输入/输出 Token：**NOT_MEASURED**，未用字符数替代。
- 真实 CI 等待期间 LLM 轮次：**NOT_MEASURED**。
- 真实网络/CI 端到端耗时对比：**NOT_MEASURED**。
- 实际 Python 合成 harness 耗时已逐任务记录，但不含真实网络与 CI 等待，不能据此声称生产速度提升。
- 合成安全误授执行权：五任务均为 false，且其余对抗测试通过；不代表已经证明所有生产故障模式。

优化机制是把重复 HTTP 查询、等待和身份匹配留在一次 Python 调用内，只让 LLM 消费最终结果。严格门禁可能增加底层 API 读取次数；本 PR 不以减少安全核验换取 Token 节省，也未声称已达到 50% 实测目标。

## 使用与限制

```
python scripts/research_loop_status.py --remote-readonly
python scripts/research_loop_status.py --local --root <checkout>
python scripts/research_loop_ci_wait.py --sha <full-sha> --workflow research-loop-validate.yml --event push --branch research-loop --timeout 600 --interval 15
```

远端工具使用调用进程环境中的可选 GH_TOKEN/GITHUB_TOKEN；GitHub 插件连接不会自动提供给 Python。未安排真实网络集成测试，本轮以 CPU/模拟网络测试为限。403 无明确限流 headers 时保守归为 PERMISSION_DENIED；404 可能代表不可见资源。缺失 run 立即 NOT_FOUND；多 run 或分页不完整停止而不猜测。

网络超时使用短生命周期 daemon transport thread 保证 CLI 有界返回。若作为库调用且宿主进程保持存活，停滞 OS 请求线程可能留到 OS 返回，但它只能完成原只读请求，不会重试或启动新动作。没有安装常驻服务。

PR-1 的四字段输出、按需读取、完整科学审计、原 Stop Conditions 兼容。状态查询没有验证 terminal artifact hashes 或所有 runtime 条件；此限制已写入输出和使用文档。后续 PI 可在审核后单独授权真实 API/发布验证，本轮不进行。

## 交回 PI

候选仅在独立开发分支本地保存。未 Push、未创建 PR、未合并、未改变实验状态、未启动 GPU/仿真/Episode，也未恢复 P2-004。交付完成后停止，等待 PI 静态审核。
