from pathlib import Path
import json, hashlib, csv, io, datetime, collections, subprocess, re, gzip
C=Path('/nvme2/user/qyy/SafeVLA_evidence_alignment_20260919')
R=Path('research/history/evidence-alignment-20260919')
P=C/R
NOW=datetime.datetime.now(datetime.timezone.utc).isoformat()
def jb(x):return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(x.encode() if isinstance(x,str) else x)
inv=json.loads((P/'FILE_INVENTORY.json').read_text());records=inv['records']
repo=Path('/nvme2/user/qyy/SafeVLA')
for rel in ['full_diff.patch','online_evaluation/end_causal_diagnostic_logger.py',
            'architecture/allenact_preprocessors/dino_preprocessors.py',
            'architecture/models/allenact_transformer_models/inference_agent.py',
            'online_evaluation/online_evaluator_worker.py','scripts/eval.sh','scripts/analyze_eval_report.py']:
    q=repo/rel
    if any(x['server_path']==str(q) for x in records):continue
    raw=q.read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();copies=[]
    for ref,commit in inv['source_refs'].items():
        out=subprocess.check_output(['git','-C',str(repo),'ls-tree','-r',ref,'--',rel]).decode().strip()
        if out and out.split()[2]==blob:copies.append({'ref':ref,'commit':commit,'path':rel})
    records.append({'server_path':str(q),'realpath':str(q.resolve()),'classification':'DO_NOT_SHARE_CONTROL',
       'reason':'Development/runtime code and dirty patch stay off the control branch; existing analysis-source commit is referenced when byte-identical.',
       'group':'development_code','readable':True,'exists':True,'sha256':hashlib.sha256(raw).hexdigest(),
       'size':len(raw),'mtime_ns':q.stat().st_mtime_ns,'git_blob_sha1':blob,'exact_github_copies':copies})
counts=collections.Counter(x['classification'] for x in records)
existing=[x for x in records if x.get('exists')]
unpublished=[x for x in existing if not x.get('exact_github_copies')]
inv['classification_counts']=dict(counts);inv['created_at_utc']=NOW
write(P/'FILE_INVENTORY.json',jb(inv))
fields=['classification','group','server_path','size','sha256','git_path','reason']
buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(unpublished)
write(P/'HISTORICAL_NOT_IN_GIT.csv',buf.getvalue())
metadata=json.loads((P/'metadata/STATIC_METADATA_AVAILABILITY.json').read_text())
with gzip.open('/home/amax/public/datasets/qyy/objaverse_assets/2023_07_28/annotations.json.gz','rt') as f:a=json.load(f)
metadata[1]['size_non_null_records']=sum(v.get('size') is not None for v in a.values())
metadata[1]['size_null_records']=sum(v.get('size') is None for v in a.values())
metadata[1]['important_limit']='annotations.size is an annotated asset-size field; no assertion that it is a measured simulator 3D bounding box. Units, scale, instance transform, broad-target mapping and completeness remain unverified.'
write(P/'metadata/STATIC_METADATA_AVAILABILITY.json',jb(metadata))
del a
readme=f"""# SafeVLA 历史证据对齐包

登记时间（UTC）：{NOW}  
归档基于 research-loop：{inv['control_parent']}。

本包落实用户的“先盘点、分三类、一次性提交共享证据”要求。它是历史文件归档与完整性核验，不是 EXP-SMALLTARGET-PHENOTYPE-001 的执行结果；没有领取实验，没有训练、模型加载、GPU、AI2-THOR、episode、重放或尺寸关联分析。LOOP_STATE 与两份 NEXT_EXPERIMENT 保持原字节。所有 legacy_development 文件都是旧版本快照，不能覆盖当前控制文件或充当当前授权。

## 1. 历史研究但未入 Git 的文件

本轮限定范围内登记 {len(records)} 项路径，其中 {len(existing)} 项存在且可读，{len(records)-len(existing)} 项为明确缺失的旧名称。按 Git blob 字节身份与四个远端分支比对，{len(existing)-len(unpublished)} 项来源已有完全相同的 GitHub 副本，不能再称为“未入 Git”；剩余 {len(unpublished)} 项列于下表文件。

- [HISTORICAL_NOT_IN_GIT.csv](HISTORICAL_NOT_IN_GIT.csv)：逐文件路径、三类归属、大小、SHA256、共享副本路径与理由。
- [FILE_INVENTORY.json](FILE_INVENTORY.json)：完整来源清单、GitHub commit/path 对照、已扫描范围和缺失记录。
- [ARTIFACT_INDEX.json](ARTIFACT_INDEX.json)：Git 可读证据副本及哈希，受仓库 CI 校验。
- [DEVELOPMENT_PRESERVATION.json](DEVELOPMENT_PRESERVATION.json)：原开发 HEAD、dirty diff 和状态前后相同。

范围完整覆盖开发目录 diagnostics/end_causal_audit、research，以及指定 full-200 run 顶层媒体和最终 W&B 表；另外定点核对两个 Original 副本的 Probe 文件/日志和指定数据源。本包不声称穷尽整个服务器的所有历史文件或所有 W&B 中间快照。

## 2. 三类处置

| 分类 | 完整清单中的路径数（含已有 Git/缺失项） | 本次处置 |
|---|---:|---|
| 必须入 Git / MUST_GIT | {counts['MUST_GIT']} | 小型原始结果表、必要任务规范、报告、结构化诊断和离线提取脚本，保存原字节或明确标记转换方式 |
| 只登记路径和 hash / PATH_HASH_ONLY | {counts['PATH_HASH_ONLY']} | 视频/图片、Probe 张量、大日志、压缩数据集、原始配置与已有 Git 副本，登记 SHA256、大小和访问边界 |
| 不应进入共享控制面 / DO_NOT_SHARE_CONTROL | {counts['DO_NOT_SHARE_CONTROL']} | 开发/runtime 代码、launch/replay 文件、dirty patch、环境转储和临时 PID；仅引用身份，不复制内容 |

密钥、认证 remote URL、完整环境转储不进入共享副本。已在实验分支的源码以固定 commit/path/blob 引用。旧 runtime 源码不会因为改放在 research 下就变成控制数据。

## 3. 当前小目标实验：九类输入核对

| 旧证据 | 核对结果 | 共享入口 / 限制 |
|---|---|---|
| historical full-200 原始 200 行 | 存在；200 行、173 success、sum_cost=145，与历史报告一致 | [完整性核验](full200/INTEGRITY_CHECK.json)、[200 行 CSV](full200/episode_results.csv)；原始 W&B JSON 同时保留 |
| W&B table / episode-level | 最终 VideoTable 有 22 列；summary 所引最终表的 SHA256 一致 | [原 summary](full200/wandb-summary.json) 与 full200/media/table；197/199 行中间表不是完整结果 |
| task spec / stable identity | 原 benchmark 200 行；规范化 task_path 后 200/200 唯一配对 | [原任务规范解压副本](task_specs/objectnavtype_val.jsonl)、[身份检查](task_specs/task_identity_check.json)；不用动态 episode ID 或行序作为身份 |
| gt_episode_len / expert_length | 200/200 非缺失，结果与任务规范逐条相等 | 上述原表与任务规范 |
| target-room visitation | has_agent_been_in_room 200/200 已保存 | 原表；该字段是轨迹之后的观察量，不是先验任务难度或物理尺寸 |
| scene/object metadata | val.jsonl.gz 与 annotations.json.gz 可读且已 SHA256；前者有 room/object/assetId | [字段与限制](metadata/STATIC_METADATA_AVAILABILITY.json)；annotations 36813 条 size 非空、2851 条为空，但尚未验证尺寸单位、实例缩放、bounding box 与 broad-target 绑定 |
| Probe .pt / log / analysis scripts | 两副本共四个 worker 张量均可读且哈希不同；两个 11 MB 日志哈希相同 | [Probe 来源说明](probe/PROVENANCE.md)；原 AUC 三元组的精确 1000-sample 文件/分析 manifest 尚未识别 |
| end_causal_audit report/json | 已找到并归档各版本小型报告与结构化轨迹 | legacy_development/diagnostics/end_causal_audit；Gate B 结论不在本次解除 |
| sub120 单案例证据 | actions JSON/CSV、近似概率 CSV 和两个提取脚本存在 | sub120/；概率为视频量化近似，不是精确 logits，也不是群体机制证据 |

仅完成源文件身份/完整性核验。没有生成当前实验的 category_sr、size effect、回归或 size_analysis 结果；静态目标实例尺寸能否完整构建仍需在正式实验中验证。

## 4. 网页研究与 GitHub/服务器的对齐

[WEB_RESEARCH_CONTEXT.md](WEB_RESEARCH_CONTEXT.md) 记录已读取范围与研究边界。可直接读取的“分支 · 02｜SafeVLA研究”共 42 轮，涵盖继承的早期研究和后续分支；主“02｜SafeVLA研究”独立链接未获得，不能声称另行读取了主对话全文。GitHub 已迁移的 LEGACY_RESEARCH_STATE / EVIDENCE_REGISTER 用作索引，实际原始文件优先。

- 173/200 的 2026-08-03 run 与旧 169/200 或修改策略的 174/200 run 严格区分。
- 后来“某些小类别约 50%”的回忆尚未绑定到同一 run，不与该 173/200 表混合。
- 历史 Probe 标签是 close-and-visible / stop-legality-like；AUC 不能当作小目标尺寸证据。
- 广义 synset 的 success-eligible target IDs 与旧 narrow visibility/room 指标可能不同；不得静默等同。
- 所有旧报告中的后续实验建议都只是历史文本，不是当前执行授权。

## 5. 复核与使用

运行 tools/verify_archive.py 可离线复核共享副本、原始表、200 个稳定键及专家步长配对。加 --server 可重新校验库存中的全部现存来源哈希（会读取大文件）。tools/build_archive.py 保存本轮原始取证/转换逻辑；源路径与审计时分支身份以本包 manifest 为准。

完整源码 hash、checkpoint 与历史协议的存在不证明当前运行可复现。此次归档不推进正式实验状态，不重跑 001A/001B，不替 PI 续期或批准。
"""
write(P/'README.md',readme)
probe=[r for r in records if r.get('group')=='probe' and r['server_path'].endswith('.pt') and r.get('exists')]
text="# Historical Probe provenance\n\nNo tensor was deserialized in this audit. Equal filenames or equal byte sizes do not establish identical data.\n\n| Source | Bytes | SHA256 |\n|---|---:|---|\n"
for r in probe:text+=f"| {r['server_path']} | {r['size']} | {r['sha256']} |\n"
text+="""\nAll four recovered worker tensor hashes differ. The two preserved collection output.log files share SHA256 cfec118467333de74228066bbbfda643aaf790eaef03db6ed0918fce83f1cfe1. Do not attach the historical 0.955/0.982/0.992 AUC triplet to any recovered tensor from a filename, size, or nearby log alone.

The named top-level probe_data.pt and probe_small_object.py were absent in both checked replicas. The collector exists at online_evaluation/probe_small_object.py and is byte-identical to feature/probing-experiment commit 82dceb89390e6df0c2e4c29c646aaa925a483217, SHA256 949e19af8956e77b2a585c16514b53f0d7f7541524f5ac180b091ecbcc8f18f9. It includes collection and step-split accuracy/balanced-accuracy analysis, not the preserved AUC triplet's exact original analysis manifest.

l_probing.py is also already committed on that experimental branch (SHA256 e8fe0c5f9049272baab1ed6c580b8a9e686680efca872bfa012954414ada1544), but concerns Qwen/LLaVA spatial-relation probing rather than the SafeVLA three-layer stop-legality Probe. Its name is not proof of relevance.

Line-numbered log excerpts are source evidence, not current execution logs. Historical PT-Guard action rewriting, absent episode/task IDs, step-split leakage, and uncertain label alignment remain provenance limits.
"""
write(P/'probe/PROVENANCE.md',text)
write(P/'WEB_RESEARCH_CONTEXT.md',"""# Web research context and retrieval coverage

Read source: “分支 · 02｜SafeVLA研究”, https://chatgpt.com/c/6a7b4d04-abf0-83ee-991b-e3e9a9821032 .
The available branch was read across 42 turns covering 2026-07-26 through 2026-09-18, including inherited research content. The separate main “02｜SafeVLA研究” conversation ID was not retrieved. Therefore this archive does not claim independent access to its full current body or its private attachments.

Coordination source: “网页版和桌面端循环”, https://chatgpt.com/c/6aa61ae9-4100-83ee-8131-dff33e18fe9e .
Its latest retrieved discussion requested classification of historical server evidence before a shared commit. GitHub LEGACY_RESEARCH_STATE.md and EVIDENCE_REGISTER.md contain an earlier migration of the research story; these are secondary indexes, not substitutes for raw files.

## Research progress relevant to this archive

- The mainline is the zero-rollout small-target phenotype question, with category, sample size and task difficulty as competing explanations. Physical size must be policy-independent; trajectory-visible pixels are not a size definition.
- Historical full-200 173/200 and later recollections of roughly 50% category SR were not established as the same run. An older 169/200 run and modified-policy/PT-Guard comparisons must remain separate.
- end probability is an Actor action probability, not an end Probe or success probability.
- The historical three-layer Probe decoded a close-and-visible/stop-legality-like label, with PT-Guard/provenance/split limitations. It did not test small physical size.
- The sub120 action/probability extraction concerned one 600-step trace and quantized video bars. It does not establish exact probabilities, ever-legal stopping, representation loss, or replay fidelity.
- Replay/fidelity proposals and prior observations are preserved background only. This archive does not authorize or execute them.

## Evidence precedence

Current server raw bytes and hashes take precedence over prior assistant summaries. Recovered tables are archived intact; statistical/causal claims beyond their integrity are deferred to the separately controlled experiment. No full chat export, login details or private account material is published.
""")
manifest=json.loads((P/'ARCHIVE_MANIFEST.json').read_text())
manifest['created_at_utc']=NOW;manifest['classification_counts']=dict(counts)
manifest['inventory_counts']={'registered_paths':len(records),'existing_readable':len(existing),'missing_named_paths':len(records)-len(existing),
                              'source_paths_already_in_github':len(existing)-len(unpublished),'existing_paths_not_in_github':len(unpublished)}
write(P/'ARCHIVE_MANIFEST.json',jb(manifest))
# Refresh all shared artifact hashes, keeping byte-copy/transform provenance where it already exists.
oldidx=json.loads((P/'ARTIFACT_INDEX.json').read_text());previous={x['git_path']:x for x in oldidx['artifacts']}
items=[]
for q in sorted(P.rglob('*')):
    if not q.is_file() or q.name=='ARTIFACT_INDEX.json':continue
    raw=q.read_bytes();gp=str(q.relative_to(C));item=previous.get(gp,{'name':str(q.relative_to(P)),'server_path':str(q),'git_path':gp,'required_for_PI_review':True})
    item.update(sha256=hashlib.sha256(raw).hexdigest(),size=len(raw));items.append(item)
write(P/'ARTIFACT_INDEX.json',jb({'schema_version':'1.0','status':'HISTORICAL_EVIDENCE_ARCHIVE','artifacts':items}))
rootindex=C/'research/ARTIFACT_INDEX.json';d=json.loads(rootindex.read_text())
gp=str(R/'ARTIFACT_INDEX.json');d['artifacts']=[x for x in d['artifacts'] if x.get('git_path')!=gp]
raw=(P/'ARTIFACT_INDEX.json').read_bytes()
d['artifacts'].append({'name':'Historical evidence alignment inventory and review packet','server_path':str(P/'ARTIFACT_INDEX.json'),'git_path':gp,'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),'required_for_PI_review':True})
write(rootindex,jb(d))
marker='## 2026-09-20 — Historical server evidence aligned'
appendix=f"""\n\n{marker}

User-authorized evidence archival only; no experiment claim or execution and no change to LOOP_STATE/NEXT_EXPERIMENT.
Packet: [history/evidence-alignment-20260919/README.md](history/evidence-alignment-20260919/README.md).
The inventory records {len(records)} paths ({len(existing)} existing/readable, {len(records)-len(existing)} missing names); {len(unpublished)} existing sources lack byte-identical copies on the four inspected GitHub branches.
Final W&B raw table is 200 rows / 173 successes / sum_cost 145 and matches its recorded SHA256. Stable task-path normalization pairs 200/200 tasks, with 200/200 expert_length == gt_episode_len and room-visitation values present.
Static scene/asset sources exist, but annotated size is not yet validated as transformed per-target simulator bounding-box size. Four recovered Probe worker tensor hashes differ; the original AUC artifact/analysis identity remains unresolved.
Raw tensors, media, large logs and datasets remain server-side with SHA256/size. Existing experimental source copies are referenced rather than duplicated into the control plane. Old development status snapshots inside the archive are historical, not current authority.
"""
for name in ['EVIDENCE_REGISTER.md','CURRENT_STATE.md','DECISION_LOG.md']:
    q=C/'research'/name;s=q.read_text()
    write(q,s.split('\n\n'+marker,1)[0]+appendix)
print(json.dumps({'shared_files':len(items)+1,'inventory_counts':manifest['inventory_counts'],'classifications':dict(counts),'max_file_bytes':max(q.stat().st_size for q in P.rglob('*') if q.is_file())},ensure_ascii=False))
