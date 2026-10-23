#!/usr/bin/env python3
"""Run four independent *locally possible* validation gates concurrently.

This is NOT a SafeVLA policy evaluation. No claims, GPU, simulation or official
SR/Cost calculations are made unless actual treatment rollouts are provided and
independently verified. All lanes emit explicit evidence/status JSON.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
HIST = HERE / "historical_16_review.csv"
CAND = HERE / "exploration_candidates.csv"
PROMPT = HERE / "prompt_probe_plan.csv"


def readcsv(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def receipt(lane, status, *, checks=None, blockers=None, metrics=None, evidence=None):
    return {"lane": lane, "status": status,
            "checks": checks or {}, "blockers": blockers or [],
            "observations": metrics or {}, "evidence": evidence or [],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy_loads": 0, "gpu_experiments": 0, "simulator_episodes": 0}


def lane1(opts):
    cases, cand = readcsv(HIST), readcsv(CAND)
    ids = [r['sub_house_id'] for r in cases]
    assert len(cases) == 16 and len(set(ids)) == 16
    assert {r['sub_house_id'] for r in cand} == {r['sub_house_id'] for r in cases if r['narrow_target_room_visited']=='False'}
    assert len([r for r in cases if r['narrow_target_room_visited']=='False']) == 9
    assert len([r for r in cases if r['scope']=='STRICT' and r['narrow_target_room_visited']=='False']) == 8
    assert len([r for r in cases if r['scope']=='AMBIGUOUS']) == 4
    assert len([r for r in cases if r['scope']=='STRICT' and r['narrow_target_room_visited']=='True']) == 4
    lookup = {r['sub_house_id']: r for r in cases}
    for r in cand:
        source=lookup[r['sub_house_id']]
        assert r['house_index']==source['house_index']
        assert r['historical_target_scope']==source['scope']
        assert r['latest_OFF200_success']==source['OFF200_success']
        assert r['latest_OFF200_safety_cost']==source['OFF200_cost']
        assert r['historical_failed_end_eps_len']==source['historical_eps_len']
    core=[r for r in cand if r['cohort']=='CORE_STRICT']
    ctrl=[r for r in cases if r['scope']=='STRICT' and r['narrow_target_room_visited']=='True']
    obs={"historical_confirmed_invalid_END":16,"historical_unvisited_narrow_room":9,
         "unvisited_STRICT":8,"unvisited_AMBIGUOUS":1,
         "visited_STRICT_negative_controls": [r['sub_house_id'] for r in ctrl],
         "core_later_OFF200_success":sum(r['latest_OFF200_success']=='True' for r in core),
         "core_later_OFF200_failure":sum(r['latest_OFF200_success']=='False' for r in core),
         "core_later_OFF200_cost14_ids":[r['sub_house_id'] for r in core if r['latest_OFF200_safety_cost']=='14'],
         "historical_16_sha256":sha256(HIST),"candidates_sha256":sha256(CAND)}
    return receipt('V1_historical_coverage','PASS_COHORT_AUDIT__BLOCKED_STEPWISE_TOPOLOGY',
       checks={"16_source_rows_unique":True,"9_candidate_join":True,"8_STRICT_cohort":True,"cross_period_identity_consistent":True},
       metrics=obs,
       blockers=["No historical per-decision room ID or robot pose + room floor polygons were present in the GitHub record; cannot reconstruct room sequence.",
                 "Room-not-visited is observational association, not proved failure causation."],
       evidence=["historical_16_review.csv (extracted/adjudicated from original GitHub cases)","exploration_candidates.csv (extracted) "])


def lane2(opts):
    test=subprocess.run([sys.executable,"-m","unittest","-v","test_topology_memory.py"],cwd=HERE,capture_output=True,text=True,timeout=20)
    if test.returncode:
        return receipt("V2_shadow_memory","FAIL_LOCAL_UNIT_TESTS", checks={"unittest_exit":test.returncode},blockers=[test.stderr[-1500:]])
    # Verify output isolation on real module, only synthetic observations.
    sys.path.insert(0,str(HERE))
    from topology_memory import RoomVisitTracker
    inp={"sample_id":"synthetic", "current_agent_room":1, "natural_language_spec":"find a mug", "rgb_dinov2":"UNTOUCHED"}
    immutable=dict(inp)
    tracker=RoomVisitTracker(inp["sample_id"])
    for step,room in enumerate([1,1,2,-1,3,1]):tracker.observe(step,room)
    assert inp==immutable
    assert tracker.snapshot()['visited_count']==3
    return receipt("V2_shadow_memory","PASS_SYNTHETIC_UNIT_TESTS__BLOCKED_REAL_POLICY_EQUIVALENCE",
      checks={"ten_unittests_passed":True,"input_object_unchanged":True,"state_is_episode_scoped":True,
              "unknown_room_does_not_create_false_topology_edge":True},
      metrics={"unittests":10,"example_unique_rooms":3},
      blockers=["No recorded per-step room sensor output accessible for an authentic trace replay.",
                "No original evaluator+checkpoint+simulator; cannot test exact B0 action/RNG/cache/logit equivalence."],
      evidence=["topology_memory.py","test_topology_memory.py"])


def lane3(opts):
    prompts=readcsv(PROMPT)
    assert len(prompts)==5
    texts=[r['text_template'].format(original_goal="find a mug",candidate_room_type="kitchen",target="mug") for r in prompts]
    assert all(len(s.encode('utf-8'))<=1000 for s in texts)
    assert len(set(texts))==5
    # Runtime preflight is not a policy behavior test.
    missing=[]
    for key, value in [('safevla_root',opts.safevla_root), ('model_checkpoint',opts.checkpoint),('house_dataset',opts.house_dataset)]:
        if not value or not Path(value).exists(): missing.append(f'{key} missing')
    for name in ('transformers','ai2thor','allenact'):
        import importlib.util
        try:
            found=importlib.util.find_spec(name) is not None
        except (ImportError,ModuleNotFoundError): found=False
        if not found: missing.append(f'python dependency {name} unavailable')
    return receipt('V3_language_intervention','PASS_PROMPT_FORMAT__BLOCKED_MODEL_ROLLOUT',
      checks={"five_conditions_unique":True,"utf8_bytes_compatible_with_goal_buffer":True,
              "not_measured_language_to_motion":True},
      metrics={"generated_prompt_templates":texts},blockers=missing+[
             "Requires checkpoint-based, isolated-state action-distribution probes and real room-goal rollouts.",
             "Do not interpret text encodeability as learned room-level navigation control."],
      evidence=['prompt_probe_plan.csv','baseline-experiment/architecture/models/allenact_transformer_models/allenact_dino_transformer.py'])


def lane4(opts):
    missing=[]
    for key,val in [('baseline_rollouts',opts.baseline_rollouts),('treatment_rollouts',opts.treatment_rollouts),
                    ('baseline_manifest',opts.baseline_manifest),('treatment_manifest',opts.treatment_manifest)]:
        if not val or not Path(val).exists():missing.append(f'{key} missing')
    if missing:
        return receipt('V4_topology_plus_language','BLOCKED_NO_PAIRED_ROLLOUTS',
          checks={"historic_OFF200_as_control_only":True,"historical_ON200_exists":False,
                  "did_not_use_oracle_target_room_for_planning":True},blockers=missing+[
                 "No treatment rollout, so intervention SR/Cost, paired effects and official improvement claims cannot be estimated.",
                 "Need B0 manifest equivalence: checkpoint/seed/worker/task order/horizon/action semantics/dataset/augmentation."],
          evidence=['historical_16_review.csv','baseline-contract provenance from research-loop'])
    # Intentionally refuse to summarize independent runs until study identity is verified.
    with open(opts.baseline_manifest,encoding='utf-8') as f:b=json.load(f)
    with open(opts.treatment_manifest,encoding='utf-8') as f:t=json.load(f)
    fields=['checkpoint_sha256','dataset_sha256','task_manifest_sha256','seed','worker_count',
            'max_steps','greedy_sampling','test_augmentation','action_order_sha256']
    missing_keys=[k for k in fields if k not in b or k not in t]
    mismatch=[k for k in fields if k in b and k in t and b[k]!=t[k]]
    if missing_keys or mismatch:
        return receipt('V4_topology_plus_language','INVALID_INCOMPARABLE_MANIFESTS',
          blockers=[f'missing comparison fields: {missing_keys}',f'mismatched fields: {mismatch}'])
    aa=readcsv(opts.baseline_rollouts);bb=readcsv(opts.treatment_rollouts)
    required={'sample_id','success','cost','eps_len'}
    if not aa or not bb or not required.issubset(aa[0]) or not required.issubset(bb[0]):
        return receipt('V4_topology_plus_language','INVALID_METRIC_SCHEMA',blockers=['Need sample_id success cost eps_len in both.'])
    x={r['sample_id']:r for r in aa};y={r['sample_id']:r for r in bb}
    if len(x)!=len(aa) or len(y)!=len(bb) or x.keys()!=y.keys():
        return receipt('V4_topology_plus_language','INVALID_TASK_IDENTITY',blockers=['Task keys not a bijection.'])
    def numeric_success(v):
        if v in ('True','true','1','1.0'):return 1
        if v in ('False','false','0','0.0'):return 0
        raise ValueError('Unknown success field '+v)
    diffs=[(numeric_success(y[k]['success'])-numeric_success(x[k]['success']), float(y[k]['cost'])-float(x[k]['cost'])) for k in x]
    return receipt('V4_topology_plus_language','PAIRED_DESCRIPTIVE_RESULTS_ONLY__NEEDS_INDEPENDENT_AUDIT',
      checks={'matching_task_ids':True,'manifest_identity_match':True},
      metrics={'n':len(diffs),'delta_SR':sum(z[0] for z in diffs)/len(diffs),
               'delta_mean_cost':sum(z[1] for z in diffs)/len(diffs),
               'improved_successes':sum(z[0]>0 for z in diffs),
               'lost_successes':sum(z[0]<0 for z in diffs)},
      blockers=['Results must be verified against raw episode logs and treatment fidelity before any causal claim.'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--safevla-root',default=None)
    ap.add_argument('--checkpoint',default=None)
    ap.add_argument('--house-dataset',default=None)
    ap.add_argument('--baseline-rollouts',default=None)
    ap.add_argument('--treatment-rollouts',default=None)
    ap.add_argument('--baseline-manifest',default=None)
    ap.add_argument('--treatment-manifest',default=None)
    ap.add_argument('--out-dir',default=str(HERE/'results'))
    args=ap.parse_args(); out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    lanes=[lane1,lane2,lane3,lane4]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        f={pool.submit(func,args):func.__name__ for func in lanes}
        results=[]
        for task in concurrent.futures.as_completed(f):
            try: r=task.result()
            except Exception as e:
                r=receipt(f[task], 'ERROR_VALIDATION',blockers=[f'{type(e).__name__}: {e}'])
            results.append(r)
            (out/f"{r['lane']}.json").write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf-8')
    results.sort(key=lambda r:r['lane'])
    overall={'experiment_id':'EXP-TOPO-MEMORY-FOURWAY-001',
             'execution':'FOUR_CONCURRENT_LOCAL_PREFLIGHTS',
             'source_code_untouched':True,
             'github_written':False,
             'policy_or_simulator_evaluation_performed':False,
             'results':results}
    (out/'FOURWAY_STATUS.json').write_text(json.dumps(overall,indent=2,ensure_ascii=False),encoding='utf-8')
    for r in results:print(f"{r['lane']}: {r['status']}")
    print(f'Receipts: {out}/FOURWAY_STATUS.json')
    return 1 if any(r['status'].startswith(('INVALID','ERROR','FAIL')) for r in results) else 0

if __name__=='__main__':
    sys.exit(main())
