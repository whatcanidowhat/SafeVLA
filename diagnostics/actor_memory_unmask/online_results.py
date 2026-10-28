"""Task-aligned official SR/Cost; mask direction is OFF minus ON."""
import math
from pathlib import Path
import random
from diagnostics.actor_logits.storage import read_json, read_jsonl, require
from diagnostics.actor_logits.validate import validate_run
from diagnostics.actor_logits.logger import COST_KEYS


def percentile(values,p):
    values=sorted(values);x=(len(values)-1)*p;i=int(x)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)


def summarize(outcomes):
    costs=[float(e['cost']) for e in outcomes]
    return {'n':len(outcomes),'successes':sum(int(e['success']) for e in outcomes),
            'SR':sum(int(e['success']) for e in outcomes)/len(outcomes),
            'cost_sum':sum(costs),'mean_cost':sum(costs)/len(costs),
            'cost_median':percentile(costs,.5),'cost_p90':percentile(costs,.9),'cost_max':max(costs),
            'cost_distribution':costs,'components':{k:sum(e[k] for e in outcomes) for k in COST_KEYS}}


def behavior(rows,official):
    require(rows,'Empty episode behavior trace')
    actions=[r['executed_action_name'] for r in rows]
    success=bool(official['success']>.1)
    terminal_end=actions[-1]=='end'
    coverage=official.get('percentage_rooms_visited')
    target_room=official.get('for_video_table/has_agent_been_in_room')
    if coverage is not None:
        coverage=float(coverage)
        require(math.isfinite(coverage) and 0<=coverage<=1.000001,'Invalid official room coverage')
    if target_room is not None:require(type(target_room) is bool,'Invalid target-room metric')
    return {'success':success,'eps_len':len(rows),'terminal_action':actions[-1],
            'subhorizon_end':terminal_end and len(rows)<600,
            'failed_early_end':not success and terminal_end and len(rows)<600,
            'failed_horizon_exhaustion':not success and len(rows)>=600,
            'official_percentage_rooms_visited':coverage,
            'official_total_rooms_visited':official.get('total_rooms_visited'),
            'official_visited_target_room':target_room,
            'failed_without_target_room_visit_proxy':None if target_room is None else (not success and not target_room),
            'adjacent_repeat_fraction':sum(a==b for a,b in zip(actions,actions[1:]))/max(1,len(rows)-1),
            'aba_oscillations':sum(actions[i]==actions[i-2]!=actions[i-1] for i in range(2,len(actions))),
            'scope':'Early END is failure+end before600. Official room coverage/target-room visits and action repetition are descriptive exploration proxies, not a causal diagnosis of insufficient exploration. No new environment queries.'}


def aggregate_behavior(episodes):
    failures=[e for e in episodes if not e['success']]
    coverage=[e['official_percentage_rooms_visited'] for e in failures if e['official_percentage_rooms_visited'] is not None]
    target=[e['failed_without_target_room_visit_proxy'] for e in episodes if e['failed_without_target_room_visit_proxy'] is not None]
    return {'episodes':len(episodes),'failures':len(failures),
            'failed_early_end':sum(e['failed_early_end'] for e in episodes),
            'failed_horizon_exhaustion':sum(e['failed_horizon_exhaustion'] for e in episodes),
            'failure_room_coverage_n':len(coverage),
            'failure_room_coverage_mean':sum(coverage)/len(coverage) if coverage else None,
            'failure_room_coverage_median':percentile(coverage,.5) if coverage else None,
            'target_room_metric_n':len(target),'failed_without_target_room_visit_proxy':sum(target) if target else None,
            'insufficient_exploration_causal_conclusion':'NOT_IDENTIFIED_FROM_THESE_PROXIES'}


def analyze(mask_on,mask_off):
    l,r=validate_run(mask_on,200),validate_run(mask_off,200)
    require(l['status']==r['status']=='PASS','Both200 integrity gates must pass')
    lm,rm=read_json(Path(mask_on)/'RUN_MANIFEST.json'),read_json(Path(mask_off)/'RUN_MANIFEST.json')
    require(lm['tasks']==rm['tasks'] and lm['action_names']==rm['action_names'],'Task/worker/order/action identity mismatch')
    lp,rp=lm['provenance'],rm['provenance']
    for k in ['git_commit','source_sha256','checkpoint_sha256','dino_sha256','manifest_sha256','seed','num_workers','shuffle','greedy','test_augmentation','physical_gpu','control_evidence','runtime_baseline']:
        require(k in lp and k in rp and lp[k]==rp[k],'Provenance mismatch: '+k)
    require(lp['actor_reset']==rp['actor_reset']=='off','Counter/KV reset treatment forbidden')
    require(lp['actor_attention_mask']=='on' and rp['actor_attention_mask']=='off','Expected MASK_ON then MASK_OFF')
    for provenance in [lp,rp]:
        baseline=provenance['runtime_baseline']
        require(baseline['source_baseline']['max_steps']==500 and baseline['source_baseline']['episode_horizon']==600,'Runtime Baseline not500/horizon600')
        for branch in baseline['loaded_branches'].values():
            require(branch['max_steps']==500 and branch['decoder_max_seq_len']==500,'Runtime branch configuration drift')
            require(all(layer['cache_k_shape'][1]==layer['cache_v_shape'][1]==500 for layer in branch['layers']),'Runtime K/V capacity drift')
    lo,ro=l['outcomes'],r['outcomes']
    ds=[int(b['success'])-int(a['success']) for a,b in zip(lo,ro)]
    dc=[float(b['cost'])-float(a['cost']) for a,b in zip(lo,ro)]
    rng=random.Random(20261028);bs=[];bc=[]
    for _ in range(10000):
        indices=[rng.randrange(200) for _ in range(200)]
        bs.append(sum(ds[i] for i in indices)/200);bc.append(sum(dc[i] for i in indices)/200)
    improve=sum(x==1 for x in ds);worse=sum(x==-1 for x in ds);n=improve+worse
    exact=min(1.,2*sum(math.comb(n,i) for i in range(min(improve,worse)+1))/2**n) if n else 1.
    left_paths=sorted((Path(mask_on)/'episodes').glob('*.complete'))
    right_paths=sorted((Path(mask_off)/'episodes').glob('*.complete'))
    require(len(left_paths)==len(right_paths)==200,'Missing complete behavior traces')
    on_behavior=[behavior(read_jsonl(p/'actions.jsonl'),read_json(p/'metrics.json')['official']) for p in left_paths]
    off_behavior=[behavior(read_jsonl(p/'actions.jsonl'),read_json(p/'metrics.json')['official']) for p in right_paths]
    return {'status':'COMPLETE','direction':'MASK_OFF minus MASK_ON','mask_on':summarize(lo),'mask_off':summarize(ro),
            'delta_SR':sum(ds)/200,'delta_mean_cost':sum(dc)/200,
            'paired_bootstrap_95_SR':[percentile(bs,.025),percentile(bs,.975)],
            'paired_bootstrap_95_cost':[percentile(bc,.025),percentile(bc,.975)],
            'success_discordance':{'improved':improve,'worsened':worse,'exact_mcnemar_p':exact},
            'behavior':{'mask_on':aggregate_behavior(on_behavior),'mask_off':aggregate_behavior(off_behavior)},
            'joint_directional_criterion':sum(ds)>0 and sum(dc)<0,
            'paired_tasks':[{'sample_id':t['sample_id'],'delta_success':s,'delta_cost':c,
                             'delta_cost_components':{k:b[k]-a[k] for k in COST_KEYS},
                             'mask_on_behavior':ab,'mask_off_behavior':bb}
                            for t,s,c,a,b,ab,bb in zip(lm['tasks'],ds,dc,lo,ro,on_behavior,off_behavior)],
            'scope':'Exploratory task-aligned comparison; sequential episode histories and stochastic trajectories violate independence assumptions. Bootstrap/McNemar are descriptive uncertainty, not full causal identification. No nonsignificance-as-equivalence claim.'}
