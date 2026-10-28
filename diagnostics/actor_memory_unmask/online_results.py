"""Task-aligned official SR/Cost; mask direction is OFF minus ON."""
import math
from pathlib import Path
import random
from diagnostics.actor_logits.storage import read_json, require
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


def analyze(mask_on,mask_off):
    l,r=validate_run(mask_on,200),validate_run(mask_off,200)
    require(l['status']==r['status']=='PASS','Both200 integrity gates must pass')
    lm,rm=read_json(Path(mask_on)/'RUN_MANIFEST.json'),read_json(Path(mask_off)/'RUN_MANIFEST.json')
    require(lm['tasks']==rm['tasks'] and lm['action_names']==rm['action_names'],'Task/worker/order/action identity mismatch')
    lp,rp=lm['provenance'],rm['provenance']
    for k in ['git_commit','source_sha256','checkpoint_sha256','dino_sha256','manifest_sha256','seed','num_workers','shuffle','greedy','test_augmentation','physical_gpu','control_evidence']:
        require(k in lp and k in rp and lp[k]==rp[k],'Provenance mismatch: '+k)
    require(lp['actor_reset']==rp['actor_reset']=='off','Counter/KV reset treatment forbidden')
    require(lp['actor_attention_mask']=='on' and rp['actor_attention_mask']=='off','Expected MASK_ON then MASK_OFF')
    lo,ro=l['outcomes'],r['outcomes']
    ds=[int(b['success'])-int(a['success']) for a,b in zip(lo,ro)]
    dc=[float(b['cost'])-float(a['cost']) for a,b in zip(lo,ro)]
    rng=random.Random(20261028);bs=[];bc=[]
    for _ in range(10000):
        indices=[rng.randrange(200) for _ in range(200)]
        bs.append(sum(ds[i] for i in indices)/200);bc.append(sum(dc[i] for i in indices)/200)
    improve=sum(x==1 for x in ds);worse=sum(x==-1 for x in ds);n=improve+worse
    exact=min(1.,2*sum(math.comb(n,i) for i in range(min(improve,worse)+1))/2**n) if n else 1.
    return {'status':'COMPLETE','direction':'MASK_OFF minus MASK_ON','mask_on':summarize(lo),'mask_off':summarize(ro),
            'delta_SR':sum(ds)/200,'delta_mean_cost':sum(dc)/200,
            'paired_bootstrap_95_SR':[percentile(bs,.025),percentile(bs,.975)],
            'paired_bootstrap_95_cost':[percentile(bc,.025),percentile(bc,.975)],
            'success_discordance':{'improved':improve,'worsened':worse,'exact_mcnemar_p':exact},
            'paired_tasks':[{'sample_id':t['sample_id'],'delta_success':s,'delta_cost':c} for t,s,c in zip(lm['tasks'],ds,dc)],
            'scope':'Exploratory task-aligned comparison; sequential episode histories and stochastic trajectories violate independence assumptions. Bootstrap/McNemar are descriptive uncertainty, not full causal identification. No nonsignificance-as-equivalence claim.'}
