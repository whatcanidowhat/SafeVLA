"""Independent post-run checks; CPU reads only, no simulator/model forward."""
import csv, hashlib, json
from pathlib import Path
import torch
OUT=Path('/nvme2/user/qyy/SafeVLA/research/handoffs/reset-001a-20261007')
RAW=Path('/nvme2/user/qyy/SafeVLA/research/runs/EXP-RESET-001A/execution_20261008')
m=json.loads((OUT/'RUN_MANIFEST.json').read_text())
assert m['claim_id']=='24f414396c8f4ef39b34b179dee8c6fa'
assert m['instruction_commit']=='0493548a70aa192bb04457bc15e9343d934bb259'
assert m['episodes_started']<=4 and m['gpu_count']<=1
assert m['official_snapshot_unchanged'] and m['development_diff_unchanged']
report={'manifest_identity':True,'budget':True,'source_preservation':True,'status':m['status'],'read_only_cpu_validation':True}
if m['status']=='AWAITING_PI_REVIEW':
    r=json.loads((OUT/'rollover_stress_result.json').read_text())
    aa=json.loads((OUT/'aa_off_equivalence.json').read_text())
    assert aa['pass']
    raw=torch.load(RAW/'offline_replays.pt',map_location='cpu',weights_only=False)
    cap=torch.load(RAW/'captured_actor_inputs.pt',map_location='cpu',weights_only=False)
    counts=json.loads((RAW/'forward_counts.json').read_text())
    assert all(n==len(cap) for n in counts.values())
    action_names=r['action_names']
    for i,record in enumerate(cap):
        assert torch.equal(record['history_action'].flatten(),record['sampled_action'].flatten())
        assert record['returned_action']==action_names[int(record['sampled_action'].item())]
        if i+1<len(cap) and cap[i+1]['episode_order']==record['episode_order']:
            assert torch.equal(cap[i+1]['prev_actions'].flatten(),record['sampled_action'].flatten())
    rows=list(csv.DictReader((OUT/'rollover_trace.csv').open()))
    n=len(rows); assert n==r['decisions']>=220
    atol=r['atol']; rtol=r['rtol']
    for tag in ['clean','carry']:
        a=raw[tag+'1']; b=raw[tag+'2']; assert len(a)==len(b)==n
        for x,y in zip(a,b):
            for field in ['hidden','logits','probs']: assert torch.allclose(x[field],y[field],atol=atol,rtol=rtol)
            for field in ['sampled','greedy','history']: assert torch.equal(x[field],y[field])
    firstlog=firstarg=firstsample=None
    for t,(a,b,row) in enumerate(zip(raw['clean1'],raw['carry1'],rows)):
        assert int(row['local_timestep'])==t
        assert a['start_pos']==t%500 and b['start_pos']==(300+t)%500
        for tag,x in [('clean',a),('carry',b)]:
            assert torch.equal(torch.tensor(json.loads(row[tag+'_logits'])),x['logits'].flatten())
            assert torch.equal(torch.tensor(json.loads(row[tag+'_probabilities'])),x['probs'].flatten())
            assert int(row[tag+'_argmax_action'])==x['logits'].argmax().item()
            assert int(row[tag+'_common_rng_sample'])==x['sampled'].item()
        dz=(a['logits']-b['logits']).abs().max().item()
        assert abs(float(row['logit_max_abs'])-dz)<1e-10
        if firstlog is None and not torch.allclose(a['logits'],b['logits'],atol=atol,rtol=rtol): firstlog=t
        if firstarg is None and not torch.equal(a['greedy'],b['greedy']): firstarg=t
        if firstsample is None and not torch.equal(a['sampled'],b['sampled']): firstsample=t
    assert firstlog==r['first_logit_divergence_step']
    assert firstarg==r['first_argmax_action_divergence_step']
    assert firstsample==r['first_common_rng_sample_divergence_step']
    expected='R1' if firstlog is not None and (firstarg is not None or firstsample is not None) else ('R2' if firstlog is not None else 'R3')
    assert expected==r['classification']
    report.update(raw_replays_validated=True,trace_matches_raw=True,csv_float32_roundtrip_exact=True,captured_sampled_returned_history_next_input_chain_exact=True,forward_counts=counts,classification=expected,decisions=n)
else:
    assert m['status'] in ['INVALID','BLOCKED','ABORTED']
    report.update(scientific_result='NOT_ESTABLISHED',failure=m.get('error'))
report['validation']='PASS'
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
