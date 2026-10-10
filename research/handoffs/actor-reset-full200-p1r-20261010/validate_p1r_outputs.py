"""CPU-only structural/evidence validation; never imports the experiment runner."""
import ast,csv,hashlib,json,sys
from pathlib import Path
O=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=(O/'run_p1r_preflight.py').read_text();tree=ast.parse(source)
 compile(source,str(O/'run_p1r_preflight.py'),'exec')
 rows=list(csv.DictReader((O/'TASK_MANIFEST.csv').open()))
 assert len(rows)==200 and len({r['sample_id'] for r in rows})==200
 assert [int(r['original_row_index']) for r in rows[:5]]==[127,103,16,6,50]
 assert sum(r['p1_selected']=='True' for r in rows)==5
 assert sha(O/'TASK_MANIFEST.csv')=='8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0'
 reset=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='root_reset')
 assert {n.func.attr for n in ast.walk(reset) if isinstance(n,ast.Call)}=={'zero_'}
 assert 'critic' not in ast.unparse(reset) and 'random' not in ast.unparse(reset)
 for token in [".open('x')","['cpu','offline','off_a','off_b']","counts['sample']+=1","counts['mode']+=1","report['initialization_attempts']<=5"]:assert token in source
 if '--static' in sys.argv:
  print(json.dumps({'status':'PASS','scope':'CPU syntax, 200-row manifest, reset write targets and static budget guards','runner_sha256':sha(O/'run_p1r_preflight.py')}));return
 m=load(O/'RUN_MANIFEST.json');assert m['status'] in ['AWAITING_PI_REVIEW','BLOCKED','INVALID']
 assert m['claim_id']=='fa92b7b37bb6457383ea4f3f19fa1b8b' and m['claim_commit']=='d35f645cabf45bfb2ad4c3dffa548cdd7da7c8e5'
 assert 0<=m['gpu_count']<=1 and 0<=m['episodes_started']<=10
 assert m['runner_sha256']==sha(O/'run_p1r_preflight.py')
 phases=m['phases'];assert sum(p.get('online_decisions',0) for p in phases)<=6000
 assert sum(p.get('offline_combined_forwards',0) for p in phases)<=2400
 if m['status']=='AWAITING_PI_REVIEW':
  assert [p['phase'] for p in phases]==['cpu','offline','off_a','off_b'] and all(p['status']=='PASS' for p in phases)
  assert m['episodes_completed']==m['episodes_started']==10 and m['source_unchanged']
  cpu=load(O/'cpu.json');assert cpu['status']=='PASS' and len(cpu['cases'])==4 and not cpu['cuda_initialized']
  gate=load(O/'logger_equivalence.json');assert gate['status']=='PASS' and gate['pairs']==600
  assert all(max(s[k] for k in ['raw','normalized','probs','cache_max_abs'])<=1e-5 and s['exact_discrete_rng_counts'] for s in gate['steps'])
  for phase in ['off_a','off_b']:
   eps=load(O/(phase+'.episodes.json'));assert [x['sample_id'] for x in eps]==[r['sample_id'] for r in rows[:5]]
   for e in eps:
    assert e['status']=='COMPLETED' and e['decisions']==e['metrics']['eps_len']
    assert sum(e['components'].values())==e['metrics']['cost'] and e['success']==(e['metrics']['success']>0.1)
  assert load(O/'aa_result.json')['status']=='PASS'
 print(json.dumps({'status':'PASS','handoff_status':m['status'],'note':'Evidence integrity is not a scientific success declaration'}))
if __name__=='__main__':main()
