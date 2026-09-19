"""Verify archived evidence without importing SafeVLA or deserializing tensors."""
from pathlib import Path
import argparse, csv, gzip, hashlib, json
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--server',action='store_true',help='Also hash every existing original server source.')
    args=parser.parse_args()
    packet=Path(__file__).resolve().parents[1]
    root=packet.parents[2]
    idx=json.loads((packet/'ARTIFACT_INDEX.json').read_text())
    for e in idx['artifacts']:
        p=root/e['git_path']
        assert p.is_file(), e['git_path']
        assert p.stat().st_size==e['size'] and sha(p)==e['sha256'],e['git_path']
    summary=json.loads((packet/'full200/wandb-summary.json').read_text())
    for v in summary.values():
        if isinstance(v,dict) and 'path' in v:
            p=packet/'full200'/v['path']
            assert sha(p)==v['sha256'],v['path']
    d=json.loads((packet/'full200'/summary['VideoTable/ObjectNavType']['path']).read_text())
    assert all(len(x)==len(d['columns']) for x in d['data'])
    rows=[dict(zip(d['columns'],r)) for r in d['data']]
    assert len(rows)==200 and sum(r['success'] for r in rows)==173
    assert sum(r['sum_cost'] for r in rows)==145
    def key(p):return p[p.index('ObjectNavType/val/'):]
    tasks=[json.loads(line) for line in (packet/'task_specs/objectnavtype_val.jsonl').read_text().splitlines() if line.strip()]
    taskmap={key(t['task_path']):t for t in tasks}
    assert len(taskmap)==len(tasks)==200
    assert len({key(r['task_path']) for r in rows})==200
    assert all(taskmap[key(r['task_path'])]['expert_length']==r['gt_episode_len'] for r in rows)
    assert all(isinstance(r['has_agent_been_in_room'],bool) for r in rows)
    with (packet/'full200/episode_results.csv').open(newline='') as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==200
    assert all(c[k]==str(r[k]) for c,r in zip(csvrows,rows) for k in d['columns'])
    inv=json.loads((packet/'FILE_INVENTORY.json').read_text())
    absent=[r for r in inv['records'] if r.get('exists') and not r.get('exact_github_copies')]
    with (packet/'HISTORICAL_NOT_IN_GIT.csv').open(newline='') as f:listed=list(csv.DictReader(f))
    assert {r['server_path'] for r in listed}=={r['server_path'] for r in absent}
    for r in inv['records']:
        if r.get('git_path'):
            assert sha(root/r['git_path'])==r['sha256'],r['server_path']
        if args.server and r.get('exists'):
            p=Path(r['server_path'])
            assert p.is_file() and p.stat().st_size==r['size'] and sha(p)==r['sha256'],r['server_path']
    print(json.dumps({'status':'PASS','shared_artifacts_verified':len(idx['artifacts']),
          'raw_rows':len(rows),'matched_tasks':len(taskmap),'source_hashes_checked':sum(r.get('exists',False) for r in inv['records']) if args.server else 0}))
if __name__=='__main__':main()
