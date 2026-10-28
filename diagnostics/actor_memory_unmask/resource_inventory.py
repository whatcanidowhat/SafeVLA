"""Bounded CPU-only inventory of selected200 houses and referenced asset files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=180)
    a=p.parse_args();assert not a.output.exists(),'Never overwrite inventory'
    deadline=time.monotonic()+a.seconds
    files=[];missing=[];unknown=[];asset_ids=set();selected={};status='INCOMPLETE'
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Inventory time budget reached')
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):check();h.update(chunk)
        return h.hexdigest()
    def add(path,kind):
        files.append({'path':str(path),'sha256':digest(path),'size':path.stat().st_size,'kind':kind})
        if len(files)%100==0:print('hashed_files',len(files),'asset_ids',len(asset_ids),flush=True)
    try:
        tasks=[json.loads(line) for line in gzip.decompress((a.repo/'benchmark/objectnavtype_val.jsonl.gz').read_bytes()).splitlines() if line]
        assert len(tasks)==200
        indices={int(t['house_index']) for t in tasks}
        houses=a.data/'objaverse_houses/houses_2023_07_28/val.jsonl.gz'
        annotations=a.data/'objaverse_assets/2023_07_28/annotations.json.gz'
        add(houses,'house_dataset');add(annotations,'asset_annotations')
        ann=json.loads(gzip.decompress(annotations.read_bytes()))
        assert isinstance(ann,dict)
        with gzip.open(houses,'rt') as stream:
            for i,line in enumerate(stream):
                check()
                if i in indices:selected[i]=json.loads(line)
                if i>=max(indices):break
        assert set(selected)==indices,'Required house indices missing'
        def visit(value):
            if isinstance(value,dict):
                if isinstance(value.get('assetId'),str):asset_ids.add(value['assetId'])
                for child in value.values():visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
        for house in selected.values():visit(house)
        asset_root=a.data/'objaverse_assets/2023_07_28/assets'
        external=[]
        for asset_id in sorted(asset_ids):
            check();directory=asset_root/asset_id
            assert directory.resolve().is_relative_to(asset_root.resolve()),'Asset path escapes root'
            if directory.is_dir():
                before=len(files)
                for path in sorted(directory.rglob('*')):
                    check()
                    if path.is_file():add(path,'selected_scene_asset')
                if len(files)==before:missing.append(asset_id)
                external.append(asset_id)
            elif asset_id in ann or re.fullmatch(r'[a-fA-F0-9]{32}|[a-fA-F0-9-]{36}',asset_id):missing.append(asset_id)
            else:unknown.append(asset_id)
        status='COMPLETE_EXTERNAL_CLOSURE_REQUIRES_BUILTIN_RENDERER_PIN' if not missing else 'MISSING_ASSETS'
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    else:error=None
    record={'status':status,'error':error,'scope':'All files under external asset directories referenced by every object in the selected200 task houses. Built-in/non-external asset IDs remain subject to simulator build/renderer review; this inventory alone is not full simulator identity proof.',
            'task_count':len(tasks) if 'tasks' in locals() else None,'house_indices':sorted(selected),
            'referenced_asset_count':len(asset_ids),'missing_external_assets':missing,
            'builtin_or_unresolved_asset_ids':unknown,'files':files,
            'hashed_files':len(files),'hashed_bytes':sum(x['size'] for x in files)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['files','house_indices','builtin_or_unresolved_asset_ids']},indent=2),flush=True)
    raise SystemExit(0 if status.startswith('COMPLETE') else 2)


if __name__=='__main__':main()
