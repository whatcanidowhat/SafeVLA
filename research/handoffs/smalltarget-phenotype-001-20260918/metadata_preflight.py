#!/usr/bin/env python3
"""Read static metadata only; never import SafeVLA, torch, or AI2-THOR."""
import collections,gzip,hashlib,json,math
from pathlib import Path
ROOT=Path('/nvme2/user/qyy/SafeVLA')
OUT=ROOT/'research/handoffs/smalltarget-phenotype-001-20260918'
HOUSE=Path('/home/amax/public/datasets/qyy/objaverse_houses/houses_2023_07_28')
ASSET=Path('/home/amax/public/datasets/qyy/objaverse_assets/2023_07_28')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def readgz(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def walk(objects):
 for o in objects:
  yield o
  yield from walk(o.get('children',[]))
def bbox(d):
 if not isinstance(d,dict):return None
 b=d.get('assetMetadata',{}).get('boundingBox')
 if not b:return None
 try:
  size=[b['max'][a]-b['min'][a] for a in 'xyz']
  if all(math.isfinite(s) and s>0 for s in size):return {'bounds':b,'side_lengths':size}
 except (KeyError,TypeError):pass
 return None
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with gzip.open(ROOT/'benchmark/objectnavtype_val.jsonl.gz','rt') as f:tasks=[json.loads(l) for l in f if l.strip()]
 needed={t['house_index'] for t in tasks}
 houses={}
 with gzip.open(HOUSE/'val.jsonl.gz','rt') as f:
  for i,line in enumerate(f):
   if i in needed:houses[i]=json.loads(line)
   if i>=max(needed):break
 house_ann=readgz(HOUSE/'annotations.json.gz')
 asset_ann=readgz(ASSET/'annotations.json.gz')
 records=[];file_sources={}
 for t in tasks:
  h=houses.get(t['house_index'])
  objs=list(walk(h['objects'])) if h else []
  mapping=collections.defaultdict(list)
  for o in objs:mapping[o['id']].append(o)
  targets=sorted(set(t['broad_synset_to_object_ids'][t['synsets'][0]]))
  for oid in targets:
   candidates=mapping.get(oid,[])
   o=candidates[0] if len(candidates)==1 else {}
   aid=o.get('assetId')
   ha=house_ann.get(aid,{})
   aa=asset_ann.get(aid,{})
   p=ASSET/'assets'/str(aid)/'thor_metadata.json'
   disk=json.loads(p.read_text()) if p.is_file() else {}
   if p.is_file():file_sources[str(p)]={'sha256':sha(p),'size':p.stat().st_size}
   rec={'task_key':t['task_path'].split('ObjectNavType/val/')[-1],'house_index':t['house_index'],'synset':t['synsets'][0],'target_id':oid,'scene_matches':len(candidates),'asset_id':aid,'object_fields':sorted(o),'position':o.get('position'),'rotation':o.get('rotation'),'instance_scale':o.get('scale'),'house_annotation_fields':sorted(ha),'annotation_size':aa.get('size'),'annotation_size_author':aa.get('size_annotated_by'),'house_annotation_bbox':bbox(ha.get('thor_metadata',{})),'disk_metadata_bbox':bbox(disk)}
   records.append(rec)
 sources={}
 for p in [ROOT/'benchmark/objectnavtype_val.jsonl.gz',HOUSE/'val.jsonl.gz',HOUSE/'annotations.json.gz',ASSET/'annotations.json.gz']:
  sources[str(p)]={'sha256':sha(p),'size':p.stat().st_size}
 sources.update(file_sources)
 summary={'tasks':len(tasks),'houses':len(needed),'houses_found':len(houses),'target_occurrences':len(records),'unique_house_target_pairs':len({(r['house_index'],r['target_id']) for r in records}),'scene_match_counts':dict(collections.Counter(r['scene_matches'] for r in records)),'house_annotation_bbox_occurrences':sum(r['house_annotation_bbox'] is not None for r in records),'disk_metadata_bbox_occurrences':sum(r['disk_metadata_bbox'] is not None for r in records),'either_bbox_occurrences':sum(r['house_annotation_bbox'] is not None or r['disk_metadata_bbox'] is not None for r in records),'tasks_all_targets_have_candidate_bbox':sum(all(r['house_annotation_bbox'] or r['disk_metadata_bbox'] for r in records if r['task_key']==t['task_path'].split('ObjectNavType/val/')[-1]) for t in tasks),'missing_bbox_assets':sorted({r['asset_id'] for r in records if not r['house_annotation_bbox'] and not r['disk_metadata_bbox']},key=str),'object_fields':sorted({k for r in records for k in r['object_fields']}),'annotations_only_are_not_validated_instance_sizes':True}
 result={'summary':summary,'sources':sources,'targets':records,'status':'CANDIDATE_METADATA_COVERAGE_ONLY_NOT_A_SIZE_VALIDATION'}
 (OUT/'metadata_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(summary))
 print('FIRST_TARGET',json.dumps(records[0]))
if __name__=='__main__':main()
