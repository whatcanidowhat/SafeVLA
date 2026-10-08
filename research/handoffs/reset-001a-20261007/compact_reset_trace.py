"""Lossless float32-readable CSV serialization; preserves the original raw CSV."""
import csv, json, shutil
from pathlib import Path
OUT=Path('/nvme2/user/qyy/SafeVLA/research/handoffs/reset-001a-20261007')
RAW=Path('/nvme2/user/qyy/SafeVLA/research/runs/EXP-RESET-001A/execution_20261008')
src=OUT/'rollover_trace.csv'
original=RAW/'rollover_trace_full_precision.csv'
assert not original.exists()
shutil.copyfile(src,original)
with src.open() as f:
    reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
for row in rows:
    for col in ['clean_logits','carry_logits','clean_probabilities','carry_probabilities']:
        row[col]='['+','.join(format(x,'.9g') for x in json.loads(row[col]))+']'
with src.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
assert src.stat().st_size<=1048576
p=OUT/'rollover_stress_result.json'; r=json.loads(p.read_text())
r['trace_serialization']='Logit/probability arrays use 9 significant decimal digits, exactly round-tripping float32 values. Unmodified CSV and tensors remain server-side.'
p.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'original_bytes':original.stat().st_size,'shared_bytes':src.stat().st_size}))
