# Historical Probe provenance

No tensor was deserialized in this audit. Equal filenames or equal byte sizes do not establish identical data.

| Source | Bytes | SHA256 |
|---|---:|---|
| /nvme2/user/qyy/SafeVLA_Original/probe_data_worker0.pt | 53628840 | b9a9fed3a2c423164ab1e19e655a0d41fd2d373965d3fff59ac3abed7fc8cfa2 |
| /nvme2/user/qyy/SafeVLA_Original/probe_data_worker1.pt | 61642152 | ed0152ed3395057a573f4050d7eb16b357b0af977f346acca19459eec0b211a7 |
| /home/amax/public/users/qyy/SafeVLA_Original/probe_data_worker0.pt | 61642152 | 80d4ae64dfc4a1a0d0b4865e03f23fe593ee72c212f7716da3328755b4c71417 |
| /home/amax/public/users/qyy/SafeVLA_Original/probe_data_worker1.pt | 61642152 | 7f1bcf86a47bc4aef651cf7e63f6ce0b5ff57c8687bf71e6c33612137b7d5eff |

All four recovered worker tensor hashes differ. The two preserved collection output.log files share SHA256 cfec118467333de74228066bbbfda643aaf790eaef03db6ed0918fce83f1cfe1. Do not attach the historical 0.955/0.982/0.992 AUC triplet to any recovered tensor from a filename, size, or nearby log alone.

The named top-level probe_data.pt and probe_small_object.py were absent in both checked replicas. The collector exists at online_evaluation/probe_small_object.py and is byte-identical to feature/probing-experiment commit 82dceb89390e6df0c2e4c29c646aaa925a483217, SHA256 949e19af8956e77b2a585c16514b53f0d7f7541524f5ac180b091ecbcc8f18f9. It includes collection and step-split accuracy/balanced-accuracy analysis, not the preserved AUC triplet's exact original analysis manifest.

l_probing.py is also already committed on that experimental branch (SHA256 e8fe0c5f9049272baab1ed6c580b8a9e686680efca872bfa012954414ada1544), but concerns Qwen/LLaVA spatial-relation probing rather than the SafeVLA three-layer stop-legality Probe. Its name is not proof of relevance.

Line-numbered log excerpts are source evidence, not current execution logs. Historical PT-Guard action rewriting, absent episode/task IDs, step-split leakage, and uncertain label alignment remain provenance limits.
