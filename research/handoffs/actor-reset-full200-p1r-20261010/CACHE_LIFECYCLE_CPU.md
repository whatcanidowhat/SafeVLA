# Gate A CPU cache lifecycle: PASS

Genuine frozen B0 Attention source `training/online/third_party_models/llama/model.py`, SHA256 `f12e1f97b83b58cf8cb99b500e4afbda7e5a7daf048c3065095d0dedaf5ab69c`. Shared restore_temporal clones/rebinds K/V and restores counter. Three Attention layers match dim512, heads8, head_dim64, max_seq_len500; initial max_batch_size0. Fixture weights are deterministic CPU engineering weights, not the SafeVLA checkpoint. CUDA_VISIBLE_DEVICES empty; torch.cuda.is_initialized stayed false.

| Weight dtype | Initial cache case | Layers | Repeat output max_abs | Full invariant result |
| --- | --- | --- | --- | --- |
| torch.float32 | empty | 3 | 0.0 | PASS |
| torch.float32 | allocated | 3 | 0.0 | PASS |
| torch.float64 | empty | 3 | 0.0 | PASS |
| torch.float64 | allocated | 3 | 0.0 | PASS |

Each empty case followed[0,500,8,64]→first forward[1,500,8,64]→restore[0,500,8,64]→identical repeat. Allocated cases saved counter1/cache1, advanced to2, restored1, repeated to2. All before/restored and first/repeated tensor shape/dtype/device/value hashes match; counter, weights, RNG and snapshot immutability match. Rebound tensors have no snapshot/live-object aliases; stale references remain unchanged. Float64 weights also test restoring initially float32 ordinary cache tensors before their original dtype conversion on forward.
24 genuine Attention calls total (4 cases×2 forwards×3 layers), 0 combined policy forwards,0 GPU. Full per-layer hashes/metadata in cpu.json. Independent report reconciliation passed without rerunning Gate A.
Scope: this CPU lifecycle fix is supported. CUDA replay, logger nonintrusiveness, critic isolation and online performance are NOT validated by Gate A.
