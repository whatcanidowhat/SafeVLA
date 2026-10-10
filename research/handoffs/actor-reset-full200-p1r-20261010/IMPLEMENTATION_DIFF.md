# P1R implementation changes

Parent implementation: P1 INVALID result a137eafbb5931190b68ff042a5b0c9e34dbfd68d, runner SHA25682339bc9aac45af1d9232f3d6a44b34d4c92b93e8a1163ff2df45980090cae1a.
Complete diff: IMPLEMENTATION.patch.txt. B0 source112 hashes remain unchanged, including genuine Attention source f12e1f97b83b58cf8cb99b500e4afbda7e5a7daf048c3065095d0dedaf5ab69c.

The sole restoration correction replaces cache_k.copy_(snapshot_k)/cache_v.copy_(snapshot_v) with attention.cache_k=snapshot_k.clone() and analogous V, preserving counter, shape, dtype, device and values without retaining aliases. The function is shared between new CPU Gate A and existing GPU replay.
Additional changes are CPU lifecycle fixtures/guards, ordered A/B/C bookkeeping, fresh paths/claim identity, partial-pair audit persistence, and truthful reporting of critic cache shapes. No other offline or online model behavior is repaired. No B0 source, frozen old runner or old artifact is edited.

Gate A covers three genuine attention layers, two CPU weight dtypes, initial empty and allocated caches, identical input/RNG repeats, complete shape/hash restoration, all weights, immutable snapshots and stale-reference alias safety. Source code and syntax/manifest checks alone do not establish a passing runtime gate.
Potential inherited latent issues remain subject to original fail-closed gates; the approved scope does not permit opportunistic repairs after failure. In particular, empty critic cache tensors must not be misrepresented as nonzero critic temporal trajectories.
