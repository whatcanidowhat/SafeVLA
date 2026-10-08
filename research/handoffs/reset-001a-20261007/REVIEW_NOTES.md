# PI review notes

- Verify the recovered claim and green CI precede runtime work; no new claim was generated during recovery.
- Review Actor-only state ownership, reset completeness, and OFF identity against the supplied patch.
- Review input source/selection, task IDs, forward counts, RNG observation scope, load events, runtime paths and preserved hashes.
- Scientific status: AWAITING_PI_REVIEW / R1. A valid Actor carry package reproducibly changes the policy distribution and at least one decision under the tested identical current inputs. This supports the state-carry-to-Actor-decision pathway; it does not establish SR or Safety Cost impact.
- A/A uses a saved-encoder-output boundary and exact original root forward, with decoder-input reconstruction checked. It does not rerun the visual frontend or the whole simulator for A/A.
- Derived attention masks differ in cache-coordinate shape between CLEAN and CARRY by design; episode masks, timesteps, recorded previous actions and all captured feature tensors remain paired.
- The reset harness uses uninitialized rollout bookkeeping for the isolated root test; unchanged original action/history code and untouched OFF branch support its restricted scope. This is not a whole-agent long-run equivalence proof.
- The independent validator performs CPU-only recomputation of saved result arithmetic; it is not a new model experiment.
- Review-only proposal: decide the next step from the frozen R1/R2/R3/R4 decision table. No follow-up is approved here.
