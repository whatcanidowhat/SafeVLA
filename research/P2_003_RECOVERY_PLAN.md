# P2-003 OFF reuse and transport review packet (PI draft)

The recorded P2-002 failure was a **pre-model remote Claim check** transport timeout, not an OFF output integrity failure and not a discovered Actor Reset model bug.

**Reference:** [P2-002 reviewed BLOCKED handoff](https://github.com/whatcanidowhat/SafeVLA/commit/88453def76051736edce2e8887f26b08d5b0c5e6). OFF200 validation PASS, 200/200, 19,270 decisions. 1203 archived files were SHA-checked at handoff. Their live server bytes remain to be rechecked by a future authorized executor.

**Approved recovery direction:** reuse OFF200 read-only; prepare a new independently authorized ON200-only run on identical TaskSpecs and science code. This document is not itself authorization.

**Operational distinction:** git/GitHub remote read failures may be caused by reliance on a short-lived SSH reverse proxy, but the exact causal timing has not been verified. Before new ON: Codex investigates proxy ownership and lifetime and proposes persistent direct or managed outbound GitHub HTTPS; remote Claim and exact successful CI checks must remain fail-closed. Verify connectivity from the *actual launcher environment*, including after ending the monitor SSH session. CPU-only network preflight is allowed for preparation, not GPU evaluation.

**Code blocking issue:** historical entry `run_p2_full400.sh` mandates a new OFF before ON and uses P2-002 Claim/cycle; it must NOT be invoked. New ON-only source SHA and entry do not yet exist. Pending exact code review/PI pin and green CI; budget proposal max 1 GPU/200 ON Episodes.

**Final comparison:** old OFF root `/nvme2/user/qyy/SafeVLA_p2_002_results_20261014/OFF200` stays untouched. New ON root `/nvme2/user/qyy/SafeVLA_p2_003_on200_results_20261014` must be new. Source/task/weight/action/scientific-code parity and hash evidence required; label paired TaskSpecs but cross-cycle environments as an explicit limitation.

**Status:** PI_REVIEW / NOT_AUTHORIZED, no 003 Claim, GPU use zero.
