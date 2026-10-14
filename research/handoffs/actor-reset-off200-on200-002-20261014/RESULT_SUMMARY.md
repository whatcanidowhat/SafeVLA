# P2-002 terminal BLOCKED: OFF200 complete, ON blocked before model load

OFF200 completed 200/200 tasks and passed the frozen integrity validator (19,270 decisions). Official success is 169/200, SR 84.5%; Safety Cost total 177, mean 0.885. Five component totals: danger 2, corner 87, blind 56, fragile 13, critical 19. These are OFF-only results, not a completed intervention comparison.

The same one-shot launcher attempted ON200 after OFF validation at 2026-10-14T11:36:48Z. ON stopped inside evaluate.py:79 at the remote Claim check: git ls-remote timed out after 60 seconds, with Proxy CONNECT timeout. execute():116 invokes this guard before model/CUDA/simulator initialization. ON result root was never created; ON Episodes=0. Launcher exited 1 at 2026-10-14T11:38:18.881016+00:00. No retry, second claim, alternate GPU, parameter edit or continuation was executed.

Approval 7fc797645d00fc38ab4ac8e08f4b8ed98053449f; unique claim 918170d26cca2fc385fb6f8fd87d0109b4a167b2, claim ID b89ac28dffc3490c9672afc9ef1afd15; exact execution code 06cde9adeb91d47c062a11e72e90c7edf56fb398. Approval and claim CI both SUCCESS before launch. One physical GPU 0; 200 total Episode starts; 19,270 decisions; 0 offline replay. Conservative start-to-exit wall-time upper bound 4.1600 hours. Source tree remains clean at frozen SHA; source hashes and all 1203 OFF indexed artifacts rechecked unchanged after exit.

The original monitoring SSH connection reported Connection reset; the server launcher continued and completed OFF. The later ON GitHub/HTTP proxy failure is directly evidenced. A lost session-dependent reverse tunnel is consistent with the traceback, but exact proxy/tunnel uptime was not recorded; this causal link is an inference, not a measured certainty.

Both GPU Logger equivalence and its unresolved noninterference limitation remain NOT_RUN_PI_WAIVED. Stored-log PASS does not establish GPU equivalence. ON/OFF SR, Cost, logits, early-END and oscillation differences are unavailable; H-RESET is not tested by this incomplete pair.

All required review files accompany this handoff. Full raw tensors/logs stay at /nvme2/user/qyy/SafeVLA_p2_002_results_20261014, hashed in SERVER_RAW_INVENTORY.json and OFF200_RAW_HASHES.json; compact actual metrics, validation, source/runtime manifest and complete console evidence are Git-readable. The original P2_LAUNCH_CONTEXT still says RUNNING because the launcher never updates it; preserved verbatim and superseded for outcome by EXIT.json and this terminal manifest.

Next actor PI. Review the valid OFF-only evidence and transport stop; any continuation/recovery design, code change, reuse of OFF as a control, or fresh execution requires an explicit separate PI decision and protocol authorization. No successor experiment is self-approved. STOP.
