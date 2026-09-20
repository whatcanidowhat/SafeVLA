# GitHub PI–Codex Handoff Protocol

Status: IMPLEMENTED CONTROL PROTOCOL; initial cycle is NOT_AUTHORIZED.
Canonical repository: https://github.com/whatcanidowhat/SafeVLA
Control branch: research-loop
Server control worktree: /nvme2/user/qyy/SafeVLA_loop_control
This orphan branch contains control data and validation tools only. Never run SafeVLA here.

## Authority and reading order

Read AGENTS.md, this protocol, LOOP_STATE.json, NEXT_EXPERIMENT.md and NEXT_EXPERIMENT.json,
then current evidence. The JSON design is machine-readable; the Markdown is the PI-facing design.
GitHub control branch HEAD is the shared instruction/state source; server files/hashes remain runtime evidence.
001A is already COMPLETED / provenance PASS / awaiting PI review. Never repeat it from a stale handoff.
001B remains NOT_AUTHORIZED. Bootstrap authorization permits control commits/push only.

Only PI may approve a task. Only one experiment is active in a cycle. Executor may propose, never self-approve.
updated_by and authorization.approved_by describe the protocol role; they are not cryptographic actor authentication.
GitHub must authenticate the real writer using their existing connection. Sharing one GitHub account does not
technically separate PI from Executor. Branch protection/restricted writer identities are a deployment hardening
boundary, not proof supplied by a JSON label. Do not impersonate a PI acknowledgement.

## State transitions

| From | To | Writer | next_actor |
| --- | --- | --- | --- |
| PI_REVIEW | APPROVED_FOR_CODEX | PI | CODEX |
| APPROVED_FOR_CODEX | CODEX_RUNNING | CODEX | CODEX |
| CODEX_RUNNING | AWAITING_PI_REVIEW | CODEX | PI |
| AWAITING_PI_REVIEW | PI_REVIEW | PI | PI |
| CODEX_RUNNING | BLOCKED / INVALID / ABORTED | CODEX | PI |
| BLOCKED / INVALID / ABORTED | PI_REVIEW | PI | PI |

Draft edits may remain PI_REVIEW and are PI-owned. A same-state edit increments state_version if state JSON changes.
Unchanged state allows documentation maintenance; approved design content cannot change.
Exception states remain non-executable and cannot transition directly to APPROVED_FOR_CODEX. PI may acknowledge a
complete BLOCKED/INVALID/ABORTED handoff back to PI_REVIEW only after independently reading the result commit and
required artifacts. That acknowledgement must set reviewed_result_commit to the immediate result HEAD, clear active
authorization metadata, retain instruction_commit/claim_id as historical provenance, and return both design files to
DRAFT. A subsequent experiment must use a fresh unused cycle_id; no automatic retry or implicit approval is allowed.

state_version increments exactly once per state mutation. Retain cycle/experiment, approval, budgets,
execution_worktree and required_outputs during execution. Prior cycle IDs cannot be reused for another claim.
Control history must be linear; no force push. CI validates every commit, so bundling an invalid transition
between valid endpoints is also rejected.

## PI staged design commits (CONTROL-PROTOCOL-FIX-001)

A PI may publish NEXT_EXPERIMENT.json and NEXT_EXPERIMENT.md in separate commits.
Only when ALL five guards hold is staging permitted:
LOOP_STATE.status=PI_REVIEW, next_actor=PI, authorization.status=NOT_AUTHORIZED,
instruction_commit=null, and claim_id=null.
The Markdown/JSON status pairs may be DRAFT/DRAFT, APPROVED/DRAFT,
DRAFT/APPROVED, or APPROVED/APPROVED. None grants execution authority.
The final PI authorization commit must update LOOP_STATE separately or together
with the designs; once LOOP_STATE is not PI_REVIEW, both designs must be
APPROVED and authorization.status must be APPROVED. Existing complete-design,
identity, transition, expiry and claim checks remain in force.
A PI_REVIEW state retaining a previous instruction/claim does not qualify for
this staging exception and retains the existing DRAFT/DRAFT requirement.
All historical commits are validated with these rules, without skipping,
rewriting or squashing staging commits. No state transition is added.

## PI approval A

PI reads the exact current research-loop commit and evidence packet using the PI-side GitHub connection.
For a draft, PI completes the design, sets NEXT_EXPERIMENT.json status=APPROVED and updates the Markdown,
sets LOOP_STATE.status=APPROVED_FOR_CODEX, next_actor=CODEX, authorization.status=APPROVED,
approved_by=PI, approval/expiration timestamps, explicit scope/budget, updated_by=PI, and increments state_version.
Publish only against the reviewed HEAD, using a non-force update. Resolve stale drafts by re-reading, not overwriting.
The resulting approval commit is A. instruction_commit and claim_id are null in A; no self-referential SHA.
The bootstrap leaves this step unperformed.

## Executor claim B

Wait for approval A's control CI to pass. In a clean control checkout on research-loop:

    python scripts/research_loop_claim.py

Use Python 3.10+ with jsonschema==4.25.1. On this server the existing interpreter
/home/amax/.conda/envs/safevla/bin/python has that control dependency; this tool imports no model code.
The command never executes a design command, model, simulator or episode.

The tool fetches origin/research-loop, requires local HEAD == fetched HEAD, checks approved state,
next_actor, design identity, expiry and history, then creates a claim under an exclusive local lock.
Claim commit B has instruction_commit=A (the full pre-claim remote HEAD), a unique claim_id,
status=CODEX_RUNNING, next_actor=CODEX, updated_by=CODEX, incremented state_version.
Only LOOP_STATE is staged by the claim command.
A normal atomic push publishes B. Competing sibling claims cannot both fast-forward the remote.
If push fails or its result is uncertain: fetch, retain the local claim, report STALE_CLAIM or transport failure,
STOP without retry, reset, force, or execution. A timeout is not permission to execute.

After successful push, verify remote HEAD is B and B's CI succeeded. Only then execute exactly the approved
task once in the approved execution_worktree; no work begins merely because a local claim commit exists.
A claim interrupted after publication does not auto-resume. Hand the uncertain state to PI.
For CONTROL_ONLY_NOOP, execution_worktree=null, GPU/episode budgets are exactly zero;
the future dry run has no SafeVLA execution tree.

## Result C

Executor preserves approved design and instruction_commit=A, and writes:

    research/handoffs/<cycle_id>/RESULT_SUMMARY.md
    research/handoffs/<cycle_id>/RUN_MANIFEST.json
    research/handoffs/<cycle_id>/ARTIFACT_INDEX.json
    research/handoffs/<cycle_id>/REVIEW_NOTES.md

RUN_MANIFEST binds cycle_id, experiment_id, instruction_commit, claim_id, status, actual command list,
gpu_count and episodes_started; add timestamps, exit state and runtime provenance when applicable.
Required outputs are also mandatory for BLOCKED/INVALID/ABORTED, with explicit missing/not-started fields.
ARTIFACT_INDEX entries contain name, server_path, sha256, size and required_for_PI_review.
Small Git-shared artifacts have git_path and hash/size verified by CI.
Every PI-required artifact must have a Git-readable copy; server-only paths are not PI-readable proof.
Large outputs stay on the server and are marked optional for this review, with hashes and access boundaries.
Never upload checkpoints, tensors, weights, video or raw large logs. Do not mark unavailable evidence as reviewed.

Before publishing: fetch, ensure no unexpected remote advancement; validate working files against HEAD,
scan the complete staged/committed tree and outgoing history, commit only the whitelist, then push without force.
Set status=AWAITING_PI_REVIEW, next_actor=PI, updated_by=CODEX, increment state_version.
The resulting commit is C; never embed C's own SHA inside C. After push, STOP.
Failure to publish leaves local evidence, not a claimed successful handoff.

## PI acknowledgement D

PI independently reads result commit C (including BLOCKED/INVALID/ABORTED handoffs) and its required artifacts through GitHub.
PI updates status=PI_REVIEW, next_actor=PI, reviewed_result_commit=C, updated_by=PI,
authorization.status=NOT_AUTHORIZED and clears approved_by/approved_at_utc/expires_at_utc; retain instruction_commit
and claim_id as historical provenance. The design returns to DRAFT. Increment state_version and publish review commit D.
The validator checks reviewed_result_commit against D's immediate result parent.
If intervening result corrections exist, PI reads and acknowledges their latest result HEAD.
A new experiment requires a fresh unused cycle_id and explicit subsequent PI approval, never approval by Executor.

## Validation and transport boundaries

scripts/validate_research_loop.py checks schema, design agreement, states/roles, transitions, commit bindings,
frozen execution identity, cycle reuse, required outputs, artifact hashes, the control file whitelist,
file size and known credential patterns. scripts/test_research_loop.py uses temporary Git repositories and
synthetic EXP-TEST tasks, including competing claims. It does not run EXP-LOOP-HANDOFF-001.
GitHub workflow only runs these control checks, has contents:read and no experiment/SSH dispatch credentials.

CI is a validator, not a PI agent, a task dispatcher, or a persistent server watcher.
PI/Executor use their existing GitHub connections and task entrypoints. No always-on process or automatic
experiment runner is installed by this bootstrap. The future dry run must verify the actual two-sided
read/write path and a real PI receipt. Executor-side API readability alone does not prove PI has read anything.

## Secrets and whitelist

Allowed: AGENTS.md, .codex/skills/safevla-research-loop/SKILL.md, research/, the three control Python tools,
.github/workflows/research-loop-validate.yml and a control .gitignore.
The control scripts/workflow are the explicitly authorized extensions to the earlier three-directory proposal.
No runtime sources or development dirty changes are copied. No symlinks/submodules. Maximum tracked file size 1 MiB.

Never print authenticated remote URLs, dump env/config, or embed credentials in files/command reports.
Report sanitized identity only: https://github.com/whatcanidowhat/SafeVLA.git.
Existing credentials confined to local .git/config must not be copied into the control branch.
Known secret patterns or oversized/disallowed artifacts in tracked/outgoing content trigger SECURITY_BLOCKED
or OVERSIZED_ARTIFACT before publication. Pattern scanning reduces exposure; it is not a proof that arbitrary
secrets are detectable. Inspect the exact whitelist and outgoing diff as well.


## Verified server Git transport (bootstrap environment)

Direct github.com Git transport timed out on this server. An ephemeral SSH reverse tunnel to the user's
existing local HTTP proxy was tested successfully with git ls-remote. No credential or global Git config is changed.
For a later authorized claim session, the local Windows caller can establish the same tunnel and run the
claim tool with HTTPS_PROXY=http://127.0.0.1:17896 (and HTTP_PROXY at the same endpoint) in that SSH session.
The actual tunnel parameters are -R 127.0.0.1:17896:127.0.0.1:7896 with ExitOnForwardFailure=yes.
Local proxy port 7896 must still be available; verify connectivity before any mutation. The tunnel is session-bound,
not a service installed by bootstrap. The claim tool inherits proxy environment without recording full env or secrets.
No proxy URL containing authentication may be persisted. A failed tunnel/fetch/push stops the claim; never bypass
publication acknowledgement or claim CI. This transport setup alone does not approve or execute a task.
