# PR-2 read-only status and CI waiting

Python 3.10+, existing jsonschema==4.25.1. No model/runtime dependencies. These helpers query
only whatcanidowhat/SafeVLA through https://api.github.com. Run from a reviewed checkout.
They never claim, dispatch/rerun CI, push, change state, or start experiments.

## Commands

```
python scripts/research_loop_status.py --remote-readonly
python scripts/research_loop_status.py --local --root /path/to/control-checkout
python scripts/research_loop_ci_wait.py --sha FULL_40_CHARACTER_SHA --workflow research-loop-validate.yml --event push --branch research-loop --timeout 600 --interval 15
```

For a PR, explicitly choose `--event pull_request --branch codex/your-branch` and its full head SHA.
The branch tip must equal that SHA throughout observation. Historical successful CI on an older
commit is deliberately rejected. `--workflow` accepts a numeric workflow ID or exact YAML filename,
not an ambiguous display name. Workflow ID/name/path and event are all verified on the run.
If a rerun is intended, pass `--attempt N`; without it only attempt 1 is accepted. A changed run ID
or attempt, multiple matching runs, or incomplete/paginated listings stop with UNKNOWN. A single
explicit attempt must be the current attempt; prior success cannot mask a failed/in-progress rerun.

Optional GH_TOKEN or GITHUB_TOKEN comes only from the caller's environment; it is never printed,
persisted or passed on a command line. A GitHub connector session does not automatically populate
these variables. Public read access may work anonymously subject to GitHub rate limits. Standard
HTTPS_PROXY is inherited. Default TLS CA verification remains enabled, and redirects are refused.
Errors never print response bodies, raw exception text, credentials or full environment/config.

## Output and exit contract

One final JSON line, under 2 KiB, with VERDICT, EVIDENCE, BLOCKER, NEXT_AUTHORIZED_ACTION.
EVIDENCE.CAN_EXECUTE is always false. CLI `--help` is usage text, not a query result.

| Verdict | Exit | Meaning |
| --- | ---: | --- |
| SUCCESS | 0 | Exact CI observation succeeded; no execution permission |
| OBSERVED | 0 | Current valid-looking approval observed; CI/Claim/runtime NOT checked |
| BLOCKED / UNKNOWN / REMOTE_UNVERIFIED | 2 | Non-executable state, incomplete/conflicting evidence, or local-only cache |
| FAILURE | 3 | CI failure/action_required/startup_failure |
| CANCELLED | 4 | CI cancelled or caller interrupted |
| TIMED_OUT | 5 | Wait deadline or CI timed_out |
| NETWORK_ERROR | 6 | HTTPS/DNS/TLS/proxy/transport failure or other HTTP error |
| NOT_FOUND | 7 | No visible run/file/ref; 404 can also conceal permission denial |
| RATE_LIMITED | 8 | HTTP 429, or 403 with quota/retry headers |
| PERMISSION_DENIED | 9 | HTTP 401/403 without rate-limit evidence |
| INVALID_INPUT | 64 | CLI argument parsing failed |

Neutral/skipped/unknown conclusions are UNKNOWN, never SUCCESS. A missing run returns NOT_FOUND
for CI waiting. Status transport/evidence errors return UNKNOWN and preserve the error category in
EVIDENCE.error_kind. A missing CI run returns NOT_FOUND
immediately; it does not poll indefinitely for future CI creation. On any HTTP/network/rate-limit
failure there is no automatic retry, credential fallback or CI rerun. Ambiguous 403 is conservatively
PERMISSION_DENIED. Only an existing, identified pending run is polled. Waiting is bounded by the
specified deadline; each request also has a 10-second cap. A temporary daemon transport thread bounds
even a stalled OS DNS/TLS operation. On timeout the CLI returns once and exits without waiting for
that thread or issuing another request. A library caller that stays alive may retain that single
read-only request thread until the OS unblocks it. No persistent background service is installed.

## Status scope and limitations

Remote status reads the branch ref, fetches LOOP_STATE and both NEXT_EXPERIMENT files at that immutable
SHA, verifies file Git Blob hashes, then checks the ref again. It uses the checked-in state schema and
checks role/design identity, approval timestamps, scope, required-output paths and Claim bindings.
`--expected-cycle` / `--expected-claim` pin caller expectations; mismatch is UNKNOWN/STALE_*.
BLOCKED/INVALID/ABORTED/review and existing running Claims remain non-executable. An expired approval
cannot yield OBSERVED. CODEX and OPERATOR are reported distinctly and never substituted.

This is a compact state/design observer, not the complete historical/terminal-artifact validator. It
does not verify archived output hashes, full provenance, approval/Claim CI, runtime identity, or
historical uniqueness. Its scope is explicit in EVIDENCE. Those existing mandatory gates still apply
separately; no claim/launch command may treat this exit code as authorization. Observations are valid
as of the final reads, not atomic locks against changes immediately afterward.

Local mode reads files from one committed HEAD using read-only git show, ignores uncommitted files,
does not fetch/update refs, and always reports REMOTE_UNVERIFIED for a coherent snapshot. Corrupt or
missing snapshots report UNKNOWN. Repeated invocations recheck current evidence; no authorization
cache or mutation is created. A CI timeout/failure does not revoke or mutate any actual state.

## CPU tests and measurements

```
python scripts/test_research_loop.py
python scripts/test_research_loop_readonly.py
python scripts/test_research_loop_readonly.py --benchmark
python scripts/validate_research_loop.py --history
```

Tests use synthetic fixtures and fake HTTP only. The benchmark reports five synthetic tasks with
virtual polling time and actual Python harness duration. The comparison models the old approach as
one LLM tool invocation per polling round versus one invocation of the new bounded helper. This is
an explicitly modeled mechanism, not a measured Codex session. Real LLM tool calls, input/output
tokens, LLM wait turns and real end-to-end latency are NOT_MEASURED. Characters are not token proxies.
Safety observations include CAN_EXECUTE=false on every outcome and no mutation transport capability.

PR-1 compatibility: its four-field output and conditional reading remain. Full scientific audit and
all original Stop Conditions remain unchanged. The only validator change is four exact whitelist
entries; existing 1 MiB/secret/symlink checks and full-history traversal remain intact. The workflow
adds offline tests without changing any existing validation step. PR-2 does not provide an Executor,
network launcher witness, evidence pack, or approval for P2-004 recovery.
