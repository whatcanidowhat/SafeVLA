# PI review notes

- Review all 16 eligibility confirmations and exact video/input hashes.
- Review source_audit.json: table fields, history/summary keys, raw console scan,
  original log scans and bounded historical inventory search.
- In historical_source_excerpts.json, distinguish transient worker_id/iter queue
  fields from persisted global tables. num_workers is an aggregate, not assignment.
- All counters and exposure labels remain unknown. Do not relabel unknown as unexposed.
- Empty paired outputs explicitly represent NOT_STARTED. B4 is unreconstructable,
  not a failed policy experiment or evidence for B3.
- A legitimate future source must establish worker assignment and within-worker
  predecessor order/counters. Global completion order alone is insufficient.
- Review or replace the design as PI; no automatic execution or PI acknowledgement.
