# PI review notes and blocking risks

1. P1 BLOCKER: executor snapshot restoration is not shape-aware across lazy initial cache allocation. See runner75/169 and frozen decoder280-284. Proposed repair only: restore complete tensor state including shape/dtype/device by replacing isolated cache attributes with cloned snapshots; test empty-to-active lifecycle on CPU and exact invariants before a fresh approved run. Not implemented after failure.
2. P1/P2 BLOCKER: all paired logger controls, reset critic invariants and live task/metric controls remain unverified. Failed static checks coverage must be expanded; syntax alone cannot validate state lifecycle.
3. Sentinel critic states are explicitly engineering fixtures; future passing invariant tests would not prove valid critic trajectory behavior. Do not present them as scientific carry histories.
4. Existing A/A design measures a five-task engineering prefix, not full200 performance. Independent stochastic differences require explanation. Current n=0 prevents any SR/Cost inference.
5. Budget planning lacks actual online peak memory/runtime; do not extrapolate the single offline step. Recommend retaining original ceilings only if independently reapproved, with offline gate first.

001C remains revoked by PI commit e93a9cdff70fe537cfa377c2d1f63c7b432d45be. This P1 run does not claim, reinstate or execute001C. Current unique claim is terminal after this handoff; unused episode budget cannot authorize retry. PI acknowledgement and any new approval must use protocol transitions; executor STOP after publication.
