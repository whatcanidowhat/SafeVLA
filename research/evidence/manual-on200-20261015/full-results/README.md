# Full OFF200 / manual ON200 results (videos excluded)

Archived on 2026-11-12 from the two original result directories on the evaluation server. All 400 videos are excluded by explicit user request; screenshots and top views are included.

## Download and verify

Full archives: [GitHub Release](https://github.com/whatcanidowhat/SafeVLA/releases/tag/actor-reset-off200-on200-20261015-no-video). Download both `*-no-video.tar.gz` files plus `FILE_MANIFEST.json` and `SHA256SUMS`, then run `sha256sum -c SHA256SUMS` before extraction. The archives retain the original root directory names and relative paths.

- `SafeVLA_p2_002_results_20261014`: 1419 regular files, 108467148 source bytes; 200 videos excluded. Archive: [SafeVLA_p2_002_results_20261014-no-video.tar.gz](https://github.com/whatcanidowhat/SafeVLA/releases/download/actor-reset-off200-on200-20261015-no-video/SafeVLA_p2_002_results_20261014-no-video.tar.gz).
- `SafeVLA_manual_on200_20261015`: 1417 regular files, 104180654 source bytes; 200 videos excluded. Archive: [SafeVLA_manual_on200_20261015-no-video.tar.gz](https://github.com/whatcanidowhat/SafeVLA/releases/download/actor-reset-off200-on200-20261015-no-video/SafeVLA_manual_on200_20261015-no-video.tar.gz).

## Contents and verification

Includes episode metrics and task/run/artifact manifests, per-step actions and boundary logs, all recorded probability NPZ files, official evaluation screenshots/top views and offline tracking files, console logs, and the manual ON launcher. The `metadata/` directory contains exact source copies of the top-level run records for direct browsing. `FILE_MANIFEST.json` lists every included source file with SHA256 and every excluded video separately. Symlinks are preserved; external linked log contents are additionally retained under `_external_symlink_targets/` in each archive.

Every regular file in each archive was read back and checked against its source SHA256. Release asset sizes and SHA256 digests were verified after upload. This checks archival fidelity, not scientific validity of the original experiment or analyses.

## Historical status

This upload does not run a new experiment, amend research-loop state, or reclassify the manual ON run. See the [original evidence report](../REPORT.md) for historical execution status and limitations. The OFF directory's `ON200.console.log` is retained as an original file; it is not substituted for the manual ON run's `console.log`.
