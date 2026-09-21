# Runtime geometry coverage: partial / BLOCKED

| Measure | Observed |
| --- | --- |
| preflight_tasks | 12 |
| preflight_initializations | 24 |
| preflight_target_observations | 36 |
| preflight_target_comparisons | 18 |
| preflight_exact_equal | 18 |
| max_absolute_dimension_difference_m | 0 |
| max_relative_dimension_difference | 0 |
| full_initializations | 20 |
| full_exact_mapping | 26 |
| full_valid_aabb | 26 |
| expected_full_targets | 368 |
| unattempted_targets | 342 |
| full_complete_tasks | 20 |
| expected_tasks | 200 |
| unattempted_tasks | 180 |
| static_candidate_total | 42 |
| static_candidate_full_valid | 18 |
| full_obb_available | 26 |
| observed_mapping_or_geometry_failures | 0 |

All 26 attempted full targets map exactly and have valid AABB. The remaining 342 targets were not attempted after the output-channel stop; they are not observed missing/ambiguous geometry. Preflight repeat observations are separate from full-pass coverage. Static candidates are checked for runtime availability only; no static/runtime numeric equivalence is asserted. OBB is secondary only.

Source ID form describes identifier syntax, not validated asset-provider provenance.

| Synset | Source ID form | Full status | Count |
| --- | --- | --- | --- |
| alarm_clock.n.01 | hex32_asset_id | NOT_ATTEMPTED_AFTER_STOP | 1 |
| alarm_clock.n.01 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 20 |
| apple.n.01 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 13 |
| ashcan.n.01 | hex32_asset_id | NOT_ATTEMPTED_AFTER_STOP | 14 |
| ashcan.n.01 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 6 |
| atomizer.n.01 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 18 |
| basketball.n.02 | hex32_asset_id | NOT_ATTEMPTED_AFTER_STOP | 3 |
| basketball.n.02 | hex32_asset_id | VALID_CREATION_STATE_AABB | 6 |
| basketball.n.02 | other_asset_id | VALID_CREATION_STATE_AABB | 1 |
| bed.n.01 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 24 |
| bowl.n.03 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 23 |
| houseplant.n.01 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 34 |
| laptop.n.01 | hex32_asset_id | NOT_ATTEMPTED_AFTER_STOP | 2 |
| laptop.n.01 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 20 |
| mug.n.04 | hex32_asset_id | NOT_ATTEMPTED_AFTER_STOP | 2 |
| mug.n.04 | hex32_asset_id | VALID_CREATION_STATE_AABB | 12 |
| sofa.n.01 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 18 |
| straight_chair.n.01 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 81 |
| television_receiver.n.01 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 32 |
| toilet.n.02 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 15 |
| vase.n.01 | hex32_asset_id | NOT_ATTEMPTED_AFTER_STOP | 2 |
| vase.n.01 | other_asset_id | NOT_ATTEMPTED_AFTER_STOP | 14 |
| vase.n.01 | other_asset_id | VALID_CREATION_STATE_AABB | 7 |
