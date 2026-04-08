# DeLinker Baseline Archive (2026-04-08)

## Summary
This archive records a bounded attempt to use the official pretrained DeLinker model as an external baseline for the current PROTAC linker task.

Outcome:
- The official pretrained DeLinker pipeline was made runnable on the server.
- It can generate chemically valid full molecules from a subset of our source conditions.
- It is not a good fit for the current PROTAC benchmark in its official pretrained form, mainly because fragment preservation is poor and source coverage is limited.

The experiment is therefore archived rather than promoted into the active baseline pipeline.

## Server Setup
- Server repo clone: `/home/yanglh/L/DeLinker`
- Server project repo: `/home/yanglh/L/protac_diffusion`
- DeLinker env used: `delinker`
- Main local sync directory for results:
  - [`outputs/server_sync/2026-04-07`](/Users/lintianjian/diffusion/outputs/server_sync/2026-04-07)

## What Was Run
### Input construction
Source conditions came from the current token model evaluation pool:
- source file on server: `/home/yanglh/L/protac_diffusion/outputs/linker_token_eval/all_generations.json`

Helper used to build DeLinker test inputs:
- [build_delinker_test_inputs.py](/Users/lintianjian/diffusion/archive/delinker_baseline_2026_04_08/build_delinker_test_inputs.py)

This assembled full source PROTACs from:
- `source_left_fragment_smiles`
- `source_anchored_linker_smiles`
- `source_right_fragment_smiles`

and then built:
- `pairs.smi`
- `reference_full.sdf`

### DeLinker mode
The run used the official pretrained same-length mode.
That means DeLinker had access to source geometry and reference linker size through the standard preprocessing route.

This is useful as a bounded external baseline, but it is not perfectly apples-to-apples with the token diffusion model.

## Coverage Through the Pipeline
Starting pool:
- 32 unique source conditions

After DeLinker geometry preprocessing:
- 15 usable examples

After filtering for official pretrained `zinc` bucket limits:
- 14 usable examples

Coverage summary:
- geometry-usable coverage: `15 / 32 = 46.88%`
- pretrained-runnable coverage: `14 / 32 = 43.75%`

One example exceeded the pretrained graph size ceiling:
- max graph-out index `89`
- official `zinc` pretrained bucket maximum `84`

## Generated Output
Same-length pretrained generation ran successfully on the filtered subset:
- 14 source conditions
- 4 generations per source
- total generated full molecules: `56`

Server output file:
- `/home/yanglh/L/DeLinker/protac_compare/delinker_32x4_same_length_bucket84.smi`

Synced local converted file:
- [delinker_32x4_same_length_bucket84.json](/Users/lintianjian/diffusion/outputs/server_sync/2026-04-07/delinker_32x4_same_length_bucket84.json)

Conversion helper used:
- [convert_delinker_sampling_to_generated_json.py](/Users/lintianjian/diffusion/archive/delinker_baseline_2026_04_08/convert_delinker_sampling_to_generated_json.py)

## Key Results
### Full-molecule metrics
File:
- [delinker_full_eval.json](/Users/lintianjian/diffusion/outputs/server_sync/2026-04-07/delinker_full_eval.json)

Results:
- total generated: `56`
- validity: `1.0000`
- uniqueness: `1.0000`
- novelty: `1.0000`
- mean QED: `0.0716`
- mean SA: `5.3184`

Interpretation:
- DeLinker produced chemically valid full molecules.
- As full PROTAC-like molecules, these outputs were weak by simple property metrics.

### Task-relevant linker extraction / fragment preservation
File:
- [delinker_32x4_same_length_bucket84.json](/Users/lintianjian/diffusion/outputs/server_sync/2026-04-07/delinker_32x4_same_length_bucket84.json)
- [delinker_linker_eval_extracted.json](/Users/lintianjian/diffusion/outputs/server_sync/2026-04-07/delinker_linker_eval_extracted.json)

Out of 56 generated full molecules:
- successfully re-extracted anchored linkers: `4`
- extractable linker rate: `4 / 56 = 0.0714`

Failure reasons:
- `linker_extraction_failed`: `19`
- `right_fragment_no_match`: `17`
- `left_fragment_no_match`: `16`
- `OK`: `4`

Interpretation:
- The main failure mode is not molecule validity.
- The main failure mode is that the generated molecule often no longer preserves the intended left/right fragment identity cleanly enough to recover the linker under the task definition.

### Extracted-linker metrics on the 4 recoverable cases
File:
- [delinker_linker_eval_extracted.json](/Users/lintianjian/diffusion/outputs/server_sync/2026-04-07/delinker_linker_eval_extracted.json)

Results on recoverable linkers only:
- valid count: `4`
- validity: `0.0714` when measured against all 56 generations
- uniqueness: `1.0000`
- novelty: `1.0000`
- mean QED: `0.5454`
- mean SA: `4.5213`

Interpretation:
- The few outputs that preserved fragments well enough to recover a linker were not obviously bad.
- The issue is the very low preservation rate, not the quality of the tiny recoverable subset.

### Free-full feasibility on recoverable outputs
File:
- [delinker_feasibility_24.summary.json](/Users/lintianjian/diffusion/outputs/server_sync/2026-04-07/delinker_feasibility_24.summary.json)

Results on 4 recoverable outputs:
- pass: `2`
- borderline: `1`
- fail: `1`
- pass rate: `0.5000`

Interpretation:
- The small recoverable subset is not hopeless structurally.
- But the subset is too small to make DeLinker competitive as a baseline for the current task.

## Practical Conclusion
For the current PROTAC linker benchmark, the official pretrained DeLinker baseline should be treated as:
- a documented external lower-bound comparison,
- not an active primary baseline.

Why it is being archived:
- source coverage is low,
- fragment preservation is poor,
- linker recoverability is very low,
- official pretrained assumptions are mismatched to large PROTAC fragment conditions.

## Recommendation If Revisited Later
Only revisit DeLinker if one of these changes:
1. We train or fine-tune a DeLinker-style model directly on the PROTAC weak-anchor dataset.
2. We define a smaller-fragment benchmark that better matches the original DeLinker training regime.
3. We want a strictly historical external baseline for a paper appendix.

Without one of those changes, further work on the official pretrained setup is unlikely to be a good use of time.

## Archived Files
This archive keeps the experiment-specific helper code only:
- [build_delinker_test_inputs.py](/Users/lintianjian/diffusion/archive/delinker_baseline_2026_04_08/build_delinker_test_inputs.py)
- [convert_delinker_sampling_to_generated_json.py](/Users/lintianjian/diffusion/archive/delinker_baseline_2026_04_08/convert_delinker_sampling_to_generated_json.py)
- [test_convert_delinker_sampling_to_generated_json.py](/Users/lintianjian/diffusion/archive/delinker_baseline_2026_04_08/test_convert_delinker_sampling_to_generated_json.py)
