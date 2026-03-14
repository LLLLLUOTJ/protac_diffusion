# Oriented Token Dataset

Date: 2026-03-14

## Why This Exists

The original `token_smiles_list_json` keeps fragment identity, but drops left/right attachment orientation for asymmetric tokens.

That causes ambiguity when decoding predicted token sequences back into one anchored linker.

Examples of ambiguous motifs:

- `*c1cn(*)nn1`
- `*/C=N/*`
- other asymmetric ring or unsaturated motifs with two attachment sites

## What Was Added

- token/linker codec: [sampling/token_linker_codec.py](/Users/lintianjian/diffusion/sampling/token_linker_codec.py)
- dataset converter: [build_oriented_token_dataset.py](/Users/lintianjian/diffusion/build_oriented_token_dataset.py)
- tests:
  - [tests/test_token_linker_codec.py](/Users/lintianjian/diffusion/tests/test_token_linker_codec.py)
  - [tests/test_build_oriented_token_dataset.py](/Users/lintianjian/diffusion/tests/test_build_oriented_token_dataset.py)

The converter replaces `token_smiles_list_json` with oriented token templates derived from `token_smiles_with_maps_list_json`, while preserving the original base tokens in:

- `base_token_smiles_list_json`
- `oriented_token_smiles_list_json`

## Generated Files

- [data/processed/linker_anchor_tokenized_oriented_multi.csv](/Users/lintianjian/diffusion/data/processed/linker_anchor_tokenized_oriented_multi.csv)
- [data/processed/linker_anchor_tokenized_oriented_multi.summary.json](/Users/lintianjian/diffusion/data/processed/linker_anchor_tokenized_oriented_multi.summary.json)
- [data/processed/linker_anchor_tokenized_oriented_core10.csv](/Users/lintianjian/diffusion/data/processed/linker_anchor_tokenized_oriented_core10.csv)
- [data/processed/linker_anchor_tokenized_oriented_core10.summary.json](/Users/lintianjian/diffusion/data/processed/linker_anchor_tokenized_oriented_core10.summary.json)
- [data/processed/task/linker_anchor_tokenized_oriented_dropped_only.csv](/Users/lintianjian/diffusion/data/processed/task/linker_anchor_tokenized_oriented_dropped_only.csv)
- [data/processed/task/linker_anchor_tokenized_oriented_dropped_only.summary.json](/Users/lintianjian/diffusion/data/processed/task/linker_anchor_tokenized_oriented_dropped_only.summary.json)

## Real-Data Reconstruction Check

Reference dataset:

- [data/processed/linker_anchor_tokenized_multi.csv](/Users/lintianjian/diffusion/data/processed/linker_anchor_tokenized_multi.csv)

Roundtrip comparison over all `2721` rows:

- plain token reconstruction:
  - exact canonical match: `2357 / 2721`
  - rate: `86.62%`
- oriented token reconstruction:
  - exact canonical match: `2638 / 2721`
  - rate: `96.95%`
- oriented token reconstruction with `isomericSmiles=False`:
  - exact match: `2721 / 2721`
  - rate: `100%`

Interpretation:

- oriented tokens fix the main connectivity/interface-direction problem
- remaining mismatches are stereochemical slash-direction losses across stitched token boundaries, not linker backbone mismatches

## Training Implication

For the next downstream model version:

- keep left/right fragment conditioning unchanged
- replace linker graph targets with oriented token sequences or oriented token embeddings
- reuse the current embedding training script by switching the input CSV to an oriented-token file

Smoke training already works with the existing embedding trainer on:

- [data/processed/linker_anchor_tokenized_oriented_multi.csv](/Users/lintianjian/diffusion/data/processed/linker_anchor_tokenized_oriented_multi.csv)

## Suggested Default Input

If we continue with the token-embedding route, use:

- all samples: [data/processed/linker_anchor_tokenized_oriented_multi.csv](/Users/lintianjian/diffusion/data/processed/linker_anchor_tokenized_oriented_multi.csv)
- core subset: [data/processed/linker_anchor_tokenized_oriented_core10.csv](/Users/lintianjian/diffusion/data/processed/linker_anchor_tokenized_oriented_core10.csv)
