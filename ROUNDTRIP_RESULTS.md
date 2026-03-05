# MolGraph Roundtrip Results

Date: 2026-03-05  
Dataset: `/Users/lintianjian/diffusion/data/sdf_dataset/AAK1.sdf`

## Run Scope

- Input files scanned: `1`
- Total molecule records: `22`
- Parsed valid molecules: `22`
- Invalid input records: `0`

## Encode/Decode Outcome

- Decode returned `None`: `0`
- Canonical SMILES match (isomeric default): `15 / 22`
- Canonical SMILES mismatch (isomeric default): `7 / 22`

## Additional Fidelity Check

Compared with `isomericSmiles=False` (ignore stereochemistry):

- Match: `20 / 22`
- Mismatch: `2 / 22`

## Observations

1. Core graph reconstruction is stable (`decode_none = 0`).
2. Most mismatches come from stereochemistry not yet encoded/decoded.
3. Remaining non-stereo mismatches are mainly aromatic `[nH]` handling cases.

## Suggested Next Improvements

1. Add chirality features (`Atom ChiralTag`, bond stereo) to improve isomeric roundtrip.
2. Use encoded hydrogen-related information during decode (especially aromatic `[nH]` cases).
