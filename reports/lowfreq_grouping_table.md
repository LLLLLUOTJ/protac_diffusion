# Low-Frequency Token Grouping Table

Date: 2026-03-11

## Summary

- core threshold: freq >= 10
- core tokens: 30
- low-frequency tokens: 73
- assigned high confidence: 5
- assigned medium confidence: 10
- uncertain: 58

## Core Tokens (Top 15 by frequency)

| Rank | Token | Frequency |
|---:|---|---:|
| 1 | `*C*` | 19151 |
| 2 | `*O*` | 3770 |
| 3 | `*C(*)=O` | 2389 |
| 4 | `*N*` | 1961 |
| 5 | `*c1cn(*)nn1` | 395 |
| 6 | `*N1CCN(*)CC1` | 372 |
| 7 | `*c1ccc(*)cc1` | 300 |
| 8 | `*C1CCN(*)CC1` | 298 |
| 9 | `*C#C*` | 198 |
| 10 | `*N(*)C` | 92 |
| 11 | `*C1CN(*)C1` | 86 |
| 12 | `*/C=N/*` | 45 |
| 13 | `*c1ccnc(*)c1` | 38 |
| 14 | `*C1(*)CC1` | 29 |
| 15 | `*C(*)C` | 27 |

## Low-Frequency -> Core (High/Medium Confidence)

| Low Token | Low Freq | Assigned Core | Confidence | Score |
|---|---:|---|---|---:|
| `*[C@H]1CC[C@H](*)CC1` | 4 | `*C1CCC(*)CC1` | high | 1.000 |
| `*[C@H]1CC[C@@H](*)CC1` | 1 | `*C1CCC(*)CC1` | high | 1.000 |
| `*C1CCN(*)C1` | 7 | `*C1CCN(*)CC1` | medium | 0.757 |
| `*[C@H]1CCN(*)C1` | 2 | `*C1CCN(*)CC1` | medium | 0.757 |
| `*[C@@H]1CCN(*)C1` | 1 | `*C1CCN(*)CC1` | medium | 0.757 |
| `*N1CCC2(CC1)CCN(*)C2` | 2 | `*N1CCC2(CC1)CCN(*)CC2` | high | 0.785 |
| `*N1CCC2(CC1)CN(*)C2` | 3 | `*N1CCC2(CC1)CCN(*)CC2` | medium | 0.749 |
| `*C1CCC2(CC1)CCN(*)CC2` | 7 | `*N1CCC2(CC1)CCN(*)CC2` | medium | 0.745 |
| `*N1CCC2(CCCN(*)C2)CC1` | 1 | `*N1CCC2(CC1)CCN(*)CC2` | medium | 0.731 |
| `*N1CCC2(C1)CN(*)C2` | 3 | `*N1CCC2(CC1)CCN(*)CC2` | medium | 0.650 |
| `*C1CC2(CCN(*)CC2)C1` | 5 | `*N1CCC2(CC1)CCN(*)CC2` | medium | 0.642 |
| `*N1CCCN(*)CC1` | 2 | `*N1CCN(*)CC1` | high | 0.801 |
| `*N1CCCN(*)CCC1` | 1 | `*N1CCN(*)CC1` | medium | 0.699 |
| `*C1CC(*)C1` | 3 | `*[C@H]1C[C@H](*)C1` | high | 1.000 |
| `*C1CC1*` | 1 | `*[C@H]1C[C@H](*)C1` | medium | 0.637 |

## Uncertain Tokens (Need Manual Review)

| Low Token | Low Freq | Suggested Core | Score | Reason |
|---|---:|---|---:|---|
| `*c1ccc(=O)n(*)c1` | 9 | `*c1ccc(*)cc1` | 0.378 | AMBIGUOUS_NEAREST_CORES |
| `*N(*)CCS` | 9 | `*N(*)C` | 0.314 | LOW_SIMILARITY_SCORE |
| `*/C=C/*` | 8 | `*/C=N/*` | 0.393 | AMBIGUOUS_NEAREST_CORES |
| `*C(*)(F)F` | 7 | `*C(*)(C)C` | 0.494 | LOW_SIMILARITY_SCORE |
| `*c1cnc(*)cn1` | 6 | `*c1ccc(*)nc1` | 0.637 | AMBIGUOUS_NEAREST_CORES |
| `*N1CC2(C1)CN(*)C2` | 6 | `*N1CCN(*)CC1` | 0.498 | AMBIGUOUS_NEAREST_CORES |
| `*C1CC2(C1)CN(*)C2` | 6 | `*C1CCN(*)CC1` | 0.491 | AMBIGUOUS_NEAREST_CORES |
| `*C(*)N1CCOCC1` | 5 | `*N1CCN(*)CC1` | 0.314 | AMBIGUOUS_NEAREST_CORES |
| `*N1CCC(*)(F)CC1` | 4 | `*N1CCN(*)CC1` | 0.531 | AMBIGUOUS_NEAREST_CORES |
| `*C1CCC2(CC1)CN(*)C2` | 4 | `*C1CCN(*)CC1` | 0.497 | AMBIGUOUS_NEAREST_CORES |
| `*c1cc(*)cc(N)c1` | 3 | `*c1cccc(*)c1` | 0.521 | LOW_SIMILARITY_SCORE |
| `*N1CCC2(CC1)CCN(*)C2=O` | 3 | `*N1CCC2(CC1)CCN(*)CC2` | 0.444 | LOW_SIMILARITY_SCORE |
| `*C(*)C(=O)NC` | 3 | `*C(*)=O` | 0.413 | LOW_SIMILARITY_SCORE |
| `*C(*)=S` | 3 | `*S*` | 0.357 | AMBIGUOUS_NEAREST_CORES |
| `*C1CCCN(*)C1` | 2 | `*C1CCN(*)CC1` | 0.647 | AMBIGUOUS_NEAREST_CORES |
| `*[C@@H]1CCCC[C@@H]1*` | 2 | `*C1CCC(*)CC1` | 0.545 | LOW_SIMILARITY_SCORE |
| `*c1cnnn1*` | 2 | `*c1cn(*)nn1` | 0.542 | AMBIGUOUS_NEAREST_CORES |
| `*C1CC2(CCN(*)C2)C1` | 2 | `*N1CCC2(CC1)CCN(*)CC2` | 0.528 | LOW_SIMILARITY_SCORE |
| `*N1CCOC2(CC1)CCN(*)CC2` | 2 | `*N1CCC2(CC1)CCN(*)CC2` | 0.528 | LOW_SIMILARITY_SCORE |
| `*N(*)C(C)=O` | 2 | `*C(*)=O` | 0.491 | LOW_SIMILARITY_SCORE |
| `*N1CCC(*)(O)CC1` | 2 | `*N1CCN(*)CC1` | 0.466 | AMBIGUOUS_NEAREST_CORES |
| `*N1CCC(*)(OC)CC1` | 2 | `*N1CCC2(CC1)CCN(*)CC2` | 0.430 | AMBIGUOUS_NEAREST_CORES |
| `*c1cc(F)c(*)c(F)c1` | 2 | `*c1cccc(*)c1` | 0.410 | AMBIGUOUS_NEAREST_CORES |
| `*C=C*` | 2 | `*/C=N/*` | 0.393 | AMBIGUOUS_NEAREST_CORES |
| `*c1c(C)nn(*)c1C` | 2 | `*c1cnn(*)c1` | 0.390 | AMBIGUOUS_NEAREST_CORES |
| `*c1ccc(*)nn1` | 1 | `*c1ccc(*)nc1` | 0.580 | LOW_SIMILARITY_SCORE |
| `*C1CCCCN1*` | 1 | `*C1CCN(*)CC1` | 0.570 | LOW_SIMILARITY_SCORE |
| `*N1CCC2(CC1)CCN(*)CO2` | 1 | `*N1CCC2(CC1)CCN(*)CC2` | 0.564 | LOW_SIMILARITY_SCORE |
| `*c1cccc(*)n1` | 1 | `*c1cccc(*)c1` | 0.557 | AMBIGUOUS_NEAREST_CORES |
| `*N1CCN(*)[C@@H](C)C1` | 1 | `*N1CCN(*)CC1` | 0.542 | LOW_SIMILARITY_SCORE |
| `*N1CCN(*)[C@H](C)C1` | 1 | `*N1CCN(*)CC1` | 0.542 | LOW_SIMILARITY_SCORE |
| `*c1ccn(*)n1` | 1 | `*c1cnn(*)c1` | 0.542 | AMBIGUOUS_NEAREST_CORES |
| `*N(*)CC` | 1 | `*N(*)C` | 0.539 | LOW_SIMILARITY_SCORE |
| `*N1CCC(*)(C)CC1` | 1 | `*N1CCN(*)CC1` | 0.537 | AMBIGUOUS_NEAREST_CORES |
| `*c1ccc(*)c(F)c1` | 1 | `*c1ccc(*)cc1` | 0.527 | AMBIGUOUS_NEAREST_CORES |
| `*c1cc(*)cc(O)c1` | 1 | `*c1cccc(*)c1` | 0.521 | LOW_SIMILARITY_SCORE |
| `*[C@@H]1CNC[C@H](*)C1` | 1 | `*[C@H]1C[C@H](*)C1` | 0.506 | AMBIGUOUS_NEAREST_CORES |
| `*C1CN(*)CCC1O` | 1 | `*C1CCN(*)CC1` | 0.501 | LOW_SIMILARITY_SCORE |
| `*C(*)N` | 1 | `*C(*)C` | 0.477 | LOW_SIMILARITY_SCORE |
| `*C1CN(*)CCO1` | 1 | `*C1CCN(*)CC1` | 0.467 | AMBIGUOUS_NEAREST_CORES |
| `*N1CC2CC(C1)CN(*)C2` | 1 | `*C1CN(*)C1` | 0.466 | LOW_SIMILARITY_SCORE |
| `*c1csc(*)n1` | 1 | `*c1ccnc(*)c1` | 0.435 | AMBIGUOUS_NEAREST_CORES |
| `*N1C[C@H]2C[C@@H]1CN2*` | 1 | `*N1CCN(*)CC1` | 0.421 | AMBIGUOUS_NEAREST_CORES |
| `*N1CC2CC1CN2*` | 1 | `*N1CCN(*)CC1` | 0.410 | AMBIGUOUS_NEAREST_CORES |
| `*N1C[C@@H]2C[C@H]1CN2*` | 1 | `*N1CCN(*)CC1` | 0.410 | AMBIGUOUS_NEAREST_CORES |
| `*C(*)C#N` | 1 | `*C#C*` | 0.401 | LOW_SIMILARITY_SCORE |
| `*N1CC(*)(C#N)C1` | 1 | `*N1CCN(*)CC1` | 0.385 | LOW_SIMILARITY_SCORE |
| `*N1CC(*)(CCN2CCN(C)CC2)C1` | 1 | `*N1CCC2(CC1)CCN(*)CC2` | 0.372 | LOW_SIMILARITY_SCORE |
| `*N(*)CC(=O)NC` | 1 | `*C(*)=O` | 0.347 | LOW_SIMILARITY_SCORE |
| `*c1ccc2nc(*)sc2c1` | 1 | `*c1ccc(*)nc1` | 0.323 | AMBIGUOUS_NEAREST_CORES |
| `*C(*)COCC` | 1 | `*C(*)C` | 0.313 | AMBIGUOUS_NEAREST_CORES |
| `*C(*)NC(=O)OC(C)(C)C` | 1 | `*C(*)=O` | 0.288 | LOW_SIMILARITY_SCORE |
| `*C1=NC(*)SC1` | 1 | `*C1CN(*)C1` | 0.256 | AMBIGUOUS_NEAREST_CORES |
| `*C1=NNC=C2C3CC(*)C(C3)C21` | 1 | `*N1CCC2(CC1)CCN(*)CC2` | 0.253 | LOW_SIMILARITY_SCORE |
| `*C1=NNC=C2C3CC(CC3*)C21` | 1 | `*N1CCC2(CC1)CCN(*)CC2` | 0.253 | LOW_SIMILARITY_SCORE |
| `*c1ccc2c(c1)CCc1cc(*)ccc1/N=N\2` | 1 | `*c1cccc(*)c1` | 0.248 | AMBIGUOUS_NEAREST_CORES |
| `*C(*)COCCOCC` | 1 | `*C(*)C` | 0.218 | AMBIGUOUS_NEAREST_CORES |
| `*C(*)CCCCNC(=O)c1ccc2c(c1)C(=O)OC21c2ccc(O)cc2Oc2cc(O)ccc21` | 1 | `*c1ccccc1*` | 0.056 | AMBIGUOUS_NEAREST_CORES |

## Files

- `/Users/lintianjian/diffusion/data/processed/core_token_table.csv`
- `/Users/lintianjian/diffusion/data/processed/lowfreq_to_core_assignment.csv`
- `/Users/lintianjian/diffusion/data/processed/lowfreq_uncertain_tokens.csv`
- `/Users/lintianjian/diffusion/data/processed/lowfreq_grouped_by_core.json`
- `/Users/lintianjian/diffusion/data/processed/lowfreq_grouping_summary.json`