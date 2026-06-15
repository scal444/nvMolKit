# RDKit FMCS Measurement Breakdown: Pair 851

Source lines: A=10051291 B=9397592
Original 1k CSV timings: nvMolKit=13987.898 ms, RDKit=284.915 ms

## Pair

A: `C[C@@]12CCC[C@H]1N(C(=O)C1=CC3=C(C=N1)NC=C3)CCN(C(=O)C1=CN=C3CCCCN13)C2`

B: `CON1CCC(C(=O)N2CC(N3CCN(C(=O)C45CCCCC4CCC5)CC3)C2)CC1`

## Measurement Table

| label | order | wallTimeSeconds | elapsedTimeSeconds | numAtoms | numBonds | totalSteps | mcsFoundStep | stepsAfterMcsFound | mcsFoundStepFraction | inspectedSeeds | seedChecks | remainingSizeRejected | individualBondExcluded | matchCalls | matchFound | matchHitRate | fastMatchCalls | fastMatchFound | fastMatchHitRate | slowMatchCalls | slowMatchFound | slowMatchHitRate | exactMatchCalls | exactMatchFound | exactMatchHitRate | findHashInCache | hashKeyFoundInCache | hashCacheHitRate | dupCacheFound | dupCacheFoundMatch | dupCacheHitRate | hashCacheKeys | hashCacheEntries | duplicateCacheEntries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | A,B | 0.392782 | 0.386856 | 21 | 20 | 16277 | 287 | 15990 | 0.0176322 | 37709 | 29498 | 8236 | 990 | 14597 | 1176 | 0.0805645 | 9236 | 786 | 0.0851018 | 13811 | 390 | 0.0282384 | 13957 | 13957 | 1 | 28554 | 13957 | 0.488793 | 944 | 1 | 0.00105932 | 1176 | 1176 | 28508 |
| no_fast_cache | A,B | 0.263685 | 0.259586 | 21 | 20 | 16277 | 287 | 15990 | 0.0176322 | 37709 | 29498 | 8236 | 990 | 28554 | 15133 | 0.529978 | 28519 | 10984 | 0.385147 | 17570 | 4149 | 0.236141 | 0 | 0 |  | 0 | 0 |  | 944 | 1 | 0.00105932 | 0 | 0 | 28508 |
| no_duplicate_cache | A,B | 0.250025 | 0.24928 | 21 | 20 | 16277 | 287 | 15990 | 0.0176322 | 37709 | 29498 | 8236 | 990 | 15540 | 1176 | 0.0756757 | 9808 | 786 | 0.0801387 | 14754 | 390 | 0.0264335 | 13958 | 13958 | 1 | 29498 | 13958 | 0.473185 | 0 | 0 |  | 1176 | 1176 | 0 |
| no_caches | A,B | 0.232717 | 0.232685 | 21 | 20 | 16277 | 287 | 15990 | 0.0176322 | 37709 | 29498 | 8236 | 990 | 29498 | 15134 | 0.513052 | 29463 | 10984 | 0.372807 | 18514 | 4150 | 0.224155 | 0 | 0 |  | 0 | 0 |  | 0 | 0 |  | 0 | 0 | 0 |
| swapped_default | B,A | 0.279137 | 0.273817 | 21 | 20 | 16277 | 287 | 15990 | 0.0176322 | 37709 | 29498 | 8236 | 990 | 14597 | 1176 | 0.0805645 | 9236 | 786 | 0.0851018 | 13811 | 390 | 0.0282384 | 13957 | 13957 | 1 | 28554 | 13957 | 0.488793 | 944 | 1 | 0.00105932 | 1176 | 1176 | 28508 |

## Default Run SMARTS

```
[#6]-[#6]-[#6](-[#6]-[#7](-[#6]-[#6]-[#7](-[#6]-[#6]-,:[#7]-,:[#6]-,:[#6]-,:[#6](-,:[#6]-,:[#6])-,:[#6])-[#6]-[#6])-[#6])-[#6]
```

## Repeated Timing Medians

Medians exclude repeat 0 in the repeated run to avoid one-time process/setup effects.

| label | median elapsed seconds |
| --- | --- |
| default | 0.269570 |
| no_fast_cache | 0.253543 |
| no_duplicate_cache | 0.247536 |
| no_caches | 0.219689 |
