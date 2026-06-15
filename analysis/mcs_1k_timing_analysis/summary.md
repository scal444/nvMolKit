# fMCS 1k Pair Timing Analysis

## Dataset

- pairs: 1000
- nvMolKit timing (ms): min=1.325, median=42.060, p95=320.006, p99=744.891, max=13987.898, mean=101.999
- RDKit timing (ms): min=0.175, median=1.985, p95=10.362, p99=23.748, max=284.915, mean=3.533
- Spearman nvMolKit vs RDKit time: 0.929
- Spearman log10(nvMolKit) vs log10(RDKit): 0.929

## Strongest Descriptor Correlations With log10(nvMolKit time)

| feature | log_nvmolkit_time_ms_spearman |
| --- | --- |
| mcs_bonds | 0.734223 |
| mcs_atoms | 0.730833 |
| pair_min_carbon_atoms | 0.595025 |
| pair_avg_carbon_atoms | 0.523953 |
| pair_min_bonds | 0.519182 |
| pair_avg_bonds | 0.513258 |
| mcs_atom_fraction_small | 0.508889 |
| pair_min_ring_bonds | 0.498344 |
| mcs_bond_fraction_small | 0.490925 |
| pair_min_ring_atoms | 0.48749 |
| pair_avg_atoms | 0.473756 |
| pair_avg_ring_bonds | 0.471048 |

## Strongest Descriptor Correlations With log10(nvMolKit/RDKit ratio)

| feature | log_time_ratio_nv_over_rdkit_spearman |
| --- | --- |
| mcs_bonds | 0.628265 |
| mcs_atoms | 0.626305 |
| mcs_atom_fraction_small | 0.529462 |
| mcs_bond_fraction_small | 0.522852 |
| pair_avg_carbon_atoms | 0.365404 |
| pair_avg_bonds | 0.363004 |
| pair_avg_ring_bonds | 0.357243 |
| pair_avg_ring_atoms | 0.352278 |
| pair_avg_rings | 0.349567 |
| pair_max_bonds | 0.340729 |
| pair_avg_atoms | 0.327984 |
| pair_max_atoms | 0.314504 |

## Largest Descriptor Differences In The Slowest 5% nvMolKit Pairs

| feature | slow_top5_mean | rest_mean | difference | ratio |
| --- | --- | --- | --- | --- |
| mcs_bonds | 17.42 | 11.6884 | 5.73158 | 1.49036 |
| mcs_atoms | 18.2 | 12.5905 | 5.60947 | 1.44553 |
| pair_min_bonds | 29.5 | 25.6389 | 3.86105 | 1.15059 |
| pair_avg_ring_atoms | 18.48 | 14.6368 | 3.84316 | 1.26257 |
| pair_avg_bonds | 31.28 | 27.9274 | 3.35263 | 1.12005 |
| pair_max_ring_atoms | 20.68 | 17.3958 | 3.28421 | 1.18879 |
| pair_min_atoms | 26.98 | 24.0937 | 2.88632 | 1.1198 |
| pair_max_bonds | 33.06 | 30.2158 | 2.84421 | 1.09413 |
| pair_avg_atoms | 28.52 | 25.9995 | 2.52053 | 1.09695 |
| pair_max_atoms | 30.06 | 27.9053 | 2.15474 | 1.07722 |
| pair_max_fused_ring_atoms | 2.8 | 1.41158 | 1.38842 | 1.98359 |
| pair_avg_fused_ring_atoms | 1.9 | 0.810526 | 1.08947 | 2.34416 |

## Ten Slowest nvMolKit Pairs

| pair_index | nvmolkit_time_ms | rdkit_time_ms | input_line_a | input_line_b | pair_min_atoms | pair_avg_atoms | pair_max_atoms | pair_max_rings | pair_max_ring_atoms | pair_max_longest_acyclic_carbon_chain | mcs_atoms | mcs_bonds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 851 | 13987.9 | 284.915 | 1.00513e+07 | 9.39759e+06 | 31 | 32 | 33 | 6 | 28 | 1 | 21 | 20 |
| 835 | 1904.85 | 30.657 | 6.00284e+06 | 9.82725e+06 | 27 | 29.5 | 32 | 6 | 26 | 4 | 20 | 20 |
| 754 | 1497.3 | 23.7417 | 1.01629e+07 | 9.70036e+06 | 32 | 33 | 34 | 3 | 18 | 3 | 26 | 26 |
| 362 | 1409.27 | 33.0115 | 8.6266e+06 | 9.49603e+06 | 29 | 30 | 31 | 4 | 21 | 5 | 18 | 17 |
| 56 | 1184.67 | 23.4155 | 3.10093e+06 | 6.74694e+06 | 25 | 26.5 | 28 | 5 | 18 | 3 | 21 | 20 |
| 747 | 1129.46 | 34.4444 | 7.25859e+06 | 3.28563e+06 | 25 | 26.5 | 28 | 3 | 14 | 5 | 17 | 16 |
| 497 | 1076.02 | 18.154 | 5.6268e+06 | 9.41035e+06 | 27 | 29 | 31 | 4 | 23 | 2 | 23 | 22 |
| 348 | 906.373 | 13.9893 | 5.13309e+06 | 6.41997e+06 | 26 | 26.5 | 27 | 3 | 16 | 5 | 20 | 20 |
| 843 | 863.217 | 15.419 | 5.50078e+06 | 3.28281e+06 | 25 | 26 | 27 | 4 | 17 | 2 | 21 | 20 |
| 638 | 764.549 | 15.8134 | 8.67609e+06 | 9.6562e+06 | 29 | 30 | 31 | 5 | 24 | 2 | 16 | 16 |

