# Kernel Factory fMCS rewrite reference

This directory preserves the winning submitted source from campaign
`4d5smzznqh2an5ycrh413jswhw`, candidate
`d9aee2ffeead7dcc064a1b9a4606b9fb0f69baa37cfd21307a5daeece9e02ee0`.

The campaign measured 83.9915 ms versus a 536.4136 ms baseline (6.3865x) with
full CudaGym correctness on its 17 workloads. The code is intentionally not
wired into nvMolKit. It is a standalone Torch extension specialized to the
extracted ABI and B200 campaign workload, and requires production-level
semantic, resource, timeout, overflow, and tier coverage review before any
integration.

`solution.json` is the submitted artifact, `solution_metrics.json` records the
campaign result, and the remaining files are byte-identical to the submitted
source files.
