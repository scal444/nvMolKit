# Recovered gCheminfoCommands AAP/DISE reference

The 2015 paper's Supplementary File 3 contains the Apache-2.0 Java source for
Atom-Atom-Path similarity and the DISE clustering workflow. Its canonical
command-line build cannot run without the proprietary OpenEye OEChem Java JAR
and a valid OpenEye license; neither is included by the authors or present in
the benchmark image.

`GCheminfoAAPDISE.java` is a narrow compatibility runner. It ports the published
`IAAPathGeneratorChar`, `IAAPathComparatorChar`, and `SphereExclusion` behavior
to a dependency-free graph representation. RDKit is used only before timing to
supply heavy-atom types, aromatic flags, and bond topology. The runner performs:

1. AAP `DEFAULT8` descriptor construction (maximum path length 7 by default).
2. Sequential sphere-exclusion centroid selection in input order.
3. Final nearest-centroid reassignment, parallelized like `sdfCluster.pl`'s NN
   phase.

This is not claimed as the untouched canonical executable. Validate it against
the clustered SDF shipped by the authors before using its timings as a reference.

Workflow benchmark mode loads the graph stream once, then starts each warmup or
measured pass with the paper's activity cleanup and stable ascending numeric
sort. It reports ordering, descriptor, selection, and assignment times. File
parsing, JVM startup, and assignment-file writing remain outside the intervals.
Descriptor construction remains included because nvMolKit's clustering call
also constructs its AAP descriptors from pre-parsed molecules.
