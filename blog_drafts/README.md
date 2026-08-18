# RDKit blog drafts

These are pre-run draft notebooks prepared for possible publication in Greg
Landrum's RDKit blog. They are stored on this nvMolKit branch for review and are
not pushed to the RDKit blog repository.

- `nvMolKit ETKDG MMFF and TFD workflow.ipynb` benchmarks an end-to-end
  conformer workflow using RDKit with 16 CPU threads and nvMolKit 0.6 on GPU.
  The main workflow uses BFGS at 200 iterations and FIRE at 600 iterations. Its
  appendix summarizes the separate FIRE 200/400/600 parameter analysis.
- `nvMolKit fragment substructure search workflow.ipynb` compares RDKit and
  nvMolKit boolean fragment searches over one million Enamine REAL molecules.
- `images/` contains the two FIRE parameter-analysis figures referenced by the
  first notebook.

Both notebooks retain their executed outputs. The Enamine REAL input file is
not distributed here; update each notebook's `smi_file` path before rerunning.
The complete reproducible FIRE parameter analysis, including raw energies, is
kept separately in the private `scal444/fire_analysis` repository.
