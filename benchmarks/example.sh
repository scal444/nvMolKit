#!/bin/bash
python minimize_nvmolkit_conformers.py --output-prefix /home/kboyd/data/fire/nvmolkit_v1_mass --mass-weighting  /home/kboyd/data/fire/inital_confs.sdf 
python mmff_energy_comparison.py --output-dir . /home/kboyd/data/fire/rdkit /home/kboyd/data/fire/nvmolkit_v1_mass /home/kboyd/data/fire/nvmolkit_v1_nonmass
