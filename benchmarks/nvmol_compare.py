from rdkit import Chem
from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

params = {
    "dt_init": 0.1,
    "dt_increment": 1.1,
    "n_min_for_increase": 20,
    "dt_max_factor": 10.0,
}
def single_minimization(scheme, half_step, mass_weighting, steps=500, n_min_for_increase=5, dt_init=0.001, dt_increment=1.1, alpha_init=0.25, max_step=0.0, dt_max_factor=10.0, use_abc=False):
    fire_debug = []
    suppl = Chem.SDMolSupplier('/home/kboyd/data/fire/inital_confs.sdf', removeHs = False)
    mol = suppl[0]
    MMFFOptimizeMoleculesConfs([mol], maxIters=steps, optimizer_backend="FIRE",
                               optimizer_options={"time_step_increment": dt_increment,
                                                  "integration_scheme": scheme,
                                                  "take_half_step_back": half_step,
                                                  "use_masses": mass_weighting,
                                                  "n_min_for_increase": n_min_for_increase,
                                                  "dt_init": dt_init,
                                                  "alpha_init": alpha_init,
                                                  "max_step": max_step,
                                                  "dt_max_factor": dt_max_factor,
                                                  "use_abc": use_abc},
                               fire_debug_output=fire_debug)
    return fire_debug
fire_debug_ideal_noabc = single_minimization("semi_implicit_euler", True, False, steps=3, use_abc=False, **params)


import json
import matplotlib.pyplot as plt
import numpy as np


