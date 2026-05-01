# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import matplotlib.pyplot as plt
import numpy as np

with open("/home/kboyd/data/fire/rdkit_with_debinfo_1013.json", "r") as f:
    rdk = json.load(f)
with open("nvmolkit_test.json", "r") as f:
    nvmol = json.load(f)
len(nvmol[0])

rdk00e = np.array(rdk[0][0]["energies"])
nv00 = nvmol[0][0]
nv00e = nv00["energies"]
nv_alpha = nv00["alphas"]
nv_dt = nv00["dt"]
nv_power = nv00["powers"]
nv_alpha = nv00["alphas"]

plt.figure()
plt.plot(rdk00e, label="rdkit")
plt.plot(nv00e, label="nvmolkit")
plt.legend()
plt.title("Energies for first molecule")
plt.xlabel("Step")
plt.ylabel("Energy (kcal/mol)")
plt.xlim(0, 160)
plt.show()

plt.figure()
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

x_range = min(200, len(nv00e))

axes[0].plot(nv_alpha[:x_range])
axes[0].set_title("Alpha per step")
axes[0].set_ylabel("Alpha")

axes[1].plot(nv_dt[:x_range])
axes[1].set_title("dt per step")
axes[1].set_ylabel("dt")

axes[2].plot(nv_power[:x_range])
axes[2].set_title("Power per step")
axes[2].set_ylabel("Power")

axes[3].plot(rdk00e[:x_range], label="rdkit")
axes[3].plot(nv00e[:x_range], label="nvmolkit")
axes[3].set_title("Energies for first molecule")
axes[3].set_ylabel("Energy (kcal/mol)")
axes[3].set_xlabel("Step")
axes[3].legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axes[0].plot(nv_alpha[:x_range])
axes[0].set_title("Alpha per step")
axes[0].set_ylabel("Alpha")
axes[0].set_yscale("log")

axes[1].plot(nv_dt[:x_range])
axes[1].set_title("dt per step")
axes[1].set_ylabel("dt")
axes[1].set_yscale("log")

axes[2].plot(nv_power[:x_range])
axes[2].set_title("Power per step")
axes[2].set_ylabel("Power")
axes[2].set_yscale("log")

axes[3].plot(rdk00e[:x_range], label="rdkit")
axes[3].plot(nv00e[:x_range], label="nvmolkit")
axes[3].set_title("Energies for first molecule")
axes[3].set_ylabel("Energy (kcal/mol)")
axes[3].set_ylim(85, 90)
axes[3].set_xlabel("Step")
axes[3].legend()
# axes[3].set_yscale('log')

plt.tight_layout()
plt.show()


from rdkit import Chem
from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs


def single_minimization(
    scheme,
    half_step,
    mass_weighting,
    steps=500,
    n_min_for_increase=5,
    dt_init=0.001,
    dt_increment=1.1,
    alpha_init=0.25,
    max_step=0.0,
    dt_max_factor=10.0,
    use_abc=False,
):
    fire_debug = []
    suppl = Chem.SDMolSupplier("/home/kboyd/data/fire/inital_confs.sdf", removeHs=False)
    mol = suppl[0]
    MMFFOptimizeMoleculesConfs(
        [mol],
        maxIters=steps,
        optimizer_backend="FIRE",
        optimizer_options={
            "time_step_increment": dt_increment,
            "integration_scheme": scheme,
            "take_half_step_back": half_step,
            "use_masses": mass_weighting,
            "n_min_for_increase": n_min_for_increase,
            "dt_init": dt_init,
            "alpha_init": alpha_init,
            "max_step": max_step,
            "dt_max_factor": dt_max_factor,
            "use_abc": use_abc,
        },
        fire_debug_output=fire_debug,
    )
    return fire_debug


params = {"dt_init": 0.0001, "dt_increment": 1.1, "alpha_init": 0.25}
fire_debug_explicit_euler = single_minimization("explicit_euler", True, True, **params)
fire_debug_semi_implicit_euler = single_minimization("semi_implicit_euler", True, True, **params)
fire_debug_explicit_euler_no_half_step = single_minimization("explicit_euler", False, True, **params)
fire_debug_semi_implicit_euler_no_half_step = single_minimization("semi_implicit_euler", False, True, **params)

import matplotlib.pyplot as plt
import numpy as np


rdk_ref = 86.9133025616466
plt.plot(fire_debug_explicit_euler[0][0]["energies"], label="Explicit Euler + half step")
plt.plot(fire_debug_semi_implicit_euler[0][0]["energies"], label="Semi-Implicit Euler + half step")
plt.plot(fire_debug_explicit_euler_no_half_step[0][0]["energies"], label="Explicit Euler (no half step)")
plt.plot(fire_debug_semi_implicit_euler_no_half_step[0][0]["energies"], label="Semi-Implicit Euler (no half step)")
plt.plot(rdk_ref * np.ones(len(fire_debug_explicit_euler[0][0]["energies"])), "k--", label="Reference")
plt.xlabel("Step")
plt.ylabel("Energy (kcal/mol)")
plt.title("Per-step MMFF energies for single minimization")
plt.legend()
plt.tight_layout()
plt.show()


# Try scanning n_min_for_increase, 5, 10, 20, 30
for dt_init in [0.0001, 0.001, 0.002]:
    fire_debug = single_minimization("semi_implicit_euler", True, True, dt_init=dt_init)
    plt.plot(fire_debug[0][0]["energies"], label="Implict Euler + half step + dt_init = " + str(dt_init))
    plt.xlabel("Step")
    plt.ylabel("Energy (kcal/mol)")
    plt.title("Per-step MMFF energies for single minimization")
plt.legend()
plt.show()

for dt_increment in [1.001, 1.01, 1.05, 1.1]:
    fire_debug = single_minimization("explicit_euler", True, True, dt_increment=dt_increment)
    plt.plot(fire_debug[0][0]["energies"], label="Explicit Euler + half step + dt_increment = " + str(dt_increment))
    plt.xlabel("Step")
    plt.ylabel("Energy (kcal/mol)")
    plt.title("Per-step MMFF energies for single minimization")
plt.legend()
plt.show()

for alpha_init in [
    0.1,
    0.25,
    0.5,
]:
    fire_debug = single_minimization("explicit_euler", True, True, alpha_init=alpha_init)
    plt.plot(fire_debug[0][0]["energies"], label="Explicit Euler + half step + alpha_init = " + str(alpha_init))
    plt.xlabel("Step")
    plt.ylabel("Energy (kcal/mol)")
    plt.title("Per-step MMFF energies for single minimization")
plt.legend()
plt.show()


import itertools

scan_params = {
    "dt_init": [0.0001, 0.001],
    "dt_increment": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    "n_min_for_increase": [0, 1, 2, 5, 10],
    "half_step": [
        True,
    ],
    "mass_weighting": [
        True,
    ],
    "max_step": [
        0.0,
    ],
    "dt_max_factor": [5.0, 10.0, 20.0],
    "use_abc": [True, False],
}

scan_param_keys = list(scan_params.keys())
scan_param_values = [scan_params[key] for key in scan_param_keys]
scan_param_tuples = list(itertools.product(*scan_param_values))
import tqdm

plt.figure()
scan_results = []
for scan_param_tuple in tqdm.tqdm(scan_param_tuples):
    fire_debug = single_minimization(
        "semi_implicit_euler",
        scan_param_tuple[3],  # half_step
        scan_param_tuple[4],  # mass_weighting
        dt_init=scan_param_tuple[0],
        dt_increment=scan_param_tuple[1],
        n_min_for_increase=scan_param_tuple[2],
        max_step=scan_param_tuple[5],
        dt_max_factor=scan_param_tuple[6],
        use_abc=scan_param_tuple[7],
    )

    energies = fire_debug[0][0]["energies"]
    result_row = dict(
        dt_init=scan_param_tuple[0],
        dt_increment=scan_param_tuple[1],
        n_min_for_increase=scan_param_tuple[2],
        half_step=scan_param_tuple[3],
        mass_weighting=scan_param_tuple[4],
        max_step=scan_param_tuple[5],
        dt_max_factor=scan_param_tuple[6],
        energies=energies,
        final_energy=energies[-1] if len(energies) > 0 else None,
        energy_at_200=energies[200] if len(energies) > 200 else (energies[-1] if len(energies) > 0 else None),
        use_abc=scan_param_tuple[7],
        n_steps=len(energies),
    )
    scan_results.append(result_row)

    # Optionally save/convert to DataFrame at the end of your loop or script:
    # df = pd.DataFrame(scan_results)
    # df.to_pickle("scan_results.pkl")
    if fire_debug[0][0]["energies"][-1] < 10**4:
        plt.plot(
            fire_debug[0][0]["energies"],
            label="Semi-Implicit Euler + half step + dt_init = "
            + str(scan_param_tuple[0])
            + " + dt_increment = "
            + str(scan_param_tuple[1])
            + " + n_min_for_increase = "
            + str(scan_param_tuple[2])
            + " + max_step = "
            + str(scan_param_tuple[3])
            + " + dt_max_factor = "
            + str(scan_param_tuple[4]),
        )

import pandas as pd

df = pd.DataFrame(scan_results)

plt.xlabel("Step")
plt.ylabel("Energy (kcal/mol)")
plt.title("Per-step MMFF energies for single minimization")
plt.legend()
plt.show()

ideal_params = {
    "dt_init": 0.0001,
    "dt_increment": 1.2,
    "n_min_for_increase": 5,
    "max_step": 0.0,
    "dt_max_factor": 100.0,
}

fire_debug_ideal_abc = single_minimization("semi_implicit_euler", True, True, use_abc=True, **ideal_params)
fire_debug_ideal_noabc = single_minimization("semi_implicit_euler", True, True, use_abc=False, **ideal_params)


# Debugging with ASE
params = {
    "dt_init": 0.001,
    "dt_increment": 1.1,
    "n_min_for_increase": 20,
    "dt_max_factor": 10.0,
}

from rdkit import Chem
from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs


def single_minimization(
    scheme,
    half_step,
    mass_weighting,
    steps=500,
    n_min_for_increase=5,
    dt_init=0.001,
    dt_increment=1.1,
    alpha_init=0.25,
    max_step=0.0,
    dt_max_factor=10.0,
    use_abc=False,
):
    fire_debug = []
    suppl = Chem.SDMolSupplier("/home/kboyd/data/fire/inital_confs.sdf", removeHs=False)
    mol = suppl[0]
    MMFFOptimizeMoleculesConfs(
        [mol],
        maxIters=steps,
        optimizer_backend="FIRE",
        optimizer_options={
            "time_step_increment": dt_increment,
            "integration_scheme": scheme,
            "take_half_step_back": half_step,
            "use_masses": mass_weighting,
            "n_min_for_increase": n_min_for_increase,
            "dt_init": dt_init,
            "alpha_init": alpha_init,
            "max_step": max_step,
            "dt_max_factor": dt_max_factor,
            "use_abc": use_abc,
        },
        fire_debug_output=fire_debug,
    )
    return fire_debug


fire_debug_ideal_noabc = single_minimization("semi_implicit_euler", True, True, steps=10, use_abc=False, **params)
