import json
import matplotlib.pyplot as plt
import numpy as np

with open('rdkit_with_debinfo_1013.json', 'r') as f:
    rdk = json.load(f)
with open('nvmolkit_test.json', 'r') as f:
    nvmol = json.load(f)
len(nvmol[0])

rdk00e = np.array(rdk[0][0]['energies'])
nv00 = nvmol[0][0]
nv00e = nv00['energies']
nv_alpha = nv00['alphas']
nv_dt = nv00['dt']
nv_power = nv00['powers']
nv_alpha = nv00['alphas']

plt.figure()
plt.plot(rdk00e, label='rdkit')
plt.plot(nv00e, label='nvmolkit')
plt.legend()
plt.title('Energies for first molecule')
plt.xlabel('Step')
plt.ylabel('Energy (kcal/mol)')
plt.show()

plt.figure()
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

x_range = min(200, len(nv00e))

axes[0].plot(nv_alpha[:x_range])
axes[0].set_title('Alpha per step')
axes[0].set_ylabel('Alpha')

axes[1].plot(nv_dt[:x_range])
axes[1].set_title('dt per step')
axes[1].set_ylabel('dt')

axes[2].plot(nv_power[:x_range])
axes[2].set_title('Power per step')
axes[2].set_ylabel('Power')

axes[3].plot(rdk00e[:x_range], label='rdkit')
axes[3].plot(nv00e[:x_range], label='nvmolkit')
axes[3].set_title('Energies for first molecule')
axes[3].set_ylabel('Energy (kcal/mol)')
axes[3].set_xlabel('Step')
axes[3].legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axes[0].plot(nv_alpha[:x_range])
axes[0].set_title('Alpha per step')
axes[0].set_ylabel('Alpha')
axes[0].set_yscale('log')

axes[1].plot(nv_dt[:x_range])
axes[1].set_title('dt per step')
axes[1].set_ylabel('dt')
axes[1].set_yscale('log')

axes[2].plot(nv_power[:x_range])
axes[2].set_title('Power per step')
axes[2].set_ylabel('Power')
axes[2].set_yscale('log')

axes[3].plot(rdk00e[:x_range], label='rdkit')
axes[3].plot(nv00e[:x_range], label='nvmolkit')
axes[3].set_title('Energies for first molecule')
axes[3].set_ylabel('Energy (kcal/mol)')
axes[3].set_ylim(85, 90)
axes[3].set_xlabel('Step')
axes[3].legend()
#axes[3].set_yscale('log')

plt.tight_layout()
plt.show()




from rdkit import Chem
from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

def single_minimization(scheme, half_step, mass_weighting, n_min_for_increase=5, dt_init=0.001, dt_increment=1.1, alpha_init=0.25):
    fire_debug = []
    suppl = Chem.SDMolSupplier('/home/kboyd/data/fire/inital_confs.sdf', removeHs = False)
    mol = suppl[0]
    MMFFOptimizeMoleculesConfs([mol], maxIters=10000, optimizer_backend="FIRE", optimizer_options={"time_step_increment": dt_increment, "integration_scheme": scheme, "take_half_step_back": half_step, "use_masses": mass_weighting, "n_min_for_increase": n_min_for_increase, "dt_init": dt_init, "alpha_init": alpha_init}, fire_debug_output=fire_debug)
    return fire_debug

fire_debug_explicit_euler = single_minimization("explicit_euler", True, True)
fire_debug_semi_implicit_euler = single_minimization("semi_implicit_euler", True, True)
fire_debug_explicit_euler_no_half_step = single_minimization("explicit_euler", False, True)
fire_debug_semi_implicit_euler_no_half_step = single_minimization("semi_implicit_euler", False, True)

import matplotlib.pyplot as plt
import numpy as np


plt.plot(fire_debug_explicit_euler[0][0]['energies'], label='Explicit Euler + half step')
plt.plot(fire_debug_semi_implicit_euler[0][0]['energies'], label='Semi-Implicit Euler + half step')
plt.plot(fire_debug_explicit_euler_no_half_step[0][0]['energies'], label='Explicit Euler (no half step)')
plt.plot(fire_debug_semi_implicit_euler_no_half_step[0][0]['energies'], label='Semi-Implicit Euler (no half step)')
plt.xlabel('Step')
plt.ylabel('Energy (kcal/mol)')
plt.title('Per-step MMFF energies for single minimization')
plt.legend()
plt.tight_layout()
plt.show()


# Try scanning n_min_for_increase, 5, 10, 20, 30
for dt_init in [0.0001, 0.001, 0.002]:
    fire_debug = single_minimization("explicit_euler", True, True, dt_init=dt_init)
    plt.plot(fire_debug[0][0]['energies'], label='Explicit Euler + half step + dt_init = ' + str(dt_init))
    plt.xlabel('Step')
    plt.ylabel('Energy (kcal/mol)')
    plt.title('Per-step MMFF energies for single minimization')
plt.legend()
plt.show()

for dt_increment in [1.001, 1.01, 1.05, 1.1, 1.2]:
    fire_debug = single_minimization("explicit_euler", True, True, dt_increment=dt_increment)
    plt.plot(fire_debug[0][0]['energies'], label='Explicit Euler + half step + dt_increment = ' + str(dt_increment))
    plt.xlabel('Step')
    plt.ylabel('Energy (kcal/mol)')
    plt.title('Per-step MMFF energies for single minimization')
plt.legend()
plt.show()

for alpha_init in [0.1, 0.25, 0.5,]:
    fire_debug = single_minimization("explicit_euler", True, True, alpha_init=alpha_init)
    plt.plot(fire_debug[0][0]['energies'], label='Explicit Euler + half step + alpha_init = ' + str(alpha_init))
    plt.xlabel('Step')
    plt.ylabel('Energy (kcal/mol)')
    plt.title('Per-step MMFF energies for single minimization')
plt.legend()
plt.show()