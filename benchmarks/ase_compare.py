from ase.calculators.calculator import Calculator, all_changes
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
import numpy as np

from ase.units import kcal, eV
from ase.units import mol as avogadro_number
class RDKitMMFFCalculator(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(self, **kwargs):

        Calculator.__init__(self, **kwargs)
        sd_supplier = Chem.SDMolSupplier('/home/kboyd/data/fire/inital_confs.sdf', removeHs=False)
        rdmol = next(m for m in sd_supplier if m is not None)  # get first non-None entry
        self.rdmol = rdmol

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos  = atoms.get_positions()
        this_conf = self.rdmol.GetConformer()
        for i in range(self.rdmol.GetNumAtoms()):
            this_conf.SetAtomPosition(i, tuple(pos[i]))
        # Re-instantiate force field at each step
        ff = AllChem.MMFFGetMoleculeForceField(
            self.rdmol,
            AllChem.MMFFGetMoleculeProperties(self.rdmol),
            confId=this_conf.GetId(),
        )
        e = ff.CalcEnergy()
        g = ff.CalcGrad()
        f = -np.array(g).reshape(-1, 3)

        self.results = {
            'energy': e * (kcal / avogadro_number) / eV,  # Convert to eV
            'forces': f * (kcal / avogadro_number) / eV  # Convert to eV/Å,
        }



# 1. Read the SDF file and get the first molecule
sdf_supplier = Chem.SDMolSupplier('/home/kboyd/data/fire/inital_confs.sdf', removeHs=False)
mol = next(m for m in sdf_supplier if m is not None)  # get first non-None entry

# 2. Get atomic symbols
symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

# 3. Get 3D coordinates (needs conformer)
conf = mol.GetConformer()
positions = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]

# 4. Create ASE Atoms object
at = Atoms(symbols=symbols, positions=positions)

calc = RDKitMMFFCalculator()
at.calc = calc
energy = at.get_potential_energy()     # MMFF energy
forces = at.get_forces()               # MMFF forces

from ase.optimize import FIRE2
dyn = FIRE2(at)
res = []
for _ in range(2):
    print("\n\nStep\n\n")
    res.append(at.get_potential_energy() * avogadro_number / kcal)
    dyn.step()
    new_mol = Chem.SDMolSupplier('/home/kboyd/data/fire/inital_confs.sdf', removeHs=False)[0]
    # Update RDKit molecule with new positions
    new_conf = new_mol.GetConformer()
    for i in range(new_mol.GetNumAtoms()):
        new_conf.SetAtomPosition(i, tuple(at.get_positions()[i]))
    new_ff = AllChem.MMFFGetMoleculeForceField(
        new_mol,
        AllChem.MMFFGetMoleculeProperties(new_mol),
        confId=new_mol.GetConformer().GetId(),
    )
    new_energy = new_ff.CalcEnergy()
