import os

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from nvmolkit.batchedForcefield import DGBatchedForcefield, ETKBatchedForcefield, MMFFBatchedForcefield


def load_reference_mol():
    mol2_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tests",
        "test_data",
        "rdkit_smallmol_1.mol2",
    )
    if not os.path.exists(mol2_path):
        pytest.skip(f"Test data file not found: {mol2_path}")
    mol = Chem.MolFromMol2File(mol2_path, sanitize=False, removeHs=False)
    if mol is None:
        pytest.skip("Failed to load rdkit_smallmol_1.mol2")
    Chem.SanitizeMol(mol)
    return mol


def clone_mols(molecules):
    return [Chem.Mol(mol) for mol in molecules]


def get_mmff_reference_energy_and_grad(mol):
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props)
    return ff.CalcEnergy(), list(ff.CalcGrad())


def test_mmff_batched_forcefield_matches_rdkit():
    mol = load_reference_mol()
    energy, grad = get_mmff_reference_energy_and_grad(Chem.Mol(mol))

    ff = MMFFBatchedForcefield([Chem.Mol(mol)])
    got_energy = ff.compute_energy()
    got_grad = ff.compute_gradients()

    assert len(got_energy) == 1
    assert len(got_grad) == 1
    assert got_energy[0] == pytest.approx(energy, rel=1e-5, abs=1e-5)
    assert got_grad[0] == pytest.approx(grad, rel=1e-4, abs=1e-4)


def test_mmff_batched_forcefield_batch_matches_single():
    mols = [load_reference_mol(), load_reference_mol()]

    batch_ff = MMFFBatchedForcefield(clone_mols(mols))
    batch_energies = batch_ff.compute_energy()
    batch_grads = batch_ff.compute_gradients()

    single_energies = []
    single_grads = []
    for mol in clone_mols(mols):
        single_ff = MMFFBatchedForcefield([mol])
        single_energies.append(single_ff.compute_energy()[0])
        single_grads.append(single_ff.compute_gradients()[0])

    assert batch_energies == pytest.approx(single_energies, rel=1e-5, abs=1e-5)
    for got_grad, want_grad in zip(batch_grads, single_grads):
        assert got_grad == pytest.approx(want_grad, rel=1e-4, abs=1e-4)


def test_dg_batched_forcefield_batch_matches_single():
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    mols = [load_reference_mol(), load_reference_mol()]

    batch_ff = DGBatchedForcefield(clone_mols(mols), params)
    batch_energies = batch_ff.compute_energy()
    batch_grads = batch_ff.compute_gradients()

    single_energies = []
    single_grads = []
    for mol in clone_mols(mols):
        single_ff = DGBatchedForcefield([mol], params)
        single_energies.append(single_ff.compute_energy()[0])
        single_grads.append(single_ff.compute_gradients()[0])

    assert batch_ff.num_molecules == 2
    assert batch_ff.data_dim == 4
    assert batch_energies == pytest.approx(single_energies, rel=1e-5, abs=1e-5)
    for got_grad, want_grad in zip(batch_grads, single_grads):
        assert got_grad == pytest.approx(want_grad, rel=1e-4, abs=1e-4)


def test_etk_batched_forcefield_batch_matches_single():
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    mols = [load_reference_mol(), load_reference_mol()]

    batch_ff = ETKBatchedForcefield(clone_mols(mols), params)
    batch_energies = batch_ff.compute_energy()
    batch_grads = batch_ff.compute_gradients()

    single_energies = []
    single_grads = []
    for mol in clone_mols(mols):
        single_ff = ETKBatchedForcefield([mol], params)
        single_energies.append(single_ff.compute_energy()[0])
        single_grads.append(single_ff.compute_gradients()[0])

    assert batch_ff.num_molecules == 2
    assert batch_ff.data_dim == 3
    assert batch_energies == pytest.approx(single_energies, rel=1e-5, abs=1e-5)
    for got_grad, want_grad in zip(batch_grads, single_grads):
        assert got_grad == pytest.approx(want_grad, rel=1e-4, abs=1e-4)
