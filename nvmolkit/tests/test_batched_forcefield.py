import os

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Geometry import Point3D

from nvmolkit.batchedForcefield import DGBatchedForcefield, ETKBatchedForcefield, MMFFBatchedForcefield
from nvmolkit.types import MMFFProperties


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


def make_embedded_mol(smiles: str, num_confs: int = 1, seed: int = 0xC0FFEE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


def make_fragmented_mol():
    mol = make_embedded_mol("CC.CC")
    conf = mol.GetConformer()
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) != 2:
        raise AssertionError("Expected two fragments for interfragment interaction test")
    anchor = conf.GetAtomPosition(fragments[0][0])
    moved = conf.GetAtomPosition(fragments[1][0])
    shift = Point3D(anchor.x - moved.x + 2.0, anchor.y - moved.y, anchor.z - moved.z)
    for atom_idx in fragments[1]:
        pos = conf.GetAtomPosition(atom_idx)
        conf.SetAtomPosition(atom_idx, Point3D(pos.x + shift.x, pos.y + shift.y, pos.z + shift.z))
    return mol


def clone_mols(molecules):
    return [Chem.Mol(mol) for mol in molecules]


def make_rdkit_mmff_properties(mol, properties: MMFFProperties | None = None):
    properties = MMFFProperties() if properties is None else properties
    mmff_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=properties.variant)
    if mmff_props is None:
        raise ValueError("RDKit could not create MMFF properties for molecule")
    mmff_props.SetMMFFVariant(properties.variant)
    mmff_props.SetMMFFDielectricConstant(properties.dielectric_constant)
    mmff_props.SetMMFFDielectricModel(properties.dielectric_model)
    mmff_props.SetMMFFBondTerm(properties.bond_term)
    mmff_props.SetMMFFAngleTerm(properties.angle_term)
    mmff_props.SetMMFFStretchBendTerm(properties.stretch_bend_term)
    mmff_props.SetMMFFOopTerm(properties.oop_term)
    mmff_props.SetMMFFTorsionTerm(properties.torsion_term)
    mmff_props.SetMMFFVdWTerm(properties.vdw_term)
    mmff_props.SetMMFFEleTerm(properties.ele_term)
    return mmff_props


def make_rdkit_mmff_forcefield(
    mol,
    properties: MMFFProperties | None = None,
    conf_id: int = -1,
):
    properties = MMFFProperties() if properties is None else properties
    mmff_props = make_rdkit_mmff_properties(mol, properties)
    return rdForceFieldHelpers.MMFFGetMoleculeForceField(
        mol,
        mmff_props,
        nonBondedThresh=properties.non_bonded_threshold,
        confId=conf_id,
        ignoreInterfragInteractions=properties.ignore_interfrag_interactions,
    )


def get_mmff_reference_energy_and_grad(
    mol,
    properties: MMFFProperties | None = None,
    conf_id: int = -1,
    configure_forcefield=None,
):
    ff = make_rdkit_mmff_forcefield(mol, properties=properties, conf_id=conf_id)
    if configure_forcefield is not None:
        configure_forcefield(ff)
    return ff.CalcEnergy(), list(ff.CalcGrad())


def assert_energy_and_gradient_close(got_energy, want_energy, got_grad, want_grad):
    assert got_energy == pytest.approx(want_energy, rel=1e-5, abs=1e-5)
    assert got_grad == pytest.approx(want_grad, rel=1e-4, abs=1e-4)


def assert_single_batched_matches_rdkit(
    mol,
    properties: MMFFProperties | None = None,
    conf_id: int = -1,
    configure_batch=None,
    configure_rdkit=None,
):
    nvmolkit_mol = Chem.Mol(mol)
    ff = MMFFBatchedForcefield([nvmolkit_mol], properties=properties, conf_id=conf_id)
    if configure_batch is not None:
        configure_batch(ff[0])
    got_energy = ff.compute_energy()[0]
    got_grad = ff.compute_gradients()[0]
    want_energy, want_grad = get_mmff_reference_energy_and_grad(
        Chem.Mol(mol),
        properties=properties,
        conf_id=conf_id,
        configure_forcefield=configure_rdkit,
    )
    assert_energy_and_gradient_close(got_energy, want_energy, got_grad, want_grad)


def test_mmff_batched_forcefield_matches_rdkit():
    assert_single_batched_matches_rdkit(load_reference_mol())


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


@pytest.mark.parametrize(
    ("mol_factory", "properties"),
    [
        pytest.param(
            lambda: make_embedded_mol("CC(=O)NC"),
            MMFFProperties(variant="MMFF94s"),
            id="variant-mmff94s",
        ),
        pytest.param(
            load_reference_mol,
            MMFFProperties(dielectric_constant=2.5, dielectric_model=2),
            id="dielectric-settings",
        ),
        pytest.param(
            load_reference_mol,
            MMFFProperties(
                bond_term=False,
                angle_term=False,
                stretch_bend_term=False,
                oop_term=False,
                torsion_term=False,
            ),
            id="term-toggles",
        ),
        pytest.param(
            make_fragmented_mol,
            MMFFProperties(ignore_interfrag_interactions=False, non_bonded_threshold=25.0),
            id="interfragment-interactions",
        ),
    ],
)
def test_mmff_batched_forcefield_properties_match_rdkit(mol_factory, properties):
    assert_single_batched_matches_rdkit(mol_factory(), properties=properties)


def test_mmff_batched_forcefield_per_molecule_properties_match_rdkit():
    mols = [make_embedded_mol("CCO"), make_fragmented_mol()]
    properties = [
        MMFFProperties(dielectric_constant=3.0, dielectric_model=2),
        MMFFProperties(ignore_interfrag_interactions=False, non_bonded_threshold=20.0),
    ]

    ff = MMFFBatchedForcefield(clone_mols(mols), properties=properties)
    got_energies = ff.compute_energy()
    got_grads = ff.compute_gradients()

    for idx, (mol, prop) in enumerate(zip(mols, properties)):
        want_energy, want_grad = get_mmff_reference_energy_and_grad(Chem.Mol(mol), properties=prop)
        assert_energy_and_gradient_close(got_energies[idx], want_energy, got_grads[idx], want_grad)


def test_mmff_batched_forcefield_conf_ids_match_rdkit():
    mol = make_embedded_mol("CCCO", num_confs=2)
    ff = MMFFBatchedForcefield([Chem.Mol(mol), Chem.Mol(mol)], conf_id=[0, 1])

    got_energies = ff.compute_energy()
    got_grads = ff.compute_gradients()

    for idx, conf_id in enumerate([0, 1]):
        want_energy, want_grad = get_mmff_reference_energy_and_grad(Chem.Mol(mol), conf_id=conf_id)
        assert_energy_and_gradient_close(got_energies[idx], want_energy, got_grads[idx], want_grad)


def test_mmff_batched_forcefield_lazy_build_and_rebuild():
    mol = make_embedded_mol("CCO")
    ff = MMFFBatchedForcefield([Chem.Mol(mol)])

    assert ff._native_ff is None
    assert ff._dirty is True

    first_energy = ff.compute_energy()[0]
    first_native = ff._native_ff

    assert first_native is not None
    assert ff._dirty is False
    assert ff.compute_energy()[0] == pytest.approx(first_energy, rel=1e-5, abs=1e-5)
    assert ff._native_ff is first_native

    ff[0].add_distance_constraint(0, 2, True, 0.2, 0.4, 25.0)
    assert ff._dirty is True

    ff.rebuild()
    assert ff._dirty is False
    assert ff._native_ff is not first_native


def test_mmff_batched_forcefield_invalid_indices():
    ff = MMFFBatchedForcefield([make_embedded_mol("CCO")])

    with pytest.raises(IndexError, match="Batch element index"):
        ff[1]

    with pytest.raises(IndexError, match="Atom index"):
        ff[0].add_distance_constraint(0, 99, False, 0.0, 1.0, 10.0)


def test_mmff_distance_constraint_matches_rdkit():
    mol = make_embedded_mol("CCO")
    assert_single_batched_matches_rdkit(
        mol,
        configure_batch=lambda element: element.add_distance_constraint(0, 2, False, 0.0, 1.5, 25.0),
        configure_rdkit=lambda ff: ff.MMFFAddDistanceConstraint(0, 2, False, 0.0, 1.5, 25.0),
    )


def test_mmff_distance_relative_constraint_matches_rdkit():
    mol = make_embedded_mol("CCO")
    assert_single_batched_matches_rdkit(
        mol,
        configure_batch=lambda element: element.add_distance_constraint(0, 2, True, 0.3, 0.6, 15.0),
        configure_rdkit=lambda ff: ff.MMFFAddDistanceConstraint(0, 2, True, 0.3, 0.6, 15.0),
    )


def test_mmff_position_constraint_matches_rdkit_reference_pose():
    mol = make_embedded_mol("CCO")
    assert_single_batched_matches_rdkit(
        mol,
        configure_batch=lambda element: element.add_position_constraint(0, 0.1, 50.0),
        configure_rdkit=lambda ff: ff.MMFFAddPositionConstraint(0, 0.1, 50.0),
    )


def test_mmff_angle_constraint_matches_rdkit():
    mol = make_embedded_mol("CCC")
    assert_single_batched_matches_rdkit(
        mol,
        configure_batch=lambda element: element.add_angle_constraint(0, 1, 2, True, 5.0, 10.0, 20.0),
        configure_rdkit=lambda ff: ff.MMFFAddAngleConstraint(0, 1, 2, True, 5.0, 10.0, 20.0),
    )


def test_mmff_torsion_constraint_matches_rdkit():
    mol = make_embedded_mol("CCCC")
    assert_single_batched_matches_rdkit(
        mol,
        configure_batch=lambda element: element.add_torsion_constraint(0, 1, 2, 3, True, 15.0, 30.0, 12.0),
        configure_rdkit=lambda ff: ff.MMFFAddTorsionConstraint(0, 1, 2, 3, True, 15.0, 30.0, 12.0),
    )


def test_mmff_mixed_properties_and_constraints_batch_matches_rdkit():
    mols = [make_embedded_mol("CCO"), make_embedded_mol("CCCC")]
    properties = [
        MMFFProperties(dielectric_constant=2.0, dielectric_model=2),
        MMFFProperties(variant="MMFF94s"),
    ]
    ff = MMFFBatchedForcefield(clone_mols(mols), properties=properties)
    ff[0].add_distance_constraint(0, 2, True, 0.2, 0.5, 20.0)
    ff[1].add_torsion_constraint(0, 1, 2, 3, True, 10.0, 20.0, 8.0)

    got_energies = ff.compute_energy()
    got_grads = ff.compute_gradients()

    ref_specs = [
        lambda forcefield: forcefield.MMFFAddDistanceConstraint(0, 2, True, 0.2, 0.5, 20.0),
        lambda forcefield: forcefield.MMFFAddTorsionConstraint(0, 1, 2, 3, True, 10.0, 20.0, 8.0),
    ]
    for idx, (mol, prop, configure_forcefield) in enumerate(zip(mols, properties, ref_specs)):
        want_energy, want_grad = get_mmff_reference_energy_and_grad(
            Chem.Mol(mol),
            properties=prop,
            configure_forcefield=configure_forcefield,
        )
        assert_energy_and_gradient_close(got_energies[idx], want_energy, got_grads[idx], want_grad)


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
