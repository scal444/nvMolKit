# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import numpy as np
import pytest
import torch
from rdkit import Chem

from nvmolkit.clustering import aap_similarity, aap_similarity_clustering


def _mol(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return molecule


def test_aap_similarity_is_one_for_identical_and_renumbered_molecules():
    molecule = _mol("CC(=O)Oc1ccccc1C(=O)O")
    renumbered = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))

    assert aap_similarity(molecule, molecule) == 1.0
    assert aap_similarity(molecule, renumbered) == 1.0
    assert aap_similarity(renumbered, molecule) == 1.0
    assert aap_similarity_clustering([molecule, renumbered], threshold=1.0) == [1, 1]


def test_aap_similarity_has_expected_directed_numerical_result():
    ethane = _mol("CC")
    propane = _mol("CCC")

    assert aap_similarity(ethane, propane) == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert aap_similarity(propane, ethane) == pytest.approx(1.0 / 5.0, abs=1e-6)


def test_aap_similarity_distinguishes_atom_bond_aromaticity_and_topology():
    assert aap_similarity(_mol("C"), _mol("N")) == 0.0
    assert aap_similarity(_mol("c1ccccc1"), _mol("C1CCCCC1")) == 0.0

    single_to_double = aap_similarity(_mol("CC"), _mol("C=C"))
    single_to_triple = aap_similarity(_mol("CC"), _mol("C#N"))
    chain_to_branch = aap_similarity(_mol("CCCC"), _mol("CC(C)C"))
    assert single_to_double == pytest.approx(1.0 / 5.0, abs=1e-6)
    assert single_to_triple == pytest.approx(1.0 / 11.0, abs=1e-6)
    assert single_to_double > single_to_triple
    assert 0.0 < chain_to_branch < 1.0


def test_aap_similarity_is_finite_and_bounded_over_varied_chemistry():
    molecules = [
        _mol(smiles)
        for smiles in (
            "C",
            "CCO",
            "CC(=O)O",
            "CC(C)C",
            "c1ccccc1",
            "c1ccncc1",
            "C1CCCCC1",
            "[NH4+]",
            "ClCCCl",
        )
    ]

    scores = [aap_similarity(left, right) for left in molecules for right in molecules]
    assert all(math.isfinite(score) for score in scores)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert all(scores[index * len(molecules) + index] == 1.0 for index in range(len(molecules)))


@pytest.mark.parametrize(
    "option, value",
    [
        ("max_path_length", 1),
        ("histogram_bins", 1),
        ("sinkhorn_iterations", 1),
        ("sinkhorn_temperature", 0.5),
    ],
)
def test_aap_similarity_options_change_the_computation(option, value):
    left = _mol("CCO")
    right = _mol("CCN")
    baseline = aap_similarity(left, right)
    configured = aap_similarity(left, right, **{option: value})

    assert 0.0 <= configured <= 1.0
    assert configured != pytest.approx(baseline, abs=1e-3)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_path_length": 0}, "maxPathLength must be positive"),
        ({"histogram_bins": 0}, "histogramBins must be between 1 and 32767"),
        ({"histogram_bins": 32768}, "histogramBins must be between 1 and 32767"),
        ({"sinkhorn_iterations": 0}, "sinkhornIterations must be positive"),
        ({"sinkhorn_temperature": 0.0}, "sinkhornTemperature must be finite and positive"),
        ({"sinkhorn_temperature": float("nan")}, "sinkhornTemperature must be finite and positive"),
        ({"sinkhorn_temperature": float("inf")}, "sinkhornTemperature must be finite and positive"),
    ],
)
def test_aap_options_are_validated_for_pair_and_empty_clustering(kwargs, message):
    molecule = _mol("CCO")

    with pytest.raises(ValueError, match=message):
        aap_similarity(molecule, molecule, **kwargs)
    with pytest.raises(ValueError, match=message):
        aap_similarity_clustering([], **kwargs)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_aap_clustering_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="threshold must be in"):
        aap_similarity_clustering([], threshold=threshold)


def test_aap_rejects_empty_oversized_null_and_unsupported_molecules():
    molecule = _mol("CCO")
    empty = Chem.RWMol().GetMol()
    oversized = _mol("C" * 65)
    unsupported = Chem.RWMol()
    unsupported.AddAtom(Chem.Atom(6))
    unsupported.AddAtom(Chem.Atom(6))
    unsupported.AddBond(0, 1, Chem.BondType.UNSPECIFIED)

    with pytest.raises(ValueError, match="does not support empty molecules"):
        aap_similarity(empty, molecule)
    with pytest.raises(ValueError, match="at most 64 atoms"):
        aap_similarity(oversized, oversized)
    with pytest.raises(ValueError, match="Invalid molecule at index 0"):
        aap_similarity_clustering([None])
    with pytest.raises(ValueError, match="supports only single, double, triple, and aromatic bonds"):
        aap_similarity(unsupported.GetMol(), molecule)


def test_aap_supports_the_64_atom_boundary():
    molecule = _mol("C" * 64)

    assert aap_similarity_clustering([molecule]) == [1]


def test_aap_clustering_handles_empty_singleton_and_generator_inputs():
    ethanol = _mol("CCO")
    benzene = _mol("c1ccccc1")

    assert aap_similarity_clustering([]) == []
    assert aap_similarity_clustering([ethanol]) == [1]
    molecules = (molecule for molecule in (ethanol, ethanol, benzene))
    assert aap_similarity_clustering(molecules, threshold=1.0) == [1, 1, 2]


def test_aap_clustering_threshold_is_inclusive():
    ethane = _mol("CC")
    propane = _mol("CCC")
    score = aap_similarity(ethane, propane)

    assert aap_similarity_clustering([ethane, propane], threshold=score) == [1, 1]
    assert aap_similarity_clustering([ethane, propane], threshold=score + 1e-4) == [1, 2]


def test_aap_clustering_is_directed_and_uses_input_order_centroids():
    ethane = _mol("CC")
    propane = _mol("CCC")

    assert aap_similarity_clustering([ethane, propane], threshold=0.25) == [1, 1]
    assert aap_similarity_clustering([propane, ethane], threshold=0.25) == [1, 2]


def test_aap_clustering_is_centroid_based_not_transitive():
    ethane = _mol("CC")
    propane = _mol("CCC")
    butane = _mol("CCCC")

    assert aap_similarity(ethane, propane) >= 0.3
    assert aap_similarity(propane, butane) >= 0.3
    assert aap_similarity(ethane, butane) < 0.3
    assert aap_similarity_clustering([ethane, propane, butane], threshold=0.3) == [1, 1, 2]


def test_aap_clustering_renumbers_clusters_by_size_then_centroid_order():
    benzene = _mol("c1ccccc1")
    ethanol = _mol("CCO")
    propane = _mol("CCC")
    molecules = [benzene, ethanol, ethanol, ethanol, propane, propane]

    assert aap_similarity_clustering(molecules, threshold=1.0) == [3, 1, 1, 1, 2, 2]


def test_aap_clustering_threshold_zero_assigns_every_molecule_to_first_centroid():
    molecules = [_mol(smiles) for smiles in ("CCO", "c1ccccc1", "[NH4+]", "ClCCCl")]

    assert aap_similarity_clustering(molecules, threshold=0.0) == [1, 1, 1, 1]


def test_aap_similarity_and_clustering_are_deterministic_across_streams():
    left = _mol("CCO")
    right = _mol("CCN")
    molecules = [left, left, right, _mol("c1ccccc1")]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    scores = [aap_similarity(left, right, stream=stream) for stream in streams]
    labels = [aap_similarity_clustering(molecules, threshold=0.2, stream=stream) for stream in streams]
    assert scores[0] == scores[1]
    assert labels[0] == labels[1]


def test_aap_similarity_tracks_optional_rdkit_reference_over_corpus():
    reference = pytest.importorskip("rdkit.Contrib.AtomAtomSimilarity.AtomAtomPathSimilarity")
    molecules = [
        _mol(smiles)
        for smiles in (
            "CCO",
            "CCN",
            "CCC",
            "CCCO",
            "CC(=O)O",
            "COC",
            "c1ccccc1",
            "c1ccncc1",
            "c1ccccc1O",
            "C1CCCCC1",
            "CC(C)C",
            "ClCCCl",
        )
    ]
    pairs = [(left, right) for index, left in enumerate(molecules) for right in molecules[index + 1 :]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.asarray([reference.AtomAtomPathSimilarity(left, right) for left, right in pairs])
    actual = np.asarray([aap_similarity(left, right) for left, right in pairs])

    assert np.corrcoef(expected, actual)[0, 1] >= 0.95
    assert np.mean(np.abs(expected - actual)) <= 0.03


def test_aap_matches_optional_ligand_clustering_source():
    reference = pytest.importorskip("gpu_ligand_clustering.aap")
    smiles = ["C", "CC", "CCC", "CCCC", "CCO", "CCN", "c1ccccc1", "c1ccncc1"]
    molecules = [_mol(value) for value in smiles]
    pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)]

    expected = [
        reference.aap_similarity(smiles[left], smiles[right], sinkhorn_execution="eager") for left, right in pairs
    ]
    actual = [aap_similarity(molecules[left], molecules[right]) for left, right in pairs]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)

    for threshold in (0.2, 0.217, 0.3, 0.5):
        expected_labels = reference.aap_similarity_clustering(
            smiles, dist_thresh=threshold, sinkhorn_execution="eager"
        )
        assert aap_similarity_clustering(molecules, threshold=threshold) == expected_labels
