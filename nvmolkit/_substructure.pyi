from typing import Sequence

import numpy as np
from rdkit.Chem import Mol

class SubstructSearchConfig:
    batchSize: int
    workerThreads: int
    preprocessingThreads: int  # Also handles RDKit fallback opportunistically
    maxMatches: int
    uniquify: bool
    gpuIds: list[int]

def getSubstructMatches(
    targets: Sequence[Mol],
    queries: Sequence[Mol],
    config: SubstructSearchConfig = ...,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Perform batch substructure matching on GPU.

    Args:
        targets: List of target RDKit molecules
        queries: List of query RDKit molecules (typically from SMARTS)
        config: SubstructSearchConfig with execution settings

    Returns:
        CSR-style tuple of numpy arrays:
        (atom_indices, match_indptr, pair_indptr, shape)
    """
    ...

def countSubstructMatches(
    targets: Sequence[Mol],
    queries: Sequence[Mol],
    config: SubstructSearchConfig = ...,
) -> np.ndarray:
    """Count substructure matches per target/query pair.

    Args:
        targets: List of target RDKit molecules
        queries: List of query RDKit molecules (typically from SMARTS)
        config: SubstructSearchConfig with execution settings

    Returns:
        2D numpy array of int with shape (num_targets, num_queries).
        results[target_idx, query_idx] = match count.
    """
    ...

def hasSubstructMatch(
    targets: Sequence[Mol],
    queries: Sequence[Mol],
    config: SubstructSearchConfig = ...,
) -> np.ndarray:
    """Check if targets contain query substructures (boolean results).

    More efficient than getSubstructMatches when only existence is needed.

    Args:
        targets: List of target RDKit molecules
        queries: List of query RDKit molecules (typically from SMARTS)
        config: SubstructSearchConfig with execution settings

    Returns:
        2D numpy array of uint8 with shape (num_targets, num_queries).
        results[target_idx, query_idx] = 1 if match exists, 0 otherwise.
    """
    ...

