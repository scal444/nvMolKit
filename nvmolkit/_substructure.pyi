from typing import Sequence

import numpy as np
from rdkit.Chem import Mol

class SubstructSearchConfig:
    batchSize: int
    workerThreads: int
    preprocessingThreads: int
    rdkitFallbackThreads: int
    slotsPerRunner: int
    presort: bool
    maxMatches: int
    uniquify: bool
    gpuIds: list[int]

def getSubstructMatches(
    targets: Sequence[Mol],
    queries: Sequence[Mol],
    config: SubstructSearchConfig = ...,
) -> list[list[list[list[int]]]]:
    """Perform batch substructure matching on GPU.

    Args:
        targets: List of target RDKit molecules
        queries: List of query RDKit molecules (typically from SMARTS)
        config: SubstructSearchConfig with execution settings

    Returns:
        Nested list: results[target_idx][query_idx] = list of matches,
        where each match is a list of target atom indices (one per query atom)
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

