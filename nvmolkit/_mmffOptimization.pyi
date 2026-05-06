from typing import Any, List
from rdkit.Chem import Mol

def MMFFOptimizeMoleculesConfs(
    molecules: List[Mol],
    maxIters: int = 200,
    properties: Any = None,
    hardwareOptions: Any = None,
    backend: str = "HYBRID",
) -> List[List[float]]: ...

def MMFFOptimizeMoleculesConfsFire(
    molecules: List[Mol],
    maxIters: int = 200,
    fireOptions: Any = None,
    properties: Any = None,
    hardwareOptions: Any = None,
    fireDebugOutput: Any = None,
    backend: str = "HYBRID",
) -> List[List[float]]: ...
