from typing import Any, List
from rdkit.Chem import Mol

def MMFFOptimizeMoleculesConfs(
    molecules: List[Mol],
    maxIters: int = 200,
    nonBondedThreshold: float = 100.0,
    hardwareOptions: Any = None,
    optimizerBackend: str = "BFGS",
    optimizerOptions: dict[str, Any] | None = None,
) -> List[List[float]]: ...
