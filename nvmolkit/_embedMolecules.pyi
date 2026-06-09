from typing import Any, List, Optional
from rdkit.Chem import Mol
from rdkit.Chem.rdDistGeom import EmbedParameters

class BatchHardwareOptions: ...

def EmbedMolecules(
    molecules: List[Mol],
    params: EmbedParameters,
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: Optional[BatchHardwareOptions] = ...,
) -> None: ...

def EmbedMoleculesDevice(
    molecules: List[Mol],
    params: EmbedParameters,
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    hardwareOptions: Optional[BatchHardwareOptions] = ...,
    targetGpu: int = -1,
) -> Any: ...
