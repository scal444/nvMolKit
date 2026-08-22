class NativePrecisionOptions:
    mode: str
    forcefieldParameterStorage: str
    forcefieldCoordinateStorage: str
    forcefieldGradientStorage: str
    hessianStorage: str
    minimizerStateStorage: str
    forcefieldCompute: str
    minimizerCompute: str
    reductionCompute: str
    floatMath: str

    def __init__(self) -> None: ...


class FireOptions:
    dtInit: float
    dtMinFactor: float
    dtMaxFactor: float
    dMax: float
    timeStepIncrement: float
    timeStepDecrement: float
    nMinForIncrease: int
    alphaInit: float
    alphaDecrement: float
    useMass: bool
    gradTol: float
    takeHalfStepBack: bool
    abcCorrection: bool
    stuckDetectionEnabled: bool
    stuckEnergyRelTol: float
    stuckStreakLength: int
    stuckEvalEveryNPolls: int

    def __init__(self) -> None: ...
