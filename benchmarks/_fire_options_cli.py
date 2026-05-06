# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI helpers for exposing the full ``nvMolKit::FireOptions`` surface.

Benchmarks that drive the FIRE minimizer (MMFF or ETKDG) should use these helpers
so that every ``FireOptions`` field can be configured from the command line and
all benchmarks default to the library defaults (defined in
``src/minimizer/fire_minimizer.h``). Defaults of ``None`` here mean ``leave the
default from the C++ struct``.
"""

from __future__ import annotations

import argparse

from nvmolkit.mmffOptimization import FireOptions


def add_fire_options_args(parser: argparse.ArgumentParser, prefix: str = "fire") -> None:
    """Add one CLI argument per field of ``FireOptions``.

    Each argument's default is ``None``; ``fire_options_from_args`` only assigns
    the field on the returned ``FireOptions`` instance when the user supplied a
    value, so the C++-side default is preserved otherwise.

    Args:
        parser: argparse parser to extend.
        prefix: option prefix. Defaults to ``fire``, producing flags like
            ``--fire-dt-init``.
    """
    group = parser.add_argument_group(f"{prefix} options")
    pfx = f"--{prefix}"
    group.add_argument(f"{pfx}-dt-init", type=float, default=None, dest=f"{prefix}_dtInit")
    group.add_argument(f"{pfx}-dt-min-factor", type=float, default=None, dest=f"{prefix}_dtMinFactor")
    group.add_argument(f"{pfx}-dt-max-factor", type=float, default=None, dest=f"{prefix}_dtMaxFactor")
    group.add_argument(f"{pfx}-d-max", type=float, default=None, dest=f"{prefix}_dMax")
    group.add_argument(f"{pfx}-time-step-increment", type=float, default=None, dest=f"{prefix}_timeStepIncrement")
    group.add_argument(f"{pfx}-time-step-decrement", type=float, default=None, dest=f"{prefix}_timeStepDecrement")
    group.add_argument(f"{pfx}-n-min-for-increase", type=int, default=None, dest=f"{prefix}_nMinForIncrease")
    group.add_argument(f"{pfx}-alpha-init", type=float, default=None, dest=f"{prefix}_alphaInit")
    group.add_argument(f"{pfx}-alpha-decrement", type=float, default=None, dest=f"{prefix}_alphaDecrement")
    group.add_argument(f"{pfx}-grad-tol", type=float, default=None, dest=f"{prefix}_gradTol")
    group.add_argument(
        f"{pfx}-use-mass",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest=f"{prefix}_useMass",
        help="Mass-weight the FIRE integrator (--no-fire-use-mass to disable).",
    )
    group.add_argument(
        f"{pfx}-take-half-step-back",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest=f"{prefix}_takeHalfStepBack",
    )
    group.add_argument(
        f"{pfx}-abc-correction", action=argparse.BooleanOptionalAction, default=None, dest=f"{prefix}_abcCorrection"
    )
    group.add_argument(
        f"{pfx}-stuck-detection-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest=f"{prefix}_stuckDetectionEnabled",
    )
    group.add_argument(f"{pfx}-stuck-energy-rel-tol", type=float, default=None, dest=f"{prefix}_stuckEnergyRelTol")
    group.add_argument(f"{pfx}-stuck-streak-length", type=int, default=None, dest=f"{prefix}_stuckStreakLength")
    group.add_argument(
        f"{pfx}-stuck-eval-every-n-polls", type=int, default=None, dest=f"{prefix}_stuckEvalEveryNPolls"
    )


_FIELDS = (
    "dtInit",
    "dtMinFactor",
    "dtMaxFactor",
    "dMax",
    "timeStepIncrement",
    "timeStepDecrement",
    "nMinForIncrease",
    "alphaInit",
    "alphaDecrement",
    "gradTol",
    "useMass",
    "takeHalfStepBack",
    "abcCorrection",
    "stuckDetectionEnabled",
    "stuckEnergyRelTol",
    "stuckStreakLength",
    "stuckEvalEveryNPolls",
)


def fire_options_from_args(args: argparse.Namespace, prefix: str = "fire") -> FireOptions:
    """Build a ``FireOptions`` from a Namespace populated by ``add_fire_options_args``.

    Fields the user did not specify retain their C++-side defaults.
    """
    options = FireOptions()
    for field in _FIELDS:
        value = getattr(args, f"{prefix}_{field}", None)
        if value is not None:
            setattr(options, field, value)
    return options
