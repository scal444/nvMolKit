# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest
from bench_utils.cli import add_autotune_cpu_budget_arg, add_backend_selection_args


@pytest.mark.parametrize("flag", ["--no-rdkit", "--no_rdkit", "--skip-rdkit"])
def test_backend_selection_accepts_common_and_legacy_aliases(flag):
    parser = argparse.ArgumentParser()
    add_backend_selection_args(parser, rdkit_dest="skip_rdkit", rdkit_aliases=("--skip-rdkit",))
    args = parser.parse_args([flag])
    assert args.skip_rdkit
    assert not args.no_nvmolkit


@pytest.mark.parametrize("flag", ["--no-nvmolkit", "--no_nvmolkit"])
def test_backend_selection_accepts_nvmolkit_spellings(flag):
    parser = argparse.ArgumentParser()
    add_backend_selection_args(parser)
    args = parser.parse_args([flag])
    assert args.no_nvmolkit
    assert not args.no_rdkit


def test_autotune_cpu_budget_accepts_positive_integer():
    parser = argparse.ArgumentParser()
    add_autotune_cpu_budget_arg(parser)

    assert parser.parse_args(["--autotune_cpu_budget", "14"]).autotune_cpu_budget == 14


@pytest.mark.parametrize("value", ["0", "-1"])
def test_autotune_cpu_budget_rejects_non_positive_integer(value):
    parser = argparse.ArgumentParser()
    add_autotune_cpu_budget_arg(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--autotune_cpu_budget", value])
