# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PAINS filtering benchmark comparing nvmolkit GPU substructure search against RDKit.

Compares three approaches:
  1. nvmolkit GPU-accelerated substructure search
  2. RDKit FilterCatalog (built-in PAINS filtering)
  3. RDKit SubstructMatch API directly

Usage:
    python pains_filter_bench.py --smiles <smiles_file> --smarts <smarts_file>

    # Skip nvmolkit (for CPU-only comparison):
    python pains_filter_bench.py --smiles <smiles_file> --smarts <smarts_file> --no_nvmolkit
"""

import argparse
import pickle
import sys
import time
from functools import partial
from typing import Callable

import nvtx
from rdkit import Chem
from tqdm.contrib.concurrent import process_map
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams


def time_it(func: Callable, runs: int = 1) -> tuple[float, float]:
    """Time a function and return (avg_ms, std_ms)."""
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append((end - start) / 1.0e6)
    avg_ms = sum(times) / len(times)
    std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5
    return avg_ms, std_ms


def load_pickle(filepath: str, max_count: int = 0) -> list[Chem.Mol]:
    """Load molecules from a pickled file containing binary mol data."""
    with open(filepath, "rb") as f:
        binary_mols = pickle.load(f)
    if max_count > 0:
        binary_mols = binary_mols[:max_count]
    mols = [Chem.Mol(b) for b in binary_mols]
    print(f"  Loaded {len(mols)} molecules from {filepath}")
    return mols


def _parse_smiles(smi: str, sanitize: bool) -> Chem.Mol | None:
    """Parse a single SMILES string."""
    return Chem.MolFromSmiles(smi, sanitize=sanitize)


def load_smiles(filepath: str, max_count: int = 0, sanitize: bool = True) -> list[Chem.Mol]:
    """Load and parse molecules from a SMILES file."""
    smiles_list = []
    with open(filepath, "r") as f:
        for line in f:
            if max_count > 0 and len(smiles_list) >= max_count:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smiles_list.append(line.split()[0])
    
    parse_func = partial(_parse_smiles, sanitize=sanitize)
    parsed = process_map(parse_func, smiles_list, desc="Parsing molecules", chunksize=1000)
    
    mols = []
    parse_failures = 0
    for mol in parsed:
        if mol is None:
            parse_failures += 1
        else:
            mols.append(mol)
    
    print(f"  Loaded {len(mols)} molecules from {filepath}")
    if parse_failures > 0:
        print(f"    ({parse_failures} parse failures)")
    return mols


def load_smarts(filepath: str, max_count: int = 0) -> tuple[list[Chem.Mol], list[str]]:
    """Load and parse query patterns from a SMARTS file."""
    queries = []
    smarts_list = []
    parse_failures = 0
    
    with open(filepath, "r") as f:
        for line in f:
            if max_count > 0 and len(queries) >= max_count:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smarts = line.split()[0]
            query = Chem.MolFromSmarts(smarts)
            if query is None:
                parse_failures += 1
                continue
            queries.append(query)
            smarts_list.append(smarts)
    
    print(f"  Loaded {len(queries)} SMARTS patterns from {filepath}")
    if parse_failures > 0:
        print(f"    ({parse_failures} parse failures)")
    return queries, smarts_list


def get_builtin_pains_catalog() -> FilterCatalog.FilterCatalog:
    """Get RDKit's built-in PAINS FilterCatalog."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    print(f"  Loaded PAINS FilterCatalog with {catalog.GetNumEntries()} filter entries")
    return catalog


def extract_patterns_from_catalog(catalog: FilterCatalog.FilterCatalog) -> tuple[list[Chem.Mol], list[str]]:
    """Extract query patterns from a FilterCatalog for direct SubstructMatch comparison."""
    queries = []
    smarts_list = []
    
    def extract_from_matcher(matcher, description: str):
        """Recursively extract SMARTS patterns from a filter matcher."""
        if hasattr(matcher, "GetPattern"):
            pattern = matcher.GetPattern()
            if pattern is not None:
                queries.append(pattern)
                smarts_list.append(description)
        if hasattr(matcher, "GetMatchers"):
            for sub_matcher in matcher.GetMatchers():
                extract_from_matcher(sub_matcher, description)
    
    try:
        for i in range(catalog.GetNumEntries()):
            entry = catalog.GetEntryWithIdx(i)
            description = entry.GetDescription()
            
            if hasattr(entry, "GetNumFilters"):
                for j in range(entry.GetNumFilters()):
                    matcher = entry.GetFilter(j)
                    extract_from_matcher(matcher, description)
            elif hasattr(entry, "GetSmarts"):
                smarts = entry.GetSmarts()
                if smarts:
                    query = Chem.MolFromSmarts(smarts)
                    if query is not None:
                        queries.append(query)
                        smarts_list.append(description)
    except Exception as e:
        print(f"  Warning: Could not extract all patterns from catalog: {e}")
    
    print(f"  Extracted {len(queries)} SMARTS patterns from FilterCatalog")
    return queries, smarts_list


@nvtx.annotate("bench_rdkit_filter_catalog", color="blue")
def bench_rdkit_filter_catalog(
    mols: list[Chem.Mol], 
    catalog: FilterCatalog.FilterCatalog,
    runs: int
) -> tuple[float, float, list[bool]]:
    """Benchmark RDKit FilterCatalog for PAINS filtering."""
    flagged = []
    
    @nvtx.annotate("filter_catalog_run", color="cyan")
    def run():
        nonlocal flagged
        flagged = []
        for mol in mols:
            entry = catalog.GetFirstMatch(mol)
            flagged.append(entry is not None)
    
    avg_ms, std_ms = time_it(run, runs)
    return avg_ms, std_ms, flagged


@nvtx.annotate("bench_rdkit_substruct", color="green")
def bench_rdkit_substruct(
    mols: list[Chem.Mol], 
    queries: list[Chem.Mol], 
    runs: int
) -> tuple[float, float, list[bool]]:
    """Benchmark RDKit SubstructMatch API directly."""
    params = Chem.SubstructMatchParameters()
    params.uniquify = False
    
    flagged = []
    
    @nvtx.annotate("substruct_run", color="yellow")
    def run():
        nonlocal flagged
        flagged = []
        for mol in mols:
            is_flagged = False
            for query in queries:
                if mol.HasSubstructMatch(query, params):
                    is_flagged = True
                    break
            flagged.append(is_flagged)
    
    avg_ms, std_ms = time_it(run, runs)
    return avg_ms, std_ms, flagged


@nvtx.annotate("bench_nvmolkit", color="red")
def bench_nvmolkit(
    mols: list[Chem.Mol],
    queries: list[Chem.Mol],
    runs: int,
    config
) -> tuple[float, float, list[bool]]:
    """Benchmark nvmolkit GPU substructure search."""
    import torch
    from nvmolkit.substructure import hasSubstructMatch
    
    flagged = []
    
    @nvtx.annotate("nvmolkit_run", color="orange")
    def run():
        nonlocal flagged
        results = hasSubstructMatch(mols, queries, config)
        torch.cuda.synchronize()
        flagged = [any(query_matches) for query_matches in results]
    
    avg_ms, std_ms = time_it(run, runs)
    return avg_ms, std_ms, flagged


def main():
    parser = argparse.ArgumentParser(
        description="PAINS filtering benchmark: nvmolkit vs RDKit FilterCatalog vs RDKit SubstructMatch"
    )
    parser.add_argument("--smiles", "-s", help="Path to SMILES file with molecules to filter")
    parser.add_argument("--pickle", help="Path to pickled molecules file (alternative to --smiles)")
    parser.add_argument("--smarts", "-q", required=True, help="Path to SMARTS file with filter patterns")
    parser.add_argument("--num_mols", "-n", type=int, default=0, help="Max number of molecules (default: 0 = all)")
    parser.add_argument("--sanitize", action="store_true", dest="sanitize", help="Sanitize SMILES during parsing")
    parser.add_argument("--no_sanitize", action="store_false", dest="sanitize", help="Skip sanitization (preprocessed SMILES)")
    parser.set_defaults(sanitize=False)
    parser.add_argument("--runs", "-r", type=int, default=1, help="Number of timing runs (default: 1)")
    parser.add_argument("--no_nvmolkit", action="store_true", help="Skip nvmolkit benchmark")
    parser.add_argument("--no_filter_catalog", action="store_true", help="Skip RDKit FilterCatalog benchmark")
    parser.add_argument("--no_substruct", action="store_true", help="Skip RDKit SubstructMatch benchmark")
    parser.add_argument("--batch_size", "-b", type=int, default=1024, help="nvmolkit batch size (default: 1024)")
    parser.add_argument("--workers", "-p", type=int, default=2, help="nvmolkit worker threads (default: 2)")
    
    args = parser.parse_args()
    
    if not args.smiles and not args.pickle:
        print("Error: Either --smiles or --pickle is required")
        sys.exit(1)
    
    if args.smiles and args.pickle:
        print("Error: Cannot specify both --smiles and --pickle")
        sys.exit(1)
    
    print("\nConfiguration:")
    print(f"  Input file: {args.smiles or args.pickle} ({'pickle' if args.pickle else 'smiles'})")
    print(f"  SMARTS file: {args.smarts}")
    print(f"  Max molecules: {args.num_mols if args.num_mols > 0 else 'all'}")
    print(f"  Runs: {args.runs}")
    print(f"  Run nvmolkit: {not args.no_nvmolkit}")
    print(f"  Run FilterCatalog: {not args.no_filter_catalog}")
    print(f"  Run SubstructMatch: {not args.no_substruct}")
    
    print("\nLoading molecules...")
    if args.pickle:
        mols = load_pickle(args.pickle, args.num_mols)
    else:
        mols = load_smiles(args.smiles, args.num_mols, args.sanitize)
    
    if len(mols) == 0:
        print("Error: No valid molecules loaded")
        sys.exit(1)
    
    print("\nLoading SMARTS patterns...")
    queries, _ = load_smarts(args.smarts)
    if len(queries) == 0:
        print("Error: No valid SMARTS patterns loaded from file")
        sys.exit(1)
    
    catalog = None
    if not args.no_filter_catalog:
        print("\nLoading builtin PAINS FilterCatalog...")
        catalog = get_builtin_pains_catalog()
    
    num_patterns = len(queries)
    print(f"\nBenchmarking PAINS filtering: {len(mols)} molecules × {num_patterns} patterns")
    print("=" * 70)
    
    results = {}
    
    if not args.no_nvmolkit:
        try:
            from nvmolkit.substructure import SubstructSearchConfig, hasSubstructMatch
            import torch
            
            config = SubstructSearchConfig()
            config.batchSize = args.batch_size
            config.workerThreads = args.workers
            config.presort = True
            
            print("\nWarming up nvmolkit...")
            warmup_mols = mols[:10]
            with nvtx.annotate("nvmolkit_warmup", color="purple"):
                hasSubstructMatch(warmup_mols, queries, config)
                torch.cuda.synchronize()
            
            print("Running nvmolkit GPU benchmark...")
            nvmolkit_avg, nvmolkit_std, nvmolkit_flagged = bench_nvmolkit(
                mols, queries, args.runs, config
            )
            nvmolkit_hits = sum(nvmolkit_flagged)
            print(f"  nvmolkit:        {nvmolkit_avg:10.2f} ms (± {nvmolkit_std:.2f} ms), {nvmolkit_hits} flagged")
            results["nvmolkit"] = (nvmolkit_avg, nvmolkit_std, nvmolkit_flagged)
        except ImportError as e:
            print(f"  nvmolkit: SKIPPED (import error: {e})")
    
    if not args.no_filter_catalog:
        print("\nRunning RDKit FilterCatalog benchmark...")
        fc_avg, fc_std, fc_flagged = bench_rdkit_filter_catalog(mols, catalog, args.runs)
        fc_hits = sum(fc_flagged)
        print(f"  FilterCatalog:   {fc_avg:10.2f} ms (± {fc_std:.2f} ms), {fc_hits} flagged")
        results["filter_catalog"] = (fc_avg, fc_std, fc_flagged)
    
    if not args.no_substruct:
        print("\nRunning RDKit SubstructMatch benchmark...")
        ss_avg, ss_std, ss_flagged = bench_rdkit_substruct(mols, queries, args.runs)
        ss_hits = sum(ss_flagged)
        print(f"  SubstructMatch:  {ss_avg:10.2f} ms (± {ss_std:.2f} ms), {ss_hits} flagged")
        results["substruct"] = (ss_avg, ss_std, ss_flagged)
    
    print("\n" + "=" * 70)
    print("Summary:")
    
    if not results:
        print("  No benchmarks were run!")
        sys.exit(1)
    
    baseline = None
    if "substruct" in results:
        baseline = ("RDKit SubstructMatch", results["substruct"][0])
    elif "filter_catalog" in results:
        baseline = ("RDKit FilterCatalog", results["filter_catalog"][0])
    
    for name, (avg_ms, std_ms, flagged) in results.items():
        hits = sum(flagged)
        speedup_str = ""
        if baseline and name != baseline[0].lower().replace(" ", "_"):
            speedup = baseline[1] / avg_ms if avg_ms > 0 else 0
            speedup_str = f", {speedup:.1f}x vs {baseline[0]}"
        print(f"  {name:20s}: {avg_ms:10.2f} ms (± {std_ms:.2f} ms), {hits:5d} flagged{speedup_str}")
    
    ref_flagged = None
    ref_name = None
    for name in ["filter_catalog", "substruct"]:
        if name in results:
            ref_flagged = results[name][2]
            ref_name = name
            break
    
    if ref_flagged is not None:
        print("\nValidation:")
        for name, (_, _, flagged) in results.items():
            if name == ref_name:
                continue
            matches = sum(1 for a, b in zip(flagged, ref_flagged) if a == b)
            pct = 100.0 * matches / len(flagged) if flagged else 0
            diff = sum(1 for a, b in zip(flagged, ref_flagged) if a != b)
            print(f"  {name} vs {ref_name}: {matches}/{len(flagged)} ({pct:.1f}%) agree, {diff} differ")
    
    print("\n\nCSV Results:")
    print("method,num_mols,num_patterns,time_ms,std_ms,flagged")
    for name, (avg_ms, std_ms, flagged) in results.items():
        print(f"{name},{len(mols)},{num_patterns},{avg_ms:.2f},{std_ms:.2f},{sum(flagged)}")


if __name__ == "__main__":
    main()

