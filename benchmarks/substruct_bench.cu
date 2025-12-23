// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <getopt.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "benchmark_utils.h"
#include "cuda_error_check.h"
#include "device.h"
#include "substruct/substructure_search.cuh"
#include "testutils/substruct_validation.h"

using nvMolKit::addQueryToBatch;
using nvMolKit::addToBatch;
using nvMolKit::algorithmName;
using nvMolKit::checkReturnCode;
using nvMolKit::getRDKitSubstructMatches;
using nvMolKit::getSubstructMatches;
using nvMolKit::MoleculesDevice;
using nvMolKit::MoleculesHost;
using nvMolKit::printValidationResult;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructAlgorithm;
using nvMolKit::SubstructMatchResultsHost;
using nvMolKit::SubstructValidationResult;
using nvMolKit::validateAgainstRDKit;

namespace {

std::unique_ptr<RDKit::ROMol> makeMolFromSmiles(const std::string& smiles) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
}

std::unique_ptr<RDKit::ROMol> makeMolFromSmarts(const std::string& smarts) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
}

/**
 * @brief Read SMILES strings from a file.
 *
 * Supports .smi, .smiles, .cxsmiles file formats.
 * Lines starting with # are treated as comments.
 */
std::vector<std::string> readSmilesFile(const std::string& filePath, unsigned int maxCount) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open SMILES file: " + filePath);
  }

  std::vector<std::string> smilesList;
  std::string              line;

  while (std::getline(file, line) && smilesList.size() < maxCount) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::string smiles = line.substr(0, line.find_first_of(" \t"));
    if (!smiles.empty()) {
      smilesList.push_back(smiles);
    }
  }

  if (smilesList.empty()) {
    throw std::runtime_error("No valid SMILES found in file: " + filePath);
  }

  return smilesList;
}

/**
 * @brief Parse molecules from SMILES/SMARTS strings.
 * @param smilesList List of SMILES strings
 * @param asQuery If true, parse as SMARTS; otherwise parse as SMILES
 * @return Vector of parsed molecules
 */
std::vector<std::unique_ptr<RDKit::ROMol>> parseMolecules(const std::vector<std::string>& smilesList,
                                                          bool                            asQuery) {
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  mols.reserve(smilesList.size());

  for (const auto& smi : smilesList) {
    auto mol = asQuery ? makeMolFromSmarts(smi) : makeMolFromSmiles(smi);
    if (mol) {
      mols.push_back(std::move(mol));
    } else {
      std::cerr << "Warning: Failed to parse " << (asQuery ? "SMARTS" : "SMILES") << ": " << smi
                << std::endl;
    }
  }

  return mols;
}

/**
 * @brief Build host-side molecule batches for GPU processing.
 */
void buildBatches(const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                  const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                  MoleculesHost&                                    targetsHost,
                  MoleculesHost&                                    queriesHost) {
  for (const auto& mol : targetMols) {
    addToBatch(mol.get(), targetsHost);
  }
  for (const auto& mol : queryMols) {
    addQueryToBatch(mol.get(), queriesHost);
  }
}

/**
 * @brief Benchmark RDKit substructure matching.
 */
void benchRDKit(const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                int&                                              totalMatches,
                BenchUtils::TimingResult&                         timingOut) {
  totalMatches = 0;

  timingOut = BenchUtils::timeIt(
    [&]() {
      RDKit::SubstructMatchParameters params;
      params.uniquify = false;
      int              localMatches = 0;

      for (const auto& target : targetMols) {
        for (const auto& query : queryMols) {
          auto matches = RDKit::SubstructMatch(*target, *query, params);
          localMatches += static_cast<int>(matches.size());
        }
      }
      totalMatches = localMatches;
    },
    3, 1);

  std::cout << "RDKit SubstructMatch, targets=" << targetMols.size() << ", queries=" << queryMols.size()
            << ": " << timingOut.avgMs << " ms (±" << timingOut.stdMs << " ms)\n";
}

/**
 * @brief Benchmark nvMolKit GPU substructure matching.
 */
void benchNvMolKit(const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                   const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                   SubstructAlgorithm                                algorithm,
                   int                                               numThreads,
                   int&                                              totalMatches,
                   SubstructMatchResultsHost&                        resultsOut,
                   BenchUtils::TimingResult&                         timingOut) {
  std::string algoStr = algorithmName(algorithm);

  ScopedStream stream;

  MoleculesHost targetsHost;
  MoleculesHost queriesHost;
  buildBatches(targetMols, queryMols, targetsHost, queriesHost);

  MoleculesDevice targetsDevice(stream.stream());
  MoleculesDevice queriesDevice(stream.stream());
  targetsDevice.copyFromHost(targetsHost);
  queriesDevice.copyFromHost(queriesHost);

  timingOut = BenchUtils::timeIt(
    [&]() {
      getSubstructMatches(targetsDevice, queriesDevice, targetsHost, queriesHost, resultsOut, algorithm,
                          stream.stream(), 1024, numThreads);
    },
    3, 1);

  totalMatches = 0;
  for (int count : resultsOut.matchCounts) {
    totalMatches += count;
  }

  std::cout << "nvMolKit SubstructMatch (" << algoStr << "), targets=" << targetMols.size()
            << ", queries=" << queryMols.size() << ": " << timingOut.avgMs << " ms (±" << timingOut.stdMs
            << " ms)\n";
}

bool parseBoolArg(const std::string& arg) {
  std::string s = arg;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

SubstructAlgorithm parseAlgorithmArg(const std::string& arg) {
  std::string s = arg;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s == "vf2" || s == "0") {
    return SubstructAlgorithm::VF2;
  } else if (s == "gsi" || s == "1") {
    return SubstructAlgorithm::GSI;
  } else if (s == "warpunified" || s == "warp" || s == "2") {
    return SubstructAlgorithm::WarpUnified;
  } else {
    throw std::runtime_error("Invalid algorithm. Use 'vf2', 'gsi', or 'warpunified'");
  }
}

void printHelp(const char* progName) {
  std::cout << "Usage: " << progName << " [options]\n\n";
  std::cout << "Substructure matching benchmark comparing nvMolKit GPU algorithms to RDKit.\n\n";
  std::cout << "Options:\n";
  std::cout << "  -t, --targets <path>      Path to targets SMILES file [required]\n";
  std::cout << "  -q, --queries <path>      Path to queries SMARTS file [required]\n";
  std::cout << "  -n, --num_targets <int>   Max number of target molecules [default: 100]\n";
  std::cout << "  -m, --num_queries <int>   Max number of query molecules [default: 10]\n";
  std::cout
    << "  -a, --algorithm <str>     Algorithm: vf2, gsi, or warpunified [default: warpunified]\n";
  std::cout << "  -p, --num_threads <int>   CPU worker threads used by nvMolKit [default: 2]\n";
  std::cout << "  -r, --do_rdkit <bool>     Run RDKit benchmark comparison [default: true]\n";
  std::cout << "  -w, --do_warmup <bool>    Run warmup before benchmarking [default: true]\n";
  std::cout << "  -v, --validate <bool>     Validate GPU results against RDKit [default: false]\n";
  std::cout << "  -h, --help                Show this help message\n\n";
  std::cout << "Boolean values can be: true/false, 1/0, yes/no, on/off (case insensitive)\n";
  std::cout << "\nExamples:\n";
  std::cout << "  " << progName
            << " --targets targets.smi --queries queries.smi --num_targets 1000 --algorithm gsi\n";
  std::cout << "  " << progName << " -t targets.smi -q queries.smi -n 500 -m 20 -v true\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string        targetsPath;
  std::string        queriesPath;
  int                numTargets = 100;
  int                numQueries = 10;
  SubstructAlgorithm algorithm  = SubstructAlgorithm::WarpUnified;
  int                numThreads = 2;
  bool               doRdkit    = true;
  bool               doWarmup   = true;
  bool               doValidate = false;

  static struct option long_options[] = {
    {   "targets", required_argument, 0, 't'},
    {   "queries", required_argument, 0, 'q'},
    {"num_targets", required_argument, 0, 'n'},
    {"num_queries", required_argument, 0, 'm'},
    { "algorithm", required_argument, 0, 'a'},
    {"num_threads", required_argument, 0, 'p'},
    {  "do_rdkit", required_argument, 0, 'r'},
    { "do_warmup", required_argument, 0, 'w'},
    {  "validate", required_argument, 0, 'v'},
    {      "help",       no_argument, 0, 'h'},
    {           0,                 0, 0,   0}
  };

  int option_index = 0;
  int c;

  while ((c = getopt_long(argc, argv, "t:q:n:m:a:p:r:w:v:h", long_options, &option_index)) != -1) {
    switch (c) {
      case 't':
        targetsPath = optarg;
        break;
      case 'q':
        queriesPath = optarg;
        break;
      case 'n':
        try {
          numTargets = std::stoi(optarg);
          if (numTargets <= 0) {
            std::cerr << "Error: num_targets must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for num_targets: " << optarg << "\n";
          return 1;
        }
        break;
      case 'm':
        try {
          numQueries = std::stoi(optarg);
          if (numQueries <= 0) {
            std::cerr << "Error: num_queries must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for num_queries: " << optarg << "\n";
          return 1;
        }
        break;
      case 'a':
        try {
          algorithm = parseAlgorithmArg(optarg);
        } catch (const std::exception& e) {
          std::cerr << "Error: " << e.what() << "\n";
          return 1;
        }
        break;
      case 'p':
        try {
          numThreads = std::stoi(optarg);
          if (numThreads <= 0) {
            std::cerr << "Error: num_threads must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for num_threads: " << optarg << "\n";
          return 1;
        }
        break;
      case 'r':
        doRdkit = parseBoolArg(optarg);
        break;
      case 'w':
        doWarmup = parseBoolArg(optarg);
        break;
      case 'v':
        doValidate = parseBoolArg(optarg);
        break;
      case 'h':
        printHelp(argv[0]);
        return 0;
      case '?':
        std::cerr << "\nUse --help for usage information.\n";
        return 1;
      default:
        std::cerr << "Unknown option\n";
        return 1;
    }
  }

  if (optind < argc) {
    std::cerr << "Error: Unexpected non-option arguments: ";
    while (optind < argc) {
      std::cerr << argv[optind++] << " ";
    }
    std::cerr << "\nUse --help for usage information.\n";
    return 1;
  }

  if (targetsPath.empty() || queriesPath.empty()) {
    std::cerr << "Error: Both --targets and --queries are required.\n";
    std::cerr << "Use --help for usage information.\n";
    return 1;
  }

  if (!std::filesystem::exists(targetsPath)) {
    std::cerr << "Error: Targets file does not exist: " << targetsPath << std::endl;
    return 1;
  }
  if (!std::filesystem::exists(queriesPath)) {
    std::cerr << "Error: Queries file does not exist: " << queriesPath << std::endl;
    return 1;
  }

  std::cout << "Configuration:\n";
  std::cout << "  Targets file: " << targetsPath << "\n";
  std::cout << "  Queries file: " << queriesPath << "\n";
  std::cout << "  Max targets: " << numTargets << "\n";
  std::cout << "  Max queries: " << numQueries << "\n";
  std::cout << "  Algorithm: " << algorithmName(algorithm) << "\n";
  std::cout << "  nvMolKit threads: " << numThreads << "\n";
  std::cout << "  Run RDKit comparison: " << (doRdkit ? "yes" : "no") << "\n";
  std::cout << "  Run warmup: " << (doWarmup ? "yes" : "no") << "\n";
  std::cout << "  Validate results: " << (doValidate ? "yes" : "no") << "\n\n";

  std::vector<std::string> targetSmiles = readSmilesFile(targetsPath, numTargets);
  std::vector<std::string> querySmiles  = readSmilesFile(queriesPath, numQueries);

  std::cout << "Loaded " << targetSmiles.size() << " targets and " << querySmiles.size()
            << " queries\n\n";

  auto targetMols = parseMolecules(targetSmiles, false);
  auto queryMols  = parseMolecules(querySmiles, true);

  if (targetMols.empty()) {
    std::cerr << "Error: No valid target molecules parsed\n";
    return 1;
  }
  if (queryMols.empty()) {
    std::cerr << "Error: No valid query molecules parsed\n";
    return 1;
  }

  if (static_cast<int>(targetMols.size()) < numTargets) {
    std::cerr << "Error: Requested " << numTargets << " targets but only " << targetMols.size()
              << " valid targets available in file\n";
    return 1;
  }

  if (static_cast<int>(queryMols.size()) < numQueries) {
    std::cout << "Note: Requested " << numQueries << " queries but only " << queryMols.size()
              << " available. Duplicating to reach " << numQueries << " queries.\n";
    const size_t originalSize = queryMols.size();
    for (int i = static_cast<int>(originalSize); i < numQueries; ++i) {
      queryMols.push_back(std::make_unique<RDKit::ROMol>(*queryMols[i % originalSize]));
    }
  }

  std::cout << "Parsed " << targetMols.size() << " targets and " << queryMols.size()
            << " queries\n\n";

  if (doWarmup) {
    std::cout << "Warming up...\n";

    std::vector<std::unique_ptr<RDKit::ROMol>> warmupTargets;
    std::vector<std::unique_ptr<RDKit::ROMol>> warmupQueries;
    warmupTargets.push_back(makeMolFromSmiles("CCO"));
    warmupQueries.push_back(makeMolFromSmarts("C"));

    int                       warmupMatches;
    SubstructMatchResultsHost warmupResults;
    BenchUtils::TimingResult  warmupTiming;
    benchNvMolKit(warmupTargets, warmupQueries, algorithm, numThreads, warmupMatches, warmupResults, warmupTiming);

    if (doRdkit) {
      BenchUtils::TimingResult rdkitWarmupTiming;
      benchRDKit(warmupTargets, warmupQueries, warmupMatches, rdkitWarmupTiming);
    }

    std::cout << "Warmed up\n\n";
  }

  int                       nvmolkitMatches = 0;
  SubstructMatchResultsHost nvmolkitResults;
  BenchUtils::TimingResult  nvmolkitTiming;
  benchNvMolKit(targetMols, queryMols, algorithm, numThreads, nvmolkitMatches, nvmolkitResults, nvmolkitTiming);
  std::cout << "nvMolKit total matches: " << nvmolkitMatches << "\n";

  int                      rdkitMatches = 0;
  BenchUtils::TimingResult rdkitTiming{0.0, 0.0};

  if (doRdkit) {
    benchRDKit(targetMols, queryMols, rdkitMatches, rdkitTiming);
    std::cout << "RDKit total matches: " << rdkitMatches << "\n";

    if (nvmolkitMatches == rdkitMatches) {
      std::cout << "Match counts EQUAL\n";
    } else {
      std::cout << "Match counts DIFFER by " << std::abs(nvmolkitMatches - rdkitMatches) << "\n";
    }

    std::cout << "\nSpeedup: " << (rdkitTiming.avgMs / nvmolkitTiming.avgMs) << "x\n";
  }

  if (doValidate) {
    std::cout << "\nValidating against RDKit (per-pair comparison)...\n";
    auto validation = validateAgainstRDKit(nvmolkitResults, targetMols, queryMols);
    printValidationResult(validation, algorithmName(algorithm));
  }

  std::cout << "\n\nCSV Results:\n";
  std::cout << "algorithm,num_targets,num_queries,num_threads,nvmolkit_time_ms,nvmolkit_std_ms";
  if (doRdkit) {
    std::cout << ",rdkit_time_ms,rdkit_std_ms";
  }
  std::cout << "\n";

  std::cout << algorithmName(algorithm) << "," << targetMols.size() << "," << queryMols.size() << ","
            << numThreads << ","
            << nvmolkitTiming.avgMs << "," << nvmolkitTiming.stdMs;
  if (doRdkit) {
    std::cout << "," << rdkitTiming.avgMs << "," << rdkitTiming.stdMs;
  }
  std::cout << "\n";

  return 0;
}

