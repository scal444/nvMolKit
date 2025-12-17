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

#include "substruct_validation.h"

#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <iostream>
#include <set>

namespace nvMolKit {

std::vector<std::vector<int>> getRDKitSubstructMatches(const RDKit::ROMol& target,
                                                       const RDKit::ROMol& query,
                                                       bool                uniquify) {
  RDKit::SubstructMatchParameters params;
  params.uniquify = uniquify;

  std::vector<RDKit::MatchVectType> matches = RDKit::SubstructMatch(target, query, params);

  std::vector<std::vector<int>> result;
  result.reserve(matches.size());

  for (const auto& match : matches) {
    std::vector<int> mapping(match.size());
    for (size_t i = 0; i < match.size(); ++i) {
      mapping[match[i].first] = match[i].second;
    }
    result.push_back(std::move(mapping));
  }

  return result;
}

std::string algorithmName(SubstructAlgorithm algo) {
  switch (algo) {
    case SubstructAlgorithm::VF2:
      return "VF2";
    case SubstructAlgorithm::GSI:
      return "GSI";
    case SubstructAlgorithm::WarpUnified:
      return "WarpUnified";
  }
  return "Unknown";
}

namespace {

/**
 * @brief Extract GPU matches for a (target, query) pair from results.
 */
std::vector<std::vector<int>> extractGpuMatches(const SubstructMatchResultsHost& results,
                                                int                              targetIdx,
                                                int                              queryIdx,
                                                int                              numQueryAtoms) {
  const int pairIdx       = results.pairIndex(targetIdx, queryIdx);
  const int reportedCount = results.reportedCounts[pairIdx];
  const int startOffset   = results.pairMatchStarts[pairIdx];

  std::vector<std::vector<int>> gpuMatches;
  gpuMatches.reserve(reportedCount);

  for (int m = 0; m < reportedCount; ++m) {
    std::vector<int> mapping(numQueryAtoms);
    for (int a = 0; a < numQueryAtoms; ++a) {
      mapping[a] = results.matchIndices[startOffset + m * numQueryAtoms + a];
    }
    gpuMatches.push_back(std::move(mapping));
  }

  return gpuMatches;
}

/**
 * @brief Compare two sets of matches (order-independent).
 */
bool matchSetsEqual(const std::vector<std::vector<int>>& gpuMatches,
                    const std::vector<std::vector<int>>& rdkitMatches) {
  if (gpuMatches.size() != rdkitMatches.size()) {
    return false;
  }

  std::set<std::vector<int>> gpuSet(gpuMatches.begin(), gpuMatches.end());
  std::set<std::vector<int>> rdkitSet(rdkitMatches.begin(), rdkitMatches.end());

  return gpuSet == rdkitSet;
}

}  // namespace

SubstructValidationResult validateAgainstRDKit(
  const SubstructMatchResultsHost&                  results,
  const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
  const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols) {
  SubstructValidationResult validation;
  validation.totalPairs = results.numTargets * results.numQueries;

  for (int t = 0; t < results.numTargets; ++t) {
    for (int q = 0; q < results.numQueries; ++q) {
      const auto rdkitMatches    = getRDKitSubstructMatches(*targetMols[t], *queryMols[q], false);
      const int  pairIdx         = results.pairIndex(t, q);
      const int  gpuMatchCount   = results.matchCounts[pairIdx];
      const int  rdkitMatchCount = static_cast<int>(rdkitMatches.size());

      if (results.hasOverflow(t, q)) {
        validation.overflowPairs++;
        validation.hasOverflows = true;
      }

      if (gpuMatchCount != rdkitMatchCount) {
        validation.mismatchedPairs++;
        validation.mismatches.emplace_back(t, q, gpuMatchCount, rdkitMatchCount);
      } else if (gpuMatchCount > 0 && !results.hasOverflow(t, q)) {
        const int  numQueryAtoms = static_cast<int>(queryMols[q]->getNumAtoms());
        const auto gpuMatches    = extractGpuMatches(results, t, q, numQueryAtoms);

        if (matchSetsEqual(gpuMatches, rdkitMatches)) {
          validation.matchingPairs++;
        } else {
          validation.wrongMappingPairs++;
          validation.mappingMismatches.emplace_back(t, q);
        }
      } else {
        validation.matchingPairs++;
      }
    }
  }

  validation.allMatch = (validation.mismatchedPairs == 0 && validation.wrongMappingPairs == 0);
  return validation;
}

void printValidationResult(const SubstructValidationResult& result, const std::string& algoName) {
  std::string prefix = algoName.empty() ? "" : "[" + algoName + "] ";

  std::cout << prefix << "Validation: " << result.matchingPairs << "/" << result.totalPairs
            << " pairs match RDKit";

  if (result.overflowPairs > 0) {
    std::cout << " (" << result.overflowPairs << " overflow)";
  }

  if (result.allMatch) {
    std::cout << " - PASS" << std::endl;
  } else {
    const int totalFailures = result.mismatchedPairs + result.wrongMappingPairs;
    std::cout << " - FAIL (" << totalFailures << " failures)" << std::endl;

    const int maxPrint = 10;
    int       printed  = 0;

    if (result.mismatchedPairs > 0) {
      std::cout << "  Count mismatches:" << std::endl;
      for (const auto& [t, q, gpu, rdkit] : result.mismatches) {
        if (printed++ >= maxPrint) {
          std::cout << "    ... and " << (result.mismatchedPairs - maxPrint) << " more" << std::endl;
          break;
        }
        std::cout << "    target=" << t << " query=" << q << ": GPU=" << gpu << " RDKit=" << rdkit
                  << std::endl;
      }
    }

    if (result.wrongMappingPairs > 0) {
      std::cout << "  Mapping mismatches (count correct but indices differ):" << std::endl;
      printed = 0;
      for (const auto& [t, q] : result.mappingMismatches) {
        if (printed++ >= maxPrint) {
          std::cout << "    ... and " << (result.wrongMappingPairs - maxPrint) << " more" << std::endl;
          break;
        }
        std::cout << "    target=" << t << " query=" << q << std::endl;
      }
    }
  }
}

}  // namespace nvMolKit

