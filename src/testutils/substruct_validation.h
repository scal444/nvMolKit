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

#pragma once

#include <GraphMol/ROMol.h>

#include <memory>
#include <string>
#include <vector>

#include "substruct/substruct_types.h"

namespace nvMolKit {

/**
 * @brief Get substructure matches using RDKit.
 *
 * @param target Target molecule
 * @param query Query molecule (typically from SMARTS)
 * @param uniquify If true, return only unique matches
 * @return Vector of matches, where each match is a vector of target atom indices
 *         indexed by query atom position
 */
std::vector<std::vector<int>> getRDKitSubstructMatches(const RDKit::ROMol& target,
                                                       const RDKit::ROMol& query,
                                                       bool                uniquify = true);

/**
 * @brief Get a string name for a substruct algorithm.
 * @param algo The algorithm enum value
 * @return Human-readable name for the algorithm
 */
std::string algorithmName(SubstructAlgorithm algo);

/**
 * @brief Validation results from comparing GPU matches to RDKit ground truth.
 */
struct SubstructValidationResult {
  int  totalPairs       = 0;
  int  matchingPairs    = 0;
  int  mismatchedPairs  = 0;
  int  overflowPairs    = 0;
  bool allMatch         = false;
  bool hasOverflows     = false;

  /// Details about mismatches: (targetIdx, queryIdx, gpuCount, rdkitCount)
  std::vector<std::tuple<int, int, int, int>> mismatches;
};

/**
 * @brief Compare GPU substructure match results against RDKit ground truth.
 *
 * Uses uniquify=false to match our non-uniquifying GPU algorithm.
 *
 * @param results GPU results from getSubstructMatches
 * @param targetMols Target molecules (parallel to results.numTargets)
 * @param queryMols Query molecules (parallel to results.numQueries)
 * @return Validation result with mismatch details
 */
SubstructValidationResult validateAgainstRDKit(
  const SubstructMatchResultsHost&                  results,
  const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
  const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols);

/**
 * @brief Print validation results to stdout.
 * @param result The validation result to print
 * @param algorithmName Optional algorithm name to include in output
 */
void printValidationResult(const SubstructValidationResult& result,
                           const std::string&               algorithmName = "");

}  // namespace nvMolKit

