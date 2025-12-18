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

#include <GraphMol/QueryAtom.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <iostream>
#include <set>

#include "cuda_error_check.h"
#include "device.h"
#include "graph_labeler.cuh"
#include "molecules_device.cuh"
#include "substructure_search.cuh"

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

namespace {

void printMatches(const std::string& label, const std::vector<std::vector<int>>& matches, size_t maxToPrint = 20) {
  std::cout << "    " << label << ": ";
  if (matches.empty()) {
    std::cout << "(none)" << std::endl;
    return;
  }
  std::cout << matches.size() << " match(es)" << std::endl;
  const size_t toPrint = std::min(matches.size(), maxToPrint);
  for (size_t i = 0; i < toPrint; ++i) {
    std::cout << "      [" << i << "]: {";
    for (size_t j = 0; j < matches[i].size(); ++j) {
      if (j > 0) std::cout << ", ";
      std::cout << matches[i][j];
    }
    std::cout << "}" << std::endl;
  }
  if (matches.size() > maxToPrint) {
    std::cout << "      ... and " << (matches.size() - maxToPrint) << " more" << std::endl;
  }
}

std::vector<std::vector<int>> extractGpuMatchesForPrint(const SubstructMatchResultsHost& results,
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

using LabelMatrixStorage = FlatBitVect<kMaxTargetAtoms * kMaxQueryAtoms>;
using LabelMatrixView    = BitMatrix2DView<kMaxTargetAtoms, kMaxQueryAtoms>;

template <std::size_t MaxTarget, std::size_t MaxQuery>
__global__ void populateLabelMatrixKernel(MoleculesDeviceView                targetBatch,
                                          int                                targetMolIdx,
                                          MoleculesDeviceView                queryBatch,
                                          int                                queryMolIdx,
                                          FlatBitVect<MaxTarget * MaxQuery>* matrix,
                                          const uint32_t*                    pairRecursiveBits) {
  MoleculeView                         target = getMolecule(targetBatch, targetMolIdx);
  MoleculeView                         query  = getMolecule(queryBatch, queryMolIdx);
  BitMatrix2DView<MaxTarget, MaxQuery> view(matrix);
  populateLabelMatrixOptimized<MaxTarget, MaxQuery>(target, query, view, pairRecursiveBits);
}

}  // namespace

void printValidationResultDetailed(const SubstructValidationResult&                  result,
                                   const SubstructMatchResultsHost&                  gpuResults,
                                   const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                                   const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                                   const std::vector<std::string>&                   targetSmiles,
                                   const std::vector<std::string>&                   querySmarts,
                                   const std::string&                                algoName,
                                   int                                               maxDetails) {
  std::string prefix = algoName.empty() ? "" : "[" + algoName + "] ";

  std::cout << prefix << "Validation: " << result.matchingPairs << "/" << result.totalPairs
            << " pairs match RDKit";

  if (result.overflowPairs > 0) {
    std::cout << " (" << result.overflowPairs << " overflow)";
  }

  if (result.allMatch) {
    std::cout << " - PASS" << std::endl;
    return;
  }

  const int totalFailures = result.mismatchedPairs + result.wrongMappingPairs;
  std::cout << " - FAIL (" << totalFailures << " failures)" << std::endl;

  int printed = 0;

  if (result.mismatchedPairs > 0) {
    std::cout << "  Count mismatches:" << std::endl;
    for (const auto& [t, q, gpuCount, rdkitCount] : result.mismatches) {
      if (printed++ >= maxDetails) {
        std::cout << "  ... and " << (result.mismatchedPairs - maxDetails) << " more count mismatches"
                  << std::endl;
        break;
      }
      std::cout << "  Target[" << t << "]: " << targetSmiles[t] << std::endl;
      std::cout << "  Query[" << q << "]:  " << querySmarts[q] << std::endl;

      // Get RDKit matches
      auto rdkitMatches = getRDKitSubstructMatches(*targetMols[t], *queryMols[q], false);
      printMatches("Expected (RDKit)", rdkitMatches);

      // Get GPU matches
      const int numQueryAtoms = static_cast<int>(queryMols[q]->getNumAtoms());
      auto      gpuMatches    = extractGpuMatchesForPrint(gpuResults, t, q, numQueryAtoms);
      printMatches("Actual (GPU)", gpuMatches);

      std::cout << std::endl;
    }
  }

  if (result.wrongMappingPairs > 0) {
    std::cout << "  Mapping mismatches (count correct but indices differ):" << std::endl;
    printed = 0;
    for (const auto& [t, q] : result.mappingMismatches) {
      if (printed++ >= maxDetails) {
        std::cout << "  ... and " << (result.wrongMappingPairs - maxDetails) << " more mapping mismatches"
                  << std::endl;
        break;
      }
      std::cout << "  Target[" << t << "]: " << targetSmiles[t] << std::endl;
      std::cout << "  Query[" << q << "]:  " << querySmarts[q] << std::endl;

      // Get RDKit matches
      auto rdkitMatches = getRDKitSubstructMatches(*targetMols[t], *queryMols[q], false);
      printMatches("Expected (RDKit)", rdkitMatches);

      // Get GPU matches
      const int numQueryAtoms = static_cast<int>(queryMols[q]->getNumAtoms());
      auto      gpuMatches    = extractGpuMatchesForPrint(gpuResults, t, q, numQueryAtoms);
      printMatches("Actual (GPU)", gpuMatches);

      std::cout << std::endl;
    }
  }
}

std::vector<std::vector<uint8_t>> computeRDKitLabelMatrix(const RDKit::ROMol& targetMol,
                                                          const RDKit::ROMol& queryMol) {
  const int numTargetAtoms = static_cast<int>(targetMol.getNumAtoms());
  const int numQueryAtoms  = static_cast<int>(queryMol.getNumAtoms());

  std::vector<std::vector<uint8_t>> result(numTargetAtoms, std::vector<uint8_t>(numQueryAtoms));

  for (int ta = 0; ta < numTargetAtoms; ++ta) {
    const auto* targetAtom = targetMol.getAtomWithIdx(ta);
    for (int qa = 0; qa < numQueryAtoms; ++qa) {
      const auto* queryAtom = queryMol.getAtomWithIdx(qa);

      bool rdkitResult = false;
      if (queryAtom->hasQuery()) {
        rdkitResult = queryAtom->Match(targetAtom);
      } else {
        rdkitResult = (targetAtom->getAtomicNum() == queryAtom->getAtomicNum());
      }
      result[ta][qa] = rdkitResult ? 1 : 0;
    }
  }

  return result;
}

std::vector<std::vector<uint8_t>> computeGpuLabelMatrix(const RDKit::ROMol& targetMol,
                                                        const RDKit::ROMol& queryMol,
                                                        cudaStream_t        stream) {
  MoleculesHost targetHost;
  MoleculesHost queryHost;
  addToBatch(&targetMol, targetHost);
  addQueryToBatch(&queryMol, queryHost);

  MoleculesDevice targetDevice(stream);
  MoleculesDevice queryDevice(stream);
  targetDevice.copyFromHost(targetHost);
  queryDevice.copyFromHost(queryHost);

  const int numTargetAtoms = static_cast<int>(targetHost.totalAtoms());
  const int numQueryAtoms  = static_cast<int>(queryHost.totalAtoms());

  std::vector<int> queryAtomCounts   = {numQueryAtoms};
  std::vector<int> maxMatchesPerPair = {numTargetAtoms};

  SubstructMatchResultsDevice results(stream);
  results.allocate(1, 1, queryAtomCounts, maxMatchesPerPair);

  RecursivePatternInfo info = extractRecursivePatterns(&queryMol);
  if (!info.empty()) {
    preprocessRecursiveSmarts(targetDevice, targetHost, info, results, 0, SubstructAlgorithm::GSI, stream);
  }

  AsyncDeviceVector<LabelMatrixStorage> matrixDev(1, stream);
  const LabelMatrixStorage              hostMatrix(false);
  matrixDev.setFromVector(std::vector<LabelMatrixStorage>{hostMatrix});

  auto            resultsView       = results.view();
  const uint32_t* pairRecursiveBits = info.empty() ? nullptr : resultsView.recursiveMatchBits;

  populateLabelMatrixKernel<kMaxTargetAtoms, kMaxQueryAtoms>
    <<<1, 128, 0, stream>>>(targetDevice.view(), 0, queryDevice.view(), 0, matrixDev.data(), pairRecursiveBits);
  cudaCheckError(cudaGetLastError());

  std::vector<LabelMatrixStorage> resultMatrix(1);
  matrixDev.copyToHost(resultMatrix);
  cudaCheckError(cudaStreamSynchronize(stream));

  const LabelMatrixView view(resultMatrix[0]);

  std::vector<std::vector<uint8_t>> result(numTargetAtoms, std::vector<uint8_t>(numQueryAtoms));
  for (int ta = 0; ta < numTargetAtoms; ++ta) {
    for (int qa = 0; qa < numQueryAtoms; ++qa) {
      result[ta][qa] = view.get(ta, qa) ? 1 : 0;
    }
  }

  return result;
}

LabelMatrixComparisonResult compareLabelMatrices(const RDKit::ROMol& targetMol,
                                                 const RDKit::ROMol& queryMol,
                                                 cudaStream_t        stream) {
  auto gpuMatrix   = computeGpuLabelMatrix(targetMol, queryMol, stream);
  auto rdkitMatrix = computeRDKitLabelMatrix(targetMol, queryMol);

  LabelMatrixComparisonResult result;
  result.numTargetAtoms   = static_cast<int>(gpuMatrix.size());
  result.numQueryAtoms    = result.numTargetAtoms > 0 ? static_cast<int>(gpuMatrix[0].size()) : 0;
  result.totalComparisons = result.numTargetAtoms * result.numQueryAtoms;

  for (int ta = 0; ta < result.numTargetAtoms; ++ta) {
    for (int qa = 0; qa < result.numQueryAtoms; ++qa) {
      bool gpuResult   = gpuMatrix[ta][qa] != 0;
      bool rdkitResult = rdkitMatrix[ta][qa] != 0;

      if (gpuResult != rdkitResult) {
        if (gpuResult && !rdkitResult) {
          ++result.falsePositives;
        } else {
          ++result.falseNegatives;
        }
        result.mismatches.emplace_back(ta, qa, gpuResult, rdkitResult);
      }
    }
  }

  result.allMatch = (result.falsePositives == 0 && result.falseNegatives == 0);
  return result;
}

}  // namespace nvMolKit

