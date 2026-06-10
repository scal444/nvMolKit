// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "src/mcs/mcs_search.h"

#include "src/mcs/benchmark_data.h"
#include "src/mcs/fmcs_cuda/fmcs.cuh"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/FMCS/FMCS.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nvMolKit {
namespace {

using mcs::benchmark::MiviaGraphData;

struct AtomLabelKey {
  int atomCompareValue = 0;
  int isotope          = 0;
  int totalValence     = 0;
  int formalCharge     = 0;
  int ringState        = 0;

  [[nodiscard]] auto tie() const {
    return std::tie(atomCompareValue, isotope, totalValence, formalCharge, ringState);
  }

  bool operator<(const AtomLabelKey& other) const { return tie() < other.tie(); }
};

struct BondLabelKey {
  int bondCompareValue = 0;
  int ringState        = 0;

  [[nodiscard]] auto tie() const { return std::tie(bondCompareValue, ringState); }

  bool operator<(const BondLabelKey& other) const { return tie() < other.tie(); }
};

template <typename Key>
std::uint16_t internLabel(std::map<Key, std::uint16_t>& labels, const Key& key) {
  const auto found = labels.find(key);
  if (found != labels.end()) {
    return found->second;
  }
  if (labels.size() >= static_cast<size_t>(UINT16_MAX)) {
    throw std::runtime_error("MCS label table exceeded uint16_t capacity");
  }
  const auto label = static_cast<std::uint16_t>(labels.size() + 1);
  labels.emplace(key, label);
  return label;
}

bool usesAtomLabels(const MCSParameters& params) {
  return params.atomCompare != MCSAtomCompare::Any || params.atomCompareParameters.matchValences ||
         params.atomCompareParameters.matchFormalCharge || params.atomCompareParameters.ringMatchesRingOnly ||
         params.atomCompareParameters.matchIsotope;
}

bool usesBondLabels(const MCSParameters& params) {
  return params.bondCompare != MCSBondCompare::Any || params.bondCompareParameters.ringMatchesRingOnly;
}

AtomLabelKey makeAtomLabelKey(const RDKit::ROMol& mol, const RDKit::Atom& atom, const MCSParameters& params) {
  AtomLabelKey key;
  switch (params.atomCompare) {
    case MCSAtomCompare::Any:
      key.atomCompareValue = 0;
      break;
    case MCSAtomCompare::Elements:
      key.atomCompareValue = atom.getAtomicNum();
      break;
    case MCSAtomCompare::Isotopes:
      key.atomCompareValue = static_cast<int>(atom.getIsotope());
      break;
    case MCSAtomCompare::AnyHeavyAtom:
      key.atomCompareValue = atom.getAtomicNum() == 1 ? 1 : 2;
      break;
  }

  if (params.atomCompareParameters.matchIsotope) {
    key.isotope = static_cast<int>(atom.getIsotope());
  }
  if (params.atomCompareParameters.matchValences) {
    key.totalValence = static_cast<int>(atom.getTotalValence());
  }
  if (params.atomCompareParameters.matchFormalCharge) {
    key.formalCharge = atom.getFormalCharge();
  }
  if (params.atomCompareParameters.ringMatchesRingOnly) {
    key.ringState = mol.getRingInfo()->numAtomRings(atom.getIdx()) > 0 ? 1 : 2;
  }
  return key;
}

int bondOrderClass(const RDKit::Bond& bond, const MCSParameters& params) {
  switch (params.bondCompare) {
    case MCSBondCompare::Any:
      return 0;
    case MCSBondCompare::Order: {
      const auto bondType = static_cast<int>(bond.getBondType());
      if (bondType == static_cast<int>(RDKit::Bond::SINGLE) || bondType == static_cast<int>(RDKit::Bond::AROMATIC)) {
        return 1;
      }
      return bondType;
    }
    case MCSBondCompare::OrderExact: {
      const auto bondType = static_cast<int>(bond.getBondType());
      if (bondType == static_cast<int>(RDKit::Bond::ONEANDAHALF) ||
          bondType == static_cast<int>(RDKit::Bond::AROMATIC)) {
        return static_cast<int>(RDKit::Bond::AROMATIC);
      }
      return bondType;
    }
  }
  return 0;
}

BondLabelKey makeBondLabelKey(const RDKit::Bond& bond, const MCSParameters& params) {
  BondLabelKey key;
  key.bondCompareValue = bondOrderClass(bond, params);
  if (params.bondCompareParameters.ringMatchesRingOnly) {
    key.ringState = bond.getOwningMol().getRingInfo()->numBondRings(bond.getIdx()) > 0 ? 1 : 2;
  }
  return key;
}

MiviaGraphData buildLabeledGraph(const RDKit::ROMol&              mol,
                                 const MCSParameters&             params,
                                 std::map<AtomLabelKey, uint16_t>& atomLabels,
                                 std::map<BondLabelKey, uint16_t>& bondLabels) {
  MiviaGraphData out;
  std::vector<std::pair<size_t, size_t>> edges;
  edges.reserve(mol.getNumBonds());

  for (const auto* bond : mol.bonds()) {
    edges.emplace_back(bond->getBeginAtomIdx(), bond->getEndAtomIdx());
  }
  out.graph = mcs::buildGraphFromEdges(mol.getNumAtoms(), edges);

  out.vertexLabels.reserve(mol.getNumAtoms());
  for (const auto* atom : mol.atoms()) {
    out.vertexLabels.push_back(internLabel(atomLabels, makeAtomLabelKey(mol, *atom, params)));
  }

  const auto numAtoms = static_cast<size_t>(mol.getNumAtoms());
  out.edgeLabels.assign(numAtoms * numAtoms, 0);
  for (const auto* bond : mol.bonds()) {
    const auto label = internLabel(bondLabels, makeBondLabelKey(*bond, params));
    const auto begin = static_cast<size_t>(bond->getBeginAtomIdx());
    const auto end   = static_cast<size_t>(bond->getEndAtomIdx());
    out.edgeLabels[begin * numAtoms + end] = label;
    out.edgeLabels[end * numAtoms + begin] = label;
  }

  return out;
}

RDKit::MCSParameters buildRDKitParameters(const MCSParameters& params) {
  RDKit::MCSParameters rdParams;
  rdParams.MaximizeBonds = params.maximizeBonds;
  rdParams.Timeout       = params.timeoutSeconds;

  rdParams.AtomCompareParameters.MatchValences       = params.atomCompareParameters.matchValences;
  rdParams.AtomCompareParameters.MatchFormalCharge   = params.atomCompareParameters.matchFormalCharge;
  rdParams.AtomCompareParameters.RingMatchesRingOnly = params.atomCompareParameters.ringMatchesRingOnly;
  rdParams.AtomCompareParameters.CompleteRingsOnly   = params.atomCompareParameters.completeRingsOnly;
  rdParams.AtomCompareParameters.MatchIsotope        = params.atomCompareParameters.matchIsotope;
  rdParams.BondCompareParameters.RingMatchesRingOnly = params.bondCompareParameters.ringMatchesRingOnly;
  rdParams.BondCompareParameters.CompleteRingsOnly   = params.bondCompareParameters.completeRingsOnly;

  switch (params.atomCompare) {
    case MCSAtomCompare::Any:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareAny);
      break;
    case MCSAtomCompare::Elements:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareElements);
      break;
    case MCSAtomCompare::Isotopes:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareIsotopes);
      break;
    case MCSAtomCompare::AnyHeavyAtom:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareAnyHeavyAtom);
      break;
  }

  switch (params.bondCompare) {
    case MCSBondCompare::Any:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareAny);
      break;
    case MCSBondCompare::Order:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareOrder);
      break;
    case MCSBondCompare::OrderExact:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareOrderExact);
      break;
  }

  return rdParams;
}

void fillResultBondMapping(const RDKit::ROMol&                          molA,
                           const RDKit::ROMol&                          molB,
                           const std::vector<std::pair<int, int>>&      atomMapping,
                           std::vector<std::pair<int, int>>&            bondMapping) {
  std::vector<int> aToB(molA.getNumAtoms(), -1);
  for (const auto& [aIdx, bIdx] : atomMapping) {
    if (aIdx >= 0 && aIdx < static_cast<int>(aToB.size())) {
      aToB[static_cast<size_t>(aIdx)] = bIdx;
    }
  }

  bondMapping.clear();
  for (const auto* bondA : molA.bonds()) {
    const int aBegin = static_cast<int>(bondA->getBeginAtomIdx());
    const int aEnd   = static_cast<int>(bondA->getEndAtomIdx());
    if (aToB[static_cast<size_t>(aBegin)] < 0 || aToB[static_cast<size_t>(aEnd)] < 0) {
      continue;
    }
    const auto* bondB = molB.getBondBetweenAtoms(aToB[static_cast<size_t>(aBegin)], aToB[static_cast<size_t>(aEnd)]);
    if (bondB != nullptr) {
      bondMapping.emplace_back(static_cast<int>(bondA->getIdx()), static_cast<int>(bondB->getIdx()));
    }
  }
}

MCSResult runRDKitFallback(const RDKit::ROMol& molA, const RDKit::ROMol& molB, const MCSParameters& params) {
  MCSResult result;
  result.usedFallback = true;

  std::vector<RDKit::ROMOL_SPTR> mols;
  mols.emplace_back(new RDKit::ROMol(molA));
  mols.emplace_back(new RDKit::ROMol(molB));
  auto rdParams = buildRDKitParameters(params);
  const auto rdResult = RDKit::findMCS(mols, &rdParams);

  result.numAtoms     = rdResult.NumAtoms;
  result.numBonds     = rdResult.NumBonds;
  result.canceled     = rdResult.Canceled;
  result.smartsString = rdResult.SmartsString;

  if (rdResult.QueryMol == nullptr) {
    return result;
  }

  RDKit::SubstructMatchParameters matchParams;
  matchParams.uniquify   = false;
  matchParams.maxMatches = 1;
  const auto matchesA    = RDKit::SubstructMatch(molA, *rdResult.QueryMol, matchParams);
  const auto matchesB    = RDKit::SubstructMatch(molB, *rdResult.QueryMol, matchParams);
  if (matchesA.empty() || matchesB.empty()) {
    return result;
  }

  std::vector<int> queryToA(rdResult.QueryMol->getNumAtoms(), -1);
  std::vector<int> queryToB(rdResult.QueryMol->getNumAtoms(), -1);
  for (const auto& [queryAtomIdx, targetAtomIdx] : matchesA.front()) {
    queryToA[static_cast<size_t>(queryAtomIdx)] = targetAtomIdx;
  }
  for (const auto& [queryAtomIdx, targetAtomIdx] : matchesB.front()) {
    queryToB[static_cast<size_t>(queryAtomIdx)] = targetAtomIdx;
  }

  for (size_t queryIdx = 0; queryIdx < queryToA.size(); ++queryIdx) {
    if (queryToA[queryIdx] >= 0 && queryToB[queryIdx] >= 0) {
      result.atomMapping.emplace_back(queryToA[queryIdx], queryToB[queryIdx]);
    }
  }
  fillResultBondMapping(molA, molB, result.atomMapping, result.bondMapping);
  return result;
}

bool shouldFallbackToRDKit(const RDKit::ROMol& molA,
                           const RDKit::ROMol& molB,
                           const MCSParameters& params,
                           std::string&         reason) {
  if (!params.connectedOnly) {
    reason = "fMCS supports connected MCS only";
    return true;
  }
  if (!params.maximizeBonds) {
    reason = "fMCS currently supports MaximizeBonds only";
    return true;
  }
  if (params.timeoutSeconds > 0) {
    reason = "timeout handling is delegated to RDKit";
    return true;
  }
  if (params.atomCompare == MCSAtomCompare::AnyHeavyAtom) {
    reason = "AtomCompareAnyHeavyAtom is delegated to RDKit";
    return true;
  }
  if (params.atomCompareParameters.completeRingsOnly || params.bondCompareParameters.completeRingsOnly) {
    reason = "CompleteRingsOnly is delegated to RDKit";
    return true;
  }
  if (std::max(molA.getNumAtoms(), molB.getNumAtoms()) > 128 || std::max(molA.getNumBonds(), molB.getNumBonds()) > 128) {
    reason = "molecule exceeds fMCS tier-128 limits";
    return true;
  }
  return false;
}

MCSResult convertGpuResult(const RDKit::ROMol& molA, const RDKit::ROMol& molB, const mcs::MCSResult& gpuResult) {
  MCSResult out;
  out.numAtoms  = static_cast<unsigned int>(gpuResult.numCommonVertices);
  out.numBonds  = static_cast<unsigned int>(gpuResult.numCommonEdges);
  out.canceled  = gpuResult.timedOut || gpuResult.killed;
  out.overflowed = gpuResult.overflowed;
  out.usedGpu   = true;

  const size_t numMappedAtoms = std::min(gpuResult.mappingA.size(), gpuResult.mappingB.size());
  out.atomMapping.reserve(numMappedAtoms);
  for (size_t i = 0; i < numMappedAtoms; ++i) {
    out.atomMapping.emplace_back(static_cast<int>(gpuResult.mappingA[i]), static_cast<int>(gpuResult.mappingB[i]));
  }

  out.bondMapping.reserve(gpuResult.edgeMappingA.size());
  const size_t numMappedBonds = std::min(gpuResult.edgeMappingA.size(), gpuResult.edgeMappingB.size());
  for (size_t i = 0; i < numMappedBonds; ++i) {
    const auto& edgeA = gpuResult.edgeMappingA[i];
    const auto& edgeB = gpuResult.edgeMappingB[i];
    const auto* bondA = molA.getBondBetweenAtoms(static_cast<unsigned int>(edgeA.first),
                                                 static_cast<unsigned int>(edgeA.second));
    const auto* bondB = molB.getBondBetweenAtoms(static_cast<unsigned int>(edgeB.first),
                                                 static_cast<unsigned int>(edgeB.second));
    if (bondA != nullptr && bondB != nullptr) {
      out.bondMapping.emplace_back(static_cast<int>(bondA->getIdx()), static_cast<int>(bondB->getIdx()));
    }
  }
  return out;
}

}  // namespace

std::vector<MCSResult> findMCSBatch(const std::vector<const RDKit::ROMol*>& mols,
                                    const std::vector<MCSPair>&             pairs,
                                    cudaStream_t                            stream,
                                    const MCSParameters&                    params) {
  std::vector<MCSResult> results(pairs.size());
  std::vector<MiviaGraphData> gpuGraphsA;
  std::vector<MiviaGraphData> gpuGraphsB;
  std::vector<size_t> gpuResultIndices;
  gpuGraphsA.reserve(pairs.size());
  gpuGraphsB.reserve(pairs.size());
  gpuResultIndices.reserve(pairs.size());

  for (size_t i = 0; i < pairs.size(); ++i) {
    const auto [idxA, idxB] = pairs[i];
    if (idxA >= mols.size() || idxB >= mols.size()) {
      throw std::runtime_error("findMCSBatch pair index out of range");
    }
    const RDKit::ROMol* molA = mols[idxA];
    const RDKit::ROMol* molB = mols[idxB];
    if (molA == nullptr || molB == nullptr) {
      throw std::runtime_error("findMCSBatch pair references a null molecule pointer");
    }

    std::string fallbackReason;
    if (shouldFallbackToRDKit(*molA, *molB, params, fallbackReason)) {
      if (params.requireGpu) {
        throw std::runtime_error("GPU MCS path unavailable: " + fallbackReason);
      }
      results[i] = runRDKitFallback(*molA, *molB, params);
      continue;
    }

    std::map<AtomLabelKey, uint16_t> atomLabels;
    std::map<BondLabelKey, uint16_t> bondLabels;
    gpuGraphsA.push_back(buildLabeledGraph(*molA, params, atomLabels, bondLabels));
    gpuGraphsB.push_back(buildLabeledGraph(*molB, params, atomLabels, bondLabels));
    gpuResultIndices.push_back(i);
  }

  if (!gpuGraphsA.empty()) {
    mcs::fmcs::Parameters fmcsParams;
    fmcsParams.batchSize         = params.batchSize;
    fmcsParams.matchVertexLabels = usesAtomLabels(params);
    fmcsParams.matchEdgeLabels   = usesBondLabels(params);

    auto gpuResults = mcs::fmcs::findMCESfMCSBatchLabeled(gpuGraphsA, gpuGraphsB, fmcsParams, nullptr, stream);
    for (size_t gpuIdx = 0; gpuIdx < gpuResults.size(); ++gpuIdx) {
      const size_t resultIdx = gpuResultIndices[gpuIdx];
      const auto [idxA, idxB] = pairs[resultIdx];
      if (gpuResults[gpuIdx].overflowed) {
        if (params.requireGpu) {
          throw std::runtime_error("GPU MCS path overflowed");
        }
        results[resultIdx] = runRDKitFallback(*mols[idxA], *mols[idxB], params);
      } else {
        results[resultIdx] = convertGpuResult(*mols[idxA], *mols[idxB], gpuResults[gpuIdx]);
      }
    }
  }

  return results;
}

std::vector<MCSResult> findMCSBatch(const std::vector<const RDKit::ROMol*>& molsA,
                                    const std::vector<const RDKit::ROMol*>& molsB,
                                    cudaStream_t                            stream,
                                    const MCSParameters&                    params) {
  if (molsA.size() != molsB.size()) {
    throw std::runtime_error("findMCSBatch requires equal-sized molecule arrays");
  }

  std::vector<const RDKit::ROMol*> mols;
  mols.reserve(molsA.size() + molsB.size());
  mols.insert(mols.end(), molsA.begin(), molsA.end());
  mols.insert(mols.end(), molsB.begin(), molsB.end());

  std::vector<MCSPair> pairs;
  pairs.reserve(molsA.size());
  for (size_t i = 0; i < molsA.size(); ++i) {
    pairs.emplace_back(i, molsA.size() + i);
  }
  return findMCSBatch(mols, pairs, stream, params);
}

std::vector<MCSResult> findMCSAllPairs(const std::vector<const RDKit::ROMol*>& mols,
                                       MCSAllPairsOptions                     options,
                                       cudaStream_t                           stream,
                                       const MCSParameters&                   params) {
  std::vector<MCSPair> pairs;
  const size_t n = mols.size();
  if (options.upperTriangle) {
    const size_t maxPairs = options.includeDiagonal ? (n * (n + 1)) / 2 : (n > 1 ? (n * (n - 1)) / 2 : 0);
    pairs.reserve(maxPairs);
    for (size_t i = 0; i < n; ++i) {
      const size_t begin = options.includeDiagonal ? i : i + 1;
      for (size_t j = begin; j < n; ++j) {
        pairs.emplace_back(i, j);
      }
    }
  } else {
    pairs.reserve(options.includeDiagonal ? n * n : (n > 1 ? n * (n - 1) : 0));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (!options.includeDiagonal && i == j) {
          continue;
        }
        pairs.emplace_back(i, j);
      }
    }
  }
  return findMCSBatch(mols, pairs, stream, params);
}

}  // namespace nvMolKit
