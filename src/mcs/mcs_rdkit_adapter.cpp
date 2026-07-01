// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/mcs/mcs_rdkit_adapter.h"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/FMCS/FMCS.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "src/mcs/mcs_compile_flags.h"
#include "src/utils/nvtx.h"

namespace nvMolKit::mcs_detail {
namespace {

using mcs::benchmark::MiviaGraphData;

struct AtomLabelKey {
  int atomCompareValue = 0;
  int isotope          = 0;
  int totalValence     = 0;
  int formalCharge     = 0;
  int ringState        = 0;

  [[nodiscard]] auto tie() const { return std::tie(atomCompareValue, isotope, totalValence, formalCharge, ringState); }

  bool operator<(const AtomLabelKey& other) const { return tie() < other.tie(); }
};

struct BondLabelKey {
  int bondCompareValue = 0;
  int ringState        = 0;

  [[nodiscard]] auto tie() const { return std::tie(bondCompareValue, ringState); }

  bool operator<(const BondLabelKey& other) const { return tie() < other.tie(); }
};

template <typename Key> std::uint16_t internLabel(std::map<Key, std::uint16_t>& labels, const Key& key) {
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

MiviaGraphData buildLabeledGraph(const RDKit::ROMol&                    mol,
                                 const MCSParameters&                   params,
                                 std::map<AtomLabelKey, std::uint16_t>& atomLabels,
                                 std::map<BondLabelKey, std::uint16_t>& bondLabels) {
  MiviaGraphData                         out;
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
    const auto label                       = internLabel(bondLabels, makeBondLabelKey(*bond, params));
    const auto begin                       = static_cast<size_t>(bond->getBeginAtomIdx());
    const auto end                         = static_cast<size_t>(bond->getEndAtomIdx());
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

void fillResultBondMapping(const RDKit::ROMol&                     molA,
                           const RDKit::ROMol&                     molB,
                           const std::vector<std::pair<int, int>>& atomMapping,
                           std::vector<std::pair<int, int>>&       bondMapping) {
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

void addCompatibleSingletonIfEmpty(MCSResult&           result,
                                   const RDKit::ROMol&  molA,
                                   const RDKit::ROMol&  molB,
                                   const MCSParameters& params) {
  if (result.numAtoms != 0 || molA.getNumAtoms() == 0 || molB.getNumAtoms() == 0) {
    return;
  }

  for (const auto* atomA : molA.atoms()) {
    const auto keyA = makeAtomLabelKey(molA, *atomA, params);
    for (const auto* atomB : molB.atoms()) {
      const auto keyB = makeAtomLabelKey(molB, *atomB, params);
      if (keyA.tie() == keyB.tie()) {
        result.numAtoms = 1;
        result.atomMapping.emplace_back(static_cast<int>(atomA->getIdx()), static_cast<int>(atomB->getIdx()));
        return;
      }
    }
  }
}

}  // namespace

bool usesAtomLabels(const MCSParameters& params) {
  return params.atomCompare != MCSAtomCompare::Any || params.atomCompareParameters.matchValences ||
         params.atomCompareParameters.matchFormalCharge || params.atomCompareParameters.ringMatchesRingOnly ||
         params.atomCompareParameters.matchIsotope;
}

bool usesBondLabels(const MCSParameters& params) {
  return params.bondCompare != MCSBondCompare::Any || params.bondCompareParameters.ringMatchesRingOnly;
}

LabeledGraphPair buildLabeledGraphPair(const RDKit::ROMol&  molA,
                                       const RDKit::ROMol&  molB,
                                       const MCSParameters& params) {
  std::map<AtomLabelKey, std::uint16_t> atomLabels;
  std::map<BondLabelKey, std::uint16_t> bondLabels;
  LabeledGraphPair                      out;
  out.graphA = buildLabeledGraph(molA, params, atomLabels, bondLabels);
  out.graphB = buildLabeledGraph(molB, params, atomLabels, bondLabels);
  return out;
}

bool shouldFallbackToRDKit(const RDKit::ROMol&  molA,
                           const RDKit::ROMol&  molB,
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
  if (params.atomCompare == MCSAtomCompare::AnyHeavyAtom) {
    reason = "AtomCompareAnyHeavyAtom is delegated to RDKit";
    return true;
  }
  if (params.atomCompareParameters.completeRingsOnly || params.bondCompareParameters.completeRingsOnly) {
    reason = "CompleteRingsOnly is delegated to RDKit";
    return true;
  }
  if (std::max(molA.getNumAtoms(), molB.getNumAtoms()) > 128 ||
      std::max(molA.getNumBonds(), molB.getNumBonds()) > 128) {
    reason = "molecule exceeds fMCS tier-128 limits";
    return true;
  }
  return false;
}

MCSResult runRDKitFallback(const RDKit::ROMol& molA, const RDKit::ROMol& molB, const MCSParameters& params) {
  ScopedNvtxRange                       fallbackRange("RDKit fallback", NvtxColor::kOrange);
  std::chrono::steady_clock::time_point start;
  if constexpr (kMCSCollectTimingsEnabled) {
    if (params.collectTimings) {
      start = std::chrono::steady_clock::now();
    }
  }
  MCSResult result;
  result.usedFallback = true;
  auto finish         = [&]() {
    if constexpr (kMCSCollectTimingsEnabled) {
      if (params.collectTimings) {
        const auto end   = std::chrono::steady_clock::now();
        result.elapsedMs = static_cast<float>(std::chrono::duration<double, std::milli>(end - start).count());
      }
    }
    return result;
  };

  std::vector<RDKit::ROMOL_SPTR> mols;
  mols.emplace_back(new RDKit::ROMol(molA));
  mols.emplace_back(new RDKit::ROMol(molB));
  auto       rdParams = buildRDKitParameters(params);
  const auto rdResult = RDKit::findMCS(mols, &rdParams);

  result.numAtoms     = rdResult.NumAtoms;
  result.numBonds     = rdResult.NumBonds;
  result.canceled     = rdResult.Canceled;
  result.smartsString = rdResult.SmartsString;

  if (rdResult.QueryMol == nullptr) {
    return finish();
  }

  RDKit::SubstructMatchParameters matchParams;
  matchParams.uniquify   = false;
  matchParams.maxMatches = 1;
  const auto matchesA    = RDKit::SubstructMatch(molA, *rdResult.QueryMol, matchParams);
  const auto matchesB    = RDKit::SubstructMatch(molB, *rdResult.QueryMol, matchParams);
  if (matchesA.empty() || matchesB.empty()) {
    return finish();
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
  return finish();
}

MCSExecutionStats convertExecutionStats(const mcs::fmcs::ExecutionStats& in) {
  MCSExecutionStats out;
  out.phase2Iters                     = in.phase2Iters;
  out.initialSeeds                    = in.initialSeeds;
  out.mismatchedInitialSeeds          = in.mismatchedInitialSeeds;
  out.popped                          = in.popped;
  out.seedChecks                      = in.seedChecks;
  out.matchCalls                      = in.matchCalls;
  out.matchFound                      = in.matchFound;
  out.boundRejected                   = in.boundRejected;
  out.expanded                        = in.expanded;
  out.fillZero                        = in.fillZero;
  out.stage0Attempts                  = in.stage0Attempts;
  out.stage0Success                   = in.stage0Success;
  out.stage1Attempts                  = in.stage1Attempts;
  out.stage1Success                   = in.stage1Success;
  out.stage2Attempts                  = in.stage2Attempts;
  out.stage2Success                   = in.stage2Success;
  out.individualBondExcluded          = in.individualBondExcluded;
  out.fastAttempts                    = in.fastAttempts;
  out.fastSuccess                     = in.fastSuccess;
  out.fallbackCalls                   = in.fallbackCalls;
  out.fallbackSuccess                 = in.fallbackSuccess;
  out.fallbackFail                    = in.fallbackFail;
  out.fallbackOverflow                = in.fallbackOverflow;
  out.maxQueue                        = in.maxQueue;
  out.forcedExit                      = in.forcedExit;
  out.totalClocks                     = in.totalClocks;
  out.phase1Clocks                    = in.phase1Clocks;
  out.phase2Clocks                    = in.phase2Clocks;
  out.incrementalMatchCycles1024      = in.incrementalMatchCycles1024;
  out.substructureMatchCycles1024     = in.substructureMatchCycles1024;
  out.phase2PopSyncWaitCycles1024     = in.phase2PopSyncWaitCycles1024;
  out.phase2SyncWaitCycles1024        = in.phase2SyncWaitCycles1024;
  out.phase2IdleNoSeedWaitCycles1024  = in.phase2IdleNoSeedWaitCycles1024;
  out.phase2IdleNoMatchWaitCycles1024 = in.phase2IdleNoMatchWaitCycles1024;
  out.phase2ActiveWorkCycles1024      = in.phase2ActiveWorkCycles1024;
  out.phase2ActiveMatchCycles1024     = in.phase2ActiveMatchCycles1024;
  return out;
}

MCSResult convertGpuResult(const RDKit::ROMol&              molA,
                           const RDKit::ROMol&              molB,
                           const mcs::MCSResult&            gpuResult,
                           const MCSParameters&             params,
                           float                            elapsedMs,
                           const mcs::fmcs::ExecutionStats* kernelTimings,
                           const mcs::fmcs::ExecutionStats* executionStats) {
  MCSResult out;
  out.numAtoms   = static_cast<unsigned int>(gpuResult.numCommonVertices);
  out.numBonds   = static_cast<unsigned int>(gpuResult.numCommonEdges);
  out.canceled   = gpuResult.timedOut;
  out.overflowed = gpuResult.overflowed;
  out.usedGpu    = true;
  out.elapsedMs  = elapsedMs;
  if (kernelTimings != nullptr) {
    out.hasKernelTimings = true;
    out.kernelTimings    = convertExecutionStats(*kernelTimings);
  }
  if (executionStats != nullptr) {
    out.hasExecutionStats = true;
    out.executionStats    = convertExecutionStats(*executionStats);
  }

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
    const auto* bondA =
      molA.getBondBetweenAtoms(static_cast<unsigned int>(edgeA.first), static_cast<unsigned int>(edgeA.second));
    const auto* bondB =
      molB.getBondBetweenAtoms(static_cast<unsigned int>(edgeB.first), static_cast<unsigned int>(edgeB.second));
    if (bondA != nullptr && bondB != nullptr) {
      out.bondMapping.emplace_back(static_cast<int>(bondA->getIdx()), static_cast<int>(bondB->getIdx()));
    }
  }
  addCompatibleSingletonIfEmpty(out, molA, molB, params);
  return out;
}

}  // namespace nvMolKit::mcs_detail
