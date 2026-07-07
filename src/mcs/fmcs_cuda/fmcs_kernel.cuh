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

#ifndef FMCS_CUDA_FMCS_KERNEL_CUH
#define FMCS_CUDA_FMCS_KERNEL_CUH

#include <cooperative_groups.h>

#include <cstdint>

#include "fmcs_cuda/fmcs_config.cuh"
#include "fmcs_cuda/fmcs_debug.cuh"
#include "fmcs_cuda/fmcs_grow.cuh"
#include "fmcs_cuda/fmcs_kernel_types.cuh"
#include "fmcs_cuda/fmcs_match.cuh"
#include "fmcs_cuda/fmcs_match_tables.cuh"
#include "fmcs_cuda/fmcs_queue_cooperative.cuh"
#include "fmcs_cuda/fmcs_seed.cuh"
#include "fmcs_cuda/fmcs_seed_queue.cuh"
#include "fmcs_cuda/fmcs_stats.cuh"
#include "fmcs_cuda/fmcs_tiers.cuh"
#include "mcs_common/mcs_cooperative_copy.cuh"

namespace mcs {
namespace fmcs {

namespace cg = cooperative_groups;

template <class GroupT, class QueuedT>
__device__ __forceinline__ void updateIncumbentCooperative(const GroupT&  group,
                                                           const QueuedT& candidate,
                                                           QueuedT&       best,
                                                           unsigned int*  bestScore,
                                                           int*           bestCopyLock) {
  const int groupRank  = static_cast<int>(group.thread_rank());
  int       locked     = 0;
  int       shouldCopy = 0;
  if (groupRank == 0) {
    const unsigned int candidateScore =
      (static_cast<unsigned int>(candidate.seed.numBonds) << 16) | static_cast<unsigned int>(candidate.seed.numAtoms);
    unsigned int prev = *bestScore;
    bool         won  = false;
    while (candidateScore > prev) {
      const unsigned int seen = atomicCAS(bestScore, prev, candidateScore);
      if (seen == prev) {
        won = true;
        break;
      }
      prev = seen;
    }
    if (won) {
      while (atomicCAS(bestCopyLock, 0, 1) != 0) {
      }
      locked     = 1;
      shouldCopy = (candidateScore == *bestScore) ? 1 : 0;
    }
  }
  locked     = group.shfl(locked, 0);
  shouldCopy = group.shfl(shouldCopy, 0);
  if (shouldCopy) {
    warpAtomicStoreWords(group, &best, candidate);
  }
  group.sync();
  if (groupRank == 0 && locked) {
    __threadfence_block();
    atomicExch(bestCopyLock, 0);
  }
  group.sync();
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ bool hasOnlyCompleteRingsCooperative(
  const GroupT&                                      group,
  const QueuedT&                                     candidate,
  const DeviceCsrView&                               queryTopology,
  const std::uint32_t*                               queryRingBondFlags,
  const std::uint32_t*                               targetRingBondFlags,
  typename decltype(candidate.seed)::atom_word_type* visitedAtoms,
  int*                                               changed) {
  using SeedT             = decltype(candidate.seed);
  using AtomWord          = typename SeedT::atom_word_type;
  using BondWord          = typename SeedT::bond_word_type;
  constexpr int kAtomBits = SeedT::kAtomBitsPerWord;
  constexpr int kBondBits = SeedT::kBondBitsPerWord;
  const int     rank      = static_cast<int>(group.thread_rank());
  const int     width     = static_cast<int>(group.size());

  for (int excludedBond = 0; excludedBond < queryTopology.numBonds; ++excludedBond) {
    const BondWord excludedMask = static_cast<BondWord>(1) << (excludedBond % kBondBits);
    if ((candidate.seed.bonds[excludedBond / kBondBits] & excludedMask) == 0)
      continue;
    const std::uint8_t targetBond         = candidate.match.targetBondIdx[excludedBond];
    const bool         isOriginalRingBond = queryRingBondFlags[excludedBond] != 0 ||
                                    (targetBond != kUnmappedTargetIdx && targetRingBondFlags[targetBond] != 0);
    if (!isOriginalRingBond)
      continue;

    for (int word = rank; word < SeedT::kAtomWords; word += width)
      visitedAtoms[word] = 0;
    group.sync();
    const std::uint32_t excludedEndpoints = queryTopology.bondEndpoints[excludedBond];
    const int           from              = static_cast<int>(excludedEndpoints >> kBondEndpointShift);
    const int           goal              = static_cast<int>(excludedEndpoints & kBondEndpointMask);
    if (rank == 0) {
      visitedAtoms[from / kAtomBits] |= static_cast<AtomWord>(1) << (from % kAtomBits);
    }
    group.sync();

    while (true) {
      if (rank == 0)
        *changed = 0;
      group.sync();
      for (int bondIdx = rank; bondIdx < queryTopology.numBonds; bondIdx += width) {
        if (bondIdx == excludedBond)
          continue;
        const BondWord bondMask = static_cast<BondWord>(1) << (bondIdx % kBondBits);
        if ((candidate.seed.bonds[bondIdx / kBondBits] & bondMask) == 0)
          continue;
        const std::uint32_t endpoints = queryTopology.bondEndpoints[bondIdx];
        const int           u         = static_cast<int>(endpoints >> kBondEndpointShift);
        const int           v         = static_cast<int>(endpoints & kBondEndpointMask);
        const AtomWord      uMask     = static_cast<AtomWord>(1) << (u % kAtomBits);
        const AtomWord      vMask     = static_cast<AtomWord>(1) << (v % kAtomBits);
        const bool          uSeen     = (visitedAtoms[u / kAtomBits] & uMask) != 0;
        const bool          vSeen     = (visitedAtoms[v / kAtomBits] & vMask) != 0;
        if (uSeen == vSeen)
          continue;
        AtomWord*      dst  = &visitedAtoms[(uSeen ? v : u) / kAtomBits];
        const AtomWord mask = uSeen ? vMask : uMask;
        AtomWord       old;
        if constexpr (sizeof(AtomWord) == sizeof(unsigned int)) {
          old = static_cast<AtomWord>(atomicOr(reinterpret_cast<unsigned int*>(dst), static_cast<unsigned int>(mask)));
        } else {
          old = static_cast<AtomWord>(
            atomicOr(reinterpret_cast<unsigned long long*>(dst), static_cast<unsigned long long>(mask)));
        }
        if ((old & mask) == 0)
          atomicExch(changed, 1);
      }
      group.sync();
      if (*changed == 0)
        break;
    }
    const bool goalReached = (visitedAtoms[goal / kAtomBits] & (static_cast<AtomWord>(1) << (goal % kAtomBits))) != 0;
    group.sync();
    if (!goalReached)
      return false;
  }
  return true;
}

template <class GroupT, class QueuedT>
__device__ __forceinline__ void updateCompleteRingsIncumbentCooperative(
  const GroupT&                                      group,
  const QueuedT&                                     candidate,
  QueuedT&                                           best,
  unsigned int*                                      bestScore,
  int*                                               bestCopyLock,
  bool                                               completeRingsOnly,
  const DeviceCsrView&                               queryTopology,
  const std::uint32_t*                               queryRingBondFlags,
  const std::uint32_t*                               targetRingBondFlags,
  typename decltype(candidate.seed)::atom_word_type* visitedAtoms,
  int*                                               changed) {
  int canImprove = 0;
  if (group.thread_rank() == 0) {
    const unsigned int candidateScore =
      (static_cast<unsigned int>(candidate.seed.numBonds) << 16) | static_cast<unsigned int>(candidate.seed.numAtoms);
    canImprove = candidateScore > *bestScore ? 1 : 0;
  }
  canImprove = group.shfl(canImprove, 0);
  if (!canImprove)
    return;
  if (completeRingsOnly && !hasOnlyCompleteRingsCooperative(group,
                                                            candidate,
                                                            queryTopology,
                                                            queryRingBondFlags,
                                                            targetRingBondFlags,
                                                            visitedAtoms,
                                                            changed)) {
    return;
  }
  updateIncumbentCooperative(group, candidate, best, bestScore, bestCopyLock);
}

__device__ __forceinline__ unsigned int clockCycles1024(unsigned long long clocks) {
  const unsigned long long quanta = (clocks + 1023ULL) >> 10;
  return quanta > 0xFFFFFFFFULL ? 0xFFFFFFFFu : static_cast<unsigned int>(quanta);
}

__device__ __forceinline__ void addClockCycles1024ValueWithinThread(unsigned int& dst, unsigned int quanta) {
  const unsigned long long sum = static_cast<unsigned long long>(dst) + static_cast<unsigned long long>(quanta);
  dst                          = sum > 0xFFFFFFFFULL ? 0xFFFFFFFFu : static_cast<unsigned int>(sum);
}

__device__ __forceinline__ void addClockCycles1024WithinThread(unsigned int& dst, unsigned long long clocks) {
  addClockCycles1024ValueWithinThread(dst, clockCycles1024(clocks));
}

template <bool CollectStats,
          int  maxAtoms,
          int  maxBonds,
          int  maxTA,
          int  maxTB,
          class QueryTopology,
          class TargetTopology,
          class GroupT>
__device__ __forceinline__ bool checkSeedMatchAndAppendCooperative(
  const GroupT&                                       group,
  QueuedSeed<maxAtoms, maxBonds, maxTA, maxTB>&       candidate,
  const QueryTopology&                                queryTopology,
  const TargetTopology&                               targetTopology,
  const PairMatchTablesDevice&                        tables,
  FmcsSubstructureScratch<maxAtoms, maxBonds, maxTA>& scratch,
  std::uint8_t*                                       partialStorage,
  int                                                 partialCapacity,
  int*                                                overflowedFlag,
  ExecutionStats&                                     stats,
  bool                                                countPhase2MatchWork) {
  const int groupRank = static_cast<int>(group.thread_rank());
  if constexpr (CollectStats || kFmcsMeasure) {
    if (groupRank == 0) {
      stats.seedChecks += 1u;
      stats.matchCalls += 1u;
    }
  }

  int hasStoredMatch = 0;
  if (groupRank == 0)
    hasStoredMatch = candidate.match.empty ? 0 : 1;
  hasStoredMatch = group.shfl(hasStoredMatch, 0);
  group.sync();

  bool ok = false;
  if (hasStoredMatch != 0) {
    if constexpr (CollectStats || kFmcsMeasure) {
      if (groupRank == 0)
        stats.fastAttempts += 1u;
    }
    unsigned long long matchStartClock = 0;
    if constexpr (CollectStats || kFmcsMeasure) {
      if (groupRank == 0)
        matchStartClock = clock64();
    }
    ok = matchIncrementalFastCooperative(group, candidate.seed, queryTopology, targetTopology, tables, candidate.match);
    group.sync();
    if constexpr (CollectStats || kFmcsMeasure) {
      if (groupRank == 0) {
        const unsigned long long elapsed = clock64() - matchStartClock;
        addClockCycles1024WithinThread(stats.incrementalMatchCycles1024, elapsed);
        if (countPhase2MatchWork) {
          addClockCycles1024WithinThread(stats.phase2ActiveMatchCycles1024, elapsed);
        }
        if (ok)
          stats.fastSuccess += 1u;
      }
    }
  }

  if (!ok) {
    if constexpr (CollectStats || kFmcsMeasure) {
      if (groupRank == 0)
        stats.fallbackCalls += 1u;
    }
    const bool         overflowBefore  = overflowedFlag != nullptr ? atomicAdd(overflowedFlag, 0) != 0 : false;
    unsigned long long matchStartClock = 0;
    if constexpr (CollectStats || kFmcsMeasure) {
      if (groupRank == 0)
        matchStartClock = clock64();
    }
    ok = matchSeedSubstructureCooperative < !CollectStats && !kFmcsMeasure > (group,
                                                                              candidate.seed,
                                                                              queryTopology,
                                                                              targetTopology,
                                                                              tables,
                                                                              candidate.match,
                                                                              scratch,
                                                                              partialStorage,
                                                                              partialCapacity,
                                                                              overflowedFlag);
    group.sync();
    if constexpr (CollectStats || kFmcsMeasure) {
      if (groupRank == 0) {
        const unsigned long long elapsed = clock64() - matchStartClock;
        addClockCycles1024WithinThread(stats.substructureMatchCycles1024, elapsed);
        if (countPhase2MatchWork) {
          addClockCycles1024WithinThread(stats.phase2ActiveMatchCycles1024, elapsed);
        }
        if (ok) {
          stats.fallbackSuccess += 1u;
        } else if (overflowedFlag != nullptr && atomicAdd(overflowedFlag, 0) != 0 && !overflowBefore) {
          stats.fallbackOverflow += 1u;
        } else {
          stats.fallbackFail += 1u;
        }
      }
    }
  }

  if constexpr (CollectStats || kFmcsMeasure) {
    if (groupRank == 0 && ok)
      stats.matchFound += 1u;
  }
  return ok;
}

template <bool CollectStats,
          int  maxAtoms,
          int  maxBonds,
          int  maxTA,
          int  maxTB,
          class QueryTopology,
          class TargetTopology,
          class GroupT>
__device__ __forceinline__ bool matchInitialSeedBondCooperative(const GroupT&                                 group,
                                                                QueuedSeed<maxAtoms, maxBonds, maxTA, maxTB>& candidate,
                                                                int                          queryBondIdx,
                                                                const QueryTopology&         queryTopology,
                                                                const TargetTopology&        targetTopology,
                                                                const PairMatchTablesDevice& tables,
                                                                ExecutionStats&              stats) {
  if constexpr (!CollectStats && !kFmcsMeasure) {
    (void)stats;
  }
  const int groupRank  = static_cast<int>(group.thread_rank());
  const int groupCount = static_cast<int>(group.num_threads());

  if constexpr (CollectStats || kFmcsMeasure) {
    if (groupRank == 0) {
      stats.seedChecks += 1u;
      stats.matchCalls += 1u;
    }
  }

  int chosenTargetBond  = -1;
  int chosenTargetAtomU = -1;
  int chosenTargetAtomV = -1;
  for (int targetBondIdx = groupRank; targetBondIdx < targetTopology.numBonds && chosenTargetBond < 0;
       targetBondIdx += groupCount) {
    SingleBondMatch singleMatch{};
    if (matchSingleBondWithinThread(queryBondIdx,
                                    targetBondIdx,
                                    false,
                                    queryTopology,
                                    targetTopology,
                                    tables,
                                    singleMatch)) {
      chosenTargetBond  = targetBondIdx;
      chosenTargetAtomU = static_cast<int>(singleMatch.targetAtomU);
      chosenTargetAtomV = static_cast<int>(singleMatch.targetAtomV);
    } else if (matchSingleBondWithinThread(queryBondIdx,
                                           targetBondIdx,
                                           true,
                                           queryTopology,
                                           targetTopology,
                                           tables,
                                           singleMatch)) {
      chosenTargetBond  = targetBondIdx;
      chosenTargetAtomU = static_cast<int>(singleMatch.targetAtomU);
      chosenTargetAtomV = static_cast<int>(singleMatch.targetAtomV);
    }
  }

  const unsigned int foundMask = group.ballot(chosenTargetBond >= 0 ? 1u : 0u);
  if (foundMask == 0u) {
    return false;
  }

  const int winningLane   = __ffs(foundMask) - 1;
  const int targetBondIdx = group.shfl(chosenTargetBond, winningLane);
  const int targetAtomU   = group.shfl(chosenTargetAtomU, winningLane);
  const int targetAtomV   = group.shfl(chosenTargetAtomV, winningLane);

  if (groupRank == 0) {
    const std::uint32_t queryEndpoints   = queryTopology.bondEndpoints[queryBondIdx];
    const int           queryEndpointU   = static_cast<int>(queryEndpoints >> kBondEndpointShift);
    const int           queryEndpointV   = static_cast<int>(queryEndpoints & kBondEndpointMask);
    using MatchT                         = MatchResult<maxAtoms, maxBonds, maxTA, maxTB>;
    using TargetAtomWord                 = typename MatchT::target_atom_word;
    using TargetBondWord                 = typename MatchT::target_bond_word;
    constexpr int kTargetAtomBitsPerWord = MatchT::kTargetAtomBitsPerWord;
    constexpr int kTargetBondBitsPerWord = MatchT::kTargetBondBitsPerWord;

    matchResultClearWithinThread(candidate.match);
    candidate.match.targetAtomIdx[queryEndpointU] = static_cast<std::uint8_t>(targetAtomU);
    candidate.match.targetAtomIdx[queryEndpointV] = static_cast<std::uint8_t>(targetAtomV);
    candidate.match.targetBondIdx[queryBondIdx]   = static_cast<std::uint8_t>(targetBondIdx);
    candidate.match.visitedTargetAtoms[targetAtomU / kTargetAtomBitsPerWord] |= static_cast<TargetAtomWord>(1)
                                                                             << (targetAtomU % kTargetAtomBitsPerWord);
    candidate.match.visitedTargetAtoms[targetAtomV / kTargetAtomBitsPerWord] |= static_cast<TargetAtomWord>(1)
                                                                             << (targetAtomV % kTargetAtomBitsPerWord);
    candidate.match.visitedTargetBonds[targetBondIdx / kTargetBondBitsPerWord] |=
      static_cast<TargetBondWord>(1) << (targetBondIdx % kTargetBondBitsPerWord);
    candidate.match.matchedAtomSize = static_cast<std::uint16_t>(queryEndpointU == queryEndpointV ? 1 : 2);
    candidate.match.matchedBondSize = 1;
    candidate.match.empty           = false;
    if constexpr (CollectStats || kFmcsMeasure) {
      stats.matchFound += 1u;
    }
  }
  group.sync();
  return true;
}

__device__ __forceinline__ bool isPowerOfTwo64(unsigned long long value) {
  return value != 0ULL && (value & (value - 1ULL)) == 0ULL;
}

__device__ __forceinline__ void addExecutionStatsWithinThread(ExecutionStats& dst, const ExecutionStats& src) {
  dst.phase2Iters += src.phase2Iters;
  dst.initialSeeds += src.initialSeeds;
  dst.mismatchedInitialSeeds += src.mismatchedInitialSeeds;
  dst.popped += src.popped;
  dst.seedChecks += src.seedChecks;
  dst.matchCalls += src.matchCalls;
  dst.matchFound += src.matchFound;
  dst.boundRejected += src.boundRejected;
  dst.expanded += src.expanded;
  dst.fillZero += src.fillZero;
  dst.stage0Attempts += src.stage0Attempts;
  dst.stage0Success += src.stage0Success;
  dst.stage1Attempts += src.stage1Attempts;
  dst.stage1Success += src.stage1Success;
  dst.stage2Attempts += src.stage2Attempts;
  dst.stage2Success += src.stage2Success;
  dst.individualBondExcluded += src.individualBondExcluded;
  dst.fastAttempts += src.fastAttempts;
  dst.fastSuccess += src.fastSuccess;
  dst.fallbackCalls += src.fallbackCalls;
  dst.fallbackSuccess += src.fallbackSuccess;
  dst.fallbackFail += src.fallbackFail;
  dst.fallbackOverflow += src.fallbackOverflow;
  if (src.maxQueue > dst.maxQueue)
    dst.maxQueue = src.maxQueue;
  dst.forcedExit |= src.forcedExit;
  dst.totalClocks += src.totalClocks;
  dst.phase1Clocks += src.phase1Clocks;
  dst.phase2Clocks += src.phase2Clocks;
  addClockCycles1024ValueWithinThread(dst.incrementalMatchCycles1024, src.incrementalMatchCycles1024);
  addClockCycles1024ValueWithinThread(dst.substructureMatchCycles1024, src.substructureMatchCycles1024);
  addClockCycles1024ValueWithinThread(dst.phase2PopSyncWaitCycles1024, src.phase2PopSyncWaitCycles1024);
  addClockCycles1024ValueWithinThread(dst.phase2SyncWaitCycles1024, src.phase2SyncWaitCycles1024);
  addClockCycles1024ValueWithinThread(dst.phase2IdleNoSeedWaitCycles1024, src.phase2IdleNoSeedWaitCycles1024);
  addClockCycles1024ValueWithinThread(dst.phase2IdleNoMatchWaitCycles1024, src.phase2IdleNoMatchWaitCycles1024);
  addClockCycles1024ValueWithinThread(dst.phase2ActiveWorkCycles1024, src.phase2ActiveWorkCycles1024);
  addClockCycles1024ValueWithinThread(dst.phase2ActiveMatchCycles1024, src.phase2ActiveMatchCycles1024);
}

template <int maxAtoms, int maxBonds>
__device__ __forceinline__ void seedVisitRemainingBondWithinThread(
  Seed<maxAtoms, maxBonds>&                          seed,
  const int                                          bondIdx,
  const int                                          otherAtom,
  std::uint8_t*                                      atomStack,
  typename Seed<maxAtoms, maxBonds>::atom_word_type* visitedAtoms,
  typename Seed<maxAtoms, maxBonds>::bond_word_type* visitedBonds,
  int*                                               stackSize) {
  using SeedT                    = Seed<maxAtoms, maxBonds>;
  using AtomWord                 = typename SeedT::atom_word_type;
  using BondWord                 = typename SeedT::bond_word_type;
  constexpr int kAtomBitsPerWord = SeedT::kAtomBitsPerWord;
  constexpr int kBondBitsPerWord = SeedT::kBondBitsPerWord;

  const int      bondWordIdx = bondIdx / kBondBitsPerWord;
  const BondWord bondMask    = static_cast<BondWord>(1) << (bondIdx % kBondBitsPerWord);
  if ((visitedBonds[bondWordIdx] & bondMask) != 0)
    return;

  visitedBonds[bondWordIdx] |= bondMask;
  seed.remainingBonds += 1;
  const int      atomWordIdx = otherAtom / kAtomBitsPerWord;
  const AtomWord atomMask    = static_cast<AtomWord>(1) << (otherAtom % kAtomBitsPerWord);
  if ((visitedAtoms[atomWordIdx] & atomMask) == 0) {
    visitedAtoms[atomWordIdx] |= atomMask;
    seed.remainingAtoms += 1;
    atomStack[(*stackSize)++] = static_cast<std::uint8_t>(otherAtom);
  }
}

template <int maxAtoms, int maxBonds>
__device__ __forceinline__ void seedVisitRemainingIncidentBondsWithinThread(
  Seed<maxAtoms, maxBonds>&                          seed,
  const DeviceCsrView&                               queryTopology,
  const int                                          atomIdx,
  std::uint8_t*                                      atomStack,
  typename Seed<maxAtoms, maxBonds>::atom_word_type* visitedAtoms,
  typename Seed<maxAtoms, maxBonds>::bond_word_type* visitedBonds,
  int*                                               stackSize) {
  const int begin = static_cast<int>(queryTopology.rowOffsets[atomIdx]);
  const int end   = static_cast<int>(queryTopology.rowOffsets[atomIdx + 1]);
  for (int edgeIdx = begin; edgeIdx < end; ++edgeIdx) {
    const int bondIdx   = static_cast<int>(queryTopology.bondIndices[edgeIdx]);
    const int otherAtom = static_cast<int>(queryTopology.colIndices[edgeIdx]);
    seedVisitRemainingBondWithinThread(seed, bondIdx, otherAtom, atomStack, visitedAtoms, visitedBonds, stackSize);
  }
}

template <int maxAtoms, int maxBonds, class GroupT>
__device__ __forceinline__ void seedComputeRemainingSizeRdkitCooperative(
  const GroupT&                                      group,
  Seed<maxAtoms, maxBonds>&                          seed,
  const DeviceCsrView&                               queryTopology,
  std::uint8_t*                                      atomStack,
  typename Seed<maxAtoms, maxBonds>::atom_word_type* visitedAtoms,
  typename Seed<maxAtoms, maxBonds>::bond_word_type* visitedBonds,
  int*                                               stackSize) {
  using SeedT                    = Seed<maxAtoms, maxBonds>;
  using AtomWord                 = typename SeedT::atom_word_type;
  constexpr int kAtomBitsPerWord = SeedT::kAtomBitsPerWord;

  if (group.thread_rank() == 0) {
    seed.remainingAtoms = 0;
    seed.remainingBonds = 0;
    *stackSize          = 0;
    for (int i = 0; i < SeedT::kAtomWords; ++i) {
      visitedAtoms[i] = seed.atoms[i];
    }
    for (int i = 0; i < SeedT::kBondWords; ++i) {
      visitedBonds[i] = seed.excludedBonds[i];
    }

    for (int wordIdx = 0; wordIdx < SeedT::kAtomWords; ++wordIdx) {
      AtomWord remaining = seed.lastAddedAtoms[wordIdx];
      while (remaining != 0) {
        int bitPosInWord;
        if constexpr (sizeof(AtomWord) == 4) {
          bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
        } else {
          bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
        }
        const int atomIdx = wordIdx * kAtomBitsPerWord + bitPosInWord;
        remaining &= remaining - 1;

        seedVisitRemainingIncidentBondsWithinThread(seed,
                                                    queryTopology,
                                                    atomIdx,
                                                    atomStack,
                                                    visitedAtoms,
                                                    visitedBonds,
                                                    stackSize);
      }
    }

    while (*stackSize > 0) {
      const int atomIdx = atomStack[--(*stackSize)];
      seedVisitRemainingIncidentBondsWithinThread(seed,
                                                  queryTopology,
                                                  atomIdx,
                                                  atomStack,
                                                  visitedAtoms,
                                                  visitedBonds,
                                                  stackSize);
    }
  }
  group.sync();
}

/// One CUDA block per pair.  Three-phase seed-grow search:
///   Phase 1: RDKit makeInitialSeeds() analogue.  Build one query-bond
///   seed at a time, run checkIfMatchAndAppend() via substructure search,
///   store one witness MatchResult, and carry RDKit's initial
///   ExcludedBonds prefix behavior.
///   Phase 2: every warp group pops from the block worklist and cooperatively
///   processes the RDKit Seed::grow() stages:
///   canGrowBiggerThan, fillNewBonds, Stage 0 all-outgoing-bonds child,
///   Stage 1 individual-bond pruning, and Stage 2 subset enumeration.
///   Exit on empty queue, timeout, or queue overflow.
///   Phase 3: block lane 0 writes the incumbent into DeviceMCSResult.
///
/// @p queueStorageAll points at a host-allocated global-memory slab of
/// @p queueCapacity * numPairs @c QueuedSeed entries; this block uses
/// the slice starting at @p queueStorageAll[blockIdx.x * queueCapacity].
///
template <int                 maxAtoms,
          int                 maxBonds,
          int                 blockThreads,
          bool                CollectTimings,
          bool                CollectStats,
          FmcsScratchLocation ScratchLoc = FmcsScratchLocation::Shared>
__global__ void fmcsKernel(const DevicePerPairInput* __restrict__ pairs,
                           DeviceMCSResult<maxAtoms, maxBonds>* __restrict__ results,
                           QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>* __restrict__ queueStorageAll,
                           std::uint8_t* __restrict__ substructureStorageAll,
                           FmcsSubstructureScratch<maxAtoms, maxBonds, maxAtoms>* __restrict__ scratchStorageAll,
                           unsigned long long* __restrict__ elapsedClocks,
                           ExecutionStats* __restrict__ statsOut,
                           int                queueCapacity,
                           int                substructurePartialCapacity,
                           int                numPairs,
                           unsigned long long timeoutClocks) {
  static_assert(ScratchLoc != FmcsScratchLocation::Auto,
                "Auto is host-only; resolve to Shared or Global before kernel selection");
  if constexpr (!CollectTimings) {
    (void)elapsedClocks;
  }
  if constexpr (!CollectTimings && !CollectStats) {
    (void)statsOut;
  }

  const int pairIdx = blockIdx.x;
  if (pairIdx >= numPairs)
    return;

  auto                      block = cg::this_thread_block();
  const DevicePerPairInput& pair  = pairs[pairIdx];

  if constexpr (kFmcsDebug) {
    if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
      printf("[fmcs] kernel start: pair=%d q=(a%d,b%d) t=(a%d,b%d)\n",
             pairIdx,
             pair.queryNumAtoms,
             pair.queryNumBonds,
             pair.targetNumAtoms,
             pair.targetNumBonds);
    }
  }

  using QueuedT                      = QueuedSeed<maxAtoms, maxBonds, maxAtoms, maxBonds>;
  using SubstructureScratchT         = FmcsSubstructureScratch<maxAtoms, maxBonds, maxAtoms>;
  constexpr int  kMaxNewBondsForTier = maxBonds;
  constexpr int  kNumGroups          = FmcsBlockConfig<blockThreads>::numGroups;
  constexpr bool kStatsEnabled       = CollectStats || kFmcsMeasure;
  // Block-shared resources: the queue, the incumbent, and the early-exit
  // flags are visible to every group.  Cross-group incumbent updates use
  // atomics.  Phase-2 queue operations hold queueLock across the queue header
  // update and the cooperative payload copy so pushes cannot expose unwritten
  // slots to concurrent poppers.
  __shared__ SeedQueue<QueuedT, ThreadBlockScope> queue;
  __shared__ __align__(16) unsigned char bestStorage[sizeof(QueuedT)];
  QueuedT&                               best = *reinterpret_cast<QueuedT*>(bestStorage);
  // Atomic incumbent score: high 16 bits = numBonds, low 16 bits =
  // numAtoms.  Groups race through atomicCAS on this single int to
  // claim the right to write @c best.
  __shared__ unsigned int                bestScore;
  __shared__ int                         bestCopyLock;
  __shared__ DeviceCsrView               queryView;
  __shared__ DeviceCsrView               targetView;
  __shared__ int                         overflowed;
  __shared__ int                         timedOut;
  __shared__ int                         phase2Done;
  __shared__ int                         phase2ActiveGroups;
  __shared__ int                         queueLock;
  __shared__ unsigned long long          startClock;
  __shared__ unsigned long long          phase1StartClock;
  __shared__ unsigned long long          phase2StartClock;
  // Substructure scratch placement: static shared (historical) or a slice of
  // the per-block global slab.  A __shared__ declaration inside the
  // if-constexpr block has static storage duration, so the pointer stays valid
  // for the whole kernel; under Global no shared bytes are allocated at all.
  // TODO(blockSize>512): extend this mechanism to newBondsArr and the
  // current/biggest QueuedSeed copies (see
  // analysis/fmcs_scratch_placement_plan.md section 6).
  SubstructureScratchT*                  substructureScratch;
  if constexpr (ScratchLoc == FmcsScratchLocation::Shared) {
    (void)scratchStorageAll;
    __shared__ SubstructureScratchT substructureScratchShared[kNumGroups];
    substructureScratch = substructureScratchShared;
  } else {
    substructureScratch = scratchStorageAll + static_cast<size_t>(pairIdx) * kNumGroups;
  }
  __shared__ ExecutionStats groupStats[kStatsEnabled ? kNumGroups : 1];
  __shared__ ExecutionStats measureStats;

  // Cooperative Phase 2 working state.  Approach 1 gives each warp group an
  // independent seed workspace, fallback scratch, and global partial-storage
  // slice so groups can pop and grow seeds concurrently.
  __shared__ __align__(16) unsigned char currentStorage[sizeof(QueuedT) * kNumGroups];
  __shared__ __align__(16) unsigned char biggestStorage[sizeof(QueuedT) * kNumGroups];
  QueuedT*                               current = reinterpret_cast<QueuedT*>(currentStorage);
  QueuedT*                               biggest = reinterpret_cast<QueuedT*>(biggestStorage);
  __shared__ NewBond                     newBondsArr[kNumGroups][kMaxNewBondsForTier];
  __shared__ int                         newBondCount[kNumGroups];
  __shared__ bool                        stage0Ok[kNumGroups];
  __shared__ std::uint8_t remainingAtomStack[kNumGroups][maxAtoms];
  __shared__ typename Seed<maxAtoms, maxBonds>::atom_word_type
    remainingVisitedAtoms[kNumGroups][Seed<maxAtoms, maxBonds>::kAtomWords];
  __shared__ typename Seed<maxAtoms, maxBonds>::bond_word_type
                 remainingVisitedBonds[kNumGroups][Seed<maxAtoms, maxBonds>::kBondWords];
  __shared__ int remainingStackSize[kNumGroups];
  __shared__
    typename Seed<maxAtoms, maxBonds>::bond_word_type initialExcludedBonds[Seed<maxAtoms, maxBonds>::kBondWords];

  auto                  group     = cg::tiled_partition<kFmcsGroupSize>(block);
  const int             groupId   = mark_warp_uniform(static_cast<int>(block.thread_rank()) / kFmcsGroupSize);
  const int             groupRank = static_cast<int>(group.thread_rank());
  SubstructureScratchT& mySubstructureScratch = substructureScratch[groupId];
  ExecutionStats&       myStats               = kStatsEnabled ? groupStats[groupId] : measureStats;

  QueuedT*      myQueueStorage = queueStorageAll + static_cast<size_t>(pairIdx) * queueCapacity;
  std::uint8_t* mySubstructureStorage =
    substructureStorageAll +
    (static_cast<size_t>(pairIdx) * static_cast<size_t>(kNumGroups) + static_cast<size_t>(groupId)) * 2u *
      static_cast<size_t>(substructurePartialCapacity) * static_cast<size_t>(maxAtoms);

  if (block.thread_rank() == 0) {
    queue.init(myQueueStorage, queueCapacity);
    seedClearWithinThread(best.seed);
    matchResultClearWithinThread(best.match);
    bestScore    = 0;
    bestCopyLock = 0;
    if constexpr (CollectTimings || CollectStats || kFmcsMeasure) {
      measureStats = ExecutionStats{};
    }
    overflowed         = 0;
    timedOut           = 0;
    phase2Done         = 0;
    phase2ActiveGroups = 0;
    queueLock          = 0;
    if constexpr (CollectTimings || kStatsEnabled) {
      startClock = clock64();
    } else {
      startClock = timeoutClocks > 0 ? clock64() : 0;
    }
    phase1StartClock = startClock;
    phase2StartClock = startClock;

    queryView.rowOffsets    = pair.queryRowOffsets;
    queryView.colIndices    = pair.queryColIndices;
    queryView.bondIndices   = pair.queryBondIndices;
    queryView.bondEndpoints = pair.queryBondEndpoints;
    queryView.numAtoms      = pair.queryNumAtoms;
    queryView.numBonds      = pair.queryNumBonds;

    targetView.rowOffsets    = pair.targetRowOffsets;
    targetView.colIndices    = pair.targetColIndices;
    targetView.bondIndices   = pair.targetBondIndices;
    targetView.bondEndpoints = pair.targetBondEndpoints;
    targetView.numAtoms      = pair.targetNumAtoms;
    targetView.numBonds      = pair.targetNumBonds;
  }
  if constexpr (CollectStats || kFmcsMeasure) {
    for (int statIdx = static_cast<int>(block.thread_rank()); statIdx < kNumGroups;
         statIdx += static_cast<int>(blockDim.x)) {
      groupStats[statIdx] = ExecutionStats{};
    }
  }
  // Publish block-shared kernel initialization and stats zeroing before any
  // group enters phase 1.
  block.sync();
  if constexpr (CollectTimings || CollectStats || kFmcsMeasure) {
    if (block.thread_rank() == 0) {
      phase1StartClock = clock64();
    }
  }
  // Publish phase1StartClock for stats collection before phase-1 work begins.
  block.sync();

  if constexpr (kFmcsDebug) {
    if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
      printf("[fmcs] init done\n");
    }
  }

  QueuedT&      myCurrent               = current[groupId];
  QueuedT&      myBiggest               = biggest[groupId];
  NewBond*      myNewBonds              = newBondsArr[groupId];
  std::uint8_t* myRemainingAtomStack    = remainingAtomStack[groupId];
  auto*         myRemainingVisitedAtoms = remainingVisitedAtoms[groupId];
  auto*         myRemainingVisitedBonds = remainingVisitedBonds[groupId];

  // ---- Phase 1: RDKit makeInitialSeeds() analogue ----
  // RDKit creates one initial seed per query bond, not one per target
  // embedding.  Each candidate goes through checkIfMatchAndAppend(), which
  // runs substructure matching and stores one witness MatchResult on success.
  // Initial ExcludedBonds is prefix-like: later initial seeds exclude earlier
  // query bonds, and a mismatched initial bond is also excluded from seeds
  // already admitted.  Keep this serial qBond order for RDKit parity; the
  // one-bond match itself is specialized below instead of going through the
  // general fallback matcher.
  if (block.thread_rank() < Seed<maxAtoms, maxBonds>::kBondWords) {
    initialExcludedBonds[block.thread_rank()] = 0;
  }
  // Publish initialExcludedBonds zeroing before group 0 builds initial seeds.
  block.sync();

  if (groupId == 0) {
    for (int qBond = 0; qBond < pair.queryNumBonds && overflowed == 0 && timedOut == 0; ++qBond) {
      if (groupRank == 0) {
        seedClearWithinThread(myCurrent.seed);
        matchResultClearWithinThread(myCurrent.match);
        for (int wordIdx = 0; wordIdx < Seed<maxAtoms, maxBonds>::kBondWords; ++wordIdx) {
          myCurrent.seed.excludedBonds[wordIdx] = initialExcludedBonds[wordIdx];
        }

        const std::uint32_t queryEndpoints = queryView.bondEndpoints[qBond];
        const int           queryEndpointU = static_cast<int>(queryEndpoints >> kBondEndpointShift);
        const int           queryEndpointV = static_cast<int>(queryEndpoints & kBondEndpointMask);
        seedAddBondWithinThread(myCurrent.seed, qBond);
        seedAddAtomWithinThread(myCurrent.seed, queryEndpointU);
        seedAddAtomWithinThread(myCurrent.seed, queryEndpointV);
        myCurrent.seed.growingStage = kSeedGrowStageOuter;
        if constexpr (CollectStats || kFmcsMeasure) {
          myStats.initialSeeds += 1u;
        }
      }
      group.sync();
      seedComputeRemainingSizeRdkitCooperative(group,
                                               myCurrent.seed,
                                               queryView,
                                               myRemainingAtomStack,
                                               myRemainingVisitedAtoms,
                                               myRemainingVisitedBonds,
                                               &remainingStackSize[groupId]);

      const bool matched = matchInitialSeedBondCooperative<CollectStats>(group,
                                                                         myCurrent,
                                                                         qBond,
                                                                         queryView,
                                                                         targetView,
                                                                         pair.tables,
                                                                         myStats);
      if (matched) {
        updateCompleteRingsIncumbentCooperative(group,
                                                myCurrent,
                                                best,
                                                &bestScore,
                                                &bestCopyLock,
                                                pair.completeRingsOnly,
                                                queryView,
                                                pair.queryRingBondFlags,
                                                pair.targetRingBondFlags,
                                                myRemainingVisitedAtoms,
                                                &remainingStackSize[groupId]);
        if (!pushBackCooperative(group, queue, myCurrent)) {
          atomicExch(&overflowed, 1);
        }
      } else if (groupRank == 0) {
        const int queuedSeeds = queue.size();
        for (int i = 0; i < queuedSeeds; ++i) {
          seedExcludeBondWithinThread(queue.slot(i).seed, qBond);
        }
        if constexpr (CollectStats || kFmcsMeasure) {
          myStats.mismatchedInitialSeeds += 1u;
        }
      }
      group.sync();

      if (groupRank == 0) {
        const int  wordIdx = qBond / Seed<maxAtoms, maxBonds>::kBondBitsPerWord;
        const auto mask    = static_cast<typename Seed<maxAtoms, maxBonds>::bond_word_type>(1)
                       << (qBond % Seed<maxAtoms, maxBonds>::kBondBitsPerWord);
        initialExcludedBonds[wordIdx] |= mask;
      }
      group.sync();
    }
  }

  // Phase-1/phase-2 handoff: every group must finish initial seed queue
  // mutation before block thread 0 records phase-1 timing and checks timeout.
  block.sync();
  if constexpr (CollectTimings || CollectStats || kFmcsMeasure) {
    if (block.thread_rank() == 0) {
      const unsigned long long phase1Clocks = clock64() - phase1StartClock;
      if constexpr (CollectStats || kFmcsMeasure) {
        groupStats[0].phase1Clocks = phase1Clocks;
      }
      if constexpr (CollectTimings) {
        measureStats.phase1Clocks = phase1Clocks;
      }
    }
  }
  if (block.thread_rank() == 0 && timeoutClocks > 0 && clock64() - startClock > timeoutClocks) {
    atomicExch(&timedOut, 1);
  }
  if constexpr (CollectTimings || CollectStats || kFmcsMeasure) {
    if (block.thread_rank() == 0) {
      phase2StartClock = clock64();
    }
  }

  if constexpr (kFmcsDebug) {
    if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
      printf("[fmcs] phase1 done: acceptedInitialSeeds=%d queryBonds=%d overflowed=%d\n",
             queue.size(),
             pair.queryNumBonds,
             static_cast<int>(overflowed));
    }
  }

  // ---- Phase 2: RDKit Seed::grow() analogue ----

  // Each warp group independently pops, grows, and pushes seeds.  The queue
  // lock serializes only queue mutation; activeGroups distinguishes a truly
  // empty search from a transient empty queue while other groups may still
  // push children.
  [[maybe_unused]] int debugIter = 0;
  while (true) {
    if (readFlagCooperative(group, &phase2Done) || readFlagCooperative(group, &overflowed) ||
        readFlagCooperative(group, &timedOut)) {
      break;
    }

    if constexpr (CollectStats || kFmcsMeasure) {
      if (groupRank == 0)
        myStats.phase2Iters += 1;
    }

    if constexpr (kFmcsDebug) {
      if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
        printf("[fmcs][grp %d iter %d] best=(b%d,a%d)\n",
               groupId,
               debugIter,
               static_cast<int>(best.seed.numBonds),
               static_cast<int>(best.seed.numAtoms));
      }
    }

    unsigned long long phase2PopBarrierStartClock = 0;
    if constexpr (CollectStats || kFmcsMeasure) {
      if (groupRank == 0) {
        phase2PopBarrierStartClock = clock64();
      }
    }
    bool               phase2Finished            = false;
    const bool         poppedThisGroup           = popBackLockedOrFinishCooperative(group,
                                                                  queue,
                                                                  myCurrent,
                                                                  &queueLock,
                                                                  &phase2ActiveGroups,
                                                                  &phase2Done,
                                                                  &overflowed,
                                                                  &timedOut,
                                                                  phase2Finished);
    unsigned long long phase2GroupWorkStartClock = 0;
    unsigned int       phase2MatchCallsBefore    = 0;
    if constexpr (CollectStats || kFmcsMeasure) {
      if (groupRank == 0) {
        const unsigned long long afterPopBarrierClock = clock64();
        addClockCycles1024WithinThread(myStats.phase2PopSyncWaitCycles1024,
                                       afterPopBarrierClock - phase2PopBarrierStartClock);
        phase2GroupWorkStartClock = afterPopBarrierClock;
        phase2MatchCallsBefore    = myStats.matchCalls;
        if (poppedThisGroup)
          myStats.popped += 1u;
      }
    }
    if (phase2Finished)
      break;
    if (!poppedThisGroup)
      continue;

    do {
      if constexpr (kFmcsDebug) {
        if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
          printf("[fmcs][grp %d iter %d] A: popped seed (b%d,a%d)\n",
                 groupId,
                 debugIter,
                 static_cast<int>(myCurrent.seed.numBonds),
                 static_cast<int>(myCurrent.seed.numAtoms));
        }
      }

      // RDKit growSeeds() increments TotalSteps before Seed::grow(), then
      // Seed::grow() first checks canGrowBiggerThan().
      if constexpr (CollectStats || kFmcsMeasure) {
        if (groupRank == 0)
          myStats.expanded += 1u;
      }

      const unsigned int scoreSnapshot     = readBestScoreCooperative(group, &bestScore);
      const int          bestBondsSnapshot = static_cast<int>(scoreSnapshot >> 16);
      const int          bestAtomsSnapshot = static_cast<int>(scoreSnapshot & 0xFFFFu);
      const bool         canGrowCurrent =
        mark_warp_uniform(
          seedCanGrowBiggerThanWithinThread(myCurrent.seed, bestBondsSnapshot, bestAtomsSnapshot) ? 1 : 0) != 0;
      group.sync();
      if (!canGrowCurrent) {
        if constexpr (CollectStats || kFmcsMeasure) {
          if (groupRank == 0)
            myStats.boundRejected += 1u;
        }
        break;
      }

      updateCompleteRingsIncumbentCooperative(group,
                                              myCurrent,
                                              best,
                                              &bestScore,
                                              &bestCopyLock,
                                              pair.completeRingsOnly,
                                              queryView,
                                              pair.queryRingBondFlags,
                                              pair.targetRingBondFlags,
                                              myRemainingVisitedAtoms,
                                              &remainingStackSize[groupId]);

      if constexpr (kFmcsDebug) {
        if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
          printf("[fmcs][grp %d iter %d] B: after incumbent sync\n", groupId, debugIter);
        }
      }

      const bool fillOk = fillNewBondsCooperative(group,
                                                  myCurrent.seed,
                                                  queryView,
                                                  myNewBonds,
                                                  &newBondCount[groupId],
                                                  kMaxNewBondsForTier);
      group.sync();
      if (!fillOk) {
        if (groupRank == 0)
          atomicExch(&overflowed, 1);
        break;
      }
      const int myNewBondCount = mark_warp_uniform(newBondCount[groupId]);

      if constexpr (kFmcsDebug) {
        if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
          printf("[fmcs][grp %d iter %d] C: after fillNewBonds, count=%d stage=%u\n",
                 groupId,
                 debugIter,
                 myNewBondCount,
                 static_cast<unsigned int>(myCurrent.seed.growingStage));
        }
      }
      if (myNewBondCount == 0) {
        if constexpr (CollectStats || kFmcsMeasure) {
          if (groupRank == 0)
            myStats.fillZero += 1u;
        }
        break;
      }

      const int currentGrowStage = mark_warp_uniform(static_cast<int>(myCurrent.seed.growingStage));
      bool      runInnerStage    = currentGrowStage != kSeedGrowStageOuter;

      // RDKit Seed::grow() stage 0: build the child containing all newly
      // discovered outgoing bonds and run checkIfMatchAndAppend().  If this
      // all-bonds child matches and there is more than one new bond, RDKit
      // returns immediately with the parent left at GrowingStage=1; the next
      // outer grow loop resumes the parent at the singleton/subset stage.
      if (currentGrowStage == kSeedGrowStageOuter) {
        if constexpr (CollectStats || kFmcsMeasure) {
          if (groupRank == 0)
            myStats.stage0Attempts += 1u;
        }
        if constexpr (kFmcsDebug) {
          if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
            printf("[fmcs][grp %d iter %d] D: stage0 begin\n", groupId, debugIter);
          }
        }
        warpCopy(group, &myBiggest, &myCurrent, sizeof(QueuedT));
        group.sync();
        if (groupRank == 0) {
          seedBeginGrowStepWithinThread(myBiggest.seed);
          myBiggest.seed.growingStage = kSeedGrowStageOuter;
          for (int i = 0; i < myNewBondCount; ++i) {
            seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
          }
        }
        group.sync();
        seedComputeRemainingSizeRdkitCooperative(group,
                                                 myBiggest.seed,
                                                 queryView,
                                                 myRemainingAtomStack,
                                                 myRemainingVisitedAtoms,
                                                 myRemainingVisitedBonds,
                                                 &remainingStackSize[groupId]);

        const unsigned int childScoreSnapshot = readBestScoreCooperative(group, &bestScore);
        const int          childBestBonds     = static_cast<int>(childScoreSnapshot >> 16);
        const int          childBestAtoms     = static_cast<int>(childScoreSnapshot & 0xFFFFu);
        const bool         canGrowChild =
          mark_warp_uniform(seedCanGrowBiggerThanWithinThread(myBiggest.seed, childBestBonds, childBestAtoms) ? 1 :
                                                                                                                0) != 0;
        group.sync();
        if (!canGrowChild) {
          if constexpr (CollectStats || kFmcsMeasure) {
            if (groupRank == 0)
              myStats.boundRejected += 1u;
          }
          break;
        }

        const bool ok = mark_warp_uniform(checkSeedMatchAndAppendCooperative<CollectStats>(group,
                                                                                           myBiggest,
                                                                                           queryView,
                                                                                           targetView,
                                                                                           pair.tables,
                                                                                           mySubstructureScratch,
                                                                                           mySubstructureStorage,
                                                                                           substructurePartialCapacity,
                                                                                           &overflowed,
                                                                                           myStats,
                                                                                           true) ?
                                            1 :
                                            0) != 0;
        if (groupRank == 0)
          stage0Ok[groupId] = ok;
        group.sync();

        if constexpr (kFmcsDebug) {
          if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
            printf("[fmcs][grp %d iter %d] E: stage0 after match, ok=%d\n",
                   groupId,
                   debugIter,
                   static_cast<int>(stage0Ok[groupId]));
          }
        }
        if (stage0Ok[groupId]) {
          if constexpr (CollectStats || kFmcsMeasure) {
            if (groupRank == 0)
              myStats.stage0Success += 1u;
          }
          updateCompleteRingsIncumbentCooperative(group,
                                                  myBiggest,
                                                  best,
                                                  &bestScore,
                                                  &bestCopyLock,
                                                  pair.completeRingsOnly,
                                                  queryView,
                                                  pair.queryRingBondFlags,
                                                  pair.targetRingBondFlags,
                                                  myRemainingVisitedAtoms,
                                                  &remainingStackSize[groupId]);
          if (!pushBackLockedCooperative(group, queue, myBiggest, &queueLock, &overflowed, &timedOut, &phase2Done)) {
            atomicExch(&overflowed, 1);
          }
          group.sync();
          if (myNewBondCount > 1) {
            if (groupRank == 0) {
              myCurrent.seed.growingStage = kSeedGrowStageInner;
            }
            group.sync();
            if (!pushBackLockedCooperative(group, queue, myCurrent, &queueLock, &overflowed, &timedOut, &phase2Done)) {
              atomicExch(&overflowed, 1);
            }
            group.sync();
          }
          break;
        }
        if (myNewBondCount == 1)
          break;
        runInnerStage = true;
      }

      if (!runInnerStage)
        break;

      // RDKit Seed::grow() stage 1: try every individual outgoing bond.
      // A failed individual match excludes that NewBond from later subset
      // enumeration and increments IndividualBondExcluded.
      for (int i = 0; i < myNewBondCount; ++i) {
        if (!myNewBonds[i].alive)
          continue;

        warpCopy(group, &myBiggest, &myCurrent, sizeof(QueuedT));
        group.sync();

        if (groupRank == 0) {
          seedBeginGrowStepWithinThread(myBiggest.seed);
          myBiggest.seed.growingStage = kSeedGrowStageOuter;
          seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
        }
        group.sync();
        seedComputeRemainingSizeRdkitCooperative(group,
                                                 myBiggest.seed,
                                                 queryView,
                                                 myRemainingAtomStack,
                                                 myRemainingVisitedAtoms,
                                                 myRemainingVisitedBonds,
                                                 &remainingStackSize[groupId]);

        if constexpr (CollectStats || kFmcsMeasure) {
          if (groupRank == 0)
            myStats.stage1Attempts += 1u;
        }

        const unsigned int childScoreSnapshot = readBestScoreCooperative(group, &bestScore);
        const int          childBestBonds     = static_cast<int>(childScoreSnapshot >> 16);
        const int          childBestAtoms     = static_cast<int>(childScoreSnapshot & 0xFFFFu);
        const bool         canGrowSingle =
          mark_warp_uniform(seedCanGrowBiggerThanWithinThread(myBiggest.seed, childBestBonds, childBestAtoms) ? 1 :
                                                                                                                0) != 0;
        group.sync();
        if (!canGrowSingle) {
          if constexpr (CollectStats || kFmcsMeasure) {
            if (groupRank == 0)
              myStats.boundRejected += 1u;
          }
          continue;
        }

        const bool ok = mark_warp_uniform(checkSeedMatchAndAppendCooperative<CollectStats>(group,
                                                                                           myBiggest,
                                                                                           queryView,
                                                                                           targetView,
                                                                                           pair.tables,
                                                                                           mySubstructureScratch,
                                                                                           mySubstructureStorage,
                                                                                           substructurePartialCapacity,
                                                                                           &overflowed,
                                                                                           myStats,
                                                                                           true) ?
                                            1 :
                                            0) != 0;
        if constexpr (CollectStats || kFmcsMeasure) {
          if (groupRank == 0 && ok)
            myStats.stage1Success += 1u;
        }
        if (ok) {
          updateCompleteRingsIncumbentCooperative(group,
                                                  myBiggest,
                                                  best,
                                                  &bestScore,
                                                  &bestCopyLock,
                                                  pair.completeRingsOnly,
                                                  queryView,
                                                  pair.queryRingBondFlags,
                                                  pair.targetRingBondFlags,
                                                  myRemainingVisitedAtoms,
                                                  &remainingStackSize[groupId]);
          if (!pushBackLockedCooperative(group, queue, myBiggest, &queueLock, &overflowed, &timedOut, &phase2Done)) {
            atomicExch(&overflowed, 1);
          }
        } else if (groupRank == 0) {
          myNewBonds[i].alive = false;
          if constexpr (CollectStats || kFmcsMeasure) {
            myStats.individualBondExcluded += 1u;
          }
        }
        group.sync();
      }

      // RDKit Seed::grow() stage 2: enumerate all non-singleton subsets of
      // the surviving NewBonds.  Singletons were handled by Stage 1; the
      // all-bonds subset was handled by Stage 0 unless Stage 1 erased one or
      // more individual bonds, in which case the all-surviving-bonds subset is
      // new work and must be checked.
      int aliveCount    = 0;
      int aliveOverflow = 0;
      if (groupRank == 0) {
        for (int i = 0; i < myNewBondCount; ++i) {
          if (myNewBonds[i].alive)
            ++aliveCount;
        }
        aliveOverflow = aliveCount > 63 ? 1 : 0;
        if (aliveOverflow)
          atomicExch(&overflowed, 1);
      }
      aliveCount    = group.shfl(aliveCount, 0);
      aliveOverflow = group.shfl(aliveOverflow, 0);
      if (aliveOverflow)
        break;
      if (aliveCount > 1) {
        unsigned int erasedCount = 0;
        if (groupRank == 0) {
          erasedCount = static_cast<unsigned int>(myNewBondCount - aliveCount);
        }
        erasedCount                             = group.shfl(erasedCount, 0);
        const unsigned long long maxComposition = (1ULL << aliveCount) - 1ULL;
        for (unsigned long long composition = maxComposition; composition != 0ULL; --composition) {
          if (isPowerOfTwo64(composition))
            continue;
          if (erasedCount == 0 && composition == maxComposition)
            continue;

          const unsigned int latestScoreSnapshot = readBestScoreCooperative(group, &bestScore);
          const int          latestBestBonds     = static_cast<int>(latestScoreSnapshot >> 16);
          const int          latestBestAtoms     = static_cast<int>(latestScoreSnapshot & 0xFFFFu);
          const bool         canGrowRemaining =
            mark_warp_uniform(
              seedCanGrowBiggerThanWithinThread(myCurrent.seed, latestBestBonds, latestBestAtoms) ? 1 : 0) != 0;
          group.sync();
          if (!canGrowRemaining) {
            if constexpr (CollectStats || kFmcsMeasure) {
              if (groupRank == 0)
                myStats.boundRejected += 1u;
            }
            break;
          }

          warpCopy(group, &myBiggest, &myCurrent, sizeof(QueuedT));
          group.sync();
          if (groupRank == 0) {
            seedBeginGrowStepWithinThread(myBiggest.seed);
            myBiggest.seed.growingStage = kSeedGrowStageOuter;
            int aliveBit                = 0;
            for (int i = 0; i < myNewBondCount; ++i) {
              if (!myNewBonds[i].alive)
                continue;
              if ((composition & (1ULL << aliveBit)) != 0ULL) {
                seedAddNewBondWithinThread(myBiggest.seed, myNewBonds[i]);
              }
              ++aliveBit;
            }
          }
          group.sync();
          seedComputeRemainingSizeRdkitCooperative(group,
                                                   myBiggest.seed,
                                                   queryView,
                                                   myRemainingAtomStack,
                                                   myRemainingVisitedAtoms,
                                                   myRemainingVisitedBonds,
                                                   &remainingStackSize[groupId]);

          if constexpr (CollectStats || kFmcsMeasure) {
            if (groupRank == 0)
              myStats.stage2Attempts += 1u;
          }
          const unsigned int childScoreSnapshot = readBestScoreCooperative(group, &bestScore);
          const int          childBestBonds     = static_cast<int>(childScoreSnapshot >> 16);
          const int          childBestAtoms     = static_cast<int>(childScoreSnapshot & 0xFFFFu);
          const bool         canGrowSubset =
            mark_warp_uniform(
              seedCanGrowBiggerThanWithinThread(myBiggest.seed, childBestBonds, childBestAtoms) ? 1 : 0) != 0;
          group.sync();
          if (!canGrowSubset) {
            if constexpr (CollectStats || kFmcsMeasure) {
              if (groupRank == 0)
                myStats.boundRejected += 1u;
            }
            continue;
          }

          const bool ok =
            mark_warp_uniform(checkSeedMatchAndAppendCooperative<CollectStats>(group,
                                                                               myBiggest,
                                                                               queryView,
                                                                               targetView,
                                                                               pair.tables,
                                                                               mySubstructureScratch,
                                                                               mySubstructureStorage,
                                                                               substructurePartialCapacity,
                                                                               &overflowed,
                                                                               myStats,
                                                                               true) ?
                                1 :
                                0) != 0;
          if (ok) {
            if constexpr (CollectStats || kFmcsMeasure) {
              if (groupRank == 0)
                myStats.stage2Success += 1u;
            }
            updateCompleteRingsIncumbentCooperative(group,
                                                    myBiggest,
                                                    best,
                                                    &bestScore,
                                                    &bestCopyLock,
                                                    pair.completeRingsOnly,
                                                    queryView,
                                                    pair.queryRingBondFlags,
                                                    pair.targetRingBondFlags,
                                                    myRemainingVisitedAtoms,
                                                    &remainingStackSize[groupId]);
            if (!pushBackLockedCooperative(group, queue, myBiggest, &queueLock, &overflowed, &timedOut, &phase2Done)) {
              atomicExch(&overflowed, 1);
            }
          }
          group.sync();
          if (readFlagCooperative(group, &overflowed))
            break;
        }
      }

      if constexpr (kFmcsDebug) {
        if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
          printf("[fmcs][grp %d iter %d] K: inner stages done\n", groupId, debugIter);
        }
      }
    } while (false);

    unsigned long long phase2GroupWorkEndClock = 0;
    if constexpr (CollectStats || kFmcsMeasure) {
      if (groupRank == 0) {
        phase2GroupWorkEndClock = clock64();
        addClockCycles1024WithinThread(myStats.phase2ActiveWorkCycles1024,
                                       phase2GroupWorkEndClock - phase2GroupWorkStartClock);
        if (phase2MatchCallsBefore == myStats.matchCalls) {
          addClockCycles1024WithinThread(myStats.phase2IdleNoMatchWaitCycles1024,
                                         phase2GroupWorkEndClock - phase2GroupWorkStartClock);
        }
      }
    }

    group.sync();
    if (groupRank == 0) {
      if (timeoutClocks > 0 && clock64() - startClock > timeoutClocks) {
        atomicExch(&timedOut, 1);
        atomicExch(&phase2Done, 1);
      }
      atomicSub(&phase2ActiveGroups, 1);
    }
    group.sync();

    if (readFlagCooperative(group, &overflowed) || readFlagCooperative(group, &timedOut)) {
      if (groupRank == 0)
        atomicExch(&phase2Done, 1);
      break;
    }

    if constexpr (kFmcsDebug) {
      ++debugIter;
      if (debugIter >= kFmcsDebugMaxIters) {
        if (pairIdx == kFmcsDebugPairIdx && groupRank == 0) {
          printf("[fmcs][grp %d] WATCHDOG: hit %d iters -- forcing exit\n", groupId, debugIter);
        }
        if (groupRank == 0) {
          atomicExch(&timedOut, 1);
          atomicExch(&phase2Done, 1);
        }
        break;
      }
    }
    if constexpr (kFmcsMeasure) {
      ++debugIter;
      if (debugIter >= kFmcsMeasureMaxIters) {
        if (groupRank == 0) {
          myStats.forcedExit = 1;
          atomicExch(&timedOut, 1);
          atomicExch(&phase2Done, 1);
        }
        break;
      }
    }
  }

  if constexpr (CollectStats || kFmcsMeasure) {
    // Stats reduction reads every group's final counters after all groups have
    // left phase 2.
    block.sync();
    if (block.thread_rank() == 0) {
      const unsigned long long phase2Clocks = clock64() - phase2StartClock;
      groupStats[0].phase2Clocks            = phase2Clocks;
      measureStats                          = ExecutionStats{};
      for (int statIdx = 0; statIdx < kNumGroups; ++statIdx) {
        addExecutionStatsWithinThread(measureStats, groupStats[statIdx]);
      }
      if constexpr (CollectTimings) {
        measureStats.phase1Clocks = groupStats[0].phase1Clocks;
        measureStats.phase2Clocks = phase2Clocks;
      }
    }
    // Publish measureStats before later writeback copies it to statsOut.
    block.sync();
  } else if constexpr (CollectTimings) {
    block.sync();
    if (block.thread_rank() == 0) {
      measureStats.phase2Clocks = clock64() - phase2StartClock;
    }
    block.sync();
  }

  if constexpr (kFmcsDebug) {
    if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
      printf(
        "[fmcs] phase2 exit: iters=%d queueSize=%d overflowed=%d "
        "timedOut=%d best=(b%d,a%d)\n",
        debugIter,
        queue.size(),
        static_cast<int>(overflowed),
        static_cast<int>(timedOut),
        static_cast<int>(best.seed.numBonds),
        static_cast<int>(best.seed.numAtoms));
    }
  }
  if constexpr (kFmcsMeasure) {
    if (pairIdx == kFmcsDebugPairIdx && block.thread_rank() == 0) {
      printf(
        "[fmcs][measure] iters=%u initial=%u/%u popped=%u expanded=%u "
        "seedChecks=%u match=%u/%u boundReject=%u indivExcluded=%u "
        "fillZero=%u stage0=%u/%u stage1=%u/%u stage2=%u/%u "
        "fast=%u/%u slow=%u success=%u fail=%u overflow=%u "
        "maxQueue=%u forcedExit=%u overflowed=%d timedOut=%d "
        "best=(b%d,a%d) queue=%d\n",
        measureStats.phase2Iters,
        measureStats.initialSeeds - measureStats.mismatchedInitialSeeds,
        measureStats.initialSeeds,
        measureStats.popped,
        measureStats.expanded,
        measureStats.seedChecks,
        measureStats.matchFound,
        measureStats.matchCalls,
        measureStats.boundRejected,
        measureStats.individualBondExcluded,
        measureStats.fillZero,
        measureStats.stage0Success,
        measureStats.stage0Attempts,
        measureStats.stage1Success,
        measureStats.stage1Attempts,
        measureStats.stage2Success,
        measureStats.stage2Attempts,
        measureStats.fastSuccess,
        measureStats.fastAttempts,
        measureStats.fallbackCalls,
        measureStats.fallbackSuccess,
        measureStats.fallbackFail,
        measureStats.fallbackOverflow,
        measureStats.maxQueue,
        measureStats.forcedExit,
        static_cast<int>(overflowed),
        static_cast<int>(timedOut),
        static_cast<int>(best.seed.numBonds),
        static_cast<int>(best.seed.numAtoms),
        queue.size());
    }
  }

  // ---- Phase 3: writeback ----
  block.sync();
  if (block.thread_rank() == 0) {
    unsigned long long totalElapsedClocks = 0;
    if constexpr (CollectTimings || CollectStats) {
      totalElapsedClocks = clock64() - startClock;
    }
    if constexpr (CollectTimings) {
      if (elapsedClocks != nullptr) {
        elapsedClocks[pairIdx] = totalElapsedClocks;
      }
    }
    if constexpr (CollectTimings || CollectStats) {
      measureStats.totalClocks = totalElapsedClocks;
    }
    if constexpr (CollectStats) {
      if (statsOut != nullptr) {
        statsOut[pairIdx] = measureStats;
      }
    } else if constexpr (CollectTimings) {
      if (statsOut != nullptr) {
        statsOut[pairIdx] = measureStats;
      }
    }

    auto& dst      = results[pairIdx];
    auto* dstBytes = reinterpret_cast<unsigned char*>(&dst);
    for (int byteIdx = 0; byteIdx < static_cast<int>(sizeof(DeviceMCSResult<maxAtoms, maxBonds>)); ++byteIdx) {
      dstBytes[byteIdx] = 0;
    }
    dst.numCommonVertices = best.seed.numAtoms;
    dst.numCommonEdges    = best.seed.numBonds;
    dst.timedOut          = timedOut != 0;
    dst.overflowed        = overflowed != 0;

    // Walk set bits of best.seed.atoms (in increasing query atom idx
    // order, via __ffs/__ffsll) to fill mappingA/B.
    using BestSeedT                = decltype(best.seed);
    using AtomWord                 = typename BestSeedT::atom_word_type;
    constexpr int kAtomBitsPerWord = BestSeedT::kAtomBitsPerWord;
    constexpr int kAtomWords       = BestSeedT::kAtomWords;
    int           outIdx           = 0;
    for (int wordIdx = 0; wordIdx < kAtomWords; ++wordIdx) {
      AtomWord remaining = best.seed.atoms[wordIdx];
      while (remaining != 0) {
        int bitPosInWord;
        if constexpr (sizeof(AtomWord) == 4) {
          bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
        } else {
          bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
        }
        const int queryAtomIdx = wordIdx * kAtomBitsPerWord + bitPosInWord;
        remaining &= remaining - 1;
        dst.mappingA[outIdx] = static_cast<std::uint8_t>(queryAtomIdx);
        dst.mappingB[outIdx] = best.match.targetAtomIdx[queryAtomIdx];
        ++outIdx;
      }
    }

    // Walk set bits of best.seed.bonds to fill bondMapA/B.
    using BondWord                 = typename BestSeedT::bond_word_type;
    constexpr int kBondBitsPerWord = BestSeedT::kBondBitsPerWord;
    constexpr int kBondWords       = BestSeedT::kBondWords;
    outIdx                         = 0;
    for (int wordIdx = 0; wordIdx < kBondWords; ++wordIdx) {
      BondWord remaining = best.seed.bonds[wordIdx];
      while (remaining != 0) {
        int bitPosInWord;
        if constexpr (sizeof(BondWord) == 4) {
          bitPosInWord = __ffs(static_cast<unsigned int>(remaining)) - 1;
        } else {
          bitPosInWord = __ffsll(static_cast<unsigned long long>(remaining)) - 1;
        }
        const int queryBondIdx = wordIdx * kBondBitsPerWord + bitPosInWord;
        remaining &= remaining - 1;
        dst.bondMapA[outIdx] = static_cast<std::uint8_t>(queryBondIdx);
        dst.bondMapB[outIdx] = best.match.targetBondIdx[queryBondIdx];
        ++outIdx;
      }
    }
  }
}

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_KERNEL_CUH
