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

#ifndef NVMOLKIT_MCS_TYPES_H
#define NVMOLKIT_MCS_TYPES_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nvMolKit {

enum class MCSAtomCompare : std::uint8_t {
  Any,
  Elements,
  Isotopes,
  AnyHeavyAtom
};

enum class MCSBondCompare : std::uint8_t {
  Any,
  Order,
  OrderExact
};

struct MCSAtomCompareParameters {
  bool matchValences       = false;
  bool matchFormalCharge   = false;
  bool ringMatchesRingOnly = false;
  bool completeRingsOnly   = false;
  bool matchIsotope        = false;
};

struct MCSBondCompareParameters {
  bool ringMatchesRingOnly = false;
  bool completeRingsOnly   = false;
};

struct MCSParameters {
  bool                     maximizeBonds  = true;
  bool                     connectedOnly  = true;
  bool                     requireGpu     = false;
  bool                     collectTimings = false;
  bool                     collectStats   = false;
  unsigned int             timeoutSeconds = 0;
  int                      batchSize      = 0;
  int                      blockSize      = 128;
  int                      workerThreads  = -1;
  int                      preprocessingThreads = -1;
  int                      executorsPerRunner = -1;
  std::vector<int>         gpuIds;
  MCSAtomCompare           atomCompare    = MCSAtomCompare::Elements;
  MCSBondCompare           bondCompare    = MCSBondCompare::Order;
  MCSAtomCompareParameters atomCompareParameters;
  MCSBondCompareParameters bondCompareParameters;
};

struct MCSExecutionStats {
  unsigned int phase2Iters = 0;
  unsigned int initialSeeds = 0;
  unsigned int mismatchedInitialSeeds = 0;
  unsigned int popped = 0;
  unsigned int seedChecks = 0;
  unsigned int matchCalls = 0;
  unsigned int matchFound = 0;
  unsigned int boundRejected = 0;
  unsigned int expanded = 0;
  unsigned int fillZero = 0;
  unsigned int stage0Attempts = 0;
  unsigned int stage0Success = 0;
  unsigned int stage1Attempts = 0;
  unsigned int stage1Success = 0;
  unsigned int stage2Attempts = 0;
  unsigned int stage2Success = 0;
  unsigned int individualBondExcluded = 0;
  unsigned int fastAttempts = 0;
  unsigned int fastSuccess = 0;
  unsigned int fallbackCalls = 0;
  unsigned int fallbackSuccess = 0;
  unsigned int fallbackFail = 0;
  unsigned int fallbackOverflow = 0;
  unsigned int maxQueue = 0;
  unsigned int forcedExit = 0;
  unsigned long long totalClocks = 0;
  unsigned long long phase1Clocks = 0;
  unsigned long long phase2Clocks = 0;
  unsigned int incrementalMatchCycles1024 = 0;
  unsigned int substructureMatchCycles1024 = 0;
  unsigned int phase2PopSyncWaitCycles1024 = 0;
  unsigned int phase2SyncWaitCycles1024 = 0;
  unsigned int phase2IdleNoSeedWaitCycles1024 = 0;
  unsigned int phase2IdleNoMatchWaitCycles1024 = 0;
  unsigned int phase2ActiveWorkCycles1024 = 0;
  unsigned int phase2ActiveMatchCycles1024 = 0;
};

struct MCSResult {
  unsigned int numAtoms = 0;
  unsigned int numBonds = 0;
  bool         canceled = false;
  bool         overflowed = false;
  bool         usedGpu = false;
  bool         usedFallback = false;
  float        elapsedMs = 0.0f;
  bool         hasExecutionStats = false;
  MCSExecutionStats executionStats;
  std::string  smartsString;

  std::vector<std::pair<int, int>> atomMapping;
  std::vector<std::pair<int, int>> bondMapping;

  [[nodiscard]] bool isCompleted() const { return !canceled; }
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_MCS_TYPES_H
