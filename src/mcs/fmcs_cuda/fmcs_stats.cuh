// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_FMCS_STATS_CUH
#define FMCS_CUDA_FMCS_STATS_CUH

namespace mcs {
namespace fmcs {

/// Optional per-pair fMCS execution counters for debugging search tails.
///
/// These counters are diagnostic: enabling collection launches an instrumented
/// kernel specialization with extra atomics.  Normal fMCS calls leave this off.
struct ExecutionStats {
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

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_STATS_CUH
