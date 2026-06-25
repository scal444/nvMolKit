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
#include "src/mcs/mcs_compile_flags.h"
#include "src/utils/nvtx.h"

#include <GraphMol/ROMol.h>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace boost::python;

struct MCSResultBuffers {
  std::vector<unsigned int> numAtoms;
  std::vector<unsigned int> numBonds;
  std::vector<std::uint8_t> canceled;
  std::vector<std::uint8_t> overflowed;
  std::vector<std::uint8_t> usedGpu;
  std::vector<std::uint8_t> usedFallback;
  std::vector<float>        elapsedMs;
  std::vector<unsigned long long> timingTotalClocks;
  std::vector<unsigned long long> timingPhase1Clocks;
  std::vector<unsigned long long> timingPhase2Clocks;
  std::vector<std::uint32_t> statsPhase2Iters;
  std::vector<std::uint32_t> statsInitialSeeds;
  std::vector<std::uint32_t> statsMismatchedInitialSeeds;
  std::vector<std::uint32_t> statsPopped;
  std::vector<std::uint32_t> statsSeedChecks;
  std::vector<std::uint32_t> statsMatchCalls;
  std::vector<std::uint32_t> statsMatchFound;
  std::vector<std::uint32_t> statsBoundRejected;
  std::vector<std::uint32_t> statsExpanded;
  std::vector<std::uint32_t> statsFillZero;
  std::vector<std::uint32_t> statsStage0Attempts;
  std::vector<std::uint32_t> statsStage0Success;
  std::vector<std::uint32_t> statsStage1Attempts;
  std::vector<std::uint32_t> statsStage1Success;
  std::vector<std::uint32_t> statsStage2Attempts;
  std::vector<std::uint32_t> statsStage2Success;
  std::vector<std::uint32_t> statsIndividualBondExcluded;
  std::vector<std::uint32_t> statsFastAttempts;
  std::vector<std::uint32_t> statsFastSuccess;
  std::vector<std::uint32_t> statsFallbackCalls;
  std::vector<std::uint32_t> statsFallbackSuccess;
  std::vector<std::uint32_t> statsFallbackFail;
  std::vector<std::uint32_t> statsFallbackOverflow;
  std::vector<std::uint32_t> statsMaxQueue;
  std::vector<std::uint32_t> statsForcedExit;
  std::vector<unsigned long long> statsTotalClocks;
  std::vector<unsigned long long> statsPhase1Clocks;
  std::vector<unsigned long long> statsPhase2Clocks;
  std::vector<std::uint32_t> statsIncrementalMatchCycles1024;
  std::vector<std::uint32_t> statsSubstructureMatchCycles1024;
  std::vector<std::uint32_t> statsPhase2PopSyncWaitCycles1024;
  std::vector<std::uint32_t> statsPhase2SyncWaitCycles1024;
  std::vector<std::uint32_t> statsPhase2IdleNoSeedWaitCycles1024;
  std::vector<std::uint32_t> statsPhase2IdleNoMatchWaitCycles1024;
  std::vector<std::uint32_t> statsPhase2ActiveWorkCycles1024;
  std::vector<std::uint32_t> statsPhase2ActiveMatchCycles1024;
  std::vector<std::string>  smartsStrings;

  std::vector<std::int32_t> atomMapping;
  std::vector<std::int32_t> atomMappingIndptr;
  std::vector<std::int32_t> bondMapping;
  std::vector<std::int32_t> bondMappingIndptr;
};

nvMolKit::MCSAtomCompare parseAtomCompare(const std::string& value) {
  if (value == "any") {
    return nvMolKit::MCSAtomCompare::Any;
  }
  if (value == "elements") {
    return nvMolKit::MCSAtomCompare::Elements;
  }
  if (value == "isotopes") {
    return nvMolKit::MCSAtomCompare::Isotopes;
  }
  if (value == "any_heavy_atom") {
    return nvMolKit::MCSAtomCompare::AnyHeavyAtom;
  }
  throw std::invalid_argument("Unsupported atom_compare value: " + value);
}

nvMolKit::MCSBondCompare parseBondCompare(const std::string& value) {
  if (value == "any") {
    return nvMolKit::MCSBondCompare::Any;
  }
  if (value == "order") {
    return nvMolKit::MCSBondCompare::Order;
  }
  if (value == "order_exact") {
    return nvMolKit::MCSBondCompare::OrderExact;
  }
  throw std::invalid_argument("Unsupported bond_compare value: " + value);
}

std::vector<const RDKit::ROMol*> molsFromPythonList(const list& mols) {
  nvMolKit::ScopedNvtxRange range("Python MCS: extract mol pointers", nvMolKit::NvtxColor::kYellow);
  std::vector<const RDKit::ROMol*> out;
  out.reserve(len(mols));
  for (int i = 0; i < len(mols); ++i) {
    const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(mols[i]));
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
    }
    out.push_back(mol);
  }
  return out;
}

std::vector<nvMolKit::MCSPair> pairsFromPythonList(const list& pairs) {
  std::vector<nvMolKit::MCSPair> out;
  out.reserve(len(pairs));
  for (int i = 0; i < len(pairs); ++i) {
    object pairObj(pairs[i]);
    if (!PySequence_Check(pairObj.ptr()) || PySequence_Size(pairObj.ptr()) != 2) {
      throw std::invalid_argument("MCS pair at index " + std::to_string(i) + " must be a length-2 sequence");
    }
    object first(handle<>(PySequence_GetItem(pairObj.ptr(), 0)));
    object second(handle<>(PySequence_GetItem(pairObj.ptr(), 1)));
    out.emplace_back(extract<std::size_t>(first), extract<std::size_t>(second));
  }
  return out;
}

template <typename T>
T optionValue(const dict& options, const char* key, const T& defaultValue) {
  if (PyMapping_HasKeyString(options.ptr(), key) == 0) {
    return defaultValue;
  }
  return extract<T>(options[key]);
}

template <typename T>
std::vector<T> vectorFromIterable(const object& iterable) {
  std::vector<T> converted;
  stl_input_iterator<T> it(iterable), end;
  for (; it != end; ++it) {
    converted.push_back(*it);
  }
  return converted;
}

std::vector<int> optionIntVector(const dict& options, const char* key) {
  if (PyMapping_HasKeyString(options.ptr(), key) == 0) {
    return {};
  }
  return vectorFromIterable<int>(options[key]);
}

list stringsToPythonList(const std::vector<std::string>& values) {
  list out;
  for (const auto& value : values) {
    out.append(value);
  }
  return out;
}

template <typename T>
boost::python::numpy::ndarray make1dArray(std::vector<T>& values, const object& owner) {
  const Py_intptr_t shape  = static_cast<Py_intptr_t>(values.size());
  const Py_intptr_t stride = static_cast<Py_intptr_t>(sizeof(T));
  return boost::python::numpy::from_data(values.data(),
                                         boost::python::numpy::dtype::get_builtin<T>(),
                                         make_tuple(shape),
                                         make_tuple(stride),
                                         owner);
}

boost::python::numpy::ndarray makePairArray(std::vector<std::int32_t>& values, const object& owner) {
  const Py_intptr_t rows = static_cast<Py_intptr_t>(values.size() / 2);
  const Py_intptr_t item = static_cast<Py_intptr_t>(sizeof(std::int32_t));
  return boost::python::numpy::from_data(values.data(),
                                         boost::python::numpy::dtype::get_builtin<std::int32_t>(),
                                         make_tuple(rows, 2),
                                         make_tuple(2 * item, item),
                                         owner);
}

void reserveStats(MCSResultBuffers& buffers, std::size_t size) {
  buffers.statsPhase2Iters.reserve(size);
  buffers.statsInitialSeeds.reserve(size);
  buffers.statsMismatchedInitialSeeds.reserve(size);
  buffers.statsPopped.reserve(size);
  buffers.statsSeedChecks.reserve(size);
  buffers.statsMatchCalls.reserve(size);
  buffers.statsMatchFound.reserve(size);
  buffers.statsBoundRejected.reserve(size);
  buffers.statsExpanded.reserve(size);
  buffers.statsFillZero.reserve(size);
  buffers.statsStage0Attempts.reserve(size);
  buffers.statsStage0Success.reserve(size);
  buffers.statsStage1Attempts.reserve(size);
  buffers.statsStage1Success.reserve(size);
  buffers.statsStage2Attempts.reserve(size);
  buffers.statsStage2Success.reserve(size);
  buffers.statsIndividualBondExcluded.reserve(size);
  buffers.statsFastAttempts.reserve(size);
  buffers.statsFastSuccess.reserve(size);
  buffers.statsFallbackCalls.reserve(size);
  buffers.statsFallbackSuccess.reserve(size);
  buffers.statsFallbackFail.reserve(size);
  buffers.statsFallbackOverflow.reserve(size);
  buffers.statsMaxQueue.reserve(size);
  buffers.statsForcedExit.reserve(size);
  buffers.statsTotalClocks.reserve(size);
  buffers.statsPhase1Clocks.reserve(size);
  buffers.statsPhase2Clocks.reserve(size);
  buffers.statsIncrementalMatchCycles1024.reserve(size);
  buffers.statsSubstructureMatchCycles1024.reserve(size);
  buffers.statsPhase2PopSyncWaitCycles1024.reserve(size);
  buffers.statsPhase2SyncWaitCycles1024.reserve(size);
  buffers.statsPhase2IdleNoSeedWaitCycles1024.reserve(size);
  buffers.statsPhase2IdleNoMatchWaitCycles1024.reserve(size);
  buffers.statsPhase2ActiveWorkCycles1024.reserve(size);
  buffers.statsPhase2ActiveMatchCycles1024.reserve(size);
}

void reserveTimings(MCSResultBuffers& buffers, std::size_t size) {
  buffers.timingTotalClocks.reserve(size);
  buffers.timingPhase1Clocks.reserve(size);
  buffers.timingPhase2Clocks.reserve(size);
}

void appendTimings(MCSResultBuffers& buffers, const nvMolKit::MCSExecutionStats& timings) {
  buffers.timingTotalClocks.push_back(timings.totalClocks);
  buffers.timingPhase1Clocks.push_back(timings.phase1Clocks);
  buffers.timingPhase2Clocks.push_back(timings.phase2Clocks);
}

void appendStats(MCSResultBuffers& buffers, const nvMolKit::MCSExecutionStats& stats) {
  buffers.statsPhase2Iters.push_back(stats.phase2Iters);
  buffers.statsInitialSeeds.push_back(stats.initialSeeds);
  buffers.statsMismatchedInitialSeeds.push_back(stats.mismatchedInitialSeeds);
  buffers.statsPopped.push_back(stats.popped);
  buffers.statsSeedChecks.push_back(stats.seedChecks);
  buffers.statsMatchCalls.push_back(stats.matchCalls);
  buffers.statsMatchFound.push_back(stats.matchFound);
  buffers.statsBoundRejected.push_back(stats.boundRejected);
  buffers.statsExpanded.push_back(stats.expanded);
  buffers.statsFillZero.push_back(stats.fillZero);
  buffers.statsStage0Attempts.push_back(stats.stage0Attempts);
  buffers.statsStage0Success.push_back(stats.stage0Success);
  buffers.statsStage1Attempts.push_back(stats.stage1Attempts);
  buffers.statsStage1Success.push_back(stats.stage1Success);
  buffers.statsStage2Attempts.push_back(stats.stage2Attempts);
  buffers.statsStage2Success.push_back(stats.stage2Success);
  buffers.statsIndividualBondExcluded.push_back(stats.individualBondExcluded);
  buffers.statsFastAttempts.push_back(stats.fastAttempts);
  buffers.statsFastSuccess.push_back(stats.fastSuccess);
  buffers.statsFallbackCalls.push_back(stats.fallbackCalls);
  buffers.statsFallbackSuccess.push_back(stats.fallbackSuccess);
  buffers.statsFallbackFail.push_back(stats.fallbackFail);
  buffers.statsFallbackOverflow.push_back(stats.fallbackOverflow);
  buffers.statsMaxQueue.push_back(stats.maxQueue);
  buffers.statsForcedExit.push_back(stats.forcedExit);
  buffers.statsTotalClocks.push_back(stats.totalClocks);
  buffers.statsPhase1Clocks.push_back(stats.phase1Clocks);
  buffers.statsPhase2Clocks.push_back(stats.phase2Clocks);
  buffers.statsIncrementalMatchCycles1024.push_back(stats.incrementalMatchCycles1024);
  buffers.statsSubstructureMatchCycles1024.push_back(stats.substructureMatchCycles1024);
  buffers.statsPhase2PopSyncWaitCycles1024.push_back(stats.phase2PopSyncWaitCycles1024);
  buffers.statsPhase2SyncWaitCycles1024.push_back(stats.phase2SyncWaitCycles1024);
  buffers.statsPhase2IdleNoSeedWaitCycles1024.push_back(stats.phase2IdleNoSeedWaitCycles1024);
  buffers.statsPhase2IdleNoMatchWaitCycles1024.push_back(stats.phase2IdleNoMatchWaitCycles1024);
  buffers.statsPhase2ActiveWorkCycles1024.push_back(stats.phase2ActiveWorkCycles1024);
  buffers.statsPhase2ActiveMatchCycles1024.push_back(stats.phase2ActiveMatchCycles1024);
}

dict timingsToPythonDict(MCSResultBuffers& buffers, const object& owner) {
  dict out;
  out["total_clocks"] = make1dArray(buffers.timingTotalClocks, owner);
  out["phase1_clocks"] = make1dArray(buffers.timingPhase1Clocks, owner);
  out["phase2_clocks"] = make1dArray(buffers.timingPhase2Clocks, owner);
  return out;
}

dict statsToPythonDict(MCSResultBuffers& buffers, const object& owner) {
  dict out;
  out["phase2_iters"] = make1dArray(buffers.statsPhase2Iters, owner);
  out["initial_seeds"] = make1dArray(buffers.statsInitialSeeds, owner);
  out["mismatched_initial_seeds"] = make1dArray(buffers.statsMismatchedInitialSeeds, owner);
  out["popped"] = make1dArray(buffers.statsPopped, owner);
  out["seed_checks"] = make1dArray(buffers.statsSeedChecks, owner);
  out["match_calls"] = make1dArray(buffers.statsMatchCalls, owner);
  out["match_found"] = make1dArray(buffers.statsMatchFound, owner);
  out["bound_rejected"] = make1dArray(buffers.statsBoundRejected, owner);
  out["expanded"] = make1dArray(buffers.statsExpanded, owner);
  out["fill_zero"] = make1dArray(buffers.statsFillZero, owner);
  out["stage0_attempts"] = make1dArray(buffers.statsStage0Attempts, owner);
  out["stage0_success"] = make1dArray(buffers.statsStage0Success, owner);
  out["stage1_attempts"] = make1dArray(buffers.statsStage1Attempts, owner);
  out["stage1_success"] = make1dArray(buffers.statsStage1Success, owner);
  out["stage2_attempts"] = make1dArray(buffers.statsStage2Attempts, owner);
  out["stage2_success"] = make1dArray(buffers.statsStage2Success, owner);
  out["individual_bond_excluded"] = make1dArray(buffers.statsIndividualBondExcluded, owner);
  out["fast_attempts"] = make1dArray(buffers.statsFastAttempts, owner);
  out["fast_success"] = make1dArray(buffers.statsFastSuccess, owner);
  out["fallback_calls"] = make1dArray(buffers.statsFallbackCalls, owner);
  out["fallback_success"] = make1dArray(buffers.statsFallbackSuccess, owner);
  out["fallback_fail"] = make1dArray(buffers.statsFallbackFail, owner);
  out["fallback_overflow"] = make1dArray(buffers.statsFallbackOverflow, owner);
  out["max_queue"] = make1dArray(buffers.statsMaxQueue, owner);
  out["forced_exit"] = make1dArray(buffers.statsForcedExit, owner);
  out["total_clocks"] = make1dArray(buffers.statsTotalClocks, owner);
  out["phase1_clocks"] = make1dArray(buffers.statsPhase1Clocks, owner);
  out["phase2_clocks"] = make1dArray(buffers.statsPhase2Clocks, owner);
  out["incremental_match_cycles_1024"] = make1dArray(buffers.statsIncrementalMatchCycles1024, owner);
  out["substructure_match_cycles_1024"] = make1dArray(buffers.statsSubstructureMatchCycles1024, owner);
  out["phase2_pop_sync_wait_cycles_1024"] = make1dArray(buffers.statsPhase2PopSyncWaitCycles1024, owner);
  out["phase2_sync_wait_cycles_1024"] = make1dArray(buffers.statsPhase2SyncWaitCycles1024, owner);
  out["phase2_idle_no_seed_wait_cycles_1024"] = make1dArray(buffers.statsPhase2IdleNoSeedWaitCycles1024, owner);
  out["phase2_idle_no_match_wait_cycles_1024"] = make1dArray(buffers.statsPhase2IdleNoMatchWaitCycles1024, owner);
  out["phase2_active_work_cycles_1024"] = make1dArray(buffers.statsPhase2ActiveWorkCycles1024, owner);
  out["phase2_active_match_cycles_1024"] = make1dArray(buffers.statsPhase2ActiveMatchCycles1024, owner);
  return out;
}

}  // namespace

BOOST_PYTHON_MODULE(_mcs) {
  boost::python::numpy::initialize();
  scope().attr("_MCS_TIMINGS_ENABLED") = nvMolKit::kMCSCollectTimingsEnabled;
  scope().attr("_MCS_STATS_ENABLED")   = nvMolKit::kMCSCollectStatsEnabled;

  def(
    "_findMCSBatch",
    +[](const list&        mols,
        const list&        pairs,
        const dict&        options) {
      auto molVec  = molsFromPythonList(mols);
      auto pairVec = pairsFromPythonList(pairs);

      nvMolKit::MCSParameters params;
      params.atomCompare                                      = parseAtomCompare(optionValue<std::string>(options, "atom_compare", "elements"));
      params.bondCompare                                      = parseBondCompare(optionValue<std::string>(options, "bond_compare", "order"));
      params.maximizeBonds                                    = optionValue<bool>(options, "maximize_bonds", true);
      params.connectedOnly                                    = optionValue<bool>(options, "connected_only", true);
      params.requireGpu                                       = optionValue<bool>(options, "require_gpu", false);
      params.collectTimings                                   = optionValue<bool>(options, "collect_timings", false);
      params.collectStats                                     = optionValue<bool>(options, "collect_stats", false);
      if (params.collectTimings && !nvMolKit::kMCSCollectTimingsEnabled) {
        throw std::runtime_error(
            "fMCS timing instrumentation is not instantiated in this build");
      }
      if (params.collectStats && !nvMolKit::kMCSCollectStatsEnabled) {
        throw std::runtime_error(
            "fMCS stat instrumentation is not instantiated in this build");
      }
      params.timeoutSeconds                                   = optionValue<unsigned int>(options, "timeout_seconds", 0);
      params.batchSize                                        = optionValue<int>(options, "batch_size", 0);
      params.blockSize                                        = optionValue<int>(options, "block_size", 128);
      params.workerThreads                                    = optionValue<int>(options, "worker_threads", -1);
      params.preprocessingThreads                             = optionValue<int>(options, "preprocessing_threads", -1);
      params.executorsPerRunner                               = optionValue<int>(options, "executors_per_runner", -1);
      params.gpuIds                                           = optionIntVector(options, "gpu_ids");
      params.atomCompareParameters.matchValences              = optionValue<bool>(options, "match_valences", false);
      params.atomCompareParameters.matchFormalCharge          = optionValue<bool>(options, "match_formal_charge", false);
      params.atomCompareParameters.ringMatchesRingOnly        = optionValue<bool>(options, "atom_ring_matches_ring_only", false);
      params.atomCompareParameters.completeRingsOnly          = optionValue<bool>(options, "atom_complete_rings_only", false);
      params.atomCompareParameters.matchIsotope               = optionValue<bool>(options, "match_isotope", false);
      params.bondCompareParameters.ringMatchesRingOnly        = optionValue<bool>(options, "bond_ring_matches_ring_only", false);
      params.bondCompareParameters.completeRingsOnly          = optionValue<bool>(options, "bond_complete_rings_only", false);

      nvMolKit::ScopedNvtxRange mcsRange("Python MCS: findMCSBatch", nvMolKit::NvtxColor::kOrange);
      auto results = nvMolKit::findMCSBatch(molVec, pairVec, nullptr, params);
      mcsRange.pop();

      auto buffers = std::make_unique<MCSResultBuffers>();
      buffers->numAtoms.reserve(results.size());
      buffers->numBonds.reserve(results.size());
      buffers->canceled.reserve(results.size());
      buffers->overflowed.reserve(results.size());
      buffers->usedGpu.reserve(results.size());
      buffers->usedFallback.reserve(results.size());
      if constexpr (nvMolKit::kMCSCollectTimingsEnabled) {
        if (params.collectTimings) {
          buffers->elapsedMs.reserve(results.size());
          reserveTimings(*buffers, results.size());
        }
      }
      if constexpr (nvMolKit::kMCSCollectStatsEnabled) {
        if (params.collectStats) {
          reserveStats(*buffers, results.size());
        }
      }
      buffers->smartsStrings.reserve(results.size());
      buffers->atomMappingIndptr.reserve(results.size() + 1);
      buffers->bondMappingIndptr.reserve(results.size() + 1);
      buffers->atomMappingIndptr.push_back(0);
      buffers->bondMappingIndptr.push_back(0);

      for (const auto& result : results) {
        buffers->numAtoms.push_back(result.numAtoms);
        buffers->numBonds.push_back(result.numBonds);
        buffers->canceled.push_back(result.canceled ? 1 : 0);
        buffers->overflowed.push_back(result.overflowed ? 1 : 0);
        buffers->usedGpu.push_back(result.usedGpu ? 1 : 0);
        buffers->usedFallback.push_back(result.usedFallback ? 1 : 0);
        if constexpr (nvMolKit::kMCSCollectTimingsEnabled) {
          if (params.collectTimings) {
            buffers->elapsedMs.push_back(result.elapsedMs);
            appendTimings(*buffers, result.hasKernelTimings ? result.kernelTimings : nvMolKit::MCSExecutionStats{});
          }
        }
        if constexpr (nvMolKit::kMCSCollectStatsEnabled) {
          if (params.collectStats) {
            appendStats(*buffers, result.hasExecutionStats ? result.executionStats : nvMolKit::MCSExecutionStats{});
          }
        }
        buffers->smartsStrings.push_back(result.smartsString);

        for (const auto& [a, b] : result.atomMapping) {
          buffers->atomMapping.push_back(static_cast<std::int32_t>(a));
          buffers->atomMapping.push_back(static_cast<std::int32_t>(b));
        }
        buffers->atomMappingIndptr.push_back(static_cast<std::int32_t>(buffers->atomMapping.size() / 2));

        for (const auto& [a, b] : result.bondMapping) {
          buffers->bondMapping.push_back(static_cast<std::int32_t>(a));
          buffers->bondMapping.push_back(static_cast<std::int32_t>(b));
        }
        buffers->bondMappingIndptr.push_back(static_cast<std::int32_t>(buffers->bondMapping.size() / 2));
      }

      auto deleter = [](PyObject* cap) {
        auto* ptr = reinterpret_cast<MCSResultBuffers*>(PyCapsule_GetPointer(cap, "nvmolkit.mcs_results"));
        delete ptr;
      };
      PyObject* cap = PyCapsule_New(static_cast<void*>(buffers.get()), "nvmolkit.mcs_results", deleter);
      if (cap == nullptr) {
        throw std::runtime_error("Failed to create PyCapsule for MCS results");
      }
      object owner{handle<>(cap)};
      buffers.release();
      auto* ptr = reinterpret_cast<MCSResultBuffers*>(PyCapsule_GetPointer(cap, "nvmolkit.mcs_results"));
      object elapsedObject{handle<>(borrowed(Py_None))};
      if constexpr (nvMolKit::kMCSCollectTimingsEnabled) {
        if (params.collectTimings) {
          elapsedObject = make1dArray(ptr->elapsedMs, owner);
        }
      }
      object timingsObject{handle<>(borrowed(Py_None))};
      if constexpr (nvMolKit::kMCSCollectTimingsEnabled) {
        if (params.collectTimings) {
          timingsObject = timingsToPythonDict(*ptr, owner);
        }
      }
      object statsObject{handle<>(borrowed(Py_None))};
      if constexpr (nvMolKit::kMCSCollectStatsEnabled) {
        if (params.collectStats) {
          statsObject = statsToPythonDict(*ptr, owner);
        }
      }

      nvMolKit::ScopedNvtxRange wrapRange("Python MCS: wrap results", nvMolKit::NvtxColor::kGreen);
      return make_tuple(make1dArray(ptr->numAtoms, owner),
                        make1dArray(ptr->numBonds, owner),
                        make1dArray(ptr->canceled, owner),
                        make1dArray(ptr->overflowed, owner),
                        make1dArray(ptr->usedGpu, owner),
                        make1dArray(ptr->usedFallback, owner),
                        elapsedObject,
                        stringsToPythonList(ptr->smartsStrings),
                        makePairArray(ptr->atomMapping, owner),
                        make1dArray(ptr->atomMappingIndptr, owner),
                        makePairArray(ptr->bondMapping, owner),
                        make1dArray(ptr->bondMappingIndptr, owner),
                        timingsObject,
                        statsObject);
    },
    (arg("mols"), arg("pairs"), arg("options") = dict()));
}
