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

//! \file mmff_ef_kernel_bench.cpp
//! \brief Profiler-oriented driver that exercises only the MMFF energy and
//! gradient kernels in isolation.
//!
//! The two MMFF kernel layouts in nvMolKit are:
//!   - BATCHED: one kernel per energy/grad term plus an accumulation kernel
//!     for energies. Many launches per E or F evaluation.
//!   - PER_MOL: a single unified kernel per evaluation (block-per-molecule,
//!     all terms fused). One launch per E and one per F.
//!
//! This benchmark loads molecules, embeds them with **RDKit**
//! (\c EmbedMultipleConfs + ETKDGv3), builds the flattened batched MMFF system
//! once, and then repeatedly calls the
//! E or F entry points so a profiler (Nsight Systems for timeline / NVTX,
//! Nsight Compute for per-kernel occupancy and stalls) can isolate the
//! kernels of interest. NVTX ranges are emitted around each evaluation so
//! the regions can be filtered easily.

#include <cuda_runtime.h>
#include <getopt.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../tests/test_utils.h"
#include "bfgs_common.h"
#include "cuda_error_check.h"
#include "ff_utils.h"
#include "mmff.h"
#include "mmff_flattened_builder.h"
#include "nvtx.h"

namespace {

constexpr int kMaxAtomsForEmbed = 256;

enum class KernelMode {
  BATCHED,
  PER_MOL,
  BOTH,
};

enum class EvalKind {
  ENERGY,
  GRADIENT,
  BOTH,
};

std::string kernelModeName(KernelMode mode) {
  switch (mode) {
    case KernelMode::BATCHED:
      return "BATCHED";
    case KernelMode::PER_MOL:
      return "PER_MOL";
    case KernelMode::BOTH:
      return "BOTH";
  }
  return "UNKNOWN";
}

std::string evalKindName(EvalKind kind) {
  switch (kind) {
    case EvalKind::ENERGY:
      return "ENERGY";
    case EvalKind::GRADIENT:
      return "GRADIENT";
    case EvalKind::BOTH:
      return "BOTH";
  }
  return "UNKNOWN";
}

KernelMode parseKernelMode(const std::string& arg) {
  std::string copy = arg;
  std::transform(copy.begin(), copy.end(), copy.begin(), ::tolower);
  if (copy == "batched") {
    return KernelMode::BATCHED;
  }
  if (copy == "per_mol" || copy == "per-mol" || copy == "permol") {
    return KernelMode::PER_MOL;
  }
  if (copy == "both") {
    return KernelMode::BOTH;
  }
  throw std::runtime_error("Invalid --mode value. Expected batched, per_mol, or both.");
}

EvalKind parseEvalKind(const std::string& arg) {
  std::string copy = arg;
  std::transform(copy.begin(), copy.end(), copy.begin(), ::tolower);
  if (copy == "energy") {
    return EvalKind::ENERGY;
  }
  if (copy == "gradient" || copy == "grad" || copy == "force") {
    return EvalKind::GRADIENT;
  }
  if (copy == "both") {
    return EvalKind::BOTH;
  }
  throw std::runtime_error("Invalid --eval value. Expected energy, gradient, or both.");
}

constexpr std::streamsize kSmilesIoBufferChars = static_cast<std::streamsize>(1u << 20);

//! Reservoir sample uniformly from eligible SMILES-first-column tokens. No RDKit parse;
//! bounded memory proportional to \p reservoirSize.
std::vector<std::string> reservoirSampleSmilesTokens(std::istream&          fileStream,
                                                     const bool             skipFirstNonemptyLineAsHeader,
                                                     const unsigned int    reservoirSize,
                                                     const uint64_t         seed) {
  std::vector<std::string> reservoir;
  reservoir.reserve(static_cast<size_t>(reservoirSize));
  std::mt19937_64 rng(seed);

  bool headerDone = !skipFirstNonemptyLineAsHeader;
  std::string line;
  line.reserve(4096u);

  size_t eligibleSeen = 0;
  while (std::getline(fileStream, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const bool onlyWhitespace =
      std::all_of(line.begin(), line.end(), [](unsigned char chr) { return std::isspace(chr) != 0; });
    if (onlyWhitespace) {
      continue;
    }
    if (!headerDone) {
      headerDone = true;
      continue;
    }
    const size_t delim = line.find_first_of(" \t");
    std::string  token;
    if (delim == std::string::npos) {
      token = line;
    } else {
      token.assign(line.data(), delim);
    }
    if (token.empty()) {
      continue;
    }

    if (eligibleSeen < static_cast<size_t>(reservoirSize)) {
      reservoir.push_back(std::move(token));
    } else {
      std::uniform_int_distribution<size_t> dist(0, eligibleSeen);
      const size_t                          slot = dist(rng);
      if (slot < static_cast<size_t>(reservoirSize)) {
        reservoir[slot] = std::move(token);
      }
    }
    ++eligibleSeen;
  }
  return reservoir;
}

//! Parse sampled SMILES/CXSMILES strings after reservoir selection.
void smilesStringsToParsedMols(const std::vector<std::string>&  smilesSamples,
                               std::vector<std::unique_ptr<RDKit::ROMol>>& outMols,
                               const RDKit::SmilesParserParams& params) {
  outMols.clear();
  outMols.reserve(smilesSamples.size());
  for (const std::string& smiles : smilesSamples) {
    try {
      auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles, params));
      if (mol != nullptr && mol->getNumAtoms() <= static_cast<unsigned int>(kMaxAtomsForEmbed)) {
        outMols.push_back(std::move(mol));
      }
    } catch (const std::exception& error) {
      std::cerr << "Warning: Failed to parse SMILES '" << smiles << "': " << error.what() << "\n";
    }
  }
}

//! Slow path for SMILES: parse every qualifying line (\p maxMols == full load).
void loadAllSmilesMolsFromOpenFile(std::istream&                  fileStream,
                                   const bool                    cxsmilesHeader,
                                   const RDKit::SmilesParserParams& params,
                                   std::vector<std::unique_ptr<RDKit::ROMol>>& rawMols) {
  rawMols.clear();
  bool        headerSkipped = !cxsmilesHeader;
  std::string line;
  line.reserve(4096u);
  while (std::getline(fileStream, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const bool onlyWhitespace =
      std::all_of(line.begin(), line.end(), [](unsigned char chr) { return std::isspace(chr) != 0; });
    if (onlyWhitespace) {
      continue;
    }
    if (!headerSkipped) {
      headerSkipped = true;
      continue;
    }
    const size_t delim = line.find_first_of(" \t");
    std::string  smiles;
    if (delim == std::string::npos) {
      smiles = line;
    } else {
      smiles.assign(line.data(), delim);
    }
    if (smiles.empty()) {
      continue;
    }
    try {
      auto mol = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles, params));
      if (mol != nullptr && mol->getNumAtoms() <= static_cast<unsigned int>(kMaxAtomsForEmbed)) {
        rawMols.push_back(std::move(mol));
      }
    } catch (const std::exception& error) {
      std::cerr << "Warning: Failed to parse SMILES '" << smiles << "': " << error.what() << "\n";
    }
  }
}

//! Stream-parse SDF and retain at most \p maxMols structures via reservoir sampling
//! (\p ROMol each), so bulk files do not inflate memory beyond the reservoir.
void reservoirSampleFromSdfFile(const std::string&                               filePath,
                                const unsigned int                               maxMols,
                                const uint64_t                                   seed,
                                std::vector<std::unique_ptr<RDKit::ROMol>>& reservoirOut) {
  RDKit::SDMolSupplier suppl(filePath, true, false);
  std::mt19937_64      rng(seed);
  reservoirOut.clear();
  reservoirOut.reserve(maxMols);

  size_t eligibleSeen = 0;
  while (!suppl.atEnd()) {
    auto molHolder = std::unique_ptr<RDKit::ROMol>(suppl.next());
    if (molHolder == nullptr) {
      continue;
    }
    auto* mol = molHolder.get();
    if (mol->getNumAtoms() <= 1u || mol->getNumBonds() == 0u || mol->getNumConformers() == 0u) {
      continue;
    }

    if (eligibleSeen < static_cast<size_t>(maxMols)) {
      reservoirOut.push_back(std::move(molHolder));
    } else {
      std::uniform_int_distribution<size_t> dist(0, eligibleSeen);
      const size_t                          slot = dist(rng);
      if (slot < static_cast<size_t>(maxMols)) {
        reservoirOut[slot] = std::move(molHolder);
      }
    }
    ++eligibleSeen;
  }
}

//! \brief Convert parsed \p rawMols to sanitized hydrogenated \p RWMol with conformers cleared.
std::vector<std::unique_ptr<RDKit::RWMol>> addHydrogensAndSanitizeBatch(
  std::vector<std::unique_ptr<RDKit::ROMol>>& rawMols) {
  std::vector<std::unique_ptr<RDKit::RWMol>> mols;
  mols.reserve(rawMols.size());
  for (const auto& raw : rawMols) {
    std::unique_ptr<RDKit::ROMol> withHs(RDKit::MolOps::addHs(*raw));
    auto                          rwMol = std::make_unique<RDKit::RWMol>(*withHs);
    rwMol->clearConformers();
    RDKit::MolOps::sanitizeMol(*rwMol);
    mols.push_back(std::move(rwMol));
  }
  return mols;
}

//! \brief Load molecules from an SDF or SMILES file, sanitize, add Hs, and optionally subsample.
//!
//! SMILES streams: When \p maxMols != 0, reservoir-samples SMILES tokens from the text file
//! (no per-line parse) then parses exactly that subset. When \p maxMols == 0, every qualifying
//! line is parsed (expensive on giant corpora). SDF streams: When \p maxMols != 0, reservoirs
//! parsed records while iterating the supplier (\p ROMol footprint bounded by maxMols). When
//! maxMols == 0, uses \ref getMols and retains the legacy full-file behaviour.
//!
//! \param filePath Path to the input file. Extension is used to dispatch.
//! \param maxMols Maximum number of molecules retained (strict cap for sampled paths).
//! Zero means load all qualifying records ("full corpus" path).
//! \param seed RNG seed for reservoir sampling when \p maxMols != 0.
//! \return Sanitized molecules with hydrogens added and conformers cleared.
std::vector<std::unique_ptr<RDKit::RWMol>> loadMolecules(const std::string& filePath,
                                                         unsigned int       maxMols,
                                                         uint64_t           seed) {
  std::vector<std::unique_ptr<RDKit::ROMol>> rawMols;

  std::string extension = std::filesystem::path(filePath).extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

  const bool cxsmilesHeader = extension == ".cxsmiles";

  RDKit::SmilesParserParams smilesParams;
  smilesParams.allowCXSMILES = true;
  smilesParams.sanitize      = true;

  if (extension == ".sdf") {
    if (maxMols == 0u) {
      getMols(filePath, rawMols);
    } else {
      reservoirSampleFromSdfFile(filePath, maxMols, seed, rawMols);
    }
  } else if (extension == ".smi" || extension == ".smiles" || extension == ".cxsmiles") {
    std::ifstream fileStream;
    std::vector<char> ioBuffer(static_cast<size_t>(kSmilesIoBufferChars));
    fileStream.rdbuf()->pubsetbuf(ioBuffer.data(),
                                  static_cast<std::streamsize>(ioBuffer.size()));
    fileStream.open(filePath, std::ios::in | std::ios::binary);
    if (!fileStream.is_open()) {
      throw std::runtime_error("Could not open SMILES file: " + filePath);
    }
    if (maxMols == 0u) {
      loadAllSmilesMolsFromOpenFile(fileStream, cxsmilesHeader, smilesParams, rawMols);
    } else {
      std::vector<std::string> smilesSamples =
        reservoirSampleSmilesTokens(fileStream, cxsmilesHeader, maxMols, seed);
      smilesStringsToParsedMols(smilesSamples, rawMols, smilesParams);
    }
  } else {
    throw std::runtime_error("Unsupported file extension: '" + extension +
                             "'. Use .sdf, .smi, .smiles, or .cxsmiles.");
  }

  if (rawMols.empty()) {
    throw std::runtime_error("No molecules parsed from " + filePath);
  }

  return addHydrogensAndSanitizeBatch(rawMols);
}

//! \brief Embed conformers in-place using RDKit ETKDGv3 (\c EmbedMultipleConfs).
//!
//! Parallelized across molecules with OpenMP. Using RDKit (CPU) avoids pulling
//! nvMolKit's GPU BFGS minimizer kernels into profiler captures when benchmarking
//! MMFF energy/gradient only.
//!
//! Molecules that fail to produce any conformer are filtered out so the
//! downstream MMFF system construction doesn't see empty mol entries.
void embedConformers(std::vector<std::unique_ptr<RDKit::RWMol>>& mols,
                     int                                         confsPerMol,
                     int                                         maxIterations,
                     int                                         embedParallelThreads,
                     uint64_t                                    randomSeedForEmbed) {
  nvMolKit::ScopedNvtxRange embedRange("RDKit ETKDG embed", nvMolKit::NvtxColor::kBlue);

  const int ompPool =
    embedParallelThreads > 0 ? embedParallelThreads : omp_get_max_threads();

  const long numMols = static_cast<long>(mols.size());
#pragma omp parallel for schedule(dynamic) num_threads(ompPool)
  for (long molIndex = 0; molIndex < numMols; ++molIndex) {
    RDKit::RWMol* rwMol = mols.at(static_cast<size_t>(molIndex)).get();
    if (rwMol == nullptr) {
      continue;
    }
    rwMol->clearConformers();

    RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
    params.maxIterations   = maxIterations;
    params.useRandomCoords = true;
    params.basinThresh     = 1e8;
    params.trackFailures   = false;
    params.randomSeed      = static_cast<unsigned int>(randomSeedForEmbed);
    params.numThreads      = 1;

    try {
      std::vector<int> embedStatus;
      RDKit::DGeomHelpers::EmbedMultipleConfs(*rwMol, embedStatus, confsPerMol, params);
      (void)embedStatus;
    } catch (...) {
      // EmbedMultipleConfs can throw or leave conformers absent; molecule is dropped downstream.
    }
  }

  std::vector<std::unique_ptr<RDKit::RWMol>> kept;
  kept.reserve(mols.size());
  for (auto& mol : mols) {
    if (mol->getNumConformers() > 0) {
      kept.push_back(std::move(mol));
    }
  }
  const size_t dropped = mols.size() - kept.size();
  if (dropped > 0) {
    std::cout << "  Dropped " << dropped << " molecules that produced no conformers during embedding\n";
  }
  mols = std::move(kept);
}

//! \brief Build a single batched MMFF system covering every conformer in \p mols.
//!
//! All conformers across all molecules are flattened into one batch. The
//! returned system is ready for either the BATCHED kernels via
//! `computeEnergy/computeGradients` or the unified PER_MOL kernels via
//! `computeEnergyBlockPerMol/computeGradBlockPerMol`.
nvMolKit::MMFF::BatchedMolecularSystemHost buildBatchedHostSystem(
  std::vector<RDKit::ROMol*>&                                mols,
  std::vector<nvMolKit::ConformerInfo>&                      flatConformersOut,
  std::vector<std::vector<double>>&                          perMolEnergiesOut) {
  nvMolKit::ScopedNvtxRange buildRange("MMFF build batched system", nvMolKit::NvtxColor::kCyan);

  flatConformersOut = nvMolKit::flattenConformers(mols, perMolEnergiesOut);

  nvMolKit::MMFF::BatchedMolecularSystemHost systemHost;
  std::vector<double>                        scratchPos;
  for (const auto& confInfo : flatConformersOut) {
    auto contribs = nvMolKit::MMFF::constructForcefieldContribs(*confInfo.mol);
    nvMolKit::confPosToVect(*confInfo.conformer, scratchPos);
    nvMolKit::MMFF::addMoleculeToBatch(contribs, scratchPos, systemHost);
  }
  return systemHost;
}

//! \brief Push the batched host system to the device and allocate work buffers.
void uploadBatchedSystem(const nvMolKit::MMFF::BatchedMolecularSystemHost& systemHost,
                         nvMolKit::MMFF::BatchedMolecularDeviceBuffers&    systemDevice,
                         cudaStream_t                                      stream) {
  nvMolKit::ScopedNvtxRange uploadRange("MMFF upload to device", nvMolKit::NvtxColor::kCyan);
  nvMolKit::MMFF::sendContribsAndIndicesToDevice(systemHost, systemDevice);
  nvMolKit::MMFF::setStreams(systemDevice, stream);
  nvMolKit::MMFF::allocateIntermediateBuffers(systemHost, systemDevice);
  systemDevice.positions.resize(systemHost.positions.size());
  systemDevice.positions.copyFromHost(systemHost.positions.data(), systemHost.positions.size());
  systemDevice.grad.resize(systemHost.positions.size());
  systemDevice.grad.zero();
  systemDevice.energyOuts.zero();
  systemDevice.energyBuffer.zero();
  cudaCheckError(cudaStreamSynchronize(stream));
}

void runBatchedEnergy(nvMolKit::MMFF::BatchedMolecularDeviceBuffers& systemDevice, cudaStream_t stream) {
  nvMolKit::ScopedNvtxRange range("BATCHED energy", nvMolKit::NvtxColor::kRed);
  systemDevice.energyOuts.zero();
  systemDevice.energyBuffer.zero();
  cudaCheckError(nvMolKit::MMFF::computeEnergy(systemDevice, /*coords=*/nullptr, stream));
}

void runBatchedGradient(nvMolKit::MMFF::BatchedMolecularDeviceBuffers& systemDevice, cudaStream_t stream) {
  nvMolKit::ScopedNvtxRange range("BATCHED gradient", nvMolKit::NvtxColor::kRed);
  systemDevice.grad.zero();
  cudaCheckError(nvMolKit::MMFF::computeGradients(systemDevice, stream));
}

void runPerMolEnergy(nvMolKit::MMFF::BatchedMolecularDeviceBuffers& systemDevice, cudaStream_t stream) {
  nvMolKit::ScopedNvtxRange range("PER_MOL energy", nvMolKit::NvtxColor::kGreen);
  systemDevice.energyOuts.zero();
  cudaCheckError(nvMolKit::MMFF::computeEnergyBlockPerMol(systemDevice, /*coords=*/nullptr, stream));
}

void runPerMolGradient(nvMolKit::MMFF::BatchedMolecularDeviceBuffers& systemDevice, cudaStream_t stream) {
  nvMolKit::ScopedNvtxRange range("PER_MOL gradient", nvMolKit::NvtxColor::kGreen);
  systemDevice.grad.zero();
  cudaCheckError(nvMolKit::MMFF::computeGradBlockPerMol(systemDevice, stream));
}

void runOneEvaluation(KernelMode                                     mode,
                      EvalKind                                       kind,
                      nvMolKit::MMFF::BatchedMolecularDeviceBuffers& systemDevice,
                      cudaStream_t                                   stream) {
  if (mode == KernelMode::BATCHED || mode == KernelMode::BOTH) {
    if (kind == EvalKind::ENERGY || kind == EvalKind::BOTH) {
      runBatchedEnergy(systemDevice, stream);
    }
    if (kind == EvalKind::GRADIENT || kind == EvalKind::BOTH) {
      runBatchedGradient(systemDevice, stream);
    }
  }
  if (mode == KernelMode::PER_MOL || mode == KernelMode::BOTH) {
    if (kind == EvalKind::ENERGY || kind == EvalKind::BOTH) {
      runPerMolEnergy(systemDevice, stream);
    }
    if (kind == EvalKind::GRADIENT || kind == EvalKind::BOTH) {
      runPerMolGradient(systemDevice, stream);
    }
  }
}

//! Accumulated CUDA-event elapsed time (milliseconds) per kernel segment over timed iterations.
struct GpuKernelTimingMs {
  double batchedEnergy    = 0.0;
  double batchedGradient  = 0.0;
  double perMolEnergy     = 0.0;
  double perMolGradient   = 0.0;
};

template <typename Fn>
float timeGpuSegment(cudaEvent_t evtStart, cudaEvent_t evtStop, cudaStream_t stream, Fn&& body) {
  cudaCheckError(cudaEventRecord(evtStart, stream));
  body();
  cudaCheckError(cudaEventRecord(evtStop, stream));
  cudaCheckError(cudaEventSynchronize(evtStop));
  float milliseconds = 0.0F;
  cudaCheckError(cudaEventElapsedTime(&milliseconds, evtStart, evtStop));
  return milliseconds;
}

//! Same dispatch as \ref runOneEvaluation but records per-segment GPU time into \p totals.
void runOneEvaluationTimed(KernelMode                                     mode,
                           EvalKind                                       kind,
                           nvMolKit::MMFF::BatchedMolecularDeviceBuffers& systemDevice,
                           cudaStream_t                                   stream,
                           cudaEvent_t                                    evtStart,
                           cudaEvent_t                                    evtStop,
                           GpuKernelTimingMs&                             totals) {
  if (mode == KernelMode::BATCHED || mode == KernelMode::BOTH) {
    if (kind == EvalKind::ENERGY || kind == EvalKind::BOTH) {
      totals.batchedEnergy +=
        timeGpuSegment(evtStart, evtStop, stream, [&]() { runBatchedEnergy(systemDevice, stream); });
    }
    if (kind == EvalKind::GRADIENT || kind == EvalKind::BOTH) {
      totals.batchedGradient +=
        timeGpuSegment(evtStart, evtStop, stream, [&]() { runBatchedGradient(systemDevice, stream); });
    }
  }
  if (mode == KernelMode::PER_MOL || mode == KernelMode::BOTH) {
    if (kind == EvalKind::ENERGY || kind == EvalKind::BOTH) {
      totals.perMolEnergy +=
        timeGpuSegment(evtStart, evtStop, stream, [&]() { runPerMolEnergy(systemDevice, stream); });
    }
    if (kind == EvalKind::GRADIENT || kind == EvalKind::BOTH) {
      totals.perMolGradient +=
        timeGpuSegment(evtStart, evtStop, stream, [&]() { runPerMolGradient(systemDevice, stream); });
    }
  }
}

void printTimingSummary(const GpuKernelTimingMs& totals,
                        KernelMode               mode,
                        EvalKind                 kind,
                        int                      timedIterations,
                        double                   wallSeconds) {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n=== MMFF GPU timing (CUDA events; mean over " << timedIterations << " timed iteration(s)) ===\n";
  const double denom = static_cast<double>(std::max(timedIterations, 1));

  auto printRow = [&](const char* label, double sumMs, bool active) {
    if (!active) {
      return;
    }
    const double meanMs = sumMs / denom;
    std::cout << "  " << std::left << std::setw(22) << label << std::right << meanMs << " ms/iter"
              << "   (total " << sumMs << " ms)\n";
  };

  const bool showBatched =
    (mode == KernelMode::BATCHED || mode == KernelMode::BOTH);
  const bool showPerMol =
    (mode == KernelMode::PER_MOL || mode == KernelMode::BOTH);
  const bool doEnergy    = (kind == EvalKind::ENERGY || kind == EvalKind::BOTH);
  const bool doGradient  = (kind == EvalKind::GRADIENT || kind == EvalKind::BOTH);

  printRow("BATCHED energy", totals.batchedEnergy, showBatched && doEnergy);
  printRow("BATCHED gradient", totals.batchedGradient, showBatched && doGradient);
  printRow("PER_MOL energy", totals.perMolEnergy, showPerMol && doEnergy);
  printRow("PER_MOL gradient", totals.perMolGradient, showPerMol && doGradient);

  std::cout << "  Wall clock (timed loop only): " << wallSeconds << " s\n";

  if (mode != KernelMode::BOTH) {
    std::cout << "(Comparison BATCHED vs PER_MOL requires --mode both.)\n";
    return;
  }

  if (doEnergy) {
    const double meanBatched = totals.batchedEnergy / denom;
    const double meanPerMol  = totals.perMolEnergy / denom;
    const double delta       = meanPerMol - meanBatched;
    std::cout << "\nEnergy (PER_MOL - BATCHED): " << delta << " ms/iter";
    if (meanBatched > 1e-9) {
      std::cout << "   time ratio PER_MOL/BATCHED = " << (meanPerMol / meanBatched);
    }
    std::cout << "\n";
  }
  if (doGradient) {
    const double meanBatched = totals.batchedGradient / denom;
    const double meanPerMol  = totals.perMolGradient / denom;
    const double delta       = meanPerMol - meanBatched;
    std::cout << "Gradient (PER_MOL - BATCHED): " << delta << " ms/iter";
    if (meanBatched > 1e-9) {
      std::cout << "   time ratio PER_MOL/BATCHED = " << (meanPerMol / meanBatched);
    }
    std::cout << "\n";
  }
  std::cout << "Interpretation: positive ms delta means PER_MOL took longer than BATCHED on average.\n"
            << "Time ratio PER_MOL/BATCHED < 1 means PER_MOL used less GPU time for that segment.\n";
}

void printHelp(const char* progName) {
  std::cout
    << "Usage: " << progName << " [options]\n\n"
    << "Profiler-oriented benchmark that times only the MMFF energy and\n"
    << "gradient kernels (no minimization). Loads molecules, embeds them with\n"
    << "RDKit (EmbedMultipleConfs + ETKDGv3, CPU), builds the flattened batched\n"
    << "MMFF system, then loops over the requested kernel(s).\n\n"
    << "Options:\n"
    << "  -f, --file_path <path>          Input .sdf / .smi / .smiles / .cxsmiles file (required)\n"
    << "  -n, --num_mols <int>            Max molecules after random subsample "
    << "(0 = entire file; parses every record, avoid on multi-GB SMILES) [default: 100]\n"
    << "  -c, --confs_per_mol <int>       Conformers per molecule [default: 10]\n"
    << "  -i, --embed_iters <int>         RDKit ETKDG EmbedParameters::maxIterations (> 0) "
    << "[default: 200]\n"
    << "  -m, --mode <batched|per_mol|both>  Which MMFF kernel layout to time [default: both]\n"
    << "  -e, --eval <energy|gradient|both>  Which evaluation to run per iteration [default: both]\n"
    << "      --warmup_iters <int>        Warmup E/F evaluations before timed loop [default: 3]\n"
    << "  -r, --iters <int>               Timed E/F evaluations to run [default: 50]\n"
    << "  -t, --num_threads <int>         OpenMP threads embedding molecules in parallel (<=0: all CPUs)\n"
    << "  -s, --seed <uint64>             Seed: subsample shuffle + RDKit embed randomSeed [default: 42]\n"
    << "  -h, --help                      Show this help.\n\n"
    << "NVTX ranges emitted: 'BATCHED energy', 'BATCHED gradient', 'PER_MOL energy',\n"
    << "'PER_MOL gradient', plus setup ranges. Run under nsys for timelines or\n"
    << "ncu for per-kernel occupancy / stall info. After the timed loop the binary\n"
    << "prints CUDA-event mean ms per segment and PER_MOL vs BATCHED deltas when\n"
    << "--mode both.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string filePath;
  int         numMols       = 100;
  int         confsPerMol   = 10;
  int         embedIters    = 200;
  KernelMode  mode          = KernelMode::BOTH;
  EvalKind    evalKind      = EvalKind::BOTH;
  int         warmupIters   = 3;
  int         timedIters    = 50;
  int         numThreads    = -1;
  uint64_t    seed          = 42;

  static const struct option longOptions[] = {
    {       "file_path", required_argument, nullptr, 'f'},
    {        "num_mols", required_argument, nullptr, 'n'},
    {   "confs_per_mol", required_argument, nullptr, 'c'},
    {     "embed_iters", required_argument, nullptr, 'i'},
    {            "mode", required_argument, nullptr, 'm'},
    {            "eval", required_argument, nullptr, 'e'},
    {    "warmup_iters", required_argument, nullptr,  1 },
    {           "iters", required_argument, nullptr, 'r'},
    {     "num_threads", required_argument, nullptr, 't'},
    {            "seed", required_argument, nullptr, 's'},
    {            "help",       no_argument, nullptr, 'h'},
    {           nullptr,                 0, nullptr,  0 }
  };

  int optionIndex = 0;
  int opt;
  while ((opt = getopt_long(argc, argv, "f:n:c:i:m:e:r:t:s:h", longOptions, &optionIndex)) != -1) {
    try {
      switch (opt) {
        case 'f':
          filePath = optarg;
          break;
        case 'n':
          numMols = std::stoi(optarg);
          break;
        case 'c':
          confsPerMol = std::stoi(optarg);
          break;
        case 'i':
          embedIters = std::stoi(optarg);
          break;
        case 'm':
          mode = parseKernelMode(optarg);
          break;
        case 'e':
          evalKind = parseEvalKind(optarg);
          break;
        case 1:
          warmupIters = std::stoi(optarg);
          break;
        case 'r':
          timedIters = std::stoi(optarg);
          break;
        case 't':
          numThreads = std::stoi(optarg);
          break;
        case 's':
          seed = static_cast<uint64_t>(std::stoull(optarg));
          break;
        case 'h':
          printHelp(argv[0]);
          return 0;
        case '?':
        default:
          std::cerr << "\nUse --help for usage information.\n";
          return 1;
      }
    } catch (const std::exception& error) {
      std::cerr << "Error parsing option: " << error.what() << "\n";
      return 1;
    }
  }

  if (filePath.empty()) {
    std::cerr << "Error: --file_path is required.\n";
    printHelp(argv[0]);
    return 1;
  }
  if (!std::filesystem::exists(filePath)) {
    std::cerr << "Error: input file does not exist: " << filePath << "\n";
    return 1;
  }
  if (confsPerMol <= 0) {
    std::cerr << "Error: --confs_per_mol must be positive.\n";
    return 1;
  }
  if (timedIters <= 0) {
    std::cerr << "Error: --iters must be positive.\n";
    return 1;
  }
  if (warmupIters < 0) {
    std::cerr << "Error: --warmup_iters must be >= 0.\n";
    return 1;
  }
  if (embedIters <= 0) {
    std::cerr << "Error: --embed_iters must be a positive integer.\n";
    return 1;
  }

  std::cout << "Configuration:\n"
            << "  Input: " << filePath << "\n"
            << "  num_mols cap: " << numMols << "\n"
            << "  confs_per_mol: " << confsPerMol << "\n"
            << "  embed_iters: " << embedIters << "\n"
            << "  mode: " << kernelModeName(mode) << "\n"
            << "  eval: " << evalKindName(evalKind) << "\n"
            << "  warmup iters: " << warmupIters << "\n"
            << "  timed iters: " << timedIters << "\n"
            << "  RDKit embed OpenMP threads: " << numThreads << " (<=0: use all CPUs)\n"
            << "  seed: " << seed << "\n\n";

  std::cout << "Loading molecules...\n";
  auto mols = loadMolecules(filePath, numMols < 0 ? 0u : static_cast<unsigned int>(numMols), seed);
  std::cout << "  " << mols.size() << " molecules loaded\n";

  std::cout << "Embedding " << confsPerMol << " conformers per molecule with RDKit ETKDGv3...\n";
  embedConformers(mols, confsPerMol, embedIters, numThreads, seed);
  if (mols.empty()) {
    std::cerr << "Error: no molecules retained after embedding.\n";
    return 1;
  }
  size_t totalConfs = 0;
  for (const auto& mol : mols) {
    totalConfs += mol->getNumConformers();
  }
  std::cout << "  " << mols.size() << " molecules with " << totalConfs << " conformers ready\n";

  std::vector<RDKit::ROMol*> molPtrs;
  molPtrs.reserve(mols.size());
  for (const auto& mol : mols) {
    molPtrs.push_back(mol.get());
  }

  std::vector<nvMolKit::ConformerInfo> flatConformers;
  std::vector<std::vector<double>>     perMolEnergies;
  auto                                 systemHost = buildBatchedHostSystem(molPtrs, flatConformers, perMolEnergies);
  std::cout << "Batched MMFF system built: " << flatConformers.size() << " conformers, "
            << systemHost.positions.size() / 3 << " total atoms (max atoms in any system: " << systemHost.maxNumAtoms
            << ")\n";

  cudaStream_t stream = nullptr;
  cudaCheckError(cudaStreamCreate(&stream));
  {
    nvMolKit::MMFF::BatchedMolecularDeviceBuffers systemDevice;
    uploadBatchedSystem(systemHost, systemDevice, stream);

    if (warmupIters > 0) {
      nvMolKit::ScopedNvtxRange warmupRange("warmup", nvMolKit::NvtxColor::kGrey);
      std::cout << "Running " << warmupIters << " warmup iteration(s)...\n";
      for (int iter = 0; iter < warmupIters; ++iter) {
        runOneEvaluation(mode, evalKind, systemDevice, stream);
      }
      cudaCheckError(cudaStreamSynchronize(stream));
    }

    std::cout << "Running " << timedIters << " timed iteration(s) with CUDA event timing...\n";
    cudaEvent_t evtStart = nullptr;
    cudaEvent_t evtStop  = nullptr;
    cudaCheckError(cudaEventCreate(&evtStart));
    cudaCheckError(cudaEventCreate(&evtStop));
    GpuKernelTimingMs timedTotals;
    const auto        wallBegin = std::chrono::steady_clock::now();
    {
      nvMolKit::ScopedNvtxRange timedRange("timed loop", nvMolKit::NvtxColor::kOrange);
      for (int iter = 0; iter < timedIters; ++iter) {
        nvMolKit::ScopedNvtxRange iterRange("eval iter", nvMolKit::NvtxColor::kYellow);
        runOneEvaluationTimed(mode, evalKind, systemDevice, stream, evtStart, evtStop, timedTotals);
      }
      cudaCheckError(cudaStreamSynchronize(stream));
    }
    const auto wallEnd = std::chrono::steady_clock::now();
    cudaCheckError(cudaEventDestroy(evtStart));
    cudaCheckError(cudaEventDestroy(evtStop));
    const double wallSeconds =
      std::chrono::duration<double>(wallEnd - wallBegin).count();
    printTimingSummary(timedTotals, mode, evalKind, timedIters, wallSeconds);
  }
  cudaCheckError(cudaStreamDestroy(stream));
  std::cout << "Done.\n";
  return 0;
}
