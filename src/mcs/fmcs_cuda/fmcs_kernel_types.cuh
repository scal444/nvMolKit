// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FMCS_CUDA_FMCS_KERNEL_TYPES_CUH
#define FMCS_CUDA_FMCS_KERNEL_TYPES_CUH

#include <cstdint>

#include "fmcs_cuda/fmcs_match_tables.cuh"

namespace mcs {
namespace fmcs {

/// Non-owning view over one side's CSR and bond-endpoint arrays.
struct DeviceCsrView {
  static constexpr bool kHasAdjacencyBondIndices = true;

  const std::uint32_t* rowOffsets    = nullptr;
  const std::uint32_t* colIndices    = nullptr;
  const std::uint32_t* bondIndices   = nullptr;
  const std::uint32_t* bondEndpoints = nullptr;
  int                  numAtoms      = 0;
  int                  numBonds      = 0;
};

/// Per-pair descriptor passed to the kernel. Non-owning: pointers refer
/// into host-uploaded device buffers. The caller chooses the smaller input
/// as the query and records that choice in swapped for host-side expansion.
struct DevicePerPairInput {
  int queryNumAtoms  = 0;
  int queryNumBonds  = 0;
  int targetNumAtoms = 0;
  int targetNumBonds = 0;

  const std::uint32_t* queryRowOffsets    = nullptr;
  const std::uint32_t* queryColIndices    = nullptr;
  const std::uint32_t* queryBondIndices   = nullptr;
  const std::uint32_t* queryBondEndpoints = nullptr;
  const std::uint32_t* queryRingBondFlags = nullptr;

  const std::uint32_t* targetRowOffsets    = nullptr;
  const std::uint32_t* targetColIndices    = nullptr;
  const std::uint32_t* targetBondIndices   = nullptr;
  const std::uint32_t* targetBondEndpoints = nullptr;
  const std::uint32_t* targetRingBondFlags = nullptr;

  PairMatchTablesDevice tables;

  bool swapped           = false;
  bool completeRingsOnly = false;
};

/// Fixed-size device-writable result. The host expands this POD into MCSResult.
template <int maxAtoms, int maxBonds> struct DeviceMCSResult {
  int  numCommonVertices = 0;
  int  numCommonEdges    = 0;
  bool timedOut          = false;
  bool overflowed        = false;

  uint8_t mappingA[maxAtoms];
  uint8_t mappingB[maxAtoms];
  uint8_t bondMapA[maxBonds];
  uint8_t bondMapB[maxBonds];
};

}  // namespace fmcs
}  // namespace mcs

#endif  // FMCS_CUDA_FMCS_KERNEL_TYPES_CUH
