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

#ifndef NVMOLKIT_MOLECULES_DEVICE_CUH
#define NVMOLKIT_MOLECULES_DEVICE_CUH

#include "atom_data_packed.h"
#include "molecules.h"

namespace nvMolKit {

/**
 * @brief Per-molecule view with pre-shifted pointers for device access.
 *
 * This is a lightweight view into a single molecule's data, with all pointers
 * already offset to this molecule's portion of the batch. Constructed via
 * getMolecule() from a MoleculesDeviceView.
 */
struct MoleculeView {
  const AtomData* __restrict__ atomData;         ///< Pointer to first atom of this molecule
  const BondData* __restrict__ bondData;         ///< Pointer to first bond of this molecule
  const AtomQuery* __restrict__ atomQueries;     ///< Pointer to this molecule's atom queries
  const int16_t* __restrict__ atomBondStarts;    ///< Pointer to this molecule's atom bond offsets
  const int16_t* __restrict__ otherAtomIndices;  ///< Pointer to this molecule's neighbor atom indices
  const int16_t* __restrict__ bondDataIndices;   ///< Pointer to this molecule's neighbor bond indices
  int numAtoms;
  int numBonds;

  // GPU-optimized packed data
  const AtomDataPacked* __restrict__ atomDataPacked;  ///< Packed atom properties for GPU matching
  const AtomQueryMask* __restrict__ atomQueryMasks;   ///< Precomputed query masks (query molecules only)
  const BondTypeCounts* __restrict__ bondTypeCounts;  ///< Precomputed bond type counts per atom

  __device__ __forceinline__ const AtomData& getAtom(int atomIdx) const { return atomData[atomIdx]; }

  __device__ __forceinline__ const BondData& getBond(int bondIdx) const { return bondData[bondIdx]; }

  __device__ __forceinline__ AtomQuery getAtomQuery(int atomIdx) const { return atomQueries[atomIdx]; }

  __device__ __forceinline__ int getAtomDegree(int atomIdx) const {
    return atomBondStarts[atomIdx + 1] - atomBondStarts[atomIdx];
  }

  __device__ __forceinline__ int getNeighborAtomIdx(int atomIdx, int neighborIdx) const {
    const int neighborListStart = atomBondStarts[atomIdx];
    return otherAtomIndices[neighborListStart + neighborIdx];
  }

  __device__ __forceinline__ int getNeighborBondIdx(int atomIdx, int neighborIdx) const {
    const int neighborListStart = atomBondStarts[atomIdx];
    return bondDataIndices[neighborListStart + neighborIdx];
  }

  /// Get packed atom data for GPU matching
  __device__ __forceinline__ const AtomDataPacked& getAtomPacked(int atomIdx) const { return atomDataPacked[atomIdx]; }

  /// Get precomputed query mask (only valid for query molecules)
  __device__ __forceinline__ const AtomQueryMask& getQueryMask(int atomIdx) const { return atomQueryMasks[atomIdx]; }

  /// Get precomputed bond type counts
  __device__ __forceinline__ const BondTypeCounts& getBondTypeCounts(int atomIdx) const {
    return bondTypeCounts[atomIdx];
  }
};

/**
 * @brief Get a per-molecule view with pre-shifted pointers from a batch view.
 * @param view The batch-level view
 * @param molIdx Index of the molecule in the batch
 * @return MoleculeView with pointers offset to this molecule's data
 */
__device__ __forceinline__ MoleculeView getMolecule(const MoleculesDeviceView& view, int molIdx) {
  MoleculeView mol;
  const int    atomStart = view.batchAtomStarts[molIdx];
  mol.atomData           = view.atomData + atomStart;
  mol.bondData           = view.bondData + view.batchBondStarts[molIdx];
  mol.atomQueries        = view.atomQueries + atomStart;
  mol.atomBondStarts     = view.atomBondStarts + view.batchAtomBondStarts[molIdx];
  mol.otherAtomIndices   = view.otherAtomIndices + view.batchOtherAtomIndicesStarts[molIdx];
  mol.bondDataIndices    = view.bondDataIndices + view.batchBondIndicesStarts[molIdx];
  mol.numAtoms           = view.batchAtomStarts[molIdx + 1] - atomStart;
  mol.numBonds           = view.batchBondStarts[molIdx + 1] - view.batchBondStarts[molIdx];

  // GPU-optimized packed data (may be nullptr if not populated)
  mol.atomDataPacked = view.atomDataPacked ? view.atomDataPacked + atomStart : nullptr;
  mol.atomQueryMasks = view.atomQueryMasks ? view.atomQueryMasks + atomStart : nullptr;
  mol.bondTypeCounts = view.bondTypeCounts ? view.bondTypeCounts + atomStart : nullptr;
  return mol;
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_MOLECULES_DEVICE_CUH
