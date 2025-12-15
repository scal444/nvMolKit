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
  const int16_t* __restrict__ atomBondStarts;    ///< Pointer to this molecule's atom bond offsets
  const int16_t* __restrict__ otherAtomIndices;  ///< Pointer to this molecule's neighbor atom indices
  const int16_t* __restrict__ bondDataIndices;   ///< Pointer to this molecule's neighbor bond indices
  int numAtoms;
  int numBonds;

  __device__ __forceinline__ const AtomData& getAtom(int atomIdx) const { return atomData[atomIdx]; }

  __device__ __forceinline__ const BondData& getBond(int bondIdx) const { return bondData[bondIdx]; }

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
};

/**
 * @brief Get a per-molecule view with pre-shifted pointers from a batch view.
 * @param view The batch-level view
 * @param molIdx Index of the molecule in the batch
 * @return MoleculeView with pointers offset to this molecule's data
 */
__device__ __forceinline__ MoleculeView getMolecule(const MoleculesDeviceView& view, int molIdx) {
  MoleculeView mol;
  mol.atomData         = view.atomData + view.batchAtomStarts[molIdx];
  mol.bondData         = view.bondData + view.batchBondStarts[molIdx];
  mol.atomBondStarts   = view.atomBondStarts + view.batchAtomBondStarts[molIdx];
  mol.otherAtomIndices = view.otherAtomIndices + view.batchOtherAtomIndicesStarts[molIdx];
  mol.bondDataIndices  = view.bondDataIndices + view.batchBondIndicesStarts[molIdx];
  mol.numAtoms         = view.batchAtomStarts[molIdx + 1] - view.batchAtomStarts[molIdx];
  mol.numBonds         = view.batchBondStarts[molIdx + 1] - view.batchBondStarts[molIdx];
  return mol;
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_MOLECULES_DEVICE_CUH
