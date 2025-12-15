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

#ifndef NVMOLKIT_MOLECULES_H
#define NVMOLKIT_MOLECULES_H

#include <cstdint>
#include <limits>
#include <vector>

#include "device_vector.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

struct AtomData {
  static constexpr uint8_t unsetValenceVal = std::numeric_limits<uint8_t>::max();

  uint8_t atomicNum           = 0;
  uint8_t numExplicitHs       = 0;
  uint8_t explicitValence     = unsetValenceVal;
  uint8_t implicitValence     = unsetValenceVal;
  int8_t  formalCharge        = 0;
  uint8_t chiralTag           = 0;
  uint8_t numRadicalElectrons = 0;
  uint8_t hybridization       = 0;
  uint8_t minRingSize         = 0;
  uint8_t numRings            = 0;
  bool    isAromatic          = false;
};

struct BondData {
  uint8_t bondType = 0;
};

/**
 * @brief Host-side batched molecule storage.
 *
 * Stores multiple molecules in a flattened format optimized for GPU transfer.
 * Each molecule's atoms and bonds are stored contiguously, with offset arrays
 * to locate each molecule's data.
 */
struct MoleculesHost {
  // Batch-level offsets (size = numMolecules + 1)
  std::vector<int> batchAtomStarts;              ///< Start index into atomData for each molecule
  std::vector<int> batchBondStarts;              ///< Start index into bondData for each molecule
  std::vector<int> batchAtomBondStarts;          ///< Start index into atomBondStarts for each molecule
  std::vector<int> batchOtherAtomIndicesStarts;  ///< Start index into otherAtomIndices for each molecule
  std::vector<int> batchBondIndicesStarts;       ///< Start index into bondDataIndices for each molecule

  // Molecule-level data (flattened across all molecules)
  std::vector<AtomData> atomData;          ///< Atom properties for all atoms
  std::vector<BondData> bondData;          ///< Bond properties for all bonds
  std::vector<int16_t>  atomBondStarts;    ///< Cumulative count of bonds per atom (prefix sum)
  std::vector<int16_t>  otherAtomIndices;  ///< For each atom-bond pair, the other atom index
  std::vector<int16_t>  bondDataIndices;   ///< For each atom-bond pair, index into bondData

  MoleculesHost();

  [[nodiscard]] size_t numMolecules() const { return batchAtomStarts.empty() ? 0 : batchAtomStarts.size() - 1; }
  [[nodiscard]] size_t totalAtoms() const { return atomData.size(); }
  [[nodiscard]] size_t totalBonds() const { return bondData.size(); }
};

/**
 * @brief Device-side view into batched molecule data.
 *
 * This structure contains pointers to device memory for the full batch.
 * This is a POD struct that can be passed to CUDA kernels by value.
 * Use getMolecule() from molecules_device.cuh to get per-molecule views.
 */
struct MoleculesDeviceView {
  const int*      batchAtomStarts;
  const int*      batchBondStarts;
  const int*      batchAtomBondStarts;
  const int*      batchOtherAtomIndicesStarts;
  const int*      batchBondIndicesStarts;
  const AtomData* atomData;
  const BondData* bondData;
  const int16_t*  atomBondStarts;
  const int16_t*  otherAtomIndices;
  const int16_t*  bondDataIndices;
  int             numMolecules;
};

/**
 * @brief Device-side storage for batched molecules using AsyncDeviceVector.
 *
 * Owns the device memory and provides a view for kernel access.
 */
class MoleculesDevice {
 public:
  MoleculesDevice() = default;
  explicit MoleculesDevice(cudaStream_t stream) { setStream(stream); }

  /**
   * @brief Copy molecule data from host to device.
   * @param host The host-side molecule batch to copy
   * @param stream CUDA stream for async operations (optional, uses stored stream if not provided)
   */
  void copyFromHost(const MoleculesHost& host, cudaStream_t stream);
  void copyFromHost(const MoleculesHost& host) { copyFromHost(host, stream_); }

  /**
   * @brief Get a view suitable for passing to CUDA kernels.
   */
  [[nodiscard]] MoleculesDeviceView view() const;

  void setStream(cudaStream_t stream);

 private:
  cudaStream_t stream_       = nullptr;
  int          numMolecules_ = 0;

  AsyncDeviceVector<int>      batchAtomStarts_;
  AsyncDeviceVector<int>      batchBondStarts_;
  AsyncDeviceVector<int>      batchAtomBondStarts_;
  AsyncDeviceVector<int>      batchOtherAtomIndicesStarts_;
  AsyncDeviceVector<int>      batchBondIndicesStarts_;
  AsyncDeviceVector<AtomData> atomData_;
  AsyncDeviceVector<BondData> bondData_;
  AsyncDeviceVector<int16_t>  atomBondStarts_;
  AsyncDeviceVector<int16_t>  otherAtomIndices_;
  AsyncDeviceVector<int16_t>  bondDataIndices_;
};

/**
 * @brief Add a molecule to an existing batch.
 * @param mol Pointer to the RDKit molecule to add
 * @param batch The batch to add the molecule to
 */
void addToBatch(const RDKit::ROMol* mol, MoleculesHost& batch);

}  // namespace nvMolKit

#endif  // NVMOLKIT_MOLECULES_H
