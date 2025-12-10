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

#ifndef NVMOLKIT_MOLECULE_GRAPH_LABELS_H
#define NVMOLKIT_MOLECULE_GRAPH_LABELS_H
#include <cstdint>

namespace nvMolKit {

__host__ __device__
struct AtomBaseInfo {
  //! Only takes up 128, could reuse
  uint8_t atomicNumber;
  //! C, CC, and the other specification (cn, need to figure that out)
  uint8_t chiralType;

  bool aromatic;

  uint8_t implicitConnections;
  uint8_t explicitConections;
  uint8_t charge;
  uint8_t numCycles;
  uint8_t cycleSize;
};

__host__ __device__
struct BondBaseInfo {
  uint8_t bondType;
};

enum class AtomQueryType: uint8_t {
  ATOM_NUMBER = 0,
  CHIRAL_TYPE = 1,
  AROMATICITY = 2,
  IMPLICIT_CONNECTIONS = 3,
  EXPLICIT_CONNECTIONS = 4,
  CHARGE = 5,
  NUM_CYCLES = 6,
  CYCLE_SIZE = 7,
  ANY = 8
};

struct AtomQueryBase {
  AtomQueryType type;
  uint8_t matchValue;
};


struct BondQuery {};

} // namespace nvMolKit

#endif  // NVMOLKIT_MOLECULE_GRAPH_LABELS_H
