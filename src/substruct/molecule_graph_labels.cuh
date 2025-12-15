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

struct AtomData {
  static constexpr uint8_t unsetValenceVal =
      std::numeric_limits<uint8_t>::max();

  uint8_t atomicNum = 0;
  uint8_t numExplicitHs = 0;
  uint8_t explicitValence = unsetValenceVal;
  uint8_t implicitValence = unsetValenceVal;

  int8_t formalCharge = 0;
  uint8_t chiralTag = 0;
  uint8_t numRadicalElectrons = 0;
  uint8_t hybridization = 0;
  bool isAromatic = false;
  uint8_t minRingSize = 0;
  uint8_t numRings = 0;
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
