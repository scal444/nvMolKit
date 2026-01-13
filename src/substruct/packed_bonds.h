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

#ifndef NVMOLKIT_PACKED_BONDS_H
#define NVMOLKIT_PACKED_BONDS_H

#include <cstdint>

namespace nvMolKit {

/**
 * @brief Flags for bond query matching.
 *
 * Bonds can have queries like `-&!@` (single AND not ring bond).
 */
enum BondQueryFlags : uint8_t {
  BondQueryNone             = 0,
  BondQueryIsRingBond       = 1 << 0,  ///< Bond must be in a ring (@)
  BondQueryNotRingBond      = 1 << 1,  ///< Bond must NOT be in a ring (!@)
  BondQuerySingleOrAromatic = 1 << 2,  ///< SingleOrAromaticBond query (matches single or aromatic only)
  BondQueryDoubleOrAromatic = 1 << 3,  ///< DoubleOrAromaticBond query (matches double or aromatic only)
  BondQueryAromaticOnly     = 1 << 4,  ///< Aromatic bond query (:) - matches aromatic bonds only (type 7 or 12)
  BondQueryNeverMatches     = 1 << 5,  ///< Impossible constraint (e.g., single AND aromatic)
  BondQueryUseBondMask      = 1 << 6,  ///< Use allowedBondTypes bitmask for arbitrary OR patterns
};

constexpr int kMaxBondsPerAtom = 8;
constexpr uint8_t kNoNeighbor = 0xFF;

/**
 * @brief Packed bond adjacency for a target atom.
 *
 * degree: number of valid bonds (0-8)
 * neighborIdx: 8 neighbor atom indices
 * bondInfo: 8 packed bond descriptors
 *   - bits [0-3]: bondType (0-15)
 *   - bit [4]: isInRing
 *   - bits [5-7]: unused
 */
struct TargetAtomBonds {
  uint8_t degree;
  uint8_t neighborIdx[kMaxBondsPerAtom];
  uint8_t bondInfo[kMaxBondsPerAtom];
};

/**
 * @brief Packed bond adjacency for a query atom.
 *
 * degree: number of valid bonds (0-8)
 * neighborIdx: 8 neighbor query atom indices
 * matchMask: 8 precomputed 32-bit match masks
 *   - bits [0-15]: target bond types that match when target isInRing=0
 *   - bits [16-31]: target bond types that match when target isInRing=1
 *
 * Match check: (matchMask >> (isInRing * 16 + bondType)) & 1
 */
struct QueryAtomBonds {
  uint8_t  degree;
  uint8_t  neighborIdx[kMaxBondsPerAtom];
  uint32_t matchMask[kMaxBondsPerAtom];
};

/**
 * @brief Pack target bond info into a single byte.
 */
inline uint8_t packTargetBondInfo(uint8_t bondType, bool isInRing) {
  return (bondType & 0x0F) | (isInRing ? 0x10 : 0);
}

/**
 * @brief Build query bond match mask from bond query parameters.
 *
 * @param queryBondType Bond type to match (0 = any)
 * @param queryFlags BondQueryFlags bitmask
 * @param allowedBondTypes Bitmask of allowed types (when BondQueryUseBondMask set)
 * @return 32-bit combined match mask
 */
uint32_t buildQueryBondMatchMask(uint8_t queryBondType, uint8_t queryFlags, uint16_t allowedBondTypes);

}  // namespace nvMolKit

#endif  // NVMOLKIT_PACKED_BONDS_H

