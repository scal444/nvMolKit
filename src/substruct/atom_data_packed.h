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

#ifndef NVMOLKIT_ATOM_DATA_PACKED_H
#define NVMOLKIT_ATOM_DATA_PACKED_H

#include <cstdint>
#include <limits>

#ifdef __CUDACC__
#define HD_CALLABLE __host__ __device__
#else
#define HD_CALLABLE
#endif

namespace nvMolKit {

/**
 * @brief Packed atom data for efficient GPU comparison via bitwise operations.
 *
 * This structure packs all atom properties into 128 bits (2x uint64_t) to enable
 * branchless mask-and-compare matching. All threads in a warp execute the same
 * instructions regardless of which fields are being compared.
 *
 * Bit layout (little-endian byte order within each uint64_t):
 *
 * Lower 64 bits (lo):
 *   Byte 0 [bits  0-7 ]: atomicNum (uint8_t, valid range 0-118)
 *   Byte 1 [bits  8-15]: numExplicitHs (uint8_t)
 *   Byte 2 [bits 16-23]: explicitValence (uint8_t, 0xFF = unset)
 *   Byte 3 [bits 24-31]: implicitValence (uint8_t, 0xFF = unset)
 *   Byte 4 [bits 32-39]: formalCharge (int8_t, signed, stored as uint8_t)
 *   Byte 5 [bits 40-47]: chiralTag (uint8_t)
 *   Byte 6 [bits 48-55]: numRadicalElectrons (uint8_t)
 *   Byte 7 [bits 56-63]: hybridization (uint8_t)
 *
 * Upper 64 bits (hi):
 *   Byte 0 [bits  0-7 ]: minRingSize (uint8_t)
 *   Byte 1 [bits  8-15]: numRings (uint8_t)
 *   Byte 2 [bits 16-23]: isAromatic (0x00 = false, 0x01 = true)
 *   Bytes 3-7 [bits 24-63]: reserved (padding, must be 0)
 */
struct AtomDataPacked {
  uint64_t lo = 0;
  uint64_t hi = 0;

  /// Sentinel value for unset valence fields
  static constexpr uint8_t kUnsetValence = std::numeric_limits<uint8_t>::max();

  // ============================================================================
  // Byte offset constants (within lo or hi)
  // ============================================================================

  /// @name Lower 64-bit field byte offsets
  /// @{
  static constexpr int kAtomicNumByte           = 0;
  static constexpr int kNumExplicitHsByte       = 1;
  static constexpr int kExplicitValenceByte     = 2;
  static constexpr int kImplicitValenceByte     = 3;
  static constexpr int kFormalChargeByte        = 4;
  static constexpr int kChiralTagByte           = 5;
  static constexpr int kNumRadicalElectronsByte = 6;
  static constexpr int kHybridizationByte       = 7;
  /// @}

  /// @name Upper 64-bit field byte offsets
  /// @{
  static constexpr int kMinRingSizeByte = 0;
  static constexpr int kNumRingsByte    = 1;
  static constexpr int kIsAromaticByte  = 2;
  /// @}

  // ============================================================================
  // Setters - host-side, used during molecule loading
  // ============================================================================

  HD_CALLABLE void setAtomicNum(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kAtomicNumByte * 8))) | (static_cast<uint64_t>(val) << (kAtomicNumByte * 8));
  }

  HD_CALLABLE void setNumExplicitHs(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kNumExplicitHsByte * 8))) | (static_cast<uint64_t>(val) << (kNumExplicitHsByte * 8));
  }

  HD_CALLABLE void setExplicitValence(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kExplicitValenceByte * 8))) | (static_cast<uint64_t>(val) << (kExplicitValenceByte * 8));
  }

  HD_CALLABLE void setImplicitValence(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kImplicitValenceByte * 8))) | (static_cast<uint64_t>(val) << (kImplicitValenceByte * 8));
  }

  HD_CALLABLE void setFormalCharge(int8_t val) {
    uint8_t uval = static_cast<uint8_t>(val);
    lo = (lo & ~(0xFFULL << (kFormalChargeByte * 8))) | (static_cast<uint64_t>(uval) << (kFormalChargeByte * 8));
  }

  HD_CALLABLE void setChiralTag(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kChiralTagByte * 8))) | (static_cast<uint64_t>(val) << (kChiralTagByte * 8));
  }

  HD_CALLABLE void setNumRadicalElectrons(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kNumRadicalElectronsByte * 8))) |
         (static_cast<uint64_t>(val) << (kNumRadicalElectronsByte * 8));
  }

  HD_CALLABLE void setHybridization(uint8_t val) {
    lo = (lo & ~(0xFFULL << (kHybridizationByte * 8))) | (static_cast<uint64_t>(val) << (kHybridizationByte * 8));
  }

  HD_CALLABLE void setMinRingSize(uint8_t val) {
    hi = (hi & ~(0xFFULL << (kMinRingSizeByte * 8))) | (static_cast<uint64_t>(val) << (kMinRingSizeByte * 8));
  }

  HD_CALLABLE void setNumRings(uint8_t val) {
    hi = (hi & ~(0xFFULL << (kNumRingsByte * 8))) | (static_cast<uint64_t>(val) << (kNumRingsByte * 8));
  }

  HD_CALLABLE void setIsAromatic(bool val) {
    uint8_t uval = val ? 0x01 : 0x00;
    hi           = (hi & ~(0xFFULL << (kIsAromaticByte * 8))) | (static_cast<uint64_t>(uval) << (kIsAromaticByte * 8));
  }

  // ============================================================================
  // Getters - host and device
  // ============================================================================

  HD_CALLABLE uint8_t atomicNum() const { return static_cast<uint8_t>((lo >> (kAtomicNumByte * 8)) & 0xFF); }

  HD_CALLABLE uint8_t numExplicitHs() const { return static_cast<uint8_t>((lo >> (kNumExplicitHsByte * 8)) & 0xFF); }

  HD_CALLABLE uint8_t explicitValence() const {
    return static_cast<uint8_t>((lo >> (kExplicitValenceByte * 8)) & 0xFF);
  }

  HD_CALLABLE uint8_t implicitValence() const {
    return static_cast<uint8_t>((lo >> (kImplicitValenceByte * 8)) & 0xFF);
  }

  HD_CALLABLE int8_t formalCharge() const {
    return static_cast<int8_t>(static_cast<uint8_t>((lo >> (kFormalChargeByte * 8)) & 0xFF));
  }

  HD_CALLABLE uint8_t chiralTag() const { return static_cast<uint8_t>((lo >> (kChiralTagByte * 8)) & 0xFF); }

  HD_CALLABLE uint8_t numRadicalElectrons() const {
    return static_cast<uint8_t>((lo >> (kNumRadicalElectronsByte * 8)) & 0xFF);
  }

  HD_CALLABLE uint8_t hybridization() const { return static_cast<uint8_t>((lo >> (kHybridizationByte * 8)) & 0xFF); }

  HD_CALLABLE uint8_t minRingSize() const { return static_cast<uint8_t>((hi >> (kMinRingSizeByte * 8)) & 0xFF); }

  HD_CALLABLE uint8_t numRings() const { return static_cast<uint8_t>((hi >> (kNumRingsByte * 8)) & 0xFF); }

  HD_CALLABLE bool isAromatic() const { return ((hi >> (kIsAromaticByte * 8)) & 0xFF) != 0; }
};

static_assert(sizeof(AtomDataPacked) == 16, "AtomDataPacked must be exactly 16 bytes");

/**
 * @brief Precomputed mask and expected values for branchless atom matching.
 *
 * Generated on host from AtomQueryFlags during query molecule loading.
 * Enables single-instruction comparison: match = ((target & mask) == expected)
 *
 * For each atom property field:
 * - If the query flag is SET: corresponding mask byte = 0xFF, expected byte = query value
 * - If the query flag is NOT SET: corresponding mask byte = 0x00, expected byte = 0x00
 *
 * Special handling for aromaticity:
 * - AtomQueryIsAromatic: isAromatic mask = 0xFF, expected = 0x01 (must be true)
 * - AtomQueryIsAliphatic: isAromatic mask = 0xFF, expected = 0x00 (must be false)
 *
 * The match operation becomes:
 *   bool matches = ((target.lo & maskLo) == expectedLo) &&
 *                  ((target.hi & maskHi) == expectedHi);
 */
struct AtomQueryMask {
  uint64_t maskLo     = 0;
  uint64_t maskHi     = 0;
  uint64_t expectedLo = 0;
  uint64_t expectedHi = 0;
};

static_assert(sizeof(AtomQueryMask) == 32, "AtomQueryMask must be exactly 32 bytes");

/**
 * @brief Precomputed bond type counts per atom for efficient matching.
 *
 * Stores the count of each bond type (0-7) incident on this atom.
 * Bond types: 0=unspecified, 1=single, 2=double, 3=triple, 4=quadruple,
 *             5=quintuple, 6=hextuple, 7+=other/aromatic
 *
 * During substructure matching, a target atom can match a query atom only if
 * target.bondTypeCounts[i] >= query.bondTypeCounts[i] for all i.
 */
struct BondTypeCounts {
  static constexpr int kNumBondTypes         = 8;
  uint8_t              counts[kNumBondTypes] = {0};

  HD_CALLABLE uint8_t  operator[](int idx) const { return counts[idx]; }
  HD_CALLABLE uint8_t& operator[](int idx) { return counts[idx]; }

  /**
   * @brief Check if this atom has at least as many bonds of each type as other.
   * @return true if counts[i] >= other.counts[i] for all i
   */
  HD_CALLABLE bool hasSufficientBonds(const BondTypeCounts& other) const {
    // Pack comparison into 64-bit for efficiency
    uint64_t thisVal  = 0;
    uint64_t otherVal = 0;
    for (int i = 0; i < kNumBondTypes; ++i) {
      thisVal |= static_cast<uint64_t>(counts[i]) << (i * 8);
      otherVal |= static_cast<uint64_t>(other.counts[i]) << (i * 8);
    }
    // Branchless: check each byte for >= using saturation arithmetic
    // If any target byte < query byte, the subtraction underflows (high bit set)
    // This is a simplified uniform check
    bool sufficient = true;
    for (int i = 0; i < kNumBondTypes; ++i) {
      sufficient &= (counts[i] >= other.counts[i]);
    }
    return sufficient;
  }
};

static_assert(sizeof(BondTypeCounts) == 8, "BondTypeCounts must be exactly 8 bytes");

/**
 * @brief Branchless atom matching using precomputed mask and expected values.
 *
 * All threads in a warp execute identical instructions regardless of which
 * fields are being compared, eliminating warp divergence.
 *
 * @param target The target atom to test
 * @param query The precomputed query mask with expected values
 * @return true if target matches query for all specified fields
 */
HD_CALLABLE inline bool atomMatchesPacked(const AtomDataPacked& target, const AtomQueryMask& query) {
  return ((target.lo & query.maskLo) == query.expectedLo) && ((target.hi & query.maskHi) == query.expectedHi);
}

/**
 * @brief Check if target atom has sufficient bonds of each type to match query.
 *
 * @param target Bond counts for the target atom
 * @param query Bond counts for the query atom
 * @return true if target has >= bonds of each type compared to query
 */
HD_CALLABLE inline bool bondCountsMatchPacked(const BondTypeCounts& target, const BondTypeCounts& query) {
  return target.hasSufficientBonds(query);
}

}  // namespace nvMolKit

#undef HD_CALLABLE

#endif  // NVMOLKIT_ATOM_DATA_PACKED_H
