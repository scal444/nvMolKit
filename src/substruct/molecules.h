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
#include <memory>
#include <string>
#include <vector>

#include "atom_data_packed.h"
#include "boolean_tree.cuh"
#include "device_vector.h"

namespace RDKit {
class ROMol;
}  // namespace RDKit

namespace nvMolKit {

/**
 * @brief Bitmask specifying which atom fields to compare for query matching.
 *
 * Multiple fields can be combined with bitwise OR for composite queries.
 * For example, 'C' in SMARTS checks both AtomicNum and IsAliphatic.
 */
enum AtomQueryFlags : uint32_t {
  AtomQueryNone                = 0,
  AtomQueryAtomicNum           = 1 << 0,
  AtomQueryNumExplicitHs       = 1 << 1,
  AtomQueryExplicitValence     = 1 << 2,
  AtomQueryImplicitValence     = 1 << 3,
  AtomQueryFormalCharge        = 1 << 4,
  AtomQueryChiralTag           = 1 << 5,
  AtomQueryNumRadicalElectrons = 1 << 6,
  AtomQueryHybridization       = 1 << 7,
  AtomQueryMinRingSize         = 1 << 8,
  AtomQueryNumRings            = 1 << 9,
  AtomQueryIsAromatic          = 1 << 10,
  AtomQueryIsAliphatic         = 1 << 11,
  AtomQueryTotalValence        = 1 << 12,
  AtomQueryIsInRing            = 1 << 13,  ///< For [R] and [r] any-ring queries
  AtomQueryIsotope             = 1 << 14,  ///< For isotope/mass queries like [13C]
  AtomQueryDegree              = 1 << 15,  ///< For [D] degree queries (explicit bond count)
  AtomQueryTotalConnectivity   = 1 << 16,  ///< For [X] total connectivity queries (degree + Hs)
  AtomQueryNeverMatches        = 1 << 17,  ///< Impossible constraint (e.g., [C;a] aromatic aliphatic)
};

using AtomQuery = uint32_t;

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
  uint8_t totalValence        = 0;
  bool    isAromatic          = false;
};

struct BondData {
  uint8_t bondType  = 0;
  uint8_t isInRing  = 0;  ///< 1 if bond is in a ring, 0 otherwise
};

/**
 * @brief Query flags for bond matching in SMARTS.
 *
 * Bonds can have queries like `-&!@` (single AND not ring bond).
 */
enum BondQueryFlags : uint8_t {
  BondQueryNone            = 0,
  BondQueryIsRingBond      = 1 << 0,  ///< Bond must be in a ring (@)
  BondQueryNotRingBond     = 1 << 1,  ///< Bond must NOT be in a ring (!@)
  BondQuerySingleOrAromatic = 1 << 2,  ///< SingleOrAromaticBond query (matches single or aromatic only)
  BondQueryDoubleOrAromatic = 1 << 3,  ///< DoubleOrAromaticBond query (matches double or aromatic only)
  BondQueryAromaticOnly    = 1 << 4,  ///< Aromatic bond query (:) - matches aromatic bonds only (type 7 or 12)
  BondQueryNeverMatches    = 1 << 5,  ///< Impossible constraint (e.g., single AND aromatic)
  BondQueryUseBondMask     = 1 << 6,  ///< Use allowedBondTypes bitmask for arbitrary OR patterns
};

/**
 * @brief Bond query data for SMARTS bond queries.
 *
 * Stores the bond type to match and ring bond constraints.
 * For complex OR patterns (e.g., =,#,:), allowedBondTypes is a bitmask where
 * bit N is set if bond type N is allowed (types 0-15 supported).
 */
struct BondQueryData {
  uint8_t  bondType         = 0;  ///< 0 = any, 1 = single, 2 = double, 3 = triple, 12 = aromatic
  uint8_t  queryFlags       = 0;  ///< BondQueryFlags bitmask
  uint16_t allowedBondTypes = 0;  ///< Bitmask of allowed bond types when BondQueryUseBondMask is set
};

/**
 * @brief Information about a single recursive SMARTS pattern within a query.
 *
 * Each RecursivePatternEntry represents one $(...) pattern found in the SMARTS query.
 * The queryMol pointer is non-owning - the original query owns the pattern.
 */
struct RecursivePatternEntry {
  const RDKit::ROMol* queryMol = nullptr;  ///< The inner query molecule from $(...) 
  int queryAtomIdx = 0;                     ///< Index of the query atom containing this pattern
  int patternId = 0;                        ///< Unique ID (0-15) for this pattern in the batch
};

/**
 * @brief Collection of recursive SMARTS patterns extracted from a query.
 *
 * Used to preprocess recursive patterns before main substructure matching.
 */
struct RecursivePatternInfo {
  std::vector<RecursivePatternEntry> patterns;  ///< All recursive patterns found
  bool hasRecursivePatterns = false;            ///< Quick check for any patterns

  /**
   * @brief Check if the query has any recursive patterns.
   */
  [[nodiscard]] bool empty() const { return patterns.empty(); }

  /**
   * @brief Get the number of recursive patterns.
   */
  [[nodiscard]] size_t size() const { return patterns.size(); }
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
  std::vector<AtomData>  atomData;          ///< Atom properties for all atoms
  std::vector<BondData>  bondData;          ///< Bond properties for all bonds
  std::vector<AtomQuery> atomQueries;       ///< Query type per atom (parallel to atomData)
  std::vector<int16_t>   atomBondStarts;    ///< Cumulative count of bonds per atom (prefix sum)
  std::vector<int16_t>   otherAtomIndices;  ///< For each atom-bond pair, the other atom index
  std::vector<int16_t>   bondDataIndices;   ///< For each atom-bond pair, index into bondData

  // GPU-optimized packed data (parallel to atomData)
  std::vector<AtomDataPacked> atomDataPacked;  ///< Packed atom properties for GPU matching
  std::vector<AtomQueryMask>  atomQueryMasks;  ///< Precomputed query masks (for query molecules only)
  std::vector<BondTypeCounts> bondTypeCounts;  ///< Precomputed bond type counts per atom

  // Boolean expression tree data for compound queries (OR/NOT support)
  std::vector<AtomQueryTree>   atomQueryTrees;       ///< Tree metadata per query atom (parallel to atomData)
  std::vector<BoolInstruction> queryInstructions;    ///< Flattened instruction arrays
  std::vector<AtomQueryMask>   queryLeafMasks;       ///< Flattened leaf masks for compound queries
  std::vector<BondTypeCounts>  queryLeafBondCounts;  ///< Flattened leaf bond counts
  std::vector<int>             atomInstrStarts;      ///< Start index into queryInstructions per atom
  std::vector<int>             atomLeafMaskStarts;   ///< Start index into queryLeafMasks per atom

  // Bond query data for SMARTS (parallel to bondData, only for query molecules)
  std::vector<BondQueryData> bondQueryData;  ///< Bond query info (type + ring constraints)

  // Recursive SMARTS patterns extracted from query molecules (one per molecule in batch)
  std::vector<RecursivePatternInfo> recursivePatterns;

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
  const int*       batchAtomStarts;
  const int*       batchBondStarts;
  const int*       batchAtomBondStarts;
  const int*       batchOtherAtomIndicesStarts;
  const int*       batchBondIndicesStarts;
  const AtomData*  atomData;
  const BondData*  bondData;
  const AtomQuery* atomQueries;
  const int16_t*   atomBondStarts;
  const int16_t*   otherAtomIndices;
  const int16_t*   bondDataIndices;
  int              numMolecules;

  // GPU-optimized packed data
  const AtomDataPacked* atomDataPacked;  ///< Packed atom properties for GPU matching
  const AtomQueryMask*  atomQueryMasks;  ///< Precomputed query masks (query molecules only)
  const BondTypeCounts* bondTypeCounts;  ///< Precomputed bond type counts per atom

  // Boolean expression tree data for compound queries
  const AtomQueryTree*   atomQueryTrees;       ///< Tree metadata per query atom
  const BoolInstruction* queryInstructions;    ///< Flattened instruction arrays
  const AtomQueryMask*   queryLeafMasks;       ///< Flattened leaf masks for compound queries
  const BondTypeCounts*  queryLeafBondCounts;  ///< Flattened leaf bond counts
  const int*             atomInstrStarts;      ///< Start index into queryInstructions per atom
  const int*             atomLeafMaskStarts;   ///< Start index into queryLeafMasks per atom

  // Bond query data for SMARTS
  const BondQueryData* bondQueryData;  ///< Bond query info (query molecules only)
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

  AsyncDeviceVector<int>       batchAtomStarts_;
  AsyncDeviceVector<int>       batchBondStarts_;
  AsyncDeviceVector<int>       batchAtomBondStarts_;
  AsyncDeviceVector<int>       batchOtherAtomIndicesStarts_;
  AsyncDeviceVector<int>       batchBondIndicesStarts_;
  AsyncDeviceVector<AtomData>  atomData_;
  AsyncDeviceVector<BondData>  bondData_;
  AsyncDeviceVector<AtomQuery> atomQueries_;
  AsyncDeviceVector<int16_t>   atomBondStarts_;
  AsyncDeviceVector<int16_t>   otherAtomIndices_;
  AsyncDeviceVector<int16_t>   bondDataIndices_;

  // GPU-optimized packed data
  AsyncDeviceVector<AtomDataPacked> atomDataPacked_;
  AsyncDeviceVector<AtomQueryMask>  atomQueryMasks_;
  AsyncDeviceVector<BondTypeCounts> bondTypeCounts_;

  // Boolean expression tree data for compound queries
  AsyncDeviceVector<AtomQueryTree>   atomQueryTrees_;
  AsyncDeviceVector<BoolInstruction> queryInstructions_;
  AsyncDeviceVector<AtomQueryMask>   queryLeafMasks_;
  AsyncDeviceVector<BondTypeCounts>  queryLeafBondCounts_;
  AsyncDeviceVector<int>             atomInstrStarts_;
  AsyncDeviceVector<int>             atomLeafMaskStarts_;

  // Bond query data for SMARTS
  AsyncDeviceVector<BondQueryData> bondQueryData_;
};

/**
 * @brief Add a molecule to an existing batch.
 * @param mol Pointer to the RDKit molecule to add
 * @param batch The batch to add the molecule to
 */
void addToBatch(const RDKit::ROMol* mol, MoleculesHost& batch);

/**
 * @brief Add a query molecule (from SMARTS) to an existing batch.
 *
 * Extracts query information from QueryAtom objects and populates atomQueries.
 * Supports AND, OR, and NOT combinations of query types via boolean expression trees.
 * XOR queries and recursive SMARTS ($(...)) throw an exception.
 *
 * @param mol Pointer to the RDKit molecule (typically parsed from SMARTS)
 * @param batch The batch to add the query molecule to
 */
void addQueryToBatch(const RDKit::ROMol* mol, MoleculesHost& batch);

/**
 * @brief Convert RDKit query description string to AtomQuery flags.
 * @param description The query description from RDKit (e.g., "AtomAtomicNum")
 * @return The corresponding AtomQuery flag value, or AtomQueryNone if unsupported
 */
AtomQuery atomQueryFromDescription(const std::string& description);

/**
 * @brief Build a query mask from packed atom data and query flags.
 *
 * Creates a precomputed mask and expected value pair for branchless GPU matching.
 * For each field specified in queryFlags, sets the corresponding mask byte to 0xFF
 * and the expected byte to the query atom's value.
 *
 * @param queryAtom The packed query atom data
 * @param queryFlags Bitmask of AtomQueryFlags indicating which fields to compare
 * @return AtomQueryMask with precomputed mask and expected values
 */
AtomQueryMask buildQueryMask(const AtomDataPacked& queryAtom, AtomQuery queryFlags);
/**
 * @brief Extract recursive SMARTS patterns from a query molecule.
 *
 * Walks the query tree looking for RecursiveStructure nodes and extracts the
 * inner query molecules. Validates constraints:
 * - Maximum 8 non-nested recursive patterns per query (expandable to 16)
 * - No nested recursion (throws if $($(...)) patterns are found)
 *
 * @param mol The query molecule (typically parsed from SMARTS)
 * @return RecursivePatternInfo containing all found patterns
 * @throws std::runtime_error if constraints are violated
 */
RecursivePatternInfo extractRecursivePatterns(const RDKit::ROMol* mol);

/**
 * @brief Check if a SMARTS query contains recursive patterns.
 *
 * Quick check without full extraction. Useful for batch sorting.
 *
 * @param mol The query molecule to check
 * @return true if the query contains any recursive SMARTS ($(...))
 */
bool hasRecursiveSmarts(const RDKit::ROMol* mol);

}  // namespace nvMolKit

#endif  // NVMOLKIT_MOLECULES_H
