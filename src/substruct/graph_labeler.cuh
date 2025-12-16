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

#ifndef NVMOLKIT_GRAPH_LABELER_CUH
#define NVMOLKIT_GRAPH_LABELER_CUH

#include "flat_bit_vect.h"
#include "molecules.h"
#include "molecules_device.cuh"

namespace nvMolKit {

/**
 * @brief Check if a target atom matches a query atom based on query flags.
 *
 * Compares the target atom's properties against the query atom's properties,
 * but only for the fields specified in the query flags bitmask.
 *
 * @param target The target atom data
 * @param query The query atom data (contains values to match against)
 * @param queryFlags Bitmask of AtomQueryFlags indicating which fields to compare
 * @return true if target matches query for all specified fields
 */
__device__ __forceinline__ bool atomMatches(const AtomData& target,
                                            const AtomData& query,
                                            AtomQuery queryFlags) {
  if (queryFlags & AtomQueryAtomicNum) {
    if (target.atomicNum != query.atomicNum) {
      return false;
    }
  }
  if (queryFlags & AtomQueryNumExplicitHs) {
    if (target.numExplicitHs != query.numExplicitHs) {
      return false;
    }
  }
  if (queryFlags & AtomQueryExplicitValence) {
    if (target.explicitValence != query.explicitValence) {
      return false;
    }
  }
  if (queryFlags & AtomQueryImplicitValence) {
    if (target.implicitValence != query.implicitValence) {
      return false;
    }
  }
  if (queryFlags & AtomQueryFormalCharge) {
    if (target.formalCharge != query.formalCharge) {
      return false;
    }
  }
  if (queryFlags & AtomQueryChiralTag) {
    if (target.chiralTag != query.chiralTag) {
      return false;
    }
  }
  if (queryFlags & AtomQueryNumRadicalElectrons) {
    if (target.numRadicalElectrons != query.numRadicalElectrons) {
      return false;
    }
  }
  if (queryFlags & AtomQueryHybridization) {
    if (target.hybridization != query.hybridization) {
      return false;
    }
  }
  if (queryFlags & AtomQueryMinRingSize) {
    if (target.minRingSize != query.minRingSize) {
      return false;
    }
  }
  if (queryFlags & AtomQueryNumRings) {
    if (target.numRings != query.numRings) {
      return false;
    }
  }
  if (queryFlags & AtomQueryIsAromatic) {
    if (!target.isAromatic) {
      return false;
    }
  }
  if (queryFlags & AtomQueryIsAliphatic) {
    if (target.isAromatic) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Check if target atom has at least as many bonds of each type as query atom.
 *
 * For each bond type present in the query atom's neighborhood, verifies that the
 * target atom has at least as many bonds of that type.
 *
 * @param target The target molecule view
 * @param targetAtomIdx Index of the atom in the target molecule
 * @param query The query molecule view
 * @param queryAtomIdx Index of the atom in the query molecule
 * @return true if target has sufficient bonds of each type
 */
__device__ __forceinline__ bool bondCountsMatch(const MoleculeView& target,
                                                int targetAtomIdx,
                                                const MoleculeView& query,
                                                int queryAtomIdx) {
  const int targetDegree = target.getAtomDegree(targetAtomIdx);
  const int queryDegree = query.getAtomDegree(queryAtomIdx);

  // Target must have at least as many bonds as query
  if (targetDegree < queryDegree) {
    return false;
  }

  // Count bonds by type for query atom
  // We use a simple array for bond types (RDKit bond types are small integers)
  constexpr int kMaxBondTypes = 32;
  int queryBondCounts[kMaxBondTypes] = {0};
  int targetBondCounts[kMaxBondTypes] = {0};

  for (int i = 0; i < queryDegree; ++i) {
    const int bondIdx = query.getNeighborBondIdx(queryAtomIdx, i);
    const int bondType = query.getBond(bondIdx).bondType;
    if (bondType < kMaxBondTypes) {
      ++queryBondCounts[bondType];
    }
  }

  for (int i = 0; i < targetDegree; ++i) {
    const int bondIdx = target.getNeighborBondIdx(targetAtomIdx, i);
    const int bondType = target.getBond(bondIdx).bondType;
    if (bondType < kMaxBondTypes) {
      ++targetBondCounts[bondType];
    }
  }

  // Check that target has at least as many of each bond type
  for (int bt = 0; bt < kMaxBondTypes; ++bt) {
    if (targetBondCounts[bt] < queryBondCounts[bt]) {
      return false;
    }
  }

  return true;
}

/**
 * @brief Populate the label matrix for substructure matching.
 *
 * Given a target molecule and a query molecule, populates a 2D bit matrix where
 * bit[i][j] is set if target atom i could potentially match query atom j.
 * A match requires:
 * 1. The atom properties match according to the query flags
 * 2. The target atom has at least as many bonds of each type as the query atom
 *
 * @tparam MaxTargetAtoms Maximum number of atoms in target graph
 * @tparam MaxQueryAtoms Maximum number of atoms in query graph
 * @param target The target molecule view
 * @param query The query molecule view
 * @param labelMatrix Output 2D bit matrix view [target_atoms x query_atoms]
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ void populateLabelMatrix(const MoleculeView& target,
                                    const MoleculeView& query,
                                    BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix) {
  // Clear the matrix first
  labelMatrix.clear();

  // For each target atom
  for (int targetIdx = 0; targetIdx < target.numAtoms; ++targetIdx) {
    const AtomData& targetAtom = target.getAtom(targetIdx);

    // For each query atom
    for (int queryIdx = 0; queryIdx < query.numAtoms; ++queryIdx) {
      const AtomData& queryAtom = query.getAtom(queryIdx);
      const AtomQuery queryFlags = query.getAtomQuery(queryIdx);

      // Check if atoms match
      if (!atomMatches(targetAtom, queryAtom, queryFlags)) {
        continue;
      }

      // Check if bond counts match
      if (!bondCountsMatch(target, targetIdx, query, queryIdx)) {
        continue;
      }

      // Set the bit - this target atom can match this query atom
      labelMatrix.set(targetIdx, queryIdx, true);
    }
  }
}

/**
 * @brief Populate the label matrix for a single target atom (parallelizable version).
 *
 * This version processes a single target atom, making it suitable for parallel
 * execution where each thread handles one target atom.
 *
 * @tparam MaxTargetAtoms Maximum number of atoms in target graph
 * @tparam MaxQueryAtoms Maximum number of atoms in query graph
 * @param target The target molecule view
 * @param targetAtomIdx Index of the target atom to process
 * @param query The query molecule view
 * @param labelMatrix Output 2D bit matrix view [target_atoms x query_atoms]
 */
template <std::size_t MaxTargetAtoms, std::size_t MaxQueryAtoms>
__device__ void populateLabelMatrixForAtom(const MoleculeView& target,
                                           int targetAtomIdx,
                                           const MoleculeView& query,
                                           BitMatrix2DView<MaxTargetAtoms, MaxQueryAtoms>& labelMatrix) {
  if (targetAtomIdx >= target.numAtoms) {
    return;
  }

  const AtomData& targetAtom = target.getAtom(targetAtomIdx);

  // For each query atom
  for (int queryIdx = 0; queryIdx < query.numAtoms; ++queryIdx) {
    const AtomData& queryAtom = query.getAtom(queryIdx);
    const AtomQuery queryFlags = query.getAtomQuery(queryIdx);

    // Only set bits for matches; assumes matrix is pre-cleared
    if (atomMatches(targetAtom, queryAtom, queryFlags) &&
        bondCountsMatch(target, targetAtomIdx, query, queryIdx)) {
      labelMatrix.set(targetAtomIdx, queryIdx, true);
    }
  }
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_GRAPH_LABELER_CUH





