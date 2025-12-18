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

#ifndef NVMOLKIT_BOOLEAN_TREE_CUH
#define NVMOLKIT_BOOLEAN_TREE_CUH

#include <cstdint>

#include "atom_data_packed.h"
#include "substruct_types.h"

#ifdef __CUDACC__
#define HD_CALLABLE __host__ __device__
#else
#define HD_CALLABLE
#endif

namespace nvMolKit {

/**
 * @brief Boolean operation type for query expression evaluation.
 */
enum class BoolOp : uint8_t {
  Leaf,  ///< Evaluate AtomQueryMask against target atom
  And,   ///< Binary AND of two operands
  Or,    ///< Binary OR of two operands
  Not    ///< Unary NOT of single operand
};

/**
 * @brief A single instruction in the boolean expression evaluation sequence.
 *
 * Instructions are executed in post-order to evaluate compound SMARTS queries.
 * For simple AND-only queries, a single Leaf instruction suffices.
 *
 * Layout:
 * - Leaf: scratch[dst] = atomMatchesPacked(target, leafMasks[leafMaskIdx])
 * - And:  scratch[dst] = scratch[src1] & scratch[src2]
 * - Or:   scratch[dst] = scratch[src1] | scratch[src2]
 * - Not:  scratch[dst] = !scratch[src1]
 */
struct BoolInstruction {
  BoolOp  op;           ///< Operation type
  uint8_t dst;          ///< Destination index in scratch array
  uint8_t src1;         ///< Left operand index (or source for NOT)
  uint8_t src2;         ///< Right operand index (unused for Leaf/Not)
  uint8_t leafMaskIdx;  ///< Index into leaf masks array (for Leaf op only)

  HD_CALLABLE static BoolInstruction makeLeaf(uint8_t dst, uint8_t maskIdx) {
    return BoolInstruction{BoolOp::Leaf, dst, 0, 0, maskIdx};
  }

  HD_CALLABLE static BoolInstruction makeAnd(uint8_t dst, uint8_t src1, uint8_t src2) {
    return BoolInstruction{BoolOp::And, dst, src1, src2, 0};
  }

  HD_CALLABLE static BoolInstruction makeOr(uint8_t dst, uint8_t src1, uint8_t src2) {
    return BoolInstruction{BoolOp::Or, dst, src1, src2, 0};
  }

  HD_CALLABLE static BoolInstruction makeNot(uint8_t dst, uint8_t src) {
    return BoolInstruction{BoolOp::Not, dst, src, 0, 0};
  }
};

static_assert(sizeof(BoolInstruction) == 5, "BoolInstruction must be exactly 5 bytes");

/**
 * @brief Metadata for a single query atom's boolean expression tree.
 *
 * Each query atom has its own tree describing how to combine leaf mask checks.
 * For simple AND-only queries, numInstructions=1 and scratchSize=1.
 */
struct AtomQueryTree {
  uint8_t numLeaves;        ///< Number of AtomQueryMask entries for this atom
  uint8_t numInstructions;  ///< Length of instruction sequence
  uint8_t scratchSize;      ///< Number of scratch slots needed for evaluation
  uint8_t resultIdx;        ///< Index in scratch where final result is stored
};

static_assert(sizeof(AtomQueryTree) == 4, "AtomQueryTree must be exactly 4 bytes");

/**
 * @brief Evaluate a boolean expression tree for atom matching.
 *
 * Executes the instruction sequence to determine if a target atom matches
 * a compound query expression (AND/OR/NOT combinations).
 *
 * @param targetPacked The target atom's packed data
 * @param targetBonds The target atom's bond type counts (may be nullptr if checkBonds=false)
 * @param leafMasks Pointer to first leaf mask for this query atom
 * @param leafBondCounts Pointer to first leaf bond count for this query atom (may be nullptr if checkBonds=false)
 * @param instructions Pointer to first instruction for this query atom
 * @param tree Tree metadata (num instructions, scratch size, result index)
 * @param checkBonds If true, also check bond count requirements (for substructure search).
 *                   If false, only check atom properties (for label matrix compatibility).
 * @return true if target atom matches the compound query
 */
template <bool checkBonds = true>
HD_CALLABLE inline bool evaluateBoolTree(const AtomDataPacked*   targetPacked,
                                         const BondTypeCounts*   targetBonds,
                                         const AtomQueryMask*    leafMasks,
                                         const BondTypeCounts*   leafBondCounts,
                                         const BoolInstruction*  instructions,
                                         const AtomQueryTree&    tree) {
  // Empty tree (e.g., wildcard atom *) - atom properties always match,
  // but still need to check bond counts if requested
  if (tree.numInstructions == 0) {
    if constexpr (checkBonds) {
      // For empty trees, leaf 0 still holds the bond count requirements
      return tree.numLeaves > 0 ? bondCountsMatchPacked(*targetBonds, leafBondCounts[0]) : true;
    }
    return true;
  }

  uint8_t scratch[kMaxBoolScratchSize];

  for (int i = 0; i < tree.numInstructions; ++i) {
    const BoolInstruction& instr = instructions[i];

    switch (instr.op) {
      case BoolOp::Leaf: {
        const bool atomMatch = atomMatchesPacked(*targetPacked, leafMasks[instr.leafMaskIdx]);
        bool match = atomMatch;
        if constexpr (checkBonds) {
          const bool bondMatch = bondCountsMatchPacked(*targetBonds, leafBondCounts[instr.leafMaskIdx]);
          match = atomMatch && bondMatch;
        }
        scratch[instr.dst] = match ? 1 : 0;
        break;
      }
      case BoolOp::And:
        scratch[instr.dst] = scratch[instr.src1] & scratch[instr.src2];
        break;
      case BoolOp::Or:
        scratch[instr.dst] = scratch[instr.src1] | scratch[instr.src2];
        break;
      case BoolOp::Not:
        scratch[instr.dst] = scratch[instr.src1] ? 0 : 1;
        break;
    }
  }

  return scratch[tree.resultIdx] != 0;
}

}  // namespace nvMolKit

#undef HD_CALLABLE

#endif  // NVMOLKIT_BOOLEAN_TREE_CUH

