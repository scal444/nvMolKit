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

#include "molecules.h"

#include <GraphMol/MolOps.h>
#include <GraphMol/QueryAtom.h>
#include <GraphMol/QueryOps.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmartsWrite.h>
#include <RDGeneral/versions.h>

#include <functional>
#include <stdexcept>
#include <string>

#include "substruct_types.h"

namespace nvMolKit {

namespace {

void populateAtomData(const RDKit::Atom* atom, AtomData& atomData, const RDKit::RingInfo* ringInfo) {
  atomData.atomicNum     = atom->getAtomicNum();
  atomData.chiralTag     = atom->getChiralTag();
  // Use total H count (explicit + implicit) for SMARTS [H] queries like [NH], [CH3], etc.
  atomData.numExplicitHs = atom->getTotalNumHs();
#if RDKIT_VERSION_NUM >= 0x20240300
  atomData.explicitValence = atom->getValence(RDKit::Atom::ValenceType::EXPLICIT);
  atomData.implicitValence = atom->getValence(RDKit::Atom::ValenceType::IMPLICIT);
#else
  atomData.explicitValence = atom->getExplicitValence();
  atomData.implicitValence = atom->getImplicitValence();
#endif
  atomData.totalValence        = atom->getTotalValence();
  atomData.formalCharge        = atom->getFormalCharge();
  atomData.hybridization       = atom->getHybridization();
  atomData.isAromatic          = atom->getIsAromatic();
  atomData.numRadicalElectrons = atom->getNumRadicalElectrons();
  const int idx                = atom->getIdx();
  atomData.numRings            = ringInfo->numAtomRings(idx);
  atomData.minRingSize         = ringInfo->minAtomRingSize(idx);
}

void populateAtomDataPacked(const RDKit::Atom* atom, AtomDataPacked& packed, const RDKit::RingInfo* ringInfo) {
  packed.setAtomicNum(atom->getAtomicNum());
  packed.setChiralTag(atom->getChiralTag());
  // Use total H count (explicit + implicit) for SMARTS [H] queries like [NH], [CH3], etc.
  packed.setNumExplicitHs(atom->getTotalNumHs());
#if RDKIT_VERSION_NUM >= 0x20240300
  packed.setExplicitValence(atom->getValence(RDKit::Atom::ValenceType::EXPLICIT));
  packed.setImplicitValence(atom->getValence(RDKit::Atom::ValenceType::IMPLICIT));
#else
  packed.setExplicitValence(atom->getExplicitValence());
  packed.setImplicitValence(atom->getImplicitValence());
#endif
  packed.setTotalValence(atom->getTotalValence());
  packed.setFormalCharge(atom->getFormalCharge());
  packed.setHybridization(atom->getHybridization());
  packed.setIsAromatic(atom->getIsAromatic());
  packed.setNumRadicalElectrons(atom->getNumRadicalElectrons());
  const int idx      = atom->getIdx();
  const int numRings = ringInfo->numAtomRings(idx);
  packed.setNumRings(numRings);
  packed.setMinRingSize(ringInfo->minAtomRingSize(idx));
  packed.setIsInRing(numRings > 0);

  // Isotope (0 = natural abundance)
  unsigned int isotope = atom->getIsotope();
  if (isotope > 255) {
    throw std::runtime_error("Atom isotope " + std::to_string(isotope) + " exceeds maximum supported value of 255");
  }
  packed.setIsotope(static_cast<uint8_t>(isotope));

  // Degree (number of explicit bonds) for [D] queries
  packed.setDegree(atom->getDegree());

  // Total connectivity (degree + total H count) for [X] queries
  packed.setTotalConnectivity(atom->getTotalDegree());
}

void populateBondTypeCounts(const RDKit::ROMol* mol, const RDKit::Atom* atom, BondTypeCounts& counts) {
  auto [beg, bondEnd] = mol->getAtomBonds(atom);
  while (beg != bondEnd) {
    const auto* bond     = (*mol)[*beg];
    int         bondType = bond->getBondType();
    switch (bondType) {
      case 1:
        ++counts.single;
        break;  // SINGLE
      case 2:
        ++counts.double_;
        break;  // DOUBLE
      case 3:
        ++counts.triple;
        break;  // TRIPLE
      case 7:   // ONEANDAHALF (aromatic)
      case 12:
        ++counts.aromatic;
        break;  // AROMATIC
      default:
        throw std::runtime_error("Unsupported bond type " + std::to_string(bondType) +
                                 " in target molecule. Only single, double, triple, and aromatic bonds are supported.");
    }
    ++beg;
  }
}

/**
 * @brief Populate bond type counts for a query atom (SMARTS).
 *
 * Unlike target molecules, query molecules can have "any" bonds (~) which
 * RDKit represents as bond type 0 (UNSPECIFIED). These are counted in the
 * `any` field of BondTypeCounts.
 */
/**
 * @brief Get the effective bond type for a query bond.
 *
 * For SMARTS queries, implicit bonds (no explicit bond symbol) are represented
 * with SingleOrAromaticBond query which should match any bond type. This function
 * checks for such queries and returns 0 (any) instead of the nominal bond type.
 */
int getQueryBondEffectiveType(const RDKit::Bond* bond) {
  int bondType = bond->getBondType();

  if (bond->hasQuery()) {
    const auto* query = bond->getQuery();
    if (query != nullptr) {
      const std::string desc = query->getDescription();
      if (desc == "SingleOrAromaticBond" || desc == "DoubleOrAromaticBond" || desc == "BondNull") {
        return 0;  // Any bond (flexible match)
      }
      // For BondAnd/BondOr queries, check for flexible bond patterns
      if (desc == "BondAnd" || desc == "BondOr") {
        bool hasSingle   = false;
        bool hasDouble   = false;
        bool hasAromatic = false;
        for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
          const std::string childDesc = (*it)->getDescription();
          if (childDesc == "SingleOrAromaticBond" || childDesc == "DoubleOrAromaticBond" ||
              childDesc == "BondNull") {
            return 0;
          }
          if (childDesc == "BondOrder") {
            const auto* eqQuery   = static_cast<const RDKit::BOND_EQUALS_QUERY*>((*it).get());
            int         childType = eqQuery->getVal();
            if (childType == 1)
              hasSingle = true;
            else if (childType == 2)
              hasDouble = true;
            else if (childType == 7 || childType == 12)
              hasAromatic = true;
          } else if (childDesc == "BondIsAromatic") {
            hasAromatic = true;
          }
        }
        // If this is a "single or aromatic" or "double or aromatic" BondOr pattern
        if ((hasSingle && hasAromatic) || (hasDouble && hasAromatic)) {
          return 0;  // Treat as flexible bond for counting
        }
      }
    }
  }

  return bondType;
}

void populateQueryBondTypeCounts(const RDKit::ROMol* mol, const RDKit::Atom* atom, BondTypeCounts& counts) {
  auto [beg, bondEnd] = mol->getAtomBonds(atom);
  while (beg != bondEnd) {
    const auto* bond     = (*mol)[*beg];
    int         bondType = getQueryBondEffectiveType(bond);
    switch (bondType) {
      case 0:
        ++counts.any;
        break;  // UNSPECIFIED or any bond query
      case 1:
        ++counts.single;
        break;  // SINGLE
      case 2:
        ++counts.double_;
        break;  // DOUBLE
      case 3:
        ++counts.triple;
        break;  // TRIPLE
      case 7:   // ONEANDAHALF (aromatic)
      case 12:
        ++counts.aromatic;
        break;  // AROMATIC
      default:
        throw std::runtime_error("Unsupported bond type " + std::to_string(bondType) +
                                 " in query molecule.");
    }
    ++beg;
  }
}

void populateFromQuery(const RDKit::Atom::QUERYATOM_QUERY* query, AtomData& atomData);

void handleQueryChildren(const RDKit::Atom::QUERYATOM_QUERY* query, AtomData& atomData) {
  for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
    populateFromQuery((*it).get(), atomData);
  }
}

void populateFromQuery(const RDKit::Atom::QUERYATOM_QUERY* query, AtomData& atomData) {
  const std::string desc = query->getDescription();

  // Composite queries - recurse into children
  if (desc == "AtomAnd") {
    handleQueryChildren(query, atomData);
    return;
  }

  // AtomType is used for organic subset atoms (C, N, O, etc.) in SMARTS
  // It encodes both atomic number and aromaticity:
  // - Aliphatic: value = atomic number (e.g., C=6, O=8)
  // - Aromatic: value = 1000 + atomic number (e.g., c=1006)
  if (desc == "AtomType") {
    const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(query);
    int         typeVal = eqQuery->getVal();
    if (typeVal >= 1000) {
      atomData.atomicNum  = typeVal - 1000;
      atomData.isAromatic = true;
    } else {
      atomData.atomicNum  = typeVal;
      atomData.isAromatic = false;
    }
    return;
  }

  // Boolean queries (no value to extract)
  if (desc == "AtomIsAromatic") {
    atomData.isAromatic = true;
    return;
  }
  if (desc == "AtomIsAliphatic") {
    atomData.isAromatic = false;
    return;
  }

  // For ATOM_EQUALS_QUERY types, extract the comparison value
  const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(query);

  if (desc == "AtomAtomicNum") {
    atomData.atomicNum = eqQuery->getVal();
  } else if (desc == "AtomHCount") {
    atomData.numExplicitHs = eqQuery->getVal();
  } else if (desc == "AtomFormalCharge") {
    atomData.formalCharge = eqQuery->getVal();
  } else if (desc == "AtomHybridization") {
    atomData.hybridization = eqQuery->getVal();
  } else if (desc == "AtomInNRings") {
    atomData.numRings = eqQuery->getVal();
  } else if (desc == "AtomMinRingSize") {
    atomData.minRingSize = eqQuery->getVal();
  } else if (desc == "AtomNumRadicalElectrons") {
    atomData.numRadicalElectrons = eqQuery->getVal();
  } else if (desc == "AtomTotalValence") {
    atomData.totalValence = eqQuery->getVal();
  }
}

void populateQueryAtomData(const RDKit::Atom* atom, AtomData& atomData) {
  if (!atom->hasQuery()) {
    return;
  }

  const auto* query = atom->getQuery();
  if (query == nullptr) {
    return;
  }

  populateFromQuery(query, atomData);
}

}  // namespace

MoleculesHost::MoleculesHost() {
  batchAtomStarts.push_back(0);
  batchBondStarts.push_back(0);
  batchAtomBondStarts.push_back(0);
  batchOtherAtomIndicesStarts.push_back(0);
  batchBondIndicesStarts.push_back(0);
}

void MoleculesDevice::setStream(cudaStream_t stream) {
  stream_ = stream;
  batchAtomStarts_.setStream(stream);
  batchBondStarts_.setStream(stream);
  batchAtomBondStarts_.setStream(stream);
  batchOtherAtomIndicesStarts_.setStream(stream);
  batchBondIndicesStarts_.setStream(stream);
  atomData_.setStream(stream);
  bondData_.setStream(stream);
  atomQueries_.setStream(stream);
  atomBondStarts_.setStream(stream);
  otherAtomIndices_.setStream(stream);
  bondDataIndices_.setStream(stream);
  atomDataPacked_.setStream(stream);
  atomQueryMasks_.setStream(stream);
  bondTypeCounts_.setStream(stream);
  atomQueryTrees_.setStream(stream);
  queryInstructions_.setStream(stream);
  queryLeafMasks_.setStream(stream);
  queryLeafBondCounts_.setStream(stream);
  atomInstrStarts_.setStream(stream);
  atomLeafMaskStarts_.setStream(stream);
  bondQueryData_.setStream(stream);
}

void MoleculesDevice::copyFromHost(const MoleculesHost& host, cudaStream_t stream) {
  if (host.numMolecules() == 0) {
    throw std::invalid_argument("Cannot copy empty MoleculesHost to device");
  }

  setStream(stream);
  numMolecules_ = static_cast<int>(host.numMolecules());

  batchAtomStarts_.setFromVector(host.batchAtomStarts);
  batchBondStarts_.setFromVector(host.batchBondStarts);
  batchAtomBondStarts_.setFromVector(host.batchAtomBondStarts);
  batchOtherAtomIndicesStarts_.setFromVector(host.batchOtherAtomIndicesStarts);
  batchBondIndicesStarts_.setFromVector(host.batchBondIndicesStarts);
  atomData_.setFromVector(host.atomData);
  bondData_.setFromVector(host.bondData);
  atomQueries_.setFromVector(host.atomQueries);
  atomBondStarts_.setFromVector(host.atomBondStarts);
  otherAtomIndices_.setFromVector(host.otherAtomIndices);
  bondDataIndices_.setFromVector(host.bondDataIndices);

  // Copy GPU-optimized packed data
  if (!host.atomDataPacked.empty()) {
    atomDataPacked_.setFromVector(host.atomDataPacked);
  }
  if (!host.atomQueryMasks.empty()) {
    atomQueryMasks_.setFromVector(host.atomQueryMasks);
  }
  if (!host.bondTypeCounts.empty()) {
    bondTypeCounts_.setFromVector(host.bondTypeCounts);
  }

  // Copy boolean expression tree data for compound queries
  if (!host.atomQueryTrees.empty()) {
    atomQueryTrees_.setFromVector(host.atomQueryTrees);
    queryInstructions_.setFromVector(host.queryInstructions);
    queryLeafMasks_.setFromVector(host.queryLeafMasks);
    queryLeafBondCounts_.setFromVector(host.queryLeafBondCounts);
    atomInstrStarts_.setFromVector(host.atomInstrStarts);
    atomLeafMaskStarts_.setFromVector(host.atomLeafMaskStarts);
  }

  // Copy bond query data for SMARTS
  if (!host.bondQueryData.empty()) {
    bondQueryData_.setFromVector(host.bondQueryData);
  }
}

MoleculesDeviceView MoleculesDevice::view() const {
  MoleculesDeviceView v;
  v.batchAtomStarts             = batchAtomStarts_.data();
  v.batchBondStarts             = batchBondStarts_.data();
  v.batchAtomBondStarts         = batchAtomBondStarts_.data();
  v.batchOtherAtomIndicesStarts = batchOtherAtomIndicesStarts_.data();
  v.batchBondIndicesStarts      = batchBondIndicesStarts_.data();
  v.atomData                    = atomData_.data();
  v.bondData                    = bondData_.data();
  v.atomQueries                 = atomQueries_.data();
  v.atomBondStarts              = atomBondStarts_.data();
  v.otherAtomIndices            = otherAtomIndices_.data();
  v.bondDataIndices             = bondDataIndices_.data();
  v.numMolecules                = numMolecules_;
  v.atomDataPacked              = atomDataPacked_.data();
  v.atomQueryMasks              = atomQueryMasks_.data();
  v.bondTypeCounts              = bondTypeCounts_.data();
  v.atomQueryTrees              = atomQueryTrees_.data();
  v.queryInstructions           = queryInstructions_.data();
  v.queryLeafMasks              = queryLeafMasks_.data();
  v.queryLeafBondCounts         = queryLeafBondCounts_.data();
  v.atomInstrStarts             = atomInstrStarts_.data();
  v.atomLeafMaskStarts          = atomLeafMaskStarts_.data();
  v.bondQueryData               = bondQueryData_.data();
  return v;
}

AtomQuery atomQueryFromDescription(const std::string& description) {
  if (description == "AtomAtomicNum") {
    return AtomQueryAtomicNum;
  }
  if (description == "AtomHCount") {
    return AtomQueryNumExplicitHs;
  }
  if (description == "AtomExplicitValence") {
    return AtomQueryExplicitValence;
  }
  if (description == "AtomImplicitValence") {
    return AtomQueryImplicitValence;
  }
  if (description == "AtomFormalCharge") {
    return AtomQueryFormalCharge;
  }
  if (description == "AtomHybridization") {
    return AtomQueryHybridization;
  }
  if (description == "AtomIsAromatic") {
    return AtomQueryIsAromatic;
  }
  if (description == "AtomIsAliphatic") {
    return AtomQueryIsAliphatic;
  }
  if (description == "AtomMinRingSize") {
    return AtomQueryMinRingSize;
  }
  if (description == "AtomInNRings") {
    return AtomQueryNumRings;
  }
  if (description == "AtomNumRadicalElectrons") {
    return AtomQueryNumRadicalElectrons;
  }
  if (description == "AtomNull") {
    return AtomQueryNone;
  }

  // Degree and connectivity queries
  if (description == "AtomExplicitDegree") {
    return AtomQueryDegree;
  }
  if (description == "AtomTotalDegree") {
    return AtomQueryTotalConnectivity;
  }

  // Unsupported SMARTS primitives - throw instead of silently ignoring
  if (description == "AtomRingBondCount") {
    throw std::runtime_error("SMARTS ring connectivity query (x) is not supported");
  }
  if (description == "AtomTotalValence") {
    return AtomQueryTotalValence;
  }
  if (description == "AtomImplicitHCount") {
    throw std::runtime_error("SMARTS implicit hydrogen count query (h) is not supported");
  }
  if (description == "AtomMass" || description == "AtomIsotope") {
    return AtomQueryIsotope;
  }
  if (description == "AtomHasRingBond") {
    throw std::runtime_error("SMARTS ring bond query (@) is not supported");
  }
  if (description == "AtomUnsaturated") {
    throw std::runtime_error("SMARTS unsaturation query is not supported");
  }
  if (description == "AtomChiralTag") {
    throw std::runtime_error("SMARTS chirality query (@/@@ ) is not supported");
  }
  if (description == "AtomInRing") {
    return AtomQueryIsInRing;  // [r] any ring query
  }

  throw std::runtime_error("Unsupported SMARTS atom query: " + description);
}

AtomQueryMask buildQueryMask(const AtomDataPacked& queryAtom, AtomQuery queryFlags) {
  AtomQueryMask m = {0, 0, 0, 0};

  // Helper lambda to set mask and expected for a byte in the lower 64 bits
  auto setLoField = [&](int byteOffset, uint8_t value) {
    m.maskLo |= 0xFFULL << (byteOffset * 8);
    m.expectedLo |= static_cast<uint64_t>(value) << (byteOffset * 8);
  };

  // Helper lambda to set mask and expected for a byte in the upper 64 bits
  auto setHiField = [&](int byteOffset, uint8_t value) {
    m.maskHi |= 0xFFULL << (byteOffset * 8);
    m.expectedHi |= static_cast<uint64_t>(value) << (byteOffset * 8);
  };

  // Lower 64-bit fields
  if (queryFlags & AtomQueryAtomicNum) {
    setLoField(AtomDataPacked::kAtomicNumByte, queryAtom.atomicNum());
  }
  if (queryFlags & AtomQueryNumExplicitHs) {
    setLoField(AtomDataPacked::kNumExplicitHsByte, queryAtom.numExplicitHs());
  }
  if (queryFlags & AtomQueryExplicitValence) {
    setLoField(AtomDataPacked::kExplicitValenceByte, queryAtom.explicitValence());
  }
  if (queryFlags & AtomQueryImplicitValence) {
    setLoField(AtomDataPacked::kImplicitValenceByte, queryAtom.implicitValence());
  }
  if (queryFlags & AtomQueryFormalCharge) {
    setLoField(AtomDataPacked::kFormalChargeByte, static_cast<uint8_t>(queryAtom.formalCharge()));
  }
  if (queryFlags & AtomQueryChiralTag) {
    setLoField(AtomDataPacked::kChiralTagByte, queryAtom.chiralTag());
  }
  if (queryFlags & AtomQueryNumRadicalElectrons) {
    setLoField(AtomDataPacked::kNumRadicalElectronsByte, queryAtom.numRadicalElectrons());
  }
  if (queryFlags & AtomQueryHybridization) {
    setLoField(AtomDataPacked::kHybridizationByte, queryAtom.hybridization());
  }

  // Upper 64-bit fields
  if (queryFlags & AtomQueryMinRingSize) {
    setHiField(AtomDataPacked::kMinRingSizeByte, queryAtom.minRingSize());
  }
  if (queryFlags & AtomQueryNumRings) {
    setHiField(AtomDataPacked::kNumRingsByte, queryAtom.numRings());
  }
  if (queryFlags & AtomQueryTotalValence) {
    setHiField(AtomDataPacked::kTotalValenceByte, queryAtom.totalValence());
  }

  // Special handling for aromaticity: both flags use the same field but expect different values
  if (queryFlags & AtomQueryIsAromatic) {
    setHiField(AtomDataPacked::kIsAromaticByte, 0x01);  // expect true
  }
  if (queryFlags & AtomQueryIsAliphatic) {
    setHiField(AtomDataPacked::kIsAromaticByte, 0x00);  // expect false
  }

  // [R] and [r] any-ring queries check the isInRing field
  if (queryFlags & AtomQueryIsInRing) {
    setHiField(AtomDataPacked::kIsInRingByte, 0x01);  // expect in ring
  }

  // Isotope queries like [13C]
  if (queryFlags & AtomQueryIsotope) {
    setHiField(AtomDataPacked::kIsotopeByte, queryAtom.isotope());
  }

  // Degree queries like [D3]
  if (queryFlags & AtomQueryDegree) {
    setHiField(AtomDataPacked::kDegreeByte, queryAtom.degree());
  }

  // Total connectivity queries like [X4]
  if (queryFlags & AtomQueryTotalConnectivity) {
    setHiField(AtomDataPacked::kTotalConnectivityByte, queryAtom.totalConnectivity());
  }

  return m;
}

namespace {

/**
 * @brief Builder for constructing boolean expression trees from RDKit queries.
 *
 * Recursively processes RDKit query atoms (supporting AND, OR, NOT) and generates
 * a sequence of BoolInstructions for evaluation on the GPU.
 */
struct QueryTreeBuilder {
  std::vector<AtomQueryMask>   leafMasks;
  std::vector<BondTypeCounts>  leafBondCounts;
  std::vector<BoolInstruction> instructions;
  int                          nextScratchIdx = 0;  // Use int to allow counting beyond uint8_t max

  /**
   * @brief Allocate the next scratch slot.
   * @return Scratch index where the result will be stored
   */
  uint8_t allocateScratch() { return static_cast<uint8_t>(nextScratchIdx++); }

  /**
   * @brief Process a leaf query (primitive comparison) and add to the tree.
   * @return Scratch index where the result will be stored
   */
  uint8_t addLeaf(const AtomDataPacked& packed, AtomQuery flags, const BondTypeCounts& bondCounts) {
    const uint8_t maskIdx = static_cast<uint8_t>(leafMasks.size());
    leafMasks.push_back(buildQueryMask(packed, flags));
    leafBondCounts.push_back(bondCounts);

    const uint8_t dst = allocateScratch();
    instructions.push_back(BoolInstruction::makeLeaf(dst, maskIdx));
    return dst;
  }

  /**
   * @brief Add an AND instruction combining two operands.
   */
  uint8_t addAnd(uint8_t left, uint8_t right) {
    const uint8_t dst = allocateScratch();
    instructions.push_back(BoolInstruction::makeAnd(dst, left, right));
    return dst;
  }

  /**
   * @brief Add an OR instruction combining two operands.
   */
  uint8_t addOr(uint8_t left, uint8_t right) {
    const uint8_t dst = allocateScratch();
    instructions.push_back(BoolInstruction::makeOr(dst, left, right));
    return dst;
  }

  /**
   * @brief Add a NOT instruction.
   */
  uint8_t addNot(uint8_t src) {
    const uint8_t dst = allocateScratch();
    instructions.push_back(BoolInstruction::makeNot(dst, src));
    return dst;
  }

  /**
   * @brief Check if the tree exceeds scratch limits.
   */
  bool exceedsScratchLimit() const { return nextScratchIdx > kMaxBoolScratchSize; }

  /**
   * @brief Get the required scratch size.
   */
  int requiredScratchSize() const { return nextScratchIdx; }

  /**
   * @brief Build the final AtomQueryTree metadata.
   */
  AtomQueryTree buildTree() const {
    AtomQueryTree tree;
    tree.numLeaves       = static_cast<uint8_t>(leafMasks.size());
    tree.numInstructions = static_cast<uint8_t>(instructions.size());
    tree.scratchSize     = static_cast<uint8_t>(std::min(nextScratchIdx, 255));
    tree.resultIdx       = nextScratchIdx > 0 ? static_cast<uint8_t>(nextScratchIdx - 1) : 0;
    return tree;
  }
};

/**
 * @brief Check if a query subtree contains only AND operations (no OR/NOT).
 */
bool isAndOnlyQuery(const RDKit::Atom::QUERYATOM_QUERY* query) {
  if (query->getNegation()) {
    return false;
  }

  const std::string desc = query->getDescription();
  if (desc == "AtomOr" || desc == "AtomXor") {
    return false;
  }

  if (desc == "AtomAnd") {
    for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
      if (!isAndOnlyQuery((*it).get())) {
        return false;
      }
    }
  }

  return true;
}

/**
 * @brief Collect flags and packed data from an AND-only query subtree.
 *
 * This optimized path merges all AND conditions into a single leaf mask.
 */
void collectAndOnlyFlags(const RDKit::Atom::QUERYATOM_QUERY* query,
                         AtomQuery&                          flags,
                         AtomDataPacked&                     packed) {
  const std::string desc = query->getDescription();

  if (desc == "AtomAnd") {
    for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
      collectAndOnlyFlags((*it).get(), flags, packed);
    }
    return;
  }

  if (desc == "AtomType") {
    const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(query);
    int         typeVal = eqQuery->getVal();
    if (typeVal >= 1000) {
      flags |= AtomQueryAtomicNum | AtomQueryIsAromatic;
      packed.setAtomicNum(typeVal - 1000);
      packed.setIsAromatic(true);
    } else {
      flags |= AtomQueryAtomicNum | AtomQueryIsAliphatic;
      packed.setAtomicNum(typeVal);
      packed.setIsAromatic(false);
    }
    return;
  }

  if (desc == "AtomIsAromatic") {
    flags |= AtomQueryIsAromatic;
    packed.setIsAromatic(true);
    return;
  }
  if (desc == "AtomIsAliphatic") {
    flags |= AtomQueryIsAliphatic;
    packed.setIsAromatic(false);
    return;
  }

  if (desc == "AtomNull") {
    return;
  }

  const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(query);

  if (desc == "AtomAtomicNum") {
    flags |= AtomQueryAtomicNum;
    packed.setAtomicNum(eqQuery->getVal());
  } else if (desc == "AtomHCount") {
    flags |= AtomQueryNumExplicitHs;
    packed.setNumExplicitHs(eqQuery->getVal());
  } else if (desc == "AtomFormalCharge") {
    flags |= AtomQueryFormalCharge;
    packed.setFormalCharge(eqQuery->getVal());
  } else if (desc == "AtomHybridization") {
    flags |= AtomQueryHybridization;
    packed.setHybridization(eqQuery->getVal());
  } else if (desc == "AtomInNRings") {
    int val = eqQuery->getVal();
    if (val < 0) {
      // [R] any ring query - just check isInRing
      flags |= AtomQueryIsInRing;
      packed.setIsInRing(true);
    } else {
      flags |= AtomQueryNumRings;
      packed.setNumRings(val);
    }
  } else if (desc == "AtomInRing") {
    // [r] any ring query - just check isInRing
    flags |= AtomQueryIsInRing;
    packed.setIsInRing(true);
  } else if (desc == "AtomMinRingSize") {
    flags |= AtomQueryMinRingSize;
    packed.setMinRingSize(eqQuery->getVal());
  } else if (desc == "AtomNumRadicalElectrons") {
    flags |= AtomQueryNumRadicalElectrons;
    packed.setNumRadicalElectrons(eqQuery->getVal());
  } else if (desc == "AtomTotalValence") {
    flags |= AtomQueryTotalValence;
    packed.setTotalValence(eqQuery->getVal());
  } else if (desc == "AtomMass" || desc == "AtomIsotope") {
    int isotope = eqQuery->getVal();
    if (isotope > 255) {
      throw std::runtime_error("Isotope mass " + std::to_string(isotope) + " exceeds maximum supported value of 255");
    }
    flags |= AtomQueryIsotope;
    packed.setIsotope(static_cast<uint8_t>(isotope));
  } else if (desc == "AtomExplicitDegree") {
    flags |= AtomQueryDegree;
    packed.setDegree(eqQuery->getVal());
  } else if (desc == "AtomTotalDegree") {
    flags |= AtomQueryTotalConnectivity;
    packed.setTotalConnectivity(eqQuery->getVal());
  } else {
    AtomQuery flag = atomQueryFromDescription(desc);
    flags |= flag;
  }
}

/**
 * @brief Recursively process a query tree and build boolean instructions.
 *
 * @param query The RDKit query to process
 * @param builder The builder accumulating leaves and instructions
 * @param bondCounts Bond type counts for the atom (used for leaf nodes)
 * @return Scratch index where this subtree's result will be stored
 */
uint8_t processQueryTree(const RDKit::Atom::QUERYATOM_QUERY* query,
                         QueryTreeBuilder&                   builder,
                         const BondTypeCounts&               bondCounts) {
  const std::string desc      = query->getDescription();
  const bool        isNegated = query->getNegation();

  // Handle AND-only subtrees efficiently by merging into a single leaf
  if (!isNegated && isAndOnlyQuery(query)) {
    AtomQuery      flags  = AtomQueryNone;
    AtomDataPacked packed = {};
    collectAndOnlyFlags(query, flags, packed);
    return builder.addLeaf(packed, flags, bondCounts);
  }

  // Handle OR: process children and combine with OR instructions
  if (desc == "AtomOr") {
    std::vector<uint8_t> childResults;
    for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
      childResults.push_back(processQueryTree((*it).get(), builder, bondCounts));
    }

    if (childResults.empty()) {
      throw std::runtime_error("Empty AtomOr query");
    }

    uint8_t result = childResults[0];
    for (size_t i = 1; i < childResults.size(); ++i) {
      result = builder.addOr(result, childResults[i]);
    }

    if (isNegated) {
      result = builder.addNot(result);
    }
    return result;
  }

  // Handle AND with complex children (some may have OR or NOT)
  if (desc == "AtomAnd") {
    std::vector<uint8_t> childResults;
    for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
      childResults.push_back(processQueryTree((*it).get(), builder, bondCounts));
    }

    if (childResults.empty()) {
      throw std::runtime_error("Empty AtomAnd query");
    }

    uint8_t result = childResults[0];
    for (size_t i = 1; i < childResults.size(); ++i) {
      result = builder.addAnd(result, childResults[i]);
    }

    if (isNegated) {
      result = builder.addNot(result);
    }
    return result;
  }

  if (desc == "AtomXor") {
    throw std::runtime_error("SMARTS XOR queries are not supported");
  }

  if (desc == "RecursiveStructure") {
    throw std::runtime_error("Recursive SMARTS ($(...)) are not supported");
  }

  // Leaf node - create a single leaf mask
  AtomQuery      flags  = AtomQueryNone;
  AtomDataPacked packed = {};
  collectAndOnlyFlags(query, flags, packed);
  uint8_t result = builder.addLeaf(packed, flags, bondCounts);

  if (isNegated) {
    result = builder.addNot(result);
  }
  return result;
}

/**
 * @brief Build a complete query tree for an atom.
 *
 * @param atom The RDKit atom to process
 * @param bondCounts Bond type counts for the atom
 * @param builder Output: the populated QueryTreeBuilder
 */
void buildQueryTreeForAtom(const RDKit::Atom*    atom,
                           const BondTypeCounts& bondCounts,
                           QueryTreeBuilder&     builder) {
  // Check for chirality specified on the atom (SMARTS @/@@ notation)
  if (atom->getChiralTag() != RDKit::Atom::ChiralType::CHI_UNSPECIFIED) {
    throw std::runtime_error("SMARTS chirality query (@/@@) is not supported");
  }

  if (!atom->hasQuery()) {
    builder.addLeaf(AtomDataPacked{}, AtomQueryNone, bondCounts);
    return;
  }

  const auto* query = atom->getQuery();
  if (query == nullptr) {
    builder.addLeaf(AtomDataPacked{}, AtomQueryNone, bondCounts);
    return;
  }

  processQueryTree(query, builder, bondCounts);
}

AtomQuery getQueryFlagsFromQuery(const RDKit::Atom::QUERYATOM_QUERY* query) {
  const std::string description = query->getDescription();

  // Check for negation first - applies to any query type
  if (query->getNegation()) {
    throw std::runtime_error("Negated atom queries (!) are not supported");
  }

  // Composite query - recurse into children
  if (description == "AtomAnd") {
    AtomQuery result = AtomQueryNone;
    for (auto it = query->beginChildren(); it != query->endChildren(); ++it) {
      result |= getQueryFlagsFromQuery((*it).get());
    }
    return result;
  }

  // AtomType encodes atomic number + aromaticity in the value
  // Aliphatic: value = atomic number; Aromatic: value = 1000 + atomic number
  if (description == "AtomType") {
    const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(query);
    int         typeVal = eqQuery->getVal();
    if (typeVal >= 1000) {
      return AtomQueryAtomicNum | AtomQueryIsAromatic;
    }
    return AtomQueryAtomicNum | AtomQueryIsAliphatic;
  }

  if (description == "AtomOr" || description == "AtomXor") {
    throw std::runtime_error("Composite queries (OR/XOR) are not supported: " + description);
  }

  if (description == "RecursiveStructure") {
    throw std::runtime_error("Recursive SMARTS ($(...)) are not supported");
  }

  // [R] creates AtomInNRings with value -1 meaning "any ring" (numRings != 0)
  // Now supported via AtomQueryIsInRing
  if (description == "AtomInNRings") {
    const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(query);
    if (eqQuery->getVal() < 0) {
      return AtomQueryIsInRing;
    }
  }

  return atomQueryFromDescription(description);
}

AtomQuery getAtomQueryType(const RDKit::Atom* atom) {
  // Check for chirality specified on the atom (SMARTS @/@@ notation)
  if (atom->getChiralTag() != RDKit::Atom::ChiralType::CHI_UNSPECIFIED) {
    throw std::runtime_error("SMARTS chirality query (@/@@) is not supported");
  }

  if (!atom->hasQuery()) {
    return AtomQueryNone;
  }

  const auto* query = atom->getQuery();
  if (query == nullptr) {
    return AtomQueryNone;
  }

  return getQueryFlagsFromQuery(query);
}

void addBondsAndConnectivity(const RDKit::ROMol*    mol,
                             MoleculesHost&         batch,
                             int&                   cumulativeBondCount,
                             const RDKit::RingInfo* ringInfo) {
  auto& bondDataVec      = batch.bondData;
  auto& atomBondStarts   = batch.atomBondStarts;
  auto& bondDataIndices  = batch.bondDataIndices;
  auto& otherAtomIndices = batch.otherAtomIndices;

  bondDataVec.reserve(bondDataVec.size() + mol->getNumBonds());

  for (unsigned int i = 0; i < mol->getNumBonds(); ++i) {
    auto&       bd   = bondDataVec.emplace_back();
    const auto* bond = mol->getBondWithIdx(i);
    bd.bondType      = bond->getBondType();
    bd.isInRing      = ringInfo->numBondRings(i) > 0 ? 1 : 0;
  }

  atomBondStarts.push_back(0);

  for (const RDKit::Atom* atom : mol->atoms()) {
    const unsigned int atomIdx = atom->getIdx();

    auto [beg, bondEnd] = mol->getAtomBonds(atom);
    while (beg != bondEnd) {
      const auto*        bond        = (*mol)[*beg];
      const unsigned int bondIdx     = bond->getIdx();
      const int          otherAtomId = bond->getOtherAtomIdx(atomIdx);

      otherAtomIndices.push_back(static_cast<int16_t>(otherAtomId));
      bondDataIndices.push_back(static_cast<int16_t>(bondIdx));
      ++cumulativeBondCount;
      ++beg;
    }
    atomBondStarts.push_back(static_cast<int16_t>(cumulativeBondCount));
  }
}

}  // namespace

constexpr unsigned int kMaxMoleculeAtoms = 128;

void addToBatch(const RDKit::ROMol* mol, MoleculesHost& batch) {
  if (mol->getNumAtoms() > kMaxMoleculeAtoms) {
    throw std::runtime_error("Target molecule has " + std::to_string(mol->getNumAtoms()) +
                             " atoms, which exceeds the maximum of " + std::to_string(kMaxMoleculeAtoms));
  }

  auto& atomDataVec       = batch.atomData;
  auto& atomDataPackedVec = batch.atomDataPacked;
  auto& bondTypeCountsVec = batch.bondTypeCounts;

  atomDataVec.reserve(atomDataVec.size() + mol->getNumAtoms());
  atomDataPackedVec.reserve(atomDataPackedVec.size() + mol->getNumAtoms());
  bondTypeCountsVec.reserve(bondTypeCountsVec.size() + mol->getNumAtoms());

  const auto* ringInfo = mol->getRingInfo();

  int cumulativeBondCount = 0;
  addBondsAndConnectivity(mol, batch, cumulativeBondCount, ringInfo);

  for (const RDKit::Atom* atom : mol->atoms()) {
    auto& thisAtomData = atomDataVec.emplace_back();
    populateAtomData(atom, thisAtomData, ringInfo);

    auto& thisAtomPacked = atomDataPackedVec.emplace_back();
    populateAtomDataPacked(atom, thisAtomPacked, ringInfo);

    auto& thisBondCounts = bondTypeCountsVec.emplace_back();
    populateBondTypeCounts(mol, atom, thisBondCounts);
  }

  batch.batchAtomStarts.push_back(static_cast<int>(atomDataVec.size()));
  batch.batchBondStarts.push_back(static_cast<int>(batch.bondData.size()));
  batch.batchAtomBondStarts.push_back(static_cast<int>(batch.atomBondStarts.size()));
  batch.batchOtherAtomIndicesStarts.push_back(static_cast<int>(batch.otherAtomIndices.size()));
  batch.batchBondIndicesStarts.push_back(static_cast<int>(batch.bondDataIndices.size()));
}

namespace {

void populateQueryAtomDataPacked(const RDKit::Atom* atom, AtomDataPacked& packed) {
  if (!atom->hasQuery()) {
    return;
  }

  const auto* query = atom->getQuery();
  if (query == nullptr) {
    return;
  }

  // Use the existing populateFromQuery logic to extract values, but store in packed format
  // We'll duplicate the logic here to avoid converting back and forth
  std::function<void(const RDKit::Atom::QUERYATOM_QUERY*)> populatePacked;
  populatePacked = [&](const RDKit::Atom::QUERYATOM_QUERY* q) {
    const std::string desc = q->getDescription();

    if (desc == "AtomAnd") {
      for (auto it = q->beginChildren(); it != q->endChildren(); ++it) {
        populatePacked((*it).get());
      }
      return;
    }

    if (desc == "AtomType") {
      const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(q);
      int         typeVal = eqQuery->getVal();
      if (typeVal >= 1000) {
        packed.setAtomicNum(typeVal - 1000);
        packed.setIsAromatic(true);
      } else {
        packed.setAtomicNum(typeVal);
        packed.setIsAromatic(false);
      }
      return;
    }

    if (desc == "AtomIsAromatic") {
      packed.setIsAromatic(true);
      return;
    }
    if (desc == "AtomIsAliphatic") {
      packed.setIsAromatic(false);
      return;
    }

    const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(q);

    if (desc == "AtomAtomicNum") {
      packed.setAtomicNum(eqQuery->getVal());
    } else if (desc == "AtomHCount") {
      packed.setNumExplicitHs(eqQuery->getVal());
    } else if (desc == "AtomFormalCharge") {
      packed.setFormalCharge(eqQuery->getVal());
    } else if (desc == "AtomHybridization") {
      packed.setHybridization(eqQuery->getVal());
    } else if (desc == "AtomInNRings") {
      packed.setNumRings(eqQuery->getVal());
    } else if (desc == "AtomMinRingSize") {
      packed.setMinRingSize(eqQuery->getVal());
    } else if (desc == "AtomNumRadicalElectrons") {
      packed.setNumRadicalElectrons(eqQuery->getVal());
    } else if (desc == "AtomTotalValence") {
      packed.setTotalValence(eqQuery->getVal());
    } else if (desc == "AtomMass" || desc == "AtomIsotope") {
      int isotope = eqQuery->getVal();
      if (isotope > 255) {
        throw std::runtime_error("Isotope mass " + std::to_string(isotope) +
                                 " exceeds maximum supported value of 255");
      }
      packed.setIsotope(static_cast<uint8_t>(isotope));
    } else if (desc == "AtomExplicitDegree") {
      packed.setDegree(eqQuery->getVal());
    } else if (desc == "AtomTotalDegree") {
      packed.setTotalConnectivity(eqQuery->getVal());
    }
  };

  populatePacked(query);
}

/**
 * @brief Extract bond query flags from an RDKit bond query.
 *
 * Handles ring bond constraints (!@ and @) by examining the bond's query.
 */
void extractBondQueryFlags(const RDKit::Bond* bond, BondQueryData& queryData) {
  queryData.bondType   = bond->getBondType();
  queryData.queryFlags = BondQueryNone;

  if (!bond->hasQuery()) {
    return;
  }

  const auto* query = bond->getQuery();
  if (query == nullptr) {
    return;
  }

  // Recursive function to process bond query tree
  std::function<void(const RDKit::Bond::QUERYBOND_QUERY*)> processQuery;
  processQuery = [&](const RDKit::Bond::QUERYBOND_QUERY* q) {
    const std::string desc      = q->getDescription();
    const bool        isNegated = q->getNegation();

    if (desc == "BondAnd") {
      for (auto it = q->beginChildren(); it != q->endChildren(); ++it) {
        processQuery((*it).get());
      }
      return;
    }

    if (desc == "BondOr") {
      // Check if this is a "single or aromatic" or "double or aromatic" pattern
      bool hasSingle   = false;
      bool hasDouble   = false;
      bool hasAromatic = false;
      for (auto it = q->beginChildren(); it != q->endChildren(); ++it) {
        const std::string childDesc = (*it)->getDescription();
        if (childDesc == "BondOrder") {
          const auto* eqQuery  = static_cast<const RDKit::BOND_EQUALS_QUERY*>((*it).get());
          int         bondType = eqQuery->getVal();
          if (bondType == 1) {
            hasSingle = true;
          } else if (bondType == 2) {
            hasDouble = true;
          } else if (bondType == 7 || bondType == 12) {
            hasAromatic = true;
          }
        } else if (childDesc == "BondIsAromatic") {
          hasAromatic = true;
        }
      }
      if (hasSingle && hasAromatic) {
        queryData.queryFlags |= BondQuerySingleOrAromatic;
        queryData.bondType = 1;
        return;
      }
      if (hasDouble && hasAromatic) {
        queryData.queryFlags |= BondQueryDoubleOrAromatic;
        queryData.bondType = 2;
        return;
      }
      // Fall through to process children normally for other BondOr patterns
      for (auto it = q->beginChildren(); it != q->endChildren(); ++it) {
        processQuery((*it).get());
      }
      return;
    }

    if (desc == "BondIsInRing" || desc == "BondInRing") {
      if (isNegated) {
        queryData.queryFlags |= BondQueryNotRingBond;
      } else {
        queryData.queryFlags |= BondQueryIsRingBond;
      }
    } else if (desc == "SingleOrAromaticBond") {
      queryData.queryFlags |= BondQuerySingleOrAromatic;
      queryData.bondType = 1;  // Base type is single, flag allows aromatic too
    } else if (desc == "DoubleOrAromaticBond") {
      queryData.queryFlags |= BondQueryDoubleOrAromatic;
      queryData.bondType = 2;  // Base type is double, flag allows aromatic too
    } else if (desc == "BondOrder") {
      const auto* eqQuery = static_cast<const RDKit::BOND_EQUALS_QUERY*>(q);
      queryData.bondType  = eqQuery->getVal();
    } else if (desc == "BondNull") {
      queryData.bondType = 0;  // Any bond
    }
  };

  processQuery(query);
}

/**
 * @brief Add bonds and connectivity for a query molecule (SMARTS).
 *
 * Similar to addBondsAndConnectivity but also extracts bond query information
 * including ring bond constraints.
 */
void addQueryBondsAndConnectivity(const RDKit::ROMol* mol, MoleculesHost& batch, int& cumulativeBondCount) {
  auto& bondDataVec      = batch.bondData;
  auto& bondQueryDataVec = batch.bondQueryData;
  auto& atomBondStarts   = batch.atomBondStarts;
  auto& bondDataIndices  = batch.bondDataIndices;
  auto& otherAtomIndices = batch.otherAtomIndices;

  bondDataVec.reserve(bondDataVec.size() + mol->getNumBonds());
  bondQueryDataVec.reserve(bondQueryDataVec.size() + mol->getNumBonds());

  for (unsigned int i = 0; i < mol->getNumBonds(); ++i) {
    const auto* bond = mol->getBondWithIdx(i);

    auto& bd    = bondDataVec.emplace_back();
    bd.bondType = bond->getBondType();
    bd.isInRing = 0;  // Query molecules don't have ring info computed

    auto& bq = bondQueryDataVec.emplace_back();
    extractBondQueryFlags(bond, bq);
  }

  atomBondStarts.push_back(0);

  for (const RDKit::Atom* atom : mol->atoms()) {
    const unsigned int atomIdx = atom->getIdx();

    auto [beg, bondEnd] = mol->getAtomBonds(atom);
    while (beg != bondEnd) {
      const auto*        bond        = (*mol)[*beg];
      const unsigned int bondIdx     = bond->getIdx();
      const int          otherAtomId = bond->getOtherAtomIdx(atomIdx);

      otherAtomIndices.push_back(static_cast<int16_t>(otherAtomId));
      bondDataIndices.push_back(static_cast<int16_t>(bondIdx));
      ++cumulativeBondCount;
      ++beg;
    }
    atomBondStarts.push_back(static_cast<int16_t>(cumulativeBondCount));
  }
}

}  // namespace

void addQueryToBatch(const RDKit::ROMol* mol, MoleculesHost& batch) {
  if (mol->getNumAtoms() > kMaxMoleculeAtoms) {
    throw std::runtime_error("Query molecule has " + std::to_string(mol->getNumAtoms()) +
                             " atoms, which exceeds the maximum of " + std::to_string(kMaxMoleculeAtoms));
  }

  std::vector<int> fragMapping;
  const unsigned   numFrags = RDKit::MolOps::getMolFrags(*mol, fragMapping);
  if (numFrags > 1) {
    throw std::runtime_error("Fragment queries (disconnected SMARTS patterns) are not supported. "
                             "Query has " + std::to_string(numFrags) + " disconnected components: " +
                             RDKit::MolToSmarts(*mol));
  }

  auto& atomDataVec       = batch.atomData;
  auto& atomDataPackedVec = batch.atomDataPacked;
  auto& atomQueriesVec    = batch.atomQueries;
  auto& atomQueryMasksVec = batch.atomQueryMasks;
  auto& bondTypeCountsVec = batch.bondTypeCounts;

  // Boolean tree data
  auto& atomQueryTreesVec     = batch.atomQueryTrees;
  auto& queryInstructionsVec  = batch.queryInstructions;
  auto& queryLeafMasksVec     = batch.queryLeafMasks;
  auto& queryLeafBondCountsVec = batch.queryLeafBondCounts;
  auto& atomInstrStartsVec    = batch.atomInstrStarts;
  auto& atomLeafMaskStartsVec = batch.atomLeafMaskStarts;

  atomDataVec.reserve(atomDataVec.size() + mol->getNumAtoms());
  atomDataPackedVec.reserve(atomDataPackedVec.size() + mol->getNumAtoms());
  atomQueriesVec.reserve(atomQueriesVec.size() + mol->getNumAtoms());
  atomQueryMasksVec.reserve(atomQueryMasksVec.size() + mol->getNumAtoms());
  bondTypeCountsVec.reserve(bondTypeCountsVec.size() + mol->getNumAtoms());
  atomQueryTreesVec.reserve(atomQueryTreesVec.size() + mol->getNumAtoms());
  atomInstrStartsVec.reserve(atomInstrStartsVec.size() + mol->getNumAtoms());
  atomLeafMaskStartsVec.reserve(atomLeafMaskStartsVec.size() + mol->getNumAtoms());

  int cumulativeBondCount = 0;
  addQueryBondsAndConnectivity(mol, batch, cumulativeBondCount);

  for (const RDKit::Atom* atom : mol->atoms()) {
    auto& thisAtomData = atomDataVec.emplace_back();
    populateQueryAtomData(atom, thisAtomData);

    auto& thisAtomPacked = atomDataPackedVec.emplace_back();
    populateQueryAtomDataPacked(atom, thisAtomPacked);

    // Compute bond type counts first (needed for query tree building)
    auto& thisBondCounts = bondTypeCountsVec.emplace_back();
    populateQueryBondTypeCounts(mol, atom, thisBondCounts);

    // Build the boolean expression tree for this atom
    QueryTreeBuilder builder;
    buildQueryTreeForAtom(atom, thisBondCounts, builder);

    if (builder.exceedsScratchLimit()) {
      throw std::runtime_error("SMARTS query too complex: boolean expression requires " +
                               std::to_string(builder.requiredScratchSize()) + " scratch slots, but maximum is " +
                               std::to_string(kMaxBoolScratchSize) + ". Query: " + RDKit::MolToSmarts(*mol));
    }

    // Store offsets into global instruction/leaf arrays
    atomInstrStartsVec.push_back(static_cast<int>(queryInstructionsVec.size()));
    atomLeafMaskStartsVec.push_back(static_cast<int>(queryLeafMasksVec.size()));

    // Append this atom's data to global arrays
    queryInstructionsVec.insert(queryInstructionsVec.end(),
                                builder.instructions.begin(),
                                builder.instructions.end());
    queryLeafMasksVec.insert(queryLeafMasksVec.end(),
                             builder.leafMasks.begin(),
                             builder.leafMasks.end());
    queryLeafBondCountsVec.insert(queryLeafBondCountsVec.end(),
                                  builder.leafBondCounts.begin(),
                                  builder.leafBondCounts.end());
    atomQueryTreesVec.push_back(builder.buildTree());

    // Legacy fields: for simple queries, use the first leaf mask for backwards compatibility
    AtomQuery queryFlags = AtomQueryNone;
    if (!builder.leafMasks.empty()) {
      atomQueryMasksVec.push_back(builder.leafMasks[0]);
      // Try to get flags from the legacy path for simple queries
      if (atom->hasQuery() && atom->getQuery() != nullptr) {
        try {
          queryFlags = getQueryFlagsFromQuery(atom->getQuery());
        } catch (...) {
          // Complex query - flags not applicable
        }
      }
    } else {
      atomQueryMasksVec.push_back(AtomQueryMask{});
    }
    atomQueriesVec.push_back(queryFlags);
  }

  batch.batchAtomStarts.push_back(static_cast<int>(atomDataVec.size()));
  batch.batchBondStarts.push_back(static_cast<int>(batch.bondData.size()));
  batch.batchAtomBondStarts.push_back(static_cast<int>(batch.atomBondStarts.size()));
  batch.batchOtherAtomIndicesStarts.push_back(static_cast<int>(batch.otherAtomIndices.size()));
  batch.batchBondIndicesStarts.push_back(static_cast<int>(batch.bondDataIndices.size()));
}

}  // namespace nvMolKit
