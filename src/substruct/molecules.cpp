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

#include <GraphMol/QueryAtom.h>
#include <GraphMol/QueryOps.h>
#include <GraphMol/ROMol.h>
#include <RDGeneral/versions.h>

#include <functional>
#include <stdexcept>
#include <string>

namespace nvMolKit {

namespace {

void populateAtomData(const RDKit::Atom* atom, AtomData& atomData, const RDKit::RingInfo* ringInfo) {
  atomData.atomicNum     = atom->getAtomicNum();
  atomData.chiralTag     = atom->getChiralTag();
  atomData.numExplicitHs = atom->getNumExplicitHs();
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
  packed.setNumExplicitHs(atom->getNumExplicitHs());
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
  const int idx = atom->getIdx();
  packed.setNumRings(ringInfo->numAtomRings(idx));
  packed.setMinRingSize(ringInfo->minAtomRingSize(idx));
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
void populateQueryBondTypeCounts(const RDKit::ROMol* mol, const RDKit::Atom* atom, BondTypeCounts& counts) {
  auto [beg, bondEnd] = mol->getAtomBonds(atom);
  while (beg != bondEnd) {
    const auto* bond     = (*mol)[*beg];
    int         bondType = bond->getBondType();
    switch (bondType) {
      case 0:
        ++counts.any;
        break;  // UNSPECIFIED = any bond (~)
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

  // Unsupported SMARTS primitives - throw instead of silently ignoring
  if (description == "AtomExplicitDegree") {
    throw std::runtime_error("SMARTS degree query (D) is not supported");
  }
  if (description == "AtomTotalDegree") {
    throw std::runtime_error("SMARTS total connectivity query (X) is not supported");
  }
  if (description == "AtomRingBondCount") {
    throw std::runtime_error("SMARTS ring connectivity query (x) is not supported");
  }
  if (description == "AtomTotalValence") {
    return AtomQueryTotalValence;
  }
  if (description == "AtomImplicitHCount") {
    throw std::runtime_error("SMARTS implicit hydrogen count query (h) is not supported");
  }
  if (description == "AtomMass") {
    throw std::runtime_error("SMARTS isotope/mass query is not supported");
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
    throw std::runtime_error(
        "SMARTS [r] (any ring) query is not supported; use [r5], [r6], etc. for ring size");
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

  return m;
}

namespace {

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
  // We only support exact ring count like [R1], [R2], etc. (value >= 0)
  if (description == "AtomInNRings") {
    const auto* eqQuery = static_cast<const RDKit::ATOM_EQUALS_QUERY*>(query);
    if (eqQuery->getVal() < 0) {
      throw std::runtime_error(
          "SMARTS [R] query is not supported; use [R1], [R2], etc. for exact ring count");
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

void addBondsAndConnectivity(const RDKit::ROMol* mol, MoleculesHost& batch, int& cumulativeBondCount) {
  auto& bondDataVec      = batch.bondData;
  auto& atomBondStarts   = batch.atomBondStarts;
  auto& bondDataIndices  = batch.bondDataIndices;
  auto& otherAtomIndices = batch.otherAtomIndices;

  bondDataVec.reserve(bondDataVec.size() + mol->getNumBonds());

  for (unsigned int i = 0; i < mol->getNumBonds(); ++i) {
    auto& bd    = bondDataVec.emplace_back();
    bd.bondType = mol->getBondWithIdx(i)->getBondType();
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
  auto& otherAtomIndices  = batch.otherAtomIndices;
  auto& bondDataIndices   = batch.bondDataIndices;

  const size_t otherAtomIndicesBefore = otherAtomIndices.size();
  const size_t bondDataIndicesBefore  = bondDataIndices.size();

  atomDataVec.reserve(atomDataVec.size() + mol->getNumAtoms());
  atomDataPackedVec.reserve(atomDataPackedVec.size() + mol->getNumAtoms());
  bondTypeCountsVec.reserve(bondTypeCountsVec.size() + mol->getNumAtoms());

  int cumulativeBondCount = 0;
  addBondsAndConnectivity(mol, batch, cumulativeBondCount);

  const auto* ringInfo = mol->getRingInfo();

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
  batch.batchOtherAtomIndicesStarts.push_back(static_cast<int>(otherAtomIndicesBefore));
  batch.batchBondIndicesStarts.push_back(static_cast<int>(bondDataIndicesBefore));
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
    }
  };

  populatePacked(query);
}

}  // namespace

void addQueryToBatch(const RDKit::ROMol* mol, MoleculesHost& batch) {
  if (mol->getNumAtoms() > kMaxMoleculeAtoms) {
    throw std::runtime_error("Query molecule has " + std::to_string(mol->getNumAtoms()) +
                             " atoms, which exceeds the maximum of " + std::to_string(kMaxMoleculeAtoms));
  }

  auto& atomDataVec       = batch.atomData;
  auto& atomDataPackedVec = batch.atomDataPacked;
  auto& atomQueriesVec    = batch.atomQueries;
  auto& atomQueryMasksVec = batch.atomQueryMasks;
  auto& bondTypeCountsVec = batch.bondTypeCounts;
  auto& otherAtomIndices  = batch.otherAtomIndices;
  auto& bondDataIndices   = batch.bondDataIndices;

  const size_t otherAtomIndicesBefore = otherAtomIndices.size();
  const size_t bondDataIndicesBefore  = bondDataIndices.size();

  atomDataVec.reserve(atomDataVec.size() + mol->getNumAtoms());
  atomDataPackedVec.reserve(atomDataPackedVec.size() + mol->getNumAtoms());
  atomQueriesVec.reserve(atomQueriesVec.size() + mol->getNumAtoms());
  atomQueryMasksVec.reserve(atomQueryMasksVec.size() + mol->getNumAtoms());
  bondTypeCountsVec.reserve(bondTypeCountsVec.size() + mol->getNumAtoms());

  int cumulativeBondCount = 0;
  addBondsAndConnectivity(mol, batch, cumulativeBondCount);

  for (const RDKit::Atom* atom : mol->atoms()) {
    auto& thisAtomData = atomDataVec.emplace_back();
    populateQueryAtomData(atom, thisAtomData);

    auto& thisAtomPacked = atomDataPackedVec.emplace_back();
    populateQueryAtomDataPacked(atom, thisAtomPacked);

    AtomQuery queryFlags = getAtomQueryType(atom);
    atomQueriesVec.push_back(queryFlags);

    atomQueryMasksVec.push_back(buildQueryMask(thisAtomPacked, queryFlags));

    auto& thisBondCounts = bondTypeCountsVec.emplace_back();
    populateQueryBondTypeCounts(mol, atom, thisBondCounts);
  }

  batch.batchAtomStarts.push_back(static_cast<int>(atomDataVec.size()));
  batch.batchBondStarts.push_back(static_cast<int>(batch.bondData.size()));
  batch.batchAtomBondStarts.push_back(static_cast<int>(batch.atomBondStarts.size()));
  batch.batchOtherAtomIndicesStarts.push_back(static_cast<int>(otherAtomIndicesBefore));
  batch.batchBondIndicesStarts.push_back(static_cast<int>(bondDataIndicesBefore));
}

}  // namespace nvMolKit
