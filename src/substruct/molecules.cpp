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
  atomData.formalCharge        = atom->getFormalCharge();
  atomData.hybridization       = atom->getHybridization();
  atomData.isAromatic          = atom->getIsAromatic();
  atomData.numRadicalElectrons = atom->getNumRadicalElectrons();
  const int idx                = atom->getIdx();
  atomData.numRings            = ringInfo->numAtomRings(idx);
  atomData.minRingSize         = ringInfo->minAtomRingSize(idx);
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
    int typeVal = eqQuery->getVal();
    if (typeVal >= 1000) {
      atomData.atomicNum = typeVal - 1000;
      atomData.isAromatic = true;
    } else {
      atomData.atomicNum = typeVal;
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
  return AtomQueryNone;
}

namespace {

AtomQuery getQueryFlagsFromQuery(const RDKit::Atom::QUERYATOM_QUERY* query) {
  const std::string description = query->getDescription();

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
    int typeVal = eqQuery->getVal();
    if (typeVal >= 1000) {
      return AtomQueryAtomicNum | AtomQueryIsAromatic;
    }
    return AtomQueryAtomicNum | AtomQueryIsAliphatic;
  }

  if (description == "AtomOr" || description == "AtomXor") {
    throw std::runtime_error("Composite queries (OR/XOR) are not supported: " + description);
  }

  return atomQueryFromDescription(description);
}

AtomQuery getAtomQueryType(const RDKit::Atom* atom) {
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

void addToBatch(const RDKit::ROMol* mol, MoleculesHost& batch) {
  auto& atomDataVec      = batch.atomData;
  auto& otherAtomIndices = batch.otherAtomIndices;
  auto& bondDataIndices  = batch.bondDataIndices;

  const size_t otherAtomIndicesBefore = otherAtomIndices.size();
  const size_t bondDataIndicesBefore  = bondDataIndices.size();

  atomDataVec.reserve(atomDataVec.size() + mol->getNumAtoms());

  int              cumulativeBondCount = 0;
  addBondsAndConnectivity(mol, batch, cumulativeBondCount);

  const auto* ringInfo = mol->getRingInfo();

  size_t atomIdx = 0;
  for (const RDKit::Atom* atom : mol->atoms()) {
    auto& thisAtomData = atomDataVec.emplace_back();
    populateAtomData(atom, thisAtomData, ringInfo);
    ++atomIdx;
  }

  batch.batchAtomStarts.push_back(static_cast<int>(atomDataVec.size()));
  batch.batchBondStarts.push_back(static_cast<int>(batch.bondData.size()));
  batch.batchAtomBondStarts.push_back(static_cast<int>(batch.atomBondStarts.size()));
  batch.batchOtherAtomIndicesStarts.push_back(static_cast<int>(otherAtomIndicesBefore));
  batch.batchBondIndicesStarts.push_back(static_cast<int>(bondDataIndicesBefore));
}

void addQueryToBatch(const RDKit::ROMol* mol, MoleculesHost& batch) {
  auto& atomDataVec      = batch.atomData;
  auto& atomQueriesVec   = batch.atomQueries;
  auto& otherAtomIndices = batch.otherAtomIndices;
  auto& bondDataIndices  = batch.bondDataIndices;

  const size_t otherAtomIndicesBefore = otherAtomIndices.size();
  const size_t bondDataIndicesBefore  = bondDataIndices.size();

  atomDataVec.reserve(atomDataVec.size() + mol->getNumAtoms());
  atomQueriesVec.reserve(atomQueriesVec.size() + mol->getNumAtoms());

  int cumulativeBondCount = 0;
  addBondsAndConnectivity(mol, batch, cumulativeBondCount);

  for (const RDKit::Atom* atom : mol->atoms()) {
    auto& thisAtomData = atomDataVec.emplace_back();
    atomQueriesVec.push_back(getAtomQueryType(atom));
    populateQueryAtomData(atom, thisAtomData);
  }

  batch.batchAtomStarts.push_back(static_cast<int>(atomDataVec.size()));
  batch.batchBondStarts.push_back(static_cast<int>(batch.bondData.size()));
  batch.batchAtomBondStarts.push_back(static_cast<int>(batch.atomBondStarts.size()));
  batch.batchOtherAtomIndicesStarts.push_back(static_cast<int>(otherAtomIndicesBefore));
  batch.batchBondIndicesStarts.push_back(static_cast<int>(bondDataIndicesBefore));
}

}  // namespace nvMolKit
