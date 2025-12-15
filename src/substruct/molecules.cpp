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

#include <GraphMol/ROMol.h>
#include <RDGeneral/versions.h>

#include <stdexcept>

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

}  // namespace

MoleculesHost::MoleculesHost() {
  batchAtomStarts.push_back(0);
  batchBondStarts.push_back(0);
  batchAtomBondStarts.push_back(0);
  batchOtherAtomIndicesStarts.push_back(0);
  batchBondIndicesStarts.push_back(0);
}

void addToBatch(const RDKit::ROMol* mol, MoleculesHost& batch) {
  auto& atomDataVec      = batch.atomData;
  auto& bondDataVec      = batch.bondData;
  auto& atomBondStarts   = batch.atomBondStarts;
  auto& bondDataIndices  = batch.bondDataIndices;
  auto& otherAtomIndices = batch.otherAtomIndices;

  const size_t otherAtomIndicesBefore = otherAtomIndices.size();
  const size_t bondDataIndicesBefore  = bondDataIndices.size();

  atomDataVec.reserve(atomDataVec.size() + mol->getNumAtoms());
  bondDataVec.reserve(bondDataVec.size() + mol->getNumBonds());

  // Populate bond data first for more efficient access, rather than the random order in the atom loop.
  for (unsigned int i = 0; i < mol->getNumBonds(); ++i) {
    auto& bd    = bondDataVec.emplace_back();
    bd.bondType = mol->getBondWithIdx(i)->getBondType();
  }

  const auto* ringInfo = mol->getRingInfo();

  // Push starting 0 for this molecule's prefix sum
  atomBondStarts.push_back(0);
  int cumulativeBondCount = 0;

  for (const RDKit::Atom* atom : mol->atoms()) {
    const unsigned int atomIdx      = atom->getIdx();
    auto&              thisAtomData = atomDataVec.emplace_back();
    populateAtomData(atom, thisAtomData, ringInfo);

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

  // Update batch-level offsets
  batch.batchAtomStarts.push_back(static_cast<int>(atomDataVec.size()));
  batch.batchBondStarts.push_back(static_cast<int>(bondDataVec.size()));
  batch.batchAtomBondStarts.push_back(static_cast<int>(atomBondStarts.size()));
  batch.batchOtherAtomIndicesStarts.push_back(static_cast<int>(otherAtomIndicesBefore));
  batch.batchBondIndicesStarts.push_back(static_cast<int>(bondDataIndicesBefore));
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
  v.atomBondStarts              = atomBondStarts_.data();
  v.otherAtomIndices            = otherAtomIndices_.data();
  v.bondDataIndices             = bondDataIndices_.data();
  v.numMolecules                = numMolecules_;
  return v;
}

}  // namespace nvMolKit
