#include "molecules.h"

#include <GraphMol/ROMol.h>

namespace nvMolKit {
namespace {
void populateAtomData(const RDKit::Atom* atom, AtomData& atomData, const RDKit::RingInfo* ringInfo) {
  atomData.atomicNum = atom->getAtomicNum();
  atomData.chiralTag = atom->getChiralTag();
  atomData.explicitValence = atom->getValence(RDKit::Atom::ValenceType::EXPLICIT);
  atomData.implicitValence = atom->getValence(RDKit::Atom::ValenceType::IMPLICIT);
  atomData.formalCharge = atom->getFormalCharge();
  atomData.hybridization = atom->getHybridization();
  atomData.isAromatic = atom->getIsAromatic();
  atomData.numRadicalElectrons = atom->getNumRadicalElectrons();
  const int idx = atom->getIdx();
  atomData.numRings = ringInfo->numAtomRings(idx);
  atomData.minRingSize = ringInfo->minAtomRingSize(idx);
}
}


void addToBatch(const RDKit::ROMol* mol, MoleculesHost& batch) {
  const int lastAtomStart = batch.batchAtomStarts.back();
  const int lastBondStart = batch.batchBondStarts.back();
  const int lastAtomBondStarts = batch.atomBondStarts.back();
  const int lastOtherAtomIndiciesStarts = batch.batchOtherAtomIndicesStarts.back();
  const int lastBondIndicesStart = batch.batchBondIndicesStarts.back();

  auto& atomData = batch.atomData;
  auto& bondData = batch.bondData;
  auto& atomBondStarts = batch.atomBondStarts;
  auto& bondDataIndices = batch.bondDataIndices;
  auto& otherAtomIndices = batch.otherAtomIndices;

  atomData.reserve(atomData.size() + mol->getNumAtoms());
  bondData.reserve(bondData.size() + mol->getNumBonds());

  // First populate bond data, more efficient access than the random order
  // in the atom loop.
  for (int i = 0; i < mol->getNumBonds(); ++i) {
    auto& bd = bondData.emplace_back();
    bd.bondType = mol->getBondWithIdx(i)->getBondType();
  }

  const auto* ringInfo = mol->getRingInfo();
  for (const RDKit::Atom* atom : mol->atoms()) {
    const uint32_t atomIdx = atom->getIdx();
    auto& thisAtomData = atomData.emplace_back();
    populateAtomData(atom, thisAtomData, ringInfo);
    auto [beg, bondEnd] = mol->getAtomBonds(atom);
    int numBondsThisAtom = 0;
    while (beg != bondEnd) {
      numBondsThisAtom++;
      const auto* bond = (*mol)[*beg];
      const uint32_t idx = bond->getIdx();

      const int otherAtomIdx = bond->getOtherAtomIdx(atomIdx);

      otherAtomIndices.push_back(otherAtomIdx);
      bondDataIndices.push_back(idx);

    }
    atomBondStarts.push_back(numBondsThisAtom);

  }

  // increment all the batch variables
}
} // namespace nvMolKit
