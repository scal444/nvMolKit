// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <DataStructs/BitOps.h>
#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/Fingerprints/Fingerprints.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/substruct/pattern_screen.h"
#include "src/substruct/target_chunk.h"
#include "src/testutils/mol_data.h"
#include "src/utils/device.h"
#include "tests/test_utils.h"

namespace {

using nvMolKit::kPatternFingerprintBits;
using nvMolKit::kPatternFingerprintWords;

std::vector<std::unique_ptr<RDKit::ROMol>> chemblTargets(std::size_t count) {
  return nvMolKit::testing::readSmilesFile(getTestDataFolderPath() + "/chembl_1k.smi", count, 128);
}

std::unique_ptr<ExplicitBitVect> patternFingerprint(const RDKit::ROMol& mol) {
  return std::unique_ptr<ExplicitBitVect>(RDKit::PatternFingerprintMol(mol));
}

TEST(PatternScreenHost, PacksFingerprintsWordMajorAndTransposesThemToBitSlices) {
  // 70 targets span three 32-target slice words, the last one partial.
  const auto targets = chemblTargets(70);
  ASSERT_EQ(targets.size(), 70U);
  std::vector<std::unique_ptr<ExplicitBitVect>> fingerprints;
  std::vector<const ExplicitBitVect*>           fingerprintPtrs;
  for (const auto& target : targets) {
    fingerprints.push_back(patternFingerprint(*target));
    fingerprintPtrs.push_back(fingerprints.back().get());
  }

  const std::size_t count       = targets.size();
  const auto        words       = nvMolKit::packPatternFingerprintsWordMajor(fingerprintPtrs);
  const auto        slices      = nvMolKit::buildPatternBitSlices(words, count);
  const auto        frequencies = nvMolKit::patternBitFrequencies(slices, count);
  const std::size_t sliceWords  = nvMolKit::patternSliceWords(count);
  ASSERT_EQ(words.size(), count * kPatternFingerprintWords);
  ASSERT_EQ(sliceWords, 3U);
  ASSERT_EQ(slices.size(), kPatternFingerprintBits * sliceWords);
  ASSERT_EQ(frequencies.size(), static_cast<std::size_t>(kPatternFingerprintBits));

  for (int bit = 0; bit < kPatternFingerprintBits; ++bit) {
    std::uint32_t carriers = 0;
    for (std::size_t target = 0; target < count; ++target) {
      const bool expected = fingerprints[target]->getBit(static_cast<unsigned int>(bit));
      const auto word     = words[static_cast<std::size_t>(bit / 64) * count + target];
      const auto slice    = slices[static_cast<std::size_t>(bit) * sliceWords + target / 32];
      ASSERT_EQ(((word >> (bit % 64)) & 1U) != 0, expected) << "bit " << bit << ", target " << target;
      ASSERT_EQ(((slice >> (target % 32)) & 1U) != 0, expected) << "bit " << bit << ", target " << target;
      carriers += expected ? 1U : 0U;
    }
    ASSERT_EQ(frequencies[static_cast<std::size_t>(bit)], carriers) << "bit " << bit;
  }
}

TEST(PatternScreenHost, QueryBitsAreTheFingerprintBitsOrderedRarestFirst) {
  const auto query       = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol("c1ccccc1C(=O)N"));
  const auto fingerprint = patternFingerprint(*query);
  auto       screenQuery = nvMolKit::makePatternScreenQuery(fingerprint.get(), static_cast<int>(query->getNumAtoms()));

  std::vector<std::uint16_t> onBits;
  for (unsigned int bit = 0; bit < fingerprint->getNumBits(); ++bit) {
    if (fingerprint->getBit(bit)) {
      onBits.push_back(static_cast<std::uint16_t>(bit));
    }
  }
  ASSERT_FALSE(onBits.empty());
  EXPECT_EQ(screenQuery.bits, onBits);
  EXPECT_EQ(screenQuery.numAtoms, 9);

  // Frequencies cycle so some bits tie; ties keep ascending bit order.
  std::vector<std::uint32_t> frequencies(kPatternFingerprintBits);
  for (int bit = 0; bit < kPatternFingerprintBits; ++bit) {
    frequencies[static_cast<std::size_t>(bit)] = static_cast<std::uint32_t>((bit * 7) % 5);
  }
  nvMolKit::orderPatternScreenBits(screenQuery, frequencies);
  ASSERT_EQ(screenQuery.bits.size(), onBits.size());
  EXPECT_TRUE(std::is_permutation(screenQuery.bits.begin(), screenQuery.bits.end(), onBits.begin()));
  for (std::size_t index = 1; index < screenQuery.bits.size(); ++index) {
    const auto previous = screenQuery.bits[index - 1];
    const auto current  = screenQuery.bits[index];
    ASSERT_TRUE(frequencies[previous] < frequencies[current] ||
                (frequencies[previous] == frequencies[current] && previous < current));
  }

  EXPECT_TRUE(nvMolKit::makePatternScreenQuery(nullptr, 4).bits.empty());
}

TEST(PatternScreenDevice, KeepsExactlyTheTargetsThatCanContainTheQuery) {
  // Two generations exercise extending a device target set from its base.
  const auto targets = chemblTargets(200);
  ASSERT_EQ(targets.size(), 200U);
  std::vector<const RDKit::ROMol*> firstHalf;
  std::vector<const RDKit::ROMol*> secondHalf;
  for (std::size_t index = 0; index < targets.size(); ++index) {
    (index < 90 ? firstHalf : secondHalf).push_back(targets[index].get());
  }
  // A fallback target mid-chunk leaves a gap between molecule IDs and packed fingerprint positions.
  const auto fallbackTarget = std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol("[NH3]->[Cu]"));
  firstHalf.insert(firstHalf.begin() + 45, fallbackTarget.get());
  nvMolKit::TargetChunkBuilder firstBuilder(0, firstHalf.size());
  firstBuilder.addMols(firstHalf, 2);
  const auto                   firstChunk = firstBuilder.seal();
  nvMolKit::TargetChunkBuilder secondBuilder(firstHalf.size(), secondHalf.size());
  secondBuilder.addMols(secondHalf, 2);
  const auto secondChunk = secondBuilder.seal();

  nvMolKit::ScopedStream    stream("PatternScreenDevice");
  nvMolKit::DeviceTargetSet base(nullptr, {firstChunk.get()}, true);
  base.upload(stream.stream());
  nvMolKit::DeviceTargetSet extended(&base, {secondChunk.get()}, true);
  extended.upload(stream.stream());
  ASSERT_EQ(firstChunk->fallbackCount(), 1U);
  ASSERT_EQ(extended.size(), firstChunk->gpuTargetCount() + secondChunk->gpuTargetCount());

  int deviceId = 0;
  ASSERT_EQ(cudaGetDevice(&deviceId), cudaSuccess);
  nvMolKit::PatternScreenWorkspace workspace(deviceId);
  const int                        numTargets = static_cast<int>(extended.size());

  // Screen the smaller base set first so the workspace has to grow for the extended set.
  const auto ring = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol("c1ccccc1"));
  workspace.screen(base.devicePatternSlices(),
                   base.deviceView().batchAtomStarts,
                   static_cast<int>(base.size()),
                   nvMolKit::makePatternScreenQuery(patternFingerprint(*ring).get(), 6));

  for (const char* smarts :
       {"c1ccccc1", "[OX2H]", "C(=O)N", "[#7;R]", "S(=O)(=O)", "[Cl,Br]", "CCCCCCCCCCCC", "[$(C=O)]", "[Si]"}) {
    const auto query       = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
    const auto fingerprint = patternFingerprint(*query);
    const int  queryAtoms  = static_cast<int>(query->getNumAtoms());
    auto       screenQuery = nvMolKit::makePatternScreenQuery(fingerprint.get(), queryAtoms);
    nvMolKit::orderPatternScreenBits(screenQuery, extended.patternBitFrequencies());
    workspace.screen(extended.devicePatternSlices(), extended.deviceView().batchAtomStarts, numTargets, screenQuery);
    const std::vector<int> survivors(workspace.indices(), workspace.indices() + workspace.count());

    std::vector<int> expected;
    for (int target = 0; target < numTargets; ++target) {
      const auto& mol = *extended.targets()[static_cast<std::size_t>(target)];
      if (static_cast<int>(mol.getNumAtoms()) >= queryAtoms &&
          AllProbeBitsMatch(*fingerprint, *patternFingerprint(mol))) {
        expected.push_back(target);
      }
      RDKit::MatchVectType match;
      if (RDKit::SubstructMatch(mol, *query, match)) {
        EXPECT_TRUE(std::binary_search(survivors.begin(), survivors.end(), target))
          << smarts << " rejected matching target " << target;
      }
    }
    EXPECT_EQ(survivors, expected) << smarts;
  }

  // A query larger than every packed target leaves no survivors.
  const auto chain = std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(std::string(129, 'C')));
  workspace.screen(extended.devicePatternSlices(),
                   extended.deviceView().batchAtomStarts,
                   numTargets,
                   nvMolKit::makePatternScreenQuery(patternFingerprint(*chain).get(), 129));
  EXPECT_EQ(workspace.count(), 0);

  // Without fingerprints the screen keeps every target with enough atoms.
  workspace.screen(nullptr,
                   extended.deviceView().batchAtomStarts,
                   numTargets,
                   nvMolKit::makePatternScreenQuery(nullptr, 30));
  std::vector<int> largeEnough;
  for (int target = 0; target < numTargets; ++target) {
    if (extended.targets()[static_cast<std::size_t>(target)]->getNumAtoms() >= 30U) {
      largeEnough.push_back(target);
    }
  }
  EXPECT_EQ(std::vector<int>(workspace.indices(), workspace.indices() + workspace.count()), largeEnough);
}

}  // namespace
