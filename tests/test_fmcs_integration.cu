// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "src/mcs/mcs_search.h"
#include "src/testutils/mol_data.h"
#include "tests/test_utils.h"

#include <GraphMol/FMCS/FMCS.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using nvMolKit::MCSAtomCompare;
using nvMolKit::MCSBondCompare;
using nvMolKit::MCSPair;
using nvMolKit::MCSParameters;
using nvMolKit::testing::readSmilesFileWithStrings;

constexpr size_t kMaxAtoms                = 24;
constexpr size_t kDefaultDatasetMolecules = 48;
constexpr size_t kDefaultRandomPairs      = 1;
constexpr unsigned int kDefaultSeed       = 1337;

struct RingConfig {
  bool atomRingMatchesRingOnly = false;
  bool bondRingMatchesRingOnly = false;
  const char* name             = "NoRing";
};

using FmcsIntegrationParams = std::tuple<MCSAtomCompare, MCSBondCompare, RingConfig>;

unsigned int integrationSeed() {
  if (const char* seedStr = std::getenv("NVMOLKIT_MCS_TEST_SEED")) {
    try {
      return static_cast<unsigned int>(std::stoul(seedStr));
    } catch (const std::exception&) {
    }
  }
  return kDefaultSeed;
}

size_t envSize(const char* name, size_t defaultValue) {
  if (const char* value = std::getenv(name)) {
    try {
      return std::max<size_t>(1, std::stoull(value));
    } catch (const std::exception&) {
    }
  }
  return defaultValue;
}

const char* atomCompareName(MCSAtomCompare compare) {
  switch (compare) {
    case MCSAtomCompare::Any:
      return "AtomAny";
    case MCSAtomCompare::Elements:
      return "AtomElements";
    case MCSAtomCompare::Isotopes:
      return "AtomIsotopes";
    case MCSAtomCompare::AnyHeavyAtom:
      return "AtomAnyHeavy";
  }
  return "AtomUnknown";
}

const char* bondCompareName(MCSBondCompare compare) {
  switch (compare) {
    case MCSBondCompare::Any:
      return "BondAny";
    case MCSBondCompare::Order:
      return "BondOrder";
    case MCSBondCompare::OrderExact:
      return "BondOrderExact";
  }
  return "BondUnknown";
}

MCSParameters makeParams(MCSAtomCompare atomCompare, MCSBondCompare bondCompare, RingConfig ringConfig) {
  MCSParameters params;
  params.atomCompare                                           = atomCompare;
  params.bondCompare                                           = bondCompare;
  params.atomCompareParameters.ringMatchesRingOnly             = ringConfig.atomRingMatchesRingOnly;
  params.bondCompareParameters.ringMatchesRingOnly             = ringConfig.bondRingMatchesRingOnly;
  return params;
}

RDKit::MCSParameters makeRdkitParams(const MCSParameters& params) {
  RDKit::MCSParameters rdParams;
  rdParams.MaximizeBonds = params.maximizeBonds;
  rdParams.Timeout       = params.timeoutSeconds;
  rdParams.AtomCompareParameters.MatchValences       = params.atomCompareParameters.matchValences;
  rdParams.AtomCompareParameters.MatchFormalCharge   = params.atomCompareParameters.matchFormalCharge;
  rdParams.AtomCompareParameters.RingMatchesRingOnly = params.atomCompareParameters.ringMatchesRingOnly;
  rdParams.AtomCompareParameters.CompleteRingsOnly   = params.atomCompareParameters.completeRingsOnly;
  rdParams.AtomCompareParameters.MatchIsotope        = params.atomCompareParameters.matchIsotope;
  rdParams.BondCompareParameters.RingMatchesRingOnly = params.bondCompareParameters.ringMatchesRingOnly;
  rdParams.BondCompareParameters.CompleteRingsOnly   = params.bondCompareParameters.completeRingsOnly;

  switch (params.atomCompare) {
    case MCSAtomCompare::Any:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareAny);
      break;
    case MCSAtomCompare::Elements:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareElements);
      break;
    case MCSAtomCompare::Isotopes:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareIsotopes);
      break;
    case MCSAtomCompare::AnyHeavyAtom:
      rdParams.setMCSAtomTyperFromEnum(RDKit::AtomCompareAnyHeavyAtom);
      break;
  }

  switch (params.bondCompare) {
    case MCSBondCompare::Any:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareAny);
      break;
    case MCSBondCompare::Order:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareOrder);
      break;
    case MCSBondCompare::OrderExact:
      rdParams.setMCSBondTyperFromEnum(RDKit::BondCompareOrderExact);
      break;
  }

  return rdParams;
}

RDKit::MCSResult findRdkitMCS(const RDKit::ROMol& molA, const RDKit::ROMol& molB, const MCSParameters& params) {
  std::vector<RDKit::ROMOL_SPTR> mols;
  mols.emplace_back(new RDKit::ROMol(molA));
  mols.emplace_back(new RDKit::ROMol(molB));
  auto rdParams = makeRdkitParams(params);
  return RDKit::findMCS(mols, &rdParams);
}

bool mappingHasUniqueAtoms(const std::vector<std::pair<int, int>>& mapping) {
  std::set<int> atomsA;
  std::set<int> atomsB;
  for (const auto& [a, b] : mapping) {
    if (!atomsA.insert(a).second || !atomsB.insert(b).second) {
      return false;
    }
  }
  return true;
}

bool mappingMatchesRdkitMCS(const std::vector<std::pair<int, int>>& gpuMapping,
                            const RDKit::ROMol&                     molA,
                            const RDKit::ROMol&                     molB,
                            const RDKit::MCSResult&                 rdkitMCS) {
  if (gpuMapping.empty()) {
    return rdkitMCS.NumAtoms == 0;
  }
  if (rdkitMCS.QueryMol == nullptr || gpuMapping.size() != rdkitMCS.NumAtoms || !mappingHasUniqueAtoms(gpuMapping)) {
    return false;
  }

  RDKit::SubstructMatchParameters params;
  params.uniquify   = false;
  params.maxMatches = 0;
  const auto matchesA = RDKit::SubstructMatch(molA, *rdkitMCS.QueryMol, params);
  const auto matchesB = RDKit::SubstructMatch(molB, *rdkitMCS.QueryMol, params);

  for (const auto& matchA : matchesA) {
    std::vector<int> atomAToQuery(molA.getNumAtoms(), -1);
    for (const auto& [queryAtomIdx, targetAtomIdx] : matchA) {
      atomAToQuery[static_cast<size_t>(targetAtomIdx)] = queryAtomIdx;
    }

    std::vector<int> requiredB(rdkitMCS.QueryMol->getNumAtoms(), -1);
    bool compatibleWithA = true;
    for (const auto& [gpuAtomA, gpuAtomB] : gpuMapping) {
      if (gpuAtomA < 0 || gpuAtomA >= static_cast<int>(atomAToQuery.size())) {
        compatibleWithA = false;
        break;
      }
      const int queryIdx = atomAToQuery[static_cast<size_t>(gpuAtomA)];
      if (queryIdx < 0) {
        compatibleWithA = false;
        break;
      }
      if (requiredB[static_cast<size_t>(queryIdx)] >= 0 && requiredB[static_cast<size_t>(queryIdx)] != gpuAtomB) {
        compatibleWithA = false;
        break;
      }
      requiredB[static_cast<size_t>(queryIdx)] = gpuAtomB;
    }
    if (!compatibleWithA) {
      continue;
    }

    for (const auto& matchB : matchesB) {
      bool compatibleWithB = true;
      for (const auto& [queryAtomIdx, targetAtomIdx] : matchB) {
        if (requiredB[static_cast<size_t>(queryAtomIdx)] >= 0 &&
            requiredB[static_cast<size_t>(queryAtomIdx)] != targetAtomIdx) {
          compatibleWithB = false;
          break;
        }
      }
      if (compatibleWithB) {
        return true;
      }
    }
  }

  return false;
}

void printMismatch(const std::vector<std::string>& smiles,
                   int                             pairIdx,
                   int                             idxA,
                   int                             idxB,
                   const nvMolKit::MCSResult&      gpu,
                   const RDKit::MCSResult&         rd) {
  std::cout << "\n[fMCS mismatch] pair=" << pairIdx << " a=" << idxA << " b=" << idxB << "\n";
  std::cout << "  A: " << smiles[static_cast<size_t>(idxA)] << "\n";
  std::cout << "  B: " << smiles[static_cast<size_t>(idxB)] << "\n";
  std::cout << "  GPU atoms/bonds: " << gpu.numAtoms << "/" << gpu.numBonds << " overflowed=" << gpu.overflowed
            << " fallback=" << gpu.usedFallback << "\n";
  std::cout << "  RDKit atoms/bonds: " << rd.NumAtoms << "/" << rd.NumBonds << " canceled=" << rd.Canceled << "\n";
}

struct Dataset {
  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  std::vector<std::string> smiles;
};

Dataset loadDataset() {
  const std::string smilesPath = getTestDataFolderPath() + "/chembl_1k.smi";
  if (!std::filesystem::exists(smilesPath)) {
    throw std::runtime_error("SMILES file not found: " + smilesPath);
  }
  auto [mols, smiles] =
    readSmilesFileWithStrings(smilesPath, envSize("NVMOLKIT_MCS_TEST_MOLECULES", kDefaultDatasetMolecules), kMaxAtoms);
  if (mols.empty()) {
    throw std::runtime_error("No molecules loaded from " + smilesPath);
  }
  return Dataset{std::move(mols), std::move(smiles)};
}

const Dataset& dataset() {
  static const Dataset data = loadDataset();
  return data;
}

std::vector<const RDKit::ROMol*> moleculeTable(const Dataset& data, size_t maxMols = 0) {
  const size_t count = maxMols == 0 ? data.mols.size() : std::min(maxMols, data.mols.size());
  std::vector<const RDKit::ROMol*> mols;
  mols.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    mols.push_back(data.mols[i].get());
  }
  return mols;
}

void expectSameResultShape(const std::vector<nvMolKit::MCSResult>& expected,
                           const std::vector<nvMolKit::MCSResult>& actual) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i].numAtoms, expected[i].numAtoms) << "result " << i;
    EXPECT_EQ(actual[i].numBonds, expected[i].numBonds) << "result " << i;
    EXPECT_EQ(actual[i].usedGpu, expected[i].usedGpu) << "result " << i;
    EXPECT_EQ(actual[i].usedFallback, expected[i].usedFallback) << "result " << i;
    EXPECT_EQ(actual[i].overflowed, expected[i].overflowed) << "result " << i;
  }
}

class FMCSIntegrationTest : public ::testing::TestWithParam<FmcsIntegrationParams> {};

TEST_P(FMCSIntegrationTest, SeededChemblPairsMatchRDKit) {
  const auto& data = dataset();

  const auto atomCompare = std::get<0>(GetParam());
  const auto bondCompare = std::get<1>(GetParam());
  const auto ringConfig  = std::get<2>(GetParam());
  const auto params      = makeParams(atomCompare, bondCompare, ringConfig);

  const unsigned int seed = integrationSeed() ^ (static_cast<unsigned int>(atomCompare) << 8) ^
                            (static_cast<unsigned int>(bondCompare) << 16) ^
                            (ringConfig.atomRingMatchesRingOnly ? 0x10000u : 0u) ^
                            (ringConfig.bondRingMatchesRingOnly ? 0x20000u : 0u);
  const size_t numPairs = envSize("NVMOLKIT_MCS_TEST_PAIRS", kDefaultRandomPairs);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, static_cast<int>(data.mols.size() - 1));

  const auto mols = moleculeTable(data);
  std::vector<int> indicesA;
  std::vector<int> indicesB;
  std::vector<MCSPair> pairs;
  indicesA.reserve(numPairs);
  indicesB.reserve(numPairs);
  pairs.reserve(numPairs);

  for (size_t i = 0; i < numPairs; ++i) {
    const int idxA = dist(rng);
    const int idxB = dist(rng);
    indicesA.push_back(idxA);
    indicesB.push_back(idxB);
    pairs.emplace_back(static_cast<size_t>(idxA), static_cast<size_t>(idxB));
  }

  auto gpuResults = nvMolKit::findMCSBatch(mols, pairs, nullptr, params);
  ASSERT_EQ(gpuResults.size(), numPairs);

  for (size_t i = 0; i < numPairs; ++i) {
    const auto& molA = *data.mols[static_cast<size_t>(indicesA[i])];
    const auto& molB = *data.mols[static_cast<size_t>(indicesB[i])];
    const auto rd    = findRdkitMCS(molA, molB, params);
    const auto& gpu  = gpuResults[i];

    const bool sizeMatches = gpu.numAtoms == rd.NumAtoms && gpu.numBonds == rd.NumBonds;
    const bool mappingMatches = sizeMatches ? mappingMatchesRdkitMCS(gpu.atomMapping, molA, molB, rd) : false;
    if (!sizeMatches || !mappingMatches) {
      printMismatch(data.smiles, static_cast<int>(i), indicesA[i], indicesB[i], gpu, rd);
      std::cout << "  seed=" << seed << " atomCompare=" << atomCompareName(atomCompare)
                << " bondCompare=" << bondCompareName(bondCompare) << " ringConfig=" << ringConfig.name << "\n";
    }
    EXPECT_TRUE(sizeMatches);
    EXPECT_TRUE(mappingMatches);
  }
}

TEST(FMCSDispatchRoutes, EqualListsWrapperMatchesExplicitPairs) {
  const auto& data = dataset();
  ASSERT_GE(data.mols.size(), 3);

  std::vector<const RDKit::ROMol*> molsA = {data.mols[0].get(), data.mols[1].get(), data.mols[2].get()};
  std::vector<const RDKit::ROMol*> molsB = {data.mols[1].get(), data.mols[2].get(), data.mols[0].get()};
  std::vector<const RDKit::ROMol*> combined;
  combined.reserve(molsA.size() + molsB.size());
  combined.insert(combined.end(), molsA.begin(), molsA.end());
  combined.insert(combined.end(), molsB.begin(), molsB.end());

  std::vector<MCSPair> pairs;
  pairs.reserve(molsA.size());
  for (size_t i = 0; i < molsA.size(); ++i) {
    pairs.emplace_back(i, molsA.size() + i);
  }

  MCSParameters params;
  const auto explicitResults = nvMolKit::findMCSBatch(combined, pairs, nullptr, params);
  const auto wrapperResults  = nvMolKit::findMCSBatch(molsA, molsB, nullptr, params);
  expectSameResultShape(explicitResults, wrapperResults);
}

TEST(FMCSDispatchRoutes, AllPairsUpperTriangleWrapperMatchesExplicitPairs) {
  const auto& data = dataset();
  ASSERT_GE(data.mols.size(), 3);

  const auto mols = moleculeTable(data, 3);
  const std::vector<MCSPair> pairs = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

  MCSParameters params;
  const auto explicitResults = nvMolKit::findMCSBatch(mols, pairs, nullptr, params);
  const auto wrapperResults =
    nvMolKit::findMCSAllPairs(mols, nvMolKit::MCSAllPairsOptions{}, nullptr, params);
  expectSameResultShape(explicitResults, wrapperResults);
}

constexpr RingConfig kNoRing{false, false, "NoRing"};

std::string integrationParamName(const ::testing::TestParamInfo<FmcsIntegrationParams>& info) {
  return std::string(atomCompareName(std::get<0>(info.param))) + "_" + bondCompareName(std::get<1>(info.param)) +
         "_" + std::get<2>(info.param).name;
}

INSTANTIATE_TEST_SUITE_P(AtomBondCompareSmoke,
                         FMCSIntegrationTest,
                         ::testing::Combine(::testing::Values(MCSAtomCompare::Any,
                                                              MCSAtomCompare::Elements,
                                                              MCSAtomCompare::Isotopes),
                                            ::testing::Values(MCSBondCompare::Any,
                                                              MCSBondCompare::Order,
                                                              MCSBondCompare::OrderExact),
                                            ::testing::Values(kNoRing)),
                         integrationParamName);

INSTANTIATE_TEST_SUITE_P(
  RingCompareSmoke,
  FMCSIntegrationTest,
  ::testing::Values(FmcsIntegrationParams{MCSAtomCompare::Elements, MCSBondCompare::Order, RingConfig{true,
                                                                                                      false,
                                                                                                      "AtomRing"}},
                    FmcsIntegrationParams{MCSAtomCompare::Elements, MCSBondCompare::Order, RingConfig{false,
                                                                                                      true,
                                                                                                      "BondRing"}},
                    FmcsIntegrationParams{MCSAtomCompare::Elements, MCSBondCompare::Order, RingConfig{true,
                                                                                                      true,
                                                                                                      "AtomBondRing"}}),
  integrationParamName);

}  // namespace
