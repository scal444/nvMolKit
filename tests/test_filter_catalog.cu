// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/FilterCatalog/FilterCatalog.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/substruct/filter_catalog.h"
#include "src/substruct/rdkit_filter_catalog_data.h"

namespace {

std::unique_ptr<RDKit::ROMol> smiles(const std::string& text) {
  std::unique_ptr<RDKit::ROMol> molecule(RDKit::SmilesToMol(text));
  if (molecule == nullptr) {
    throw std::runtime_error("Could not parse test SMILES: " + text);
  }
  return molecule;
}

std::vector<const RDKit::ROMol*> pointers(const std::vector<std::unique_ptr<RDKit::ROMol>>& molecules) {
  std::vector<const RDKit::ROMol*> result;
  result.reserve(molecules.size());
  for (const auto& molecule : molecules) {
    result.push_back(molecule.get());
  }
  return result;
}

TEST(FilterCatalog, RequiresFinalizeAndPublishesPendingEntriesTogether) {
  nvMolKit::FilterCatalog catalog;
  EXPECT_EQ(catalog.addSmarts("N", "nitrogen"), 0u);
  EXPECT_EQ(catalog.size(), 0u);
  EXPECT_EQ(catalog.pendingSize(), 1u);

  auto target = smiles("CN");
  EXPECT_THROW({ [[maybe_unused]] const auto result = catalog.hasMatch({target.get()}); }, std::logic_error);

  catalog.finalize();
  EXPECT_EQ(catalog.size(), 1u);
  EXPECT_EQ(catalog.pendingSize(), 0u);
  EXPECT_EQ(catalog.hasMatch({target.get()}), std::vector<std::uint8_t>({1}));

  EXPECT_EQ(catalog.addSmarts("O", "oxygen"), 1u);
  EXPECT_EQ(catalog.size(), 1u);
  EXPECT_EQ(catalog.pendingSize(), 1u);
  EXPECT_EQ(catalog.getMatches({target.get()}), std::vector<std::vector<unsigned int>>({{0}}));

  catalog.finalize();
  auto oxygen = smiles("CO");
  EXPECT_EQ(catalog.size(), 2u);
  EXPECT_EQ(catalog.getMatches({oxygen.get()}), std::vector<std::vector<unsigned int>>({{1}}));
}

TEST(FilterCatalog, CombinesGpuCandidatesAndCountedFallbackInStableOrder) {
  nvMolKit::FilterCatalog catalog;
  EXPECT_EQ(catalog.addSmarts("C",
                              "two carbons",
                              2,
                              {
                                {"source", "custom"}
  }),
            0u);
  EXPECT_EQ(catalog.addSmarts("N", "nitrogen"), 1u);
  EXPECT_EQ(catalog.addSmarts("O", "oxygen"), 2u);
  catalog.finalize();

  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  targets.push_back(smiles("CN"));
  targets.push_back(smiles("CCN"));
  targets.push_back(smiles("O"));
  targets.push_back(smiles("S"));
  const auto targetPointers = pointers(targets);

  EXPECT_EQ(catalog.hasMatch(targetPointers), std::vector<std::uint8_t>({1, 1, 1, 0}));
  EXPECT_EQ(catalog.getFirstMatch(targetPointers), std::vector<int>({1, 0, 2, -1}));
  EXPECT_EQ(catalog.getMatches(targetPointers),
            std::vector<std::vector<unsigned int>>({
              {1},
              {0, 1},
              {2},
              {}
  }));

  const auto counted = catalog.getEntry(0);
  EXPECT_EQ(counted.id, 0u);
  EXPECT_EQ(counted.description, "two carbons");
  EXPECT_EQ(counted.smarts, "C");
  EXPECT_EQ(counted.triggerCount, 2u);
  EXPECT_EQ(counted.properties.at("source"), "custom");
}

TEST(FilterCatalog, UnsupportedRecursiveQueryUsesRdkitFallback) {
  nvMolKit::FilterCatalog catalog;
  catalog.addSmarts("[$([C]@[C])]", "ring bond recursion");
  catalog.finalize();

  auto ring  = smiles("C1CC1");
  auto chain = smiles("CCC");
  EXPECT_EQ(catalog.hasMatch({ring.get(), chain.get()}), std::vector<std::uint8_t>({1, 0}));
  EXPECT_EQ(catalog.getFirstMatch({ring.get(), chain.get()}), std::vector<int>({0, -1}));
}

TEST(FilterCatalog, OversizedTargetUsesRdkitFallback) {
  nvMolKit::FilterCatalog catalog;
  catalog.addSmarts("C", "carbon");
  catalog.finalize();

  auto target = smiles(std::string(129, 'C'));
  EXPECT_EQ(catalog.hasMatch({target.get()}), std::vector<std::uint8_t>({1}));
  EXPECT_EQ(catalog.getFirstMatch({target.get()}), std::vector<int>({0}));
}

TEST(FilterCatalog, LowestEntryIdIsStableAcrossManyMiniBatches) {
  nvMolKit::SubstructSearchConfig config;
  config.batchSize            = 8;
  config.workerThreads        = 2;
  config.preprocessingThreads = 2;
  nvMolKit::FilterCatalog catalog(config);
  catalog.addSmarts("O", "first");
  for (int index = 0; index < 40; ++index) {
    catalog.addSmarts("N", "nonmatching");
  }
  catalog.addSmarts("[O]", "later");
  catalog.finalize();

  auto target = smiles("O");
  EXPECT_EQ(catalog.getFirstMatch({target.get()}), std::vector<int>({0}));
}

TEST(FilterCatalog, SupportsConcurrentConstQueries) {
  nvMolKit::SubstructSearchConfig config;
  config.workerThreads        = 1;
  config.preprocessingThreads = 1;
  nvMolKit::FilterCatalog catalog(config);
  catalog.addSmarts("N", "nitrogen");
  catalog.addSmarts("O", "oxygen");
  catalog.finalize();

  auto                                   nitrogen = smiles("CN");
  auto                                   oxygen   = smiles("CO");
  const std::vector<const RDKit::ROMol*> targets{nitrogen.get(), oxygen.get()};
  auto first = std::async(std::launch::async, [&] { return catalog.getFirstMatch(targets); });
  auto any   = std::async(std::launch::async, [&] { return catalog.hasMatch(targets); });
  EXPECT_EQ(first.get(), std::vector<int>({0, 1}));
  EXPECT_EQ(any.get(), std::vector<std::uint8_t>({1, 1}));
}

TEST(FilterCatalog, ValidatesConstructionAndInputs) {
  nvMolKit::FilterCatalog catalog;
  auto                    carbon = smiles("C");
  EXPECT_THROW(catalog.addEntry(*carbon, "invalid", 0), std::invalid_argument);
  EXPECT_THROW(catalog.addSmarts("[", "invalid"), std::invalid_argument);
  EXPECT_THROW(catalog.addPreset(static_cast<RDKit::FilterCatalogParams::FilterCatalogs>(1u << 30)),
               std::invalid_argument);
  EXPECT_THROW({ [[maybe_unused]] const auto result = catalog.getEntry(0); }, std::out_of_range);

  catalog.finalize();
  EXPECT_EQ(catalog.hasMatch({}), std::vector<std::uint8_t>{});
  EXPECT_EQ(catalog.getFirstMatch({}), std::vector<int>{});
  EXPECT_EQ(catalog.getMatches({}), std::vector<std::vector<unsigned int>>{});
  EXPECT_THROW({ [[maybe_unused]] const auto result = catalog.hasMatch({nullptr}); }, std::invalid_argument);

  nvMolKit::SubstructSearchConfig config;
  config.algorithm = nvMolKit::SubstructAlgorithm::VF2;
  EXPECT_THROW((void)nvMolKit::FilterCatalog{config}, std::invalid_argument);
}

TEST(FilterCatalog, ImportsOfficialCompositePresetsWithoutCopyingData) {
  nvMolKit::FilterCatalog catalog(RDKit::FilterCatalogParams::PAINS);
  const auto expected = static_cast<std::size_t>(RDKit::GetNumEntries(RDKit::FilterCatalogParams::PAINS_A)) +
                        RDKit::GetNumEntries(RDKit::FilterCatalogParams::PAINS_B) +
                        RDKit::GetNumEntries(RDKit::FilterCatalogParams::PAINS_C);
  EXPECT_EQ(catalog.pendingSize(), expected);

  const auto  first = catalog.getEntry(0);
  const auto* data  = RDKit::GetFilterData(RDKit::FilterCatalogParams::PAINS_A);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(first.description, data[0].name);
  EXPECT_EQ(first.smarts, data[0].smarts);
  EXPECT_EQ(first.triggerCount, data[0].max == 0 ? 1u : data[0].max + 1);
  EXPECT_EQ(first.properties.at("FilterSet"), "PAINS_A");
}

TEST(FilterCatalog, BuiltinMatchBehaviorAgreesWithRdkit) {
  nvMolKit::FilterCatalog catalog(RDKit::FilterCatalogParams::PAINS_A);
  catalog.finalize();
  RDKit::FilterCatalog reference(RDKit::FilterCatalogParams::PAINS_A);

  std::vector<std::unique_ptr<RDKit::ROMol>> targets;
  targets.push_back(smiles("CCO"));
  targets.push_back(smiles("O=C1C=CC(=O)C=C1"));
  targets.push_back(smiles("c1ccccc1N=Nc1ccccc1"));
  const auto targetPointers = pointers(targets);
  const auto actual         = catalog.hasMatch(targetPointers);
  ASSERT_EQ(actual.size(), targets.size());
  for (std::size_t index = 0; index < targets.size(); ++index) {
    EXPECT_EQ(actual[index] != 0, reference.hasMatch(*targets[index]));
  }
}

TEST(FilterCatalog, BuiltinExplicitHydrogenSmartsUsesRdkitParserMode) {
  nvMolKit::FilterCatalog catalog(RDKit::FilterCatalogParams::CHEMBL_BMS);
  catalog.finalize();
  RDKit::FilterCatalog reference(RDKit::FilterCatalogParams::CHEMBL_BMS);
  auto                 target = smiles("CC(=O)c1ncccc1");

  std::vector<unsigned int> expected;
  for (unsigned int index = 0; index < reference.getNumEntries(); ++index) {
    if (reference.getEntry(index)->hasFilterMatch(*target)) {
      expected.push_back(index);
    }
  }
  EXPECT_NE(std::find(expected.begin(), expected.end(), 98u), expected.end());
  EXPECT_EQ(catalog.getMatches({target.get()}), std::vector<std::vector<unsigned int>>({expected}));
}

TEST(FilterCatalog, FinalizesAndQueriesOnExternalStream) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  {
    nvMolKit::FilterCatalog catalog;
    catalog.addSmarts("N", "nitrogen");
    catalog.finalize(stream);
    auto target = smiles("CN");
    EXPECT_EQ(catalog.getFirstMatch({target.get()}, stream), std::vector<int>({0}));
  }

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(FilterCatalog, FailedFinalizePreservesPublishedGeneration) {
  nvMolKit::FilterCatalog catalog;
  catalog.addSmarts("N", "nitrogen");
  catalog.finalize();
  catalog.addSmarts("O", "oxygen");

  cudaStream_t invalidStream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&invalidStream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(invalidStream), cudaSuccess);
  EXPECT_THROW(catalog.finalize(invalidStream), std::invalid_argument);
  EXPECT_EQ(catalog.size(), 1u);
  EXPECT_EQ(catalog.pendingSize(), 1u);

  auto nitrogen = smiles("CN");
  auto oxygen   = smiles("CO");
  EXPECT_EQ(catalog.getFirstMatch({nitrogen.get(), oxygen.get()}), std::vector<int>({0, -1}));
  catalog.finalize();
  EXPECT_EQ(catalog.size(), 2u);
}

}  // namespace
