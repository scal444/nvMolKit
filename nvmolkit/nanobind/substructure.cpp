// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/ROMol.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/substruct/substruct_types.h"
#include "src/utils/nvtx.h"

using cudaStream_t = struct CUstream_st*;

namespace nvMolKit {

void getSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                         const std::vector<const RDKit::ROMol*>& queries,
                         SubstructSearchResults&                 results,
                         SubstructAlgorithm                      algorithm,
                         cudaStream_t                            stream,
                         const SubstructSearchConfig&            config);

void countSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                           const std::vector<const RDKit::ROMol*>& queries,
                           std::vector<int>&                       counts,
                           SubstructAlgorithm                      algorithm,
                           cudaStream_t                            stream,
                           const SubstructSearchConfig&            config);

void hasSubstructMatch(const std::vector<const RDKit::ROMol*>& targets,
                       const std::vector<const RDKit::ROMol*>& queries,
                       HasSubstructMatchResults&               results,
                       SubstructAlgorithm                      algorithm,
                       cudaStream_t                            stream,
                       const SubstructSearchConfig&            config);

}  // namespace nvMolKit

namespace {

namespace nb = nanobind;
using namespace nb::literals;

struct SubstructMatchesCSR {
  std::vector<std::int32_t> atomIndices;
  std::vector<std::int32_t> matchIndptr;
  std::vector<std::int32_t> pairIndptr;
  int                       numTargets = 0;
  int                       numQueries = 0;
};

std::vector<const RDKit::ROMol*> moleculesFromList(const nb::list& molecules, const char* kind) {
  std::vector<const RDKit::ROMol*> converted;
  converted.reserve(molecules.size());
  for (std::size_t index = 0; index < molecules.size(); ++index) {
    const RDKit::ROMol* molecule = nb::cast<const RDKit::ROMol*>(molecules[index]);
    if (molecule == nullptr) {
      throw std::invalid_argument("Invalid " + std::string(kind) + " molecule at index " + std::to_string(index));
    }
    converted.push_back(molecule);
  }
  return converted;
}

std::vector<int> integersFromIterable(const nb::iterable& iterable) {
  std::vector<int> converted;
  for (const nb::handle item : iterable) {
    converted.push_back(nb::cast<int>(item));
  }
  return converted;
}

std::string getAlgorithm(const nvMolKit::SubstructSearchConfig& config) {
  switch (config.algorithm) {
    case nvMolKit::SubstructAlgorithm::GSI:
      return "gsi";
    case nvMolKit::SubstructAlgorithm::DFS:
      return "dfs";
    case nvMolKit::SubstructAlgorithm::VF2:
      throw std::invalid_argument("VF2 is not supported by the Python substructure bindings");
  }
  throw std::invalid_argument("Unknown substructure algorithm");
}

void setAlgorithm(nvMolKit::SubstructSearchConfig& config, const std::string& algorithm) {
  if (algorithm == "gsi") {
    config.algorithm = nvMolKit::SubstructAlgorithm::GSI;
  } else if (algorithm == "dfs") {
    config.algorithm = nvMolKit::SubstructAlgorithm::DFS;
  } else {
    throw std::invalid_argument("algorithm must be 'gsi' or 'dfs'");
  }
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> makeOneDimensionalArray(std::vector<T>& values, const nb::handle& owner) {
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(values.data(), {values.size()}, owner);
}

nb::tuple getSubstructMatches(const nb::list&                        targets,
                              const nb::list&                        queries,
                              const nvMolKit::SubstructSearchConfig& config) {
  nvMolKit::ScopedNvtxRange extractRange("Python: extract mol pointers", nvMolKit::NvtxColor::kYellow);
  const auto                targetMolecules = moleculesFromList(targets, "target");
  const auto                queryMolecules  = moleculesFromList(queries, "query");
  extractRange.pop();

  nvMolKit::SubstructSearchResults results;
  nvMolKit::getSubstructMatches(targetMolecules, queryMolecules, results, config.algorithm, nullptr, config);

  auto buffers        = std::make_unique<SubstructMatchesCSR>();
  buffers->numTargets = results.numTargets;
  buffers->numQueries = results.numQueries;

  nvMolKit::ScopedNvtxRange buildRange("Python: build CSR buffers", nvMolKit::NvtxColor::kOrange);
  const std::int64_t        numPairs = static_cast<std::int64_t>(buffers->numTargets) * buffers->numQueries;
  buffers->pairIndptr.resize(static_cast<std::size_t>(numPairs) + 1, 0);
  buffers->matchIndptr.reserve(1024);
  buffers->matchIndptr.push_back(0);

  std::int32_t matchCount = 0;
  std::int32_t atomCount  = 0;
  for (int targetIndex = 0; targetIndex < buffers->numTargets; ++targetIndex) {
    for (int queryIndex = 0; queryIndex < buffers->numQueries; ++queryIndex) {
      const std::int64_t pairIndex = static_cast<std::int64_t>(targetIndex) * buffers->numQueries + queryIndex;
      buffers->pairIndptr[static_cast<std::size_t>(pairIndex)] = matchCount;
      for (const auto& match : results.getMatches(targetIndex, queryIndex)) {
        buffers->atomIndices.insert(buffers->atomIndices.end(), match.begin(), match.end());
        atomCount += static_cast<std::int32_t>(match.size());
        buffers->matchIndptr.push_back(atomCount);
        ++matchCount;
      }
    }
  }
  buffers->pairIndptr[static_cast<std::size_t>(numPairs)] = matchCount;
  buildRange.pop();

  SubstructMatchesCSR* const bufferPointer = buffers.get();
  nb::capsule                owner(bufferPointer, [](void* pointer) noexcept {
    std::unique_ptr<SubstructMatchesCSR> owned(static_cast<SubstructMatchesCSR*>(pointer));
  });
  buffers.release();

  nvMolKit::ScopedNvtxRange wrapRange("Python: wrap CSR numpy arrays", nvMolKit::NvtxColor::kGreen);
  return nb::make_tuple(makeOneDimensionalArray(bufferPointer->atomIndices, owner),
                        makeOneDimensionalArray(bufferPointer->matchIndptr, owner),
                        makeOneDimensionalArray(bufferPointer->pairIndptr, owner),
                        nb::make_tuple(bufferPointer->numTargets, bufferPointer->numQueries));
}

nb::ndarray<nb::numpy, int, nb::ndim<2>> countSubstructMatches(const nb::list&                        targets,
                                                               const nb::list&                        queries,
                                                               const nvMolKit::SubstructSearchConfig& config) {
  nvMolKit::ScopedNvtxRange extractRange("Python: extract mol pointers", nvMolKit::NvtxColor::kYellow);
  const auto                targetMolecules = moleculesFromList(targets, "target");
  const auto                queryMolecules  = moleculesFromList(queries, "query");
  extractRange.pop();

  auto counts = std::make_unique<std::vector<int>>();
  nvMolKit::countSubstructMatches(targetMolecules, queryMolecules, *counts, config.algorithm, nullptr, config);

  std::vector<int>* const countsPointer = counts.get();
  nb::capsule             owner(countsPointer, [](void* pointer) noexcept {
    std::unique_ptr<std::vector<int>> owned(static_cast<std::vector<int>*>(pointer));
  });
  counts.release();

  nvMolKit::ScopedNvtxRange wrapRange("Python: wrap numpy array", nvMolKit::NvtxColor::kGreen);
  return nb::ndarray<nb::numpy, int, nb::ndim<2>>(countsPointer->data(),
                                                  {targetMolecules.size(), queryMolecules.size()},
                                                  owner);
}

nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>> hasSubstructMatch(const nb::list&                        targets,
                                                                    const nb::list&                        queries,
                                                                    const nvMolKit::SubstructSearchConfig& config) {
  nvMolKit::ScopedNvtxRange extractRange("Python: extract mol pointers", nvMolKit::NvtxColor::kYellow);
  const auto                targetMolecules = moleculesFromList(targets, "target");
  const auto                queryMolecules  = moleculesFromList(queries, "query");
  extractRange.pop();

  auto results = std::make_unique<nvMolKit::HasSubstructMatchResults>();
  nvMolKit::hasSubstructMatch(targetMolecules, queryMolecules, *results, config.algorithm, nullptr, config);

  nvMolKit::HasSubstructMatchResults* const resultsPointer = results.get();
  nb::capsule                               owner(resultsPointer, [](void* pointer) noexcept {
    std::unique_ptr<nvMolKit::HasSubstructMatchResults> owned(
      static_cast<nvMolKit::HasSubstructMatchResults*>(pointer));
  });
  results.release();

  nvMolKit::ScopedNvtxRange wrapRange("Python: wrap numpy array", nvMolKit::NvtxColor::kGreen);
  return nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>>(
    resultsPointer->hasMatch.data(),
    {static_cast<std::size_t>(resultsPointer->numTargets), static_cast<std::size_t>(resultsPointer->numQueries)},
    owner);
}

}  // namespace

NB_MODULE(_substructure, module) {
  namespace nb = nanobind;
  using namespace nb::literals;

  nb::module_::import_("rdkit.Chem.rdchem");

  nb::class_<nvMolKit::SubstructSearchConfig>(module, "SubstructSearchConfig")
    .def(nb::init<>())
    .def_rw("batchSize", &nvMolKit::SubstructSearchConfig::batchSize)
    .def_rw("workerThreads", &nvMolKit::SubstructSearchConfig::workerThreads)
    .def_rw("preprocessingThreads", &nvMolKit::SubstructSearchConfig::preprocessingThreads)
    .def_rw("maxMatches", &nvMolKit::SubstructSearchConfig::maxMatches)
    .def_rw("uniquify", &nvMolKit::SubstructSearchConfig::uniquify)
    .def_prop_rw(
      "gpuIds",
      [](const nvMolKit::SubstructSearchConfig& config) { return config.gpuIds; },
      [](nvMolKit::SubstructSearchConfig& config, const nb::iterable& values) {
        config.gpuIds = integersFromIterable(values);
      })
    .def_prop_rw("algorithm", &getAlgorithm, &setAlgorithm);

  module.def("getSubstructMatches", &getSubstructMatches, "targets"_a, "queries"_a, "config"_a);
  module.def("countSubstructMatches", &countSubstructMatches, "targets"_a, "queries"_a, "config"_a);
  module.def("hasSubstructMatch", &hasSubstructMatch, "targets"_a, "queries"_a, "config"_a);
}
