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

#include <GraphMol/ROMol.h>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <vector>

#include "substruct_types.h"

// Forward declarations - avoid including CUDA headers
using cudaStream_t = struct CUstream_st*;

namespace nvMolKit {

void getSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                         const std::vector<const RDKit::ROMol*>& queries,
                         SubstructSearchResults&                 results,
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

using namespace boost::python;

template <typename T>
list vectorToList(const std::vector<T>& vec) {
  list result;
  for (const auto& value : vec) {
    result.append(value);
  }
  return result;
}

template <typename T>
std::vector<T> listFromIterable(const object& iterable) {
  std::vector<T> converted;
  if (PySequence_Check(iterable.ptr())) {
    Py_ssize_t n = PySequence_Size(iterable.ptr());
    converted.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
      object item(handle<>(borrowed(PySequence_GetItem(iterable.ptr(), i))));
      converted.push_back(extract<T>(item));
    }
  } else {
    stl_input_iterator<T> it(iterable), end;
    for (; it != end; ++it) {
      converted.push_back(*it);
    }
  }
  return converted;
}

list getGpuIdsPy(nvMolKit::SubstructSearchConfig& config) {
  return vectorToList(config.gpuIds);
}

void setGpuIdsPy(nvMolKit::SubstructSearchConfig& config, const object& iterable) {
  config.gpuIds = listFromIterable<int>(iterable);
}

}  // namespace

BOOST_PYTHON_MODULE(_substructure) {
  class_<nvMolKit::SubstructSearchConfig>("SubstructSearchConfig")
    .def(init<>())
    .def_readwrite("batchSize", &nvMolKit::SubstructSearchConfig::batchSize)
    .def_readwrite("workerThreads", &nvMolKit::SubstructSearchConfig::workerThreads)
    .def_readwrite("preprocessingThreads", &nvMolKit::SubstructSearchConfig::preprocessingThreads)
    .def_readwrite("slotsPerRunner", &nvMolKit::SubstructSearchConfig::slotsPerRunner)
    .def_readwrite("presort", &nvMolKit::SubstructSearchConfig::presort)
    .def_readwrite("maxMatches", &nvMolKit::SubstructSearchConfig::maxMatches)
    .add_property("gpuIds", &getGpuIdsPy, &setGpuIdsPy);

  def(
    "getSubstructMatches",
    +[](const list& targets,
        const list& queries,
        const nvMolKit::SubstructSearchConfig& config) {
      std::vector<const RDKit::ROMol*> targetsVec;
      std::vector<const RDKit::ROMol*> queriesVec;

      targetsVec.reserve(len(targets));
      for (int i = 0; i < len(targets); ++i) {
        const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(targets[i]));
        if (mol == nullptr) {
          throw std::invalid_argument("Invalid target molecule at index " + std::to_string(i));
        }
        targetsVec.push_back(mol);
      }

      queriesVec.reserve(len(queries));
      for (int i = 0; i < len(queries); ++i) {
        const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(queries[i]));
        if (mol == nullptr) {
          throw std::invalid_argument("Invalid query molecule at index " + std::to_string(i));
        }
        queriesVec.push_back(mol);
      }

      nvMolKit::SubstructSearchResults results;
      nvMolKit::getSubstructMatches(targetsVec, queriesVec, results,
                                    nvMolKit::SubstructAlgorithm::WarpUnified, nullptr, config);

      // Convert results to Python: list[target][query] -> list of matches
      // Each match is a list of target atom indices
      list pyResults;
      for (int t = 0; t < results.numTargets; ++t) {
        list targetMatches;
        for (int q = 0; q < results.numQueries; ++q) {
          list queryMatches;
          const auto& matches = results.getMatches(t, q);
          for (const auto& match : matches) {
            queryMatches.append(vectorToList(match));
          }
          targetMatches.append(queryMatches);
        }
        pyResults.append(targetMatches);
      }
      return pyResults;
    },
    (arg("targets"),
     arg("queries"),
     arg("config") = nvMolKit::SubstructSearchConfig()),
    "Perform batch substructure matching on GPU.\n"
    "\n"
    "Args:\n"
    "    targets: List of target RDKit molecules\n"
    "    queries: List of query RDKit molecules (typically from SMARTS)\n"
    "    config: SubstructSearchConfig with execution settings\n"
    "\n"
    "Returns:\n"
    "    Nested list: results[target_idx][query_idx] = list of matches,\n"
    "    where each match is a list of target atom indices (one per query atom)");

  def(
    "hasSubstructMatch",
    +[](const list& targets,
        const list& queries,
        const nvMolKit::SubstructSearchConfig& config) {
      std::vector<const RDKit::ROMol*> targetsVec;
      std::vector<const RDKit::ROMol*> queriesVec;

      targetsVec.reserve(len(targets));
      for (int i = 0; i < len(targets); ++i) {
        const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(targets[i]));
        if (mol == nullptr) {
          throw std::invalid_argument("Invalid target molecule at index " + std::to_string(i));
        }
        targetsVec.push_back(mol);
      }

      queriesVec.reserve(len(queries));
      for (int i = 0; i < len(queries); ++i) {
        const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(object(queries[i]));
        if (mol == nullptr) {
          throw std::invalid_argument("Invalid query molecule at index " + std::to_string(i));
        }
        queriesVec.push_back(mol);
      }

      nvMolKit::HasSubstructMatchResults results;
      nvMolKit::hasSubstructMatch(targetsVec, queriesVec, results,
                                  nvMolKit::SubstructAlgorithm::WarpUnified, nullptr, config);

      // Convert results to Python: 2D list of booleans [target][query]
      list pyResults;
      for (int t = 0; t < results.numTargets; ++t) {
        list targetMatches;
        for (int q = 0; q < results.numQueries; ++q) {
          targetMatches.append(results.matches(t, q));
        }
        pyResults.append(targetMatches);
      }
      return pyResults;
    },
    (arg("targets"),
     arg("queries"),
     arg("config") = nvMolKit::SubstructSearchConfig()),
    "Check if targets contain query substructures (boolean results).\n"
    "\n"
    "More efficient than getSubstructMatches when only existence is needed.\n"
    "\n"
    "Args:\n"
    "    targets: List of target RDKit molecules\n"
    "    queries: List of query RDKit molecules (typically from SMARTS)\n"
    "    config: SubstructSearchConfig with execution settings\n"
    "\n"
    "Returns:\n"
    "    2D list of booleans: results[target_idx][query_idx] = True if match exists");
}

