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
#include <boost/python/numpy.hpp>
#include <boost/python/stl_iterator.hpp>
#include <memory>
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
  numpy::initialize();

  class_<nvMolKit::SubstructSearchConfig>("SubstructSearchConfig")
    .def(init<>())
    .def_readwrite("batchSize", &nvMolKit::SubstructSearchConfig::batchSize)
    .def_readwrite("workerThreads", &nvMolKit::SubstructSearchConfig::workerThreads)
    .def_readwrite("preprocessingThreads", &nvMolKit::SubstructSearchConfig::preprocessingThreads)
    .def_readwrite("rdkitFallbackThreads", &nvMolKit::SubstructSearchConfig::rdkitFallbackThreads)
    .def_readwrite("presort", &nvMolKit::SubstructSearchConfig::presort)
    .def_readwrite("maxMatches", &nvMolKit::SubstructSearchConfig::maxMatches)
    .def_readwrite("uniquify", &nvMolKit::SubstructSearchConfig::uniquify)
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
                                    nvMolKit::SubstructAlgorithm::GSI, nullptr, config);

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

      auto resultsPtr = std::make_unique<nvMolKit::HasSubstructMatchResults>();
      nvMolKit::hasSubstructMatch(targetsVec, queriesVec, *resultsPtr,
                                  nvMolKit::SubstructAlgorithm::GSI, nullptr, config);

      const int numTargets = resultsPtr->numTargets;
      const int numQueries = resultsPtr->numQueries;
      uint8_t* dataPtr = resultsPtr->hasMatch.data();

      auto deleter = [](PyObject* cap) {
        auto* r = reinterpret_cast<nvMolKit::HasSubstructMatchResults*>(
            PyCapsule_GetPointer(cap, "nvmolkit.hassubstruct_results"));
        delete r;
      };
      PyObject* cap = PyCapsule_New(static_cast<void*>(resultsPtr.get()),
                                    "nvmolkit.hassubstruct_results", deleter);
      if (cap == nullptr) {
        throw std::runtime_error("Failed to create PyCapsule for hasSubstructMatch results");
      }
      object owner{handle<>(cap)};
      resultsPtr.release();

      const Py_intptr_t shape_arr[2] = {static_cast<Py_intptr_t>(numTargets),
                                        static_cast<Py_intptr_t>(numQueries)};
      const Py_intptr_t strides_arr[2] = {static_cast<Py_intptr_t>(numQueries * sizeof(uint8_t)),
                                          static_cast<Py_intptr_t>(sizeof(uint8_t))};

      return numpy::from_data(dataPtr,
                              numpy::dtype::get_builtin<uint8_t>(),
                              make_tuple(shape_arr[0], shape_arr[1]),
                              make_tuple(strides_arr[0], strides_arr[1]),
                              owner);
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
    "    2D numpy array of uint8: results[target_idx, query_idx] = 1 if match exists, 0 otherwise");
}

