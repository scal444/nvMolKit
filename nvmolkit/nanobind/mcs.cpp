// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/ROMol.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/mcs/mcs_search.h"
#include "src/utils/nvtx.h"

namespace {

namespace nb = nanobind;
using namespace nb::literals;

struct MCSResultBuffers {
  std::vector<unsigned int> numAtoms;
  std::vector<unsigned int> numBonds;
  std::vector<std::uint8_t> canceled;
  std::vector<std::int32_t> atomMapping;
  std::vector<std::int32_t> atomMappingIndptr;
  std::vector<std::int32_t> bondMapping;
  std::vector<std::int32_t> bondMappingIndptr;
};

nvMolKit::MCSAtomCompare parseAtomCompare(const std::string& value) {
  if (value == "any") {
    return nvMolKit::MCSAtomCompare::Any;
  }
  if (value == "elements") {
    return nvMolKit::MCSAtomCompare::Elements;
  }
  if (value == "isotopes") {
    return nvMolKit::MCSAtomCompare::Isotopes;
  }
  if (value == "any_heavy_atom") {
    return nvMolKit::MCSAtomCompare::AnyHeavyAtom;
  }
  throw std::invalid_argument("Unsupported atom_compare value: " + value);
}

nvMolKit::MCSBondCompare parseBondCompare(const std::string& value) {
  if (value == "any") {
    return nvMolKit::MCSBondCompare::Any;
  }
  if (value == "order") {
    return nvMolKit::MCSBondCompare::Order;
  }
  if (value == "order_exact") {
    return nvMolKit::MCSBondCompare::OrderExact;
  }
  throw std::invalid_argument("Unsupported bond_compare value: " + value);
}

std::vector<const RDKit::ROMol*> moleculesFromList(const nb::list& molecules) {
  nvMolKit::ScopedNvtxRange        range("Python MCS: extract mol pointers", nvMolKit::NvtxColor::kYellow);
  std::vector<const RDKit::ROMol*> converted;
  converted.reserve(molecules.size());
  for (std::size_t index = 0; index < molecules.size(); ++index) {
    const RDKit::ROMol* molecule = nb::cast<const RDKit::ROMol*>(molecules[index]);
    if (molecule == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(index));
    }
    converted.push_back(molecule);
  }
  return converted;
}

std::size_t pythonSize(PyObject* value) {
  const std::size_t converted = PyLong_AsSize_t(value);
  if (converted == static_cast<std::size_t>(-1) && PyErr_Occurred()) {
    throw nb::python_error();
  }
  return converted;
}

std::vector<nvMolKit::MCSPair> pairsFromList(const nb::list& pairs) {
  std::vector<nvMolKit::MCSPair> converted;
  converted.reserve(pairs.size());
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    PyObject* sequence = PySequence_Fast(pairs[index].ptr(), "MCS pair must be a sequence");
    if (sequence == nullptr || PySequence_Fast_GET_SIZE(sequence) != 2) {
      Py_XDECREF(sequence);
      PyErr_Clear();
      throw std::invalid_argument("MCS pair at index " + std::to_string(index) + " must be a length-2 sequence");
    }
    nb::object       owner = nb::steal<nb::object>(sequence);
    PyObject* const* items = PySequence_Fast_ITEMS(sequence);
    converted.emplace_back(pythonSize(items[0]), pythonSize(items[1]));
  }
  return converted;
}

template <typename T> T optionValue(const nb::dict& options, const char* key, const T& defaultValue) {
  if (PyMapping_HasKeyString(options.ptr(), key) == 0) {
    return defaultValue;
  }
  return nb::cast<T>(options[key]);
}

unsigned int unsignedOptionValue(const nb::dict& options, const char* key, unsigned int defaultValue) {
  if (PyMapping_HasKeyString(options.ptr(), key) == 0) {
    return defaultValue;
  }
  const unsigned long value = PyLong_AsUnsignedLong(options[key].ptr());
  if (value == static_cast<unsigned long>(-1) && PyErr_Occurred()) {
    throw nb::python_error();
  }
  if (value > std::numeric_limits<unsigned int>::max()) {
    PyErr_SetString(PyExc_OverflowError, "Python int too large to convert to C unsigned int");
    throw nb::python_error();
  }
  return static_cast<unsigned int>(value);
}

std::vector<int> integerVectorOption(const nb::dict& options, const char* key) {
  if (PyMapping_HasKeyString(options.ptr(), key) == 0) {
    return {};
  }
  std::vector<int> converted;
  for (const nb::handle item : nb::cast<nb::iterable>(options[key])) {
    converted.push_back(nb::cast<int>(item));
  }
  return converted;
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> makeOneDimensionalArray(std::vector<T>& values, const nb::handle& owner) {
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(values.data(), {values.size()}, owner);
}

nb::ndarray<nb::numpy, std::int32_t, nb::ndim<2>> makePairArray(std::vector<std::int32_t>& values,
                                                                const nb::handle&          owner) {
  return nb::ndarray<nb::numpy, std::int32_t, nb::ndim<2>>(values.data(), {values.size() / 2, 2}, owner);
}

nb::tuple findMCSBatch(const nb::list& molecules, const nb::list& pairs, const nb::dict& options) {
  const auto moleculeVector = moleculesFromList(molecules);
  const auto pairVector     = pairsFromList(pairs);

  nvMolKit::MCSParameters parameters;
  parameters.atomCompare          = parseAtomCompare(optionValue<std::string>(options, "atom_compare", "elements"));
  parameters.bondCompare          = parseBondCompare(optionValue<std::string>(options, "bond_compare", "order"));
  parameters.maximizeBonds        = optionValue<bool>(options, "maximize_bonds", true);
  parameters.connectedOnly        = optionValue<bool>(options, "connected_only", true);
  parameters.requireGpu           = optionValue<bool>(options, "require_gpu", false);
  parameters.timeoutSeconds       = unsignedOptionValue(options, "timeout_seconds", 0);
  parameters.batchSize            = optionValue<int>(options, "batch_size", 0);
  parameters.workerThreads        = optionValue<int>(options, "worker_threads", -1);
  parameters.preprocessingThreads = optionValue<int>(options, "preprocessing_threads", -1);
  parameters.executorsPerRunner   = optionValue<int>(options, "executors_per_runner", -1);
  parameters.gpuIds               = integerVectorOption(options, "gpu_ids");
  parameters.atomCompareParameters.matchValences     = optionValue<bool>(options, "match_valences", false);
  parameters.atomCompareParameters.matchFormalCharge = optionValue<bool>(options, "match_formal_charge", false);
  parameters.atomCompareParameters.ringMatchesRingOnly =
    optionValue<bool>(options, "atom_ring_matches_ring_only", false);
  parameters.atomCompareParameters.matchIsotope = optionValue<bool>(options, "match_isotope", false);
  parameters.bondCompareParameters.ringMatchesRingOnly =
    optionValue<bool>(options, "bond_ring_matches_ring_only", false);

  nvMolKit::ScopedNvtxRange mcsRange("Python MCS: findMCSBatch", nvMolKit::NvtxColor::kOrange);
  const auto                results = nvMolKit::findMCSBatch(moleculeVector, pairVector, nullptr, parameters);
  mcsRange.pop();

  auto buffers = std::make_unique<MCSResultBuffers>();
  buffers->numAtoms.reserve(results.size());
  buffers->numBonds.reserve(results.size());
  buffers->canceled.reserve(results.size());
  buffers->atomMappingIndptr.reserve(results.size() + 1);
  buffers->bondMappingIndptr.reserve(results.size() + 1);
  buffers->atomMappingIndptr.push_back(0);
  buffers->bondMappingIndptr.push_back(0);

  for (const auto& result : results) {
    buffers->numAtoms.push_back(result.numAtoms);
    buffers->numBonds.push_back(result.numBonds);
    buffers->canceled.push_back(result.canceled ? 1 : 0);
    for (const auto& [first, second] : result.atomMapping) {
      buffers->atomMapping.push_back(static_cast<std::int32_t>(first));
      buffers->atomMapping.push_back(static_cast<std::int32_t>(second));
    }
    buffers->atomMappingIndptr.push_back(static_cast<std::int32_t>(buffers->atomMapping.size() / 2));
    for (const auto& [first, second] : result.bondMapping) {
      buffers->bondMapping.push_back(static_cast<std::int32_t>(first));
      buffers->bondMapping.push_back(static_cast<std::int32_t>(second));
    }
    buffers->bondMappingIndptr.push_back(static_cast<std::int32_t>(buffers->bondMapping.size() / 2));
  }

  MCSResultBuffers* const bufferPointer = buffers.get();
  nb::capsule             owner(bufferPointer, [](void* pointer) noexcept {
    std::unique_ptr<MCSResultBuffers> owned(static_cast<MCSResultBuffers*>(pointer));
  });
  buffers.release();

  nvMolKit::ScopedNvtxRange wrapRange("Python MCS: wrap results", nvMolKit::NvtxColor::kGreen);
  return nb::make_tuple(makeOneDimensionalArray(bufferPointer->numAtoms, owner),
                        makeOneDimensionalArray(bufferPointer->numBonds, owner),
                        makeOneDimensionalArray(bufferPointer->canceled, owner),
                        makePairArray(bufferPointer->atomMapping, owner),
                        makeOneDimensionalArray(bufferPointer->atomMappingIndptr, owner),
                        makePairArray(bufferPointer->bondMapping, owner),
                        makeOneDimensionalArray(bufferPointer->bondMappingIndptr, owner));
}

}  // namespace

NB_MODULE(_mcs, module) {
  namespace nb = nanobind;
  using namespace nb::literals;

  nb::module_::import_("rdkit.Chem.rdchem");
  module.def("_findMCSBatch", &findMCSBatch, "mols"_a, "pairs"_a, "options"_a = nb::dict());
}
