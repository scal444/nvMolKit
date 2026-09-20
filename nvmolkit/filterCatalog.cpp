// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/ROMol.h>

#include <boost/noncopyable.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/substruct/filter_catalog.h"

namespace {

using namespace boost::python;

class ScopedGilRelease {
 public:
  ScopedGilRelease() : state_(PyEval_SaveThread()) {}
  ScopedGilRelease(const ScopedGilRelease&)            = delete;
  ScopedGilRelease& operator=(const ScopedGilRelease&) = delete;
  ~ScopedGilRelease() { PyEval_RestoreThread(state_); }

 private:
  PyThreadState* state_;
};

nvMolKit::FilterCatalogProperties toProperties(const dict& metadata) {
  nvMolKit::FilterCatalogProperties result;
  const list                        keys = metadata.keys();
  for (Py_ssize_t index = 0; index < len(keys); ++index) {
    const std::string key = extract<std::string>(keys[index]);
    result.emplace(key, extract<std::string>(metadata[key]));
  }
  return result;
}

std::vector<const RDKit::ROMol*> collectMolecules(const object& molecules, std::vector<object>& owners) {
  std::vector<const RDKit::ROMol*> pointers;
  stl_input_iterator<object>       iterator(molecules), end;
  for (; iterator != end; ++iterator) {
    owners.push_back(*iterator);
    pointers.push_back(extract<const RDKit::ROMol*>(owners.back()));
  }
  return pointers;
}

unsigned int addQuery(nvMolKit::FilterCatalog& catalog,
                      const RDKit::ROMol&      query,
                      const std::string&       description,
                      unsigned int             triggerCount,
                      const dict&              metadata) {
  auto                   properties = toProperties(metadata);
  const ScopedGilRelease release;
  return catalog.addEntry(query, description, triggerCount, std::move(properties));
}

unsigned int addSmarts(nvMolKit::FilterCatalog& catalog,
                       const std::string&       smarts,
                       const std::string&       description,
                       unsigned int             triggerCount,
                       const dict&              metadata) {
  auto                   properties = toProperties(metadata);
  const ScopedGilRelease release;
  return catalog.addSmarts(smarts, description, triggerCount, std::move(properties));
}

std::size_t addPreset(nvMolKit::FilterCatalog& catalog, std::uint32_t preset) {
  const ScopedGilRelease release;
  return catalog.addPreset(static_cast<nvMolKit::FilterCatalogPreset>(preset));
}

void finalize(nvMolKit::FilterCatalog& catalog) {
  const ScopedGilRelease release;
  catalog.finalize();
}

list hasMatch(nvMolKit::FilterCatalog& catalog, const object& molecules) {
  std::vector<object>       owners;
  const auto                pointers = collectMolecules(molecules, owners);
  std::vector<std::uint8_t> matches;
  {
    const ScopedGilRelease release;
    matches = catalog.hasMatch(pointers);
  }
  list result;
  for (const std::uint8_t match : matches) {
    result.append(match != 0);
  }
  return result;
}

list getFirstMatch(nvMolKit::FilterCatalog& catalog, const object& molecules) {
  std::vector<object> owners;
  const auto          pointers = collectMolecules(molecules, owners);
  std::vector<int>    matches;
  {
    const ScopedGilRelease release;
    matches = catalog.getFirstMatch(pointers);
  }
  list result;
  for (const int match : matches) {
    if (match < 0) {
      result.append(object());
    } else {
      result.append(match);
    }
  }
  return result;
}

list getMatches(nvMolKit::FilterCatalog& catalog, const object& molecules) {
  std::vector<object>                    owners;
  const auto                             pointers = collectMolecules(molecules, owners);
  std::vector<std::vector<unsigned int>> matches;
  {
    const ScopedGilRelease release;
    matches = catalog.getMatches(pointers);
  }
  list result;
  for (const auto& moleculeMatches : matches) {
    list ids;
    for (const unsigned int id : moleculeMatches) {
      ids.append(id);
    }
    result.append(ids);
  }
  return result;
}

dict getEntry(const nvMolKit::FilterCatalog& catalog, unsigned int id) {
  const auto entry = catalog.getEntry(id);
  dict       metadata;
  for (const auto& [key, value] : entry.properties) {
    metadata[key] = value;
  }
  dict result;
  result["id"]           = entry.id;
  result["description"]  = entry.description;
  result["smarts"]       = entry.smarts;
  result["triggerCount"] = entry.triggerCount;
  result["metadata"]     = metadata;
  return result;
}

}  // namespace

BOOST_PYTHON_MODULE(_filterCatalog) {
  import("rdkit.Chem.rdchem");
  import("nvmolkit._substructure");

  class_<nvMolKit::FilterCatalog, boost::noncopyable>(
    "FilterCatalog",
    init<nvMolKit::SubstructSearchConfig>((arg("config") = nvMolKit::SubstructSearchConfig())))
    .def("addPreset", &addPreset)
    .def("addQuery",
         &addQuery,
         (arg("query"), arg("description") = "", arg("triggerCount") = 1, arg("metadata") = dict()))
    .def("addSmarts",
         &addSmarts,
         (arg("smarts"), arg("description") = "", arg("triggerCount") = 1, arg("metadata") = dict()))
    .def("finalize", &finalize)
    .def("hasMatch", &hasMatch)
    .def("getFirstMatch", &getFirstMatch)
    .def("getMatches", &getMatches)
    .def("getEntry", &getEntry)
    .def("__len__", &nvMolKit::FilterCatalog::size)
    .add_property("pendingSize", &nvMolKit::FilterCatalog::pendingSize);
}
