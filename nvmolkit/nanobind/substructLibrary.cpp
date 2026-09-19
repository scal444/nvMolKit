// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/ROMol.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <vector>

#include "src/substruct/substruct_library.h"

namespace {

namespace nb = nanobind;
using namespace nb::literals;

std::vector<unsigned int> addMolecules(nvMolKit::SubstructLibrary& library, const nb::iterable& molecules) {
  std::vector<nb::object>          moleculeOwners;
  std::vector<const RDKit::ROMol*> moleculePointers;
  for (const nb::handle moleculeHandle : molecules) {
    moleculeOwners.emplace_back(nb::borrow<nb::object>(moleculeHandle));
    moleculePointers.push_back(nb::cast<const RDKit::ROMol*>(moleculeHandle));
  }

  std::vector<unsigned int> moleculeIds;
  moleculeIds.reserve(moleculePointers.size());
  {
    nb::gil_scoped_release release;
    for (const RDKit::ROMol* molecule : moleculePointers) {
      moleculeIds.push_back(library.addMol(*molecule));
    }
  }
  return moleculeIds;
}

unsigned int addMolecule(nvMolKit::SubstructLibrary& library, const RDKit::ROMol& molecule) {
  nb::gil_scoped_release release;
  return library.addMol(molecule);
}

void finalize(nvMolKit::SubstructLibrary& library) {
  nb::gil_scoped_release release;
  library.finalize();
}

std::vector<unsigned int> getMatches(nvMolKit::SubstructLibrary& library,
                                     const RDKit::ROMol&         query,
                                     const int                   maxResults) {
  nb::gil_scoped_release release;
  return library.getMatches(query, maxResults);
}

std::size_t countMatches(nvMolKit::SubstructLibrary& library, const RDKit::ROMol& query) {
  nb::gil_scoped_release release;
  return library.countMatches(query);
}

bool hasMatch(nvMolKit::SubstructLibrary& library, const RDKit::ROMol& query) {
  nb::gil_scoped_release release;
  return library.hasMatch(query);
}

}  // namespace

NB_MODULE(_substructLibrary, module) {
  namespace nb = nanobind;
  using namespace nb::literals;

  nb::module_::import_("rdkit.Chem.rdchem");
  nb::module_::import_("nvmolkit._substructure");

  nb::class_<nvMolKit::SubstructLibrary>(module, "SubstructLibrary")
    .def(nb::init<std::size_t, nvMolKit::SubstructSearchConfig>(),
         "chunkSize"_a = 65536,
         "config"_a    = nvMolKit::SubstructSearchConfig{})
    .def("addMol", &addMolecule, "molecule"_a)
    .def("addMols", &addMolecules, "molecules"_a)
    .def("finalize", &finalize)
    .def("getMatches", &getMatches, "query"_a, "maxResults"_a = -1)
    .def("countMatches", &countMatches, "query"_a)
    .def("hasMatch", &hasMatch, "query"_a)
    .def("__len__", &nvMolKit::SubstructLibrary::size)
    .def_prop_ro("pendingSize", &nvMolKit::SubstructLibrary::pendingSize);
}
