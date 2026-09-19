// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/GraphMol.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "nvmolkit/nanobind/array_helpers.h"
#include "src/conformer_rmsd_mol.h"
#include "src/utils/device.h"

namespace {

namespace nb = nanobind;
using namespace nb::literals;

nb::object getConformerRMSMatrixBatch(const nb::list& molecules,
                                      bool            prealigned,
                                      bool            alignToFirstConformer,
                                      std::uintptr_t  streamPointer) {
  const auto stream = nvMolKit::acquireExternalStream(streamPointer);
  if (!stream) {
    throw std::invalid_argument("Invalid CUDA stream");
  }

  if (molecules.empty()) {
    return nb::list();
  }

  std::vector<const RDKit::ROMol*> moleculeVector;
  moleculeVector.reserve(molecules.size());
  for (std::size_t index = 0; index < molecules.size(); ++index) {
    const RDKit::ROMol* molecule = nb::cast<const RDKit::ROMol*>(molecules[index]);
    if (molecule == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(index));
    }
    moleculeVector.push_back(molecule);
  }

  auto     buffers = nvMolKit::conformerRmsdBatchMatrixMol(moleculeVector, prealigned, *stream, alignToFirstConformer);
  nb::list results;
  for (std::size_t moleculeIndex = 0; moleculeIndex < moleculeVector.size(); ++moleculeIndex) {
    const int conformerCount = moleculeVector[moleculeIndex]->getNumConformers();
    const int pairCount      = conformerCount >= 2 ? conformerCount * (conformerCount - 1) / 2 : 0;
    results.append(
      nb::cast(nvMolKit::nanobind_bindings::makePyArray(buffers[moleculeIndex], nb::make_tuple(pairCount))));
  }
  return std::move(results);
}

std::unique_ptr<nvMolKit::nanobind_bindings::PyArray> getConformerRMSMatrix(RDKit::ROMol&  molecule,
                                                                            bool           prealigned,
                                                                            bool           alignToFirstConformer,
                                                                            std::uintptr_t streamPointer) {
  const auto stream = nvMolKit::acquireExternalStream(streamPointer);
  if (!stream) {
    throw std::invalid_argument("Invalid CUDA stream");
  }

  const int          conformerCount = molecule.getNumConformers();
  const std::int64_t pairCount =
    conformerCount >= 2 ? static_cast<std::int64_t>(conformerCount) * (conformerCount - 1) / 2 : 0;
  auto buffer = nvMolKit::conformerRmsdMatrixMol(molecule, prealigned, *stream, alignToFirstConformer);
  return nvMolKit::nanobind_bindings::makePyArray(buffer, nb::make_tuple(pairCount));
}

}  // namespace

NB_MODULE(_conformerRmsd, module) {
  namespace nb = nanobind;
  using namespace nb::literals;

  nb::module_::import_("rdkit.Chem.rdchem");
  nb::module_::import_("nvmolkit._arrayHelpers");

  module.def("GetConformerRMSMatrixBatch",
             &getConformerRMSMatrixBatch,
             "mols"_a,
             "prealigned"_a            = false,
             "alignToFirstConformer"_a = false,
             "stream"_a                = static_cast<std::uintptr_t>(0));
  module.def("GetConformerRMSMatrix",
             &getConformerRMSMatrix,
             "mol"_a,
             "prealigned"_a            = false,
             "alignToFirstConformer"_a = false,
             "stream"_a                = static_cast<std::uintptr_t>(0));
}
