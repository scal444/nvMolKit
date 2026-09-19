// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <vector>

#include "nvmolkit/nanobind/device_result_python.h"
#include "nvmolkit/nanobind/forcefield_python_utils.h"
#include "src/etkdg.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

const RDKit::DGeomHelpers::EmbedParameters& extractEmbedParameters(const nb::object& parameters) {
  const nb::object parameterType = nb::module_::import_("rdkit.Chem.rdDistGeom").attr("EmbedParameters");
  if (!nb::isinstance(parameters, parameterType)) {
    throw nb::type_error("params must be an RDKit EmbedParameters object");
  }
  try {
    return nb::cast<const RDKit::DGeomHelpers::EmbedParameters&>(parameters);
  } catch (const nb::cast_error&) {
  }
  // Older RDKit nanobind wrappers do not register EmbedParameters as the base of
  // their private PyEmbedParameters class. This compatibility fallback assumes
  // single, non-virtual inheritance and a base subobject at offset zero.
  return *nb::inst_ptr<RDKit::DGeomHelpers::EmbedParameters>(parameters);
}

}  // namespace

NB_MODULE(_embedMolecules, module) {
  nb::class_<nvMolKit::BatchHardwareOptions>(module, "BatchHardwareOptions")
    .def(nb::init<>())
    .def_rw("preprocessingThreads", &nvMolKit::BatchHardwareOptions::preprocessingThreads)
    .def_rw("batchSize", &nvMolKit::BatchHardwareOptions::batchSize)
    .def_rw("batchesPerGpu", &nvMolKit::BatchHardwareOptions::batchesPerGpu)
    .def_rw("gpuIds", &nvMolKit::BatchHardwareOptions::gpuIds);

  module.def(
    "EmbedMolecules",
    [](const nb::list&                       molecules,
       const nb::object&                     parameters,
       const int                             confsPerMolecule,
       const int                             maxIterations,
       const nvMolKit::BatchHardwareOptions& hardwareOptions) {
      const auto moleculeVector = nvMolKit::NanobindForcefield::extractMolecules(molecules);
      nvMolKit::embedMolecules(moleculeVector,
                               extractEmbedParameters(parameters),
                               confsPerMolecule,
                               maxIterations,
                               false,
                               nullptr,
                               hardwareOptions);
    },
    "molecules"_a,
    "params"_a,
    "confsPerMolecule"_a = 1,
    "maxIterations"_a    = -1,
    "hardwareOptions"_a  = nvMolKit::BatchHardwareOptions(),
    "Embed multiple molecules with multiple conformers using ETKDG.");

  module.def(
    "EmbedMoleculesDevice",
    [](const nb::list&                       molecules,
       const nb::object&                     parameters,
       const int                             confsPerMolecule,
       const int                             maxIterations,
       const nvMolKit::BatchHardwareOptions& hardwareOptions,
       const int                             targetGpu) -> nb::object {
      const auto moleculeVector = nvMolKit::NanobindForcefield::extractMolecules(molecules);
      auto       result         = nvMolKit::embedMolecules(moleculeVector,
                                             extractEmbedParameters(parameters),
                                             confsPerMolecule,
                                             maxIterations,
                                             false,
                                             nullptr,
                                             hardwareOptions,
                                             nvMolKit::BfgsBackend::HYBRID,
                                             nvMolKit::CoordinateOutput::DEVICE,
                                             targetGpu);
      if (!result.has_value()) {
        throw std::runtime_error("embedMolecules(DEVICE) returned no device result");
      }
      return nvMolKit::nanobind_bindings::buildOwningDevice3DResult(*result);
    },
    "molecules"_a,
    "params"_a,
    "confsPerMolecule"_a = 1,
    "maxIterations"_a    = -1,
    "hardwareOptions"_a  = nvMolKit::BatchHardwareOptions(),
    "targetGpu"_a        = -1,
    "Embed molecules with ETKDG and return device-resident coordinates.");
}
