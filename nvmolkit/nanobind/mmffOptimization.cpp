// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>

#include "nvmolkit/nanobind/device_result_python.h"
#include "nvmolkit/nanobind/forcefield_python_utils.h"
#include "src/minimizer/bfgs_types.h"
#include "src/minimizer/fire_minimizer.h"
#include "src/minimizer/mmff_minimize.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

enum class MinimizerKind {
  BFGS,
  FIRE
};

nvMolKit::BfgsBackend parseBfgsBackend(const std::string& name) {
  if (name == "BATCHED") {
    return nvMolKit::BfgsBackend::BATCHED;
  }
  if (name == "PER_MOL" || name == "PER_MOLECULE") {
    return nvMolKit::BfgsBackend::PER_MOLECULE;
  }
  if (name == "HYBRID") {
    return nvMolKit::BfgsBackend::HYBRID;
  }
  throw std::invalid_argument("Unknown BFGS backend '" + name + "'. Expected 'BATCHED', 'PER_MOL', or 'HYBRID'.");
}

nvMolKit::FireBackend parseFireBackend(const std::string& name) {
  if (name == "BATCHED") {
    return nvMolKit::FireBackend::BATCHED;
  }
  if (name == "PER_MOL" || name == "PER_MOLECULE") {
    return nvMolKit::FireBackend::PER_MOLECULE;
  }
  if (name == "HYBRID") {
    return nvMolKit::FireBackend::HYBRID;
  }
  throw std::invalid_argument("Unknown FIRE backend '" + name + "'. Expected 'BATCHED', 'PER_MOL', or 'HYBRID'.");
}

MinimizerKind parseMinimizerKind(const std::string& name) {
  if (name == "BFGS" || name == "bfgs") {
    return MinimizerKind::BFGS;
  }
  if (name == "FIRE" || name == "fire") {
    return MinimizerKind::FIRE;
  }
  throw std::invalid_argument("Unknown minimizerKind '" + name + "'. Expected 'BFGS' or 'FIRE'.");
}

}  // namespace

NB_MODULE(_mmffOptimization, module) {
  nb::module_::import_("nvmolkit._embedMolecules");
  nb::module_::import_("nvmolkit._types");

  module.def(
    "MMFFOptimizeMoleculesConfs",
    [](const nb::list&                       molecules,
       const int                             maxIters,
       const nb::list&                       propertiesList,
       const nvMolKit::BatchHardwareOptions& hardwareOptions,
       const std::string&                    backend,
       const std::string&                    minimizerKind,
       const nvMolKit::FireOptions&          fireOptions) {
      auto       moleculeVector = nvMolKit::NanobindForcefield::extractMolecules(molecules);
      const auto properties =
        nvMolKit::NanobindForcefield::extractMMFFPropertiesList(propertiesList,
                                                                static_cast<int>(moleculeVector.size()));
      const auto kind = parseMinimizerKind(minimizerKind);
      return kind == MinimizerKind::FIRE ? nvMolKit::MMFF::MMFFOptimizeMoleculesConfsFire(moleculeVector,
                                                                                          maxIters,
                                                                                          fireOptions,
                                                                                          properties,
                                                                                          hardwareOptions,
                                                                                          parseFireBackend(backend)) :
                                           nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(moleculeVector,
                                                                                          maxIters,
                                                                                          properties,
                                                                                          hardwareOptions,
                                                                                          parseBfgsBackend(backend));
    },
    "molecules"_a,
    "maxIters"_a        = 200,
    "properties"_a      = nb::list(),
    "hardwareOptions"_a = nvMolKit::BatchHardwareOptions(),
    "backend"_a         = std::string("HYBRID"),
    "minimizerKind"_a   = std::string("BFGS"),
    "fireOptions"_a     = nvMolKit::FireOptions(),
    "Optimize conformers for multiple molecules using MMFF force field.");

  module.def(
    "MMFFOptimizeMoleculesConfsDevice",
    [](const nb::list&                       molecules,
       const int                             maxIters,
       const nb::list&                       propertiesList,
       const nvMolKit::BatchHardwareOptions& hardwareOptions,
       const int                             targetGpu,
       const std::string&                    backend,
       const std::string&                    minimizerKind,
       const nvMolKit::FireOptions&          fireOptions) -> nb::object {
      auto       moleculeVector = nvMolKit::NanobindForcefield::extractMolecules(molecules);
      const auto properties =
        nvMolKit::NanobindForcefield::extractMMFFPropertiesList(propertiesList,
                                                                static_cast<int>(moleculeVector.size()));
      const auto kind   = parseMinimizerKind(minimizerKind);
      auto       result = kind == MinimizerKind::FIRE ?
                            nvMolKit::MMFF::MMFFMinimizeMoleculesConfsFire(moleculeVector,
                                                                     maxIters,
                                                                     fireOptions,
                                                                     properties,
                                                                           {},
                                                                     hardwareOptions,
                                                                     parseFireBackend(backend),
                                                                     nvMolKit::CoordinateOutput::DEVICE,
                                                                     targetGpu) :
                            nvMolKit::MMFF::MMFFMinimizeMoleculesConfs(moleculeVector,
                                                                 maxIters,
                                                                 1e-4,
                                                                 properties,
                                                                       {},
                                                                 hardwareOptions,
                                                                 parseBfgsBackend(backend),
                                                                 nvMolKit::CoordinateOutput::DEVICE,
                                                                 targetGpu);
      if (!result.device.has_value()) {
        throw std::runtime_error("MMFFMinimizeMoleculesConfs(DEVICE) returned no device result");
      }
      return nvMolKit::nanobind_bindings::buildOwningDevice3DResult(*result.device);
    },
    "molecules"_a,
    "maxIters"_a,
    "properties"_a,
    "hardwareOptions"_a,
    "targetGpu"_a,
    "backend"_a       = std::string("HYBRID"),
    "minimizerKind"_a = std::string("BFGS"),
    "fireOptions"_a   = nvMolKit::FireOptions(),
    "Optimize conformers with MMFF and return device-resident results.");
}
