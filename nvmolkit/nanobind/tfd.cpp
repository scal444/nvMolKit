// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/ROMol.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "nvmolkit/nanobind/array_helpers.h"
#include "src/tfd/tfd_gpu.h"
#include "src/utils/nvtx.h"

namespace {

namespace nb = nanobind;
using namespace nb::literals;

std::vector<const RDKit::ROMol*> extractMolecules(const nb::list& molecules) {
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

nvMolKit::TFDComputeOptions buildOptions(bool               useWeights,
                                         const std::string& maxDeviation,
                                         int                symmetryRadius,
                                         bool               ignoreColinearBonds) {
  nvMolKit::TFDMaxDevMode maxDeviationMode;
  if (maxDeviation == "equal") {
    maxDeviationMode = nvMolKit::TFDMaxDevMode::Equal;
  } else if (maxDeviation == "spec") {
    maxDeviationMode = nvMolKit::TFDMaxDevMode::Spec;
  } else {
    throw std::invalid_argument("maxDev must be 'equal' or 'spec', got: " + maxDeviation);
  }

  return {
    .useWeights          = useWeights,
    .maxDevMode          = maxDeviationMode,
    .symmRadius          = symmetryRadius,
    .ignoreColinearBonds = ignoreColinearBonds,
  };
}

nvMolKit::TFDGpuGenerator& gpuGenerator() {
  static nvMolKit::TFDGpuGenerator generator;
  return generator;
}

nb::tuple getTFDMatricesGpuBuffer(const nb::list&    molecules,
                                  bool               useWeights,
                                  const std::string& maxDeviation,
                                  int                symmetryRadius,
                                  bool               ignoreColinearBonds) {
  const auto moleculeVector = extractMolecules(molecules);
  const auto options        = buildOptions(useWeights, maxDeviation, symmetryRadius, ignoreColinearBonds);
  auto       gpuResult      = gpuGenerator().GetTFDMatricesGpuBuffer(moleculeVector, options);

  nvMolKit::ScopedNvtxRange range("GPU: C++ to Python tuple", nvMolKit::NvtxColor::kYellow);
  nb::list                  outputStarts;
  for (const auto outputStart : gpuResult.tfdOutputStarts) {
    outputStarts.append(outputStart);
  }

  const std::size_t totalSize = gpuResult.tfdValues.size();
  auto              array = nvMolKit::nanobind_bindings::makePyArray(gpuResult.tfdValues, nb::make_tuple(totalSize));
  return nb::make_tuple(nb::cast(std::move(array)), std::move(outputStarts));
}

}  // namespace

NB_MODULE(_TFD, module) {
  namespace nb = nanobind;
  using namespace nb::literals;

  nb::module_::import_("rdkit.Chem.rdchem");
  nb::module_::import_("nvmolkit._arrayHelpers");

  module.def("GetTFDMatricesGpuBuffer",
             &getTFDMatricesGpuBuffer,
             "mols"_a,
             "useWeights"_a          = true,
             "maxDev"_a              = "equal",
             "symmRadius"_a          = 2,
             "ignoreColinearBonds"_a = true);
}
