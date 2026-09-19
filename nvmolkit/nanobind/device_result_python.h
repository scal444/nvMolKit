// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_NANOBIND_DEVICE_RESULT_PYTHON_H
#define NVMOLKIT_NANOBIND_DEVICE_RESULT_PYTHON_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstdint>
#include <memory>

#include "nvmolkit/nanobind/array_helpers.h"
#include "src/conformer/device_coord_result.h"
#include "src/utils/device_vector.h"

namespace nvMolKit::nanobind_bindings {

namespace nb = nanobind;

inline nb::object wrapAsync(std::unique_ptr<PyArray> array, const int gpuId, const nb::object& asyncClass) {
  return asyncClass(nb::cast(std::move(array)), gpuId);
}

inline nb::object buildOwningDevice3DResult(AsyncDeviceVector<double>&  values,
                                            AsyncDeviceVector<int32_t>& atomStarts,
                                            AsyncDeviceVector<int32_t>& molIndices,
                                            AsyncDeviceVector<int32_t>& confIndices,
                                            const int                   gpuId,
                                            const int                   nMols,
                                            AsyncDeviceVector<double>*  energies  = nullptr,
                                            AsyncDeviceVector<int8_t>*  converged = nullptr) {
  const nb::object typesModule    = nb::module_::import_("nvmolkit.types");
  const nb::object resultClass    = typesModule.attr("Device3DResult");
  const nb::object asyncClass     = typesModule.attr("AsyncGpuResult");
  const int        numAtoms       = static_cast<int>(values.size() / 3);
  auto             valuesArray    = makePyArray(values, "f8", nb::make_tuple(numAtoms, 3));
  auto             atomStartsArr  = makePyArray(atomStarts);
  auto             molIndicesArr  = makePyArray(molIndices);
  auto             confIndicesArr = makePyArray(confIndices);
  nb::object       energiesObj    = nb::none();
  nb::object       convergedObj   = nb::none();
  if (energies != nullptr) {
    energiesObj = wrapAsync(makePyArray(*energies), gpuId, asyncClass);
  }
  if (converged != nullptr) {
    convergedObj = wrapAsync(makePyArray(*converged), gpuId, asyncClass);
  }
  return resultClass(wrapAsync(std::move(valuesArray), gpuId, asyncClass),
                     wrapAsync(std::move(atomStartsArr), gpuId, asyncClass),
                     wrapAsync(std::move(molIndicesArr), gpuId, asyncClass),
                     wrapAsync(std::move(confIndicesArr), gpuId, asyncClass),
                     gpuId,
                     nMols,
                     energiesObj,
                     convergedObj);
}

inline nb::object buildOwningDevice3DResult(DeviceCoordResult& result) {
  AsyncDeviceVector<double>* energies  = result.energies.size() > 0 ? &result.energies : nullptr;
  AsyncDeviceVector<int8_t>* converged = result.converged.size() > 0 ? &result.converged : nullptr;
  return buildOwningDevice3DResult(result.positions,
                                   result.atomStarts,
                                   result.molIndices,
                                   result.confIndices,
                                   result.gpuId,
                                   result.nMols,
                                   energies,
                                   converged);
}

}  // namespace nvMolKit::nanobind_bindings

#endif  // NVMOLKIT_NANOBIND_DEVICE_RESULT_PYTHON_H
