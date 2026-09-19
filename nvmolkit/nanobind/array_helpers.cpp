// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nvmolkit/nanobind/array_helpers.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>

namespace nb = nanobind;

NB_MODULE(_arrayHelpers, module) {
  nb::class_<nvMolKit::nanobind_bindings::PyArray>(module, "_arrayHelpers")
    .def_prop_ro("__cuda_array_interface__",
                 [](const nvMolKit::nanobind_bindings::PyArray& array) { return array.cudaArrayInterface; });
}
