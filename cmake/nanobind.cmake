# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  RESULT_VARIABLE _nvmolkit_nanobind_result
  OUTPUT_VARIABLE nanobind_ROOT
  ERROR_VARIABLE _nvmolkit_nanobind_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _nvmolkit_nanobind_result EQUAL 0)
  message(
    FATAL_ERROR "The NANOBIND backend requires the nanobind Python package: "
                "${_nvmolkit_nanobind_error}")
endif()

find_package(nanobind 2 CONFIG REQUIRED)
