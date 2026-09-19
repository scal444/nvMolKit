# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

if(NVMOLKIT_BUILD_PYTHON_BINDINGS)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" -c
      "from rdkit import rdBase; print(getattr(rdBase, '_wrapperType', 'boost'))"
    RESULT_VARIABLE _nvmolkit_rdkit_wrapper_result
    OUTPUT_VARIABLE _nvmolkit_rdkit_wrapper
    ERROR_VARIABLE _nvmolkit_rdkit_wrapper_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _nvmolkit_rdkit_wrapper_result EQUAL 0)
    message(
      FATAL_ERROR
        "Could not determine the installed RDKit Python binding backend: "
        "${_nvmolkit_rdkit_wrapper_error}")
  endif()

  string(TOUPPER "${_nvmolkit_rdkit_wrapper}" _nvmolkit_rdkit_wrapper_backend)
  if(NOT _nvmolkit_rdkit_wrapper_backend STREQUAL
     NVMOLKIT_PYTHON_BINDING_BACKEND)
    message(
      FATAL_ERROR
        "nvMolKit binding backend ${NVMOLKIT_PYTHON_BINDING_BACKEND} does not "
        "match the installed RDKit backend ${_nvmolkit_rdkit_wrapper_backend}")
  endif()
  message(
    STATUS
      "Building ${NVMOLKIT_PYTHON_BINDING_BACKEND} Python bindings against "
      "${_nvmolkit_rdkit_wrapper_backend} RDKit bindings")

  if(NVMOLKIT_PYTHON_BINDING_BACKEND STREQUAL "NANOBIND")
    include(nanobind)
  endif()
endif()
