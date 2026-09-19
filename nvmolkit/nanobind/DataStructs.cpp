// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nvmolkit/nanobind/array_helpers.h"
#include "src/similarity.h"
#include "src/utils/device.h"

namespace {

namespace nb = nanobind;
using namespace nb::literals;
using nvMolKit::nanobind_bindings::getSpanFromDictElems;

template <typename ComputeFn>
nb::ndarray<nb::numpy, double, nb::ndim<2>> crossSimilarityCPUFromRawBuffers(const nb::dict& bitsOne,
                                                                             const nb::dict& bitsTwo,
                                                                             ComputeFn       compute,
                                                                             cudaStream_t    stream) {
  const nb::tuple   shapeOne   = nb::cast<nb::tuple>(bitsOne["shape"]);
  const nb::tuple   shapeTwo   = nb::cast<nb::tuple>(bitsTwo["shape"]);
  const std::size_t numMolsOne = nb::cast<std::size_t>(shapeOne[0]);
  const std::size_t numMolsTwo = nb::cast<std::size_t>(shapeTwo[0]);
  const std::size_t numWords   = nb::cast<std::size_t>(shapeOne[1]);
  if (numWords != nb::cast<std::size_t>(shapeTwo[1])) {
    throw std::invalid_argument("Shape of bitsOne and bitsTwo dim 1 must be the same");
  }

  const auto dataOne = nb::cast<nb::tuple>(bitsOne["data"]);
  const auto dataTwo = nb::cast<nb::tuple>(bitsTwo["data"]);
  const auto ptrOne  = nb::cast<std::size_t>(dataOne[0]);
  const auto ptrTwo  = nb::cast<std::size_t>(dataTwo[0]);
  const auto spanOne = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(ptrOne), shapeOne);
  const auto spanTwo = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(ptrTwo), shapeTwo);
  auto       values =
    std::make_unique<std::vector<double>>(compute(spanOne, spanTwo, static_cast<int>(numWords * 32), stream));
  double* const data = values->data();
  nb::capsule   owner(values.get(), [](void* pointer) noexcept { delete static_cast<std::vector<double>*>(pointer); });
  values.release();
  return nb::ndarray<nb::numpy, double, nb::ndim<2>>(data, {numMolsOne, numMolsTwo}, owner);
}

template <typename ComputeFn>
std::unique_ptr<nvMolKit::nanobind_bindings::PyArray> crossSimilarityGPUFromRawBuffers(const nb::dict& bitsOne,
                                                                                       const nb::dict& bitsTwo,
                                                                                       ComputeFn       compute,
                                                                                       cudaStream_t    stream) {
  const nb::tuple   shapeOne = nb::cast<nb::tuple>(bitsOne["shape"]);
  const nb::tuple   shapeTwo = nb::cast<nb::tuple>(bitsTwo["shape"]);
  const std::size_t numWords = nb::cast<std::size_t>(shapeOne[1]);
  if (numWords != nb::cast<std::size_t>(shapeTwo[1])) {
    throw std::invalid_argument("Shape of bitsOne and bitsTwo dim 1 must be the same");
  }

  const std::size_t numMolsOne = nb::cast<std::size_t>(shapeOne[0]);
  const std::size_t numMolsTwo = nb::cast<std::size_t>(shapeTwo[0]);
  const auto        dataOne    = nb::cast<nb::tuple>(bitsOne["data"]);
  const auto        dataTwo    = nb::cast<nb::tuple>(bitsTwo["data"]);
  const auto        ptrOne     = nb::cast<std::size_t>(dataOne[0]);
  const auto        ptrTwo     = nb::cast<std::size_t>(dataTwo[0]);
  const auto        spanOne    = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(ptrOne), shapeOne);
  const auto        spanTwo    = getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(ptrTwo), shapeTwo);
  auto              result     = compute(spanOne, spanTwo, static_cast<int>(numWords * 32), stream);
  return nvMolKit::nanobind_bindings::makePyArray(result, nb::make_tuple(numMolsOne, numMolsTwo));
}

cudaStream_t getStream(const std::uintptr_t streamPointer) {
  auto stream = nvMolKit::acquireExternalStream(streamPointer);
  if (!stream) {
    throw std::invalid_argument("Invalid CUDA stream");
  }
  return *stream;
}

}  // namespace

NB_MODULE(_DataStructs, module) {
  module.def(
    "CrossTanimotoSimilarityRawBuffers",
    [](const nb::dict& bitsOne, const nb::dict& bitsTwo, const std::uintptr_t streamPointer) {
      return crossSimilarityGPUFromRawBuffers(
        bitsOne,
        bitsTwo,
        [](const auto& one, const auto& two, const int fpSize, const cudaStream_t stream) {
          return nvMolKit::crossTanimotoSimilarityGpuResult(one, two, fpSize, stream);
        },
        getStream(streamPointer));
    },
    "bitsOne"_a,
    "bitsTwo"_a,
    "stream"_a = 0);
  module.def(
    "CrossCosineSimilarityRawBuffers",
    [](const nb::dict& bitsOne, const nb::dict& bitsTwo, const std::uintptr_t streamPointer) {
      return crossSimilarityGPUFromRawBuffers(
        bitsOne,
        bitsTwo,
        [](const auto& one, const auto& two, const int fpSize, const cudaStream_t stream) {
          return nvMolKit::crossCosineSimilarityGpuResult(one, two, fpSize, stream);
        },
        getStream(streamPointer));
    },
    "bitsOne"_a,
    "bitsTwo"_a,
    "stream"_a = 0);
  module.def(
    "CrossTanimotoSimilarityCPURawBuffers",
    [](const nb::dict& bitsOne, const nb::dict& bitsTwo, const std::uintptr_t streamPointer) {
      return crossSimilarityCPUFromRawBuffers(
        bitsOne,
        bitsTwo,
        [](const auto& one, const auto& two, const int fpSize, const cudaStream_t stream) {
          return nvMolKit::crossTanimotoSimilarityCPUResult(one, two, fpSize, {}, stream);
        },
        getStream(streamPointer));
    },
    "bitsOne"_a,
    "bitsTwo"_a,
    "stream"_a = 0);
  module.def(
    "CrossCosineSimilarityCPURawBuffers",
    [](const nb::dict& bitsOne, const nb::dict& bitsTwo, const std::uintptr_t streamPointer) {
      return crossSimilarityCPUFromRawBuffers(
        bitsOne,
        bitsTwo,
        [](const auto& one, const auto& two, const int fpSize, const cudaStream_t stream) {
          return nvMolKit::crossCosineSimilarityCPUResult(one, two, fpSize, {}, stream);
        },
        getStream(streamPointer));
    },
    "bitsOne"_a,
    "bitsTwo"_a,
    "stream"_a = 0);
}
