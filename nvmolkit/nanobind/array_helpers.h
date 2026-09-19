// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMOLKIT_NANOBIND_ARRAY_HELPERS_H
#define NVMOLKIT_NANOBIND_ARRAY_HELPERS_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "src/utils/device_vector.h"

namespace nvMolKit::nanobind_bindings {

namespace nb = nanobind;

template <typename BlockT> cuda::std::span<const BlockT> getSpanFromDictElems(void* data, const nb::tuple& shape) {
  std::size_t size = nb::cast<std::size_t>(shape[0]);
  for (std::size_t dimension = 1; dimension < nb::len(shape); ++dimension) {
    size *= nb::cast<std::size_t>(shape[dimension]);
  }

  return cuda::std::span<const BlockT>(reinterpret_cast<const BlockT*>(data), size);
}

struct PyArray {
  PyArray()                          = default;
  PyArray(const PyArray&)            = delete;
  PyArray& operator=(const PyArray&) = delete;

  ~PyArray() {
    if (devicePtr != nullptr && owned) {
      cudaFreeAsync(devicePtr, stream);
    }
  }

  nb::dict     cudaArrayInterface;
  void*        devicePtr = nullptr;
  cudaStream_t stream    = nullptr;
  bool         owned     = true;
};

template <typename T> std::string getNumpyType() {
  if constexpr (std::is_same_v<T, float>) {
    return "f4";
  } else if constexpr (std::is_same_v<T, double>) {
    return "f8";
  } else if constexpr (std::is_same_v<T, std::int32_t> || std::is_same_v<T, int>) {
    return "i4";
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return "u4";
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return "l8";
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return "L8";
  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    return "h2";
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return "H2";
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    return "b1";
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    return "B1";
  } else {
    throw std::runtime_error("Unsupported type for numpy array:" + std::string(typeid(T).name()));
  }
}

template <typename T>
std::unique_ptr<PyArray> makePyArray(AsyncDeviceVector<T>& deviceVector,
                                     const std::string&    dTypeStr,
                                     const nb::tuple&      shape) {
  auto array                       = std::make_unique<PyArray>();
  array->stream                    = deviceVector.stream();
  T* const releasedPtr             = deviceVector.release();
  array->devicePtr                 = releasedPtr;
  array->cudaArrayInterface        = nb::dict();
  auto&             arrayInterface = array->cudaArrayInterface;
  const std::string typeStr        = "|" + dTypeStr;
  arrayInterface["shape"]          = shape;
  arrayInterface["typestr"]        = nb::str(typeStr.c_str());
  arrayInterface["data"]           = nb::make_tuple(reinterpret_cast<std::size_t>(releasedPtr), false);
  arrayInterface["version"]        = 2;
  return array;
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>>>
std::unique_ptr<PyArray> makePyArray(AsyncDeviceVector<T>&    deviceVector,
                                     std::optional<nb::tuple> shape = std::nullopt) {
  return makePyArray(deviceVector, getNumpyType<T>(), shape.value_or(nb::make_tuple(deviceVector.size())));
}

template <typename T>
std::unique_ptr<PyArray> makePyArrayBorrowed(AsyncDeviceVector<T>& deviceVector,
                                             const std::string&    dTypeStr,
                                             const nb::tuple&      shape) {
  auto array                       = std::make_unique<PyArray>();
  array->stream                    = deviceVector.stream();
  array->devicePtr                 = static_cast<void*>(deviceVector.data());
  array->owned                     = false;
  array->cudaArrayInterface        = nb::dict();
  auto&             arrayInterface = array->cudaArrayInterface;
  const std::string typeStr        = "|" + dTypeStr;
  arrayInterface["shape"]          = shape;
  arrayInterface["typestr"]        = nb::str(typeStr.c_str());
  arrayInterface["data"]           = nb::make_tuple(reinterpret_cast<std::size_t>(deviceVector.data()), false);
  arrayInterface["version"]        = 2;
  return array;
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>>>
std::unique_ptr<PyArray> makePyArrayBorrowed(AsyncDeviceVector<T>&    deviceVector,
                                             std::optional<nb::tuple> shape = std::nullopt) {
  return makePyArrayBorrowed(deviceVector, getNumpyType<T>(), shape.value_or(nb::make_tuple(deviceVector.size())));
}

}  // namespace nvMolKit::nanobind_bindings

#endif  // NVMOLKIT_NANOBIND_ARRAY_HELPERS_H
