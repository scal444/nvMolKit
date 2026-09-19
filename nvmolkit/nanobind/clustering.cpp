// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "nvmolkit/nanobind/array_helpers.h"
#include "src/butina.h"
#include "src/utils/device.h"

namespace {

namespace nb = nanobind;
using namespace nb::literals;

nb::object wrapButinaResult(nvMolKit::ButinaResult& result, const int numItems, const bool returnCentroids) {
  auto clusterArray = nvMolKit::nanobind_bindings::makePyArray(result.clusterIds, nb::make_tuple(numItems));
  if (!returnCentroids) {
    return nb::cast(std::move(clusterArray));
  }

  auto centroidArray = nvMolKit::nanobind_bindings::makePyArray(result.centroids, nb::make_tuple(result.numClusters));
  return nb::make_tuple(nb::cast(std::move(clusterArray)), nb::cast(std::move(centroidArray)));
}

cudaStream_t getStream(const std::uintptr_t streamPointer) {
  auto stream = nvMolKit::acquireExternalStream(streamPointer);
  if (!stream) {
    throw std::invalid_argument("Invalid CUDA stream");
  }
  return *stream;
}

}  // namespace

NB_MODULE(_clustering, module) {
  module.def(
    "butina",
    [](const nb::dict&      distanceMatrix,
       const double         cutoff,
       const int            neighborlistMaxSize,
       const bool           returnCentroids,
       const bool           reordering,
       const std::uintptr_t streamPointer) {
      const nb::tuple shape      = nb::cast<nb::tuple>(distanceMatrix["shape"]);
      const int       matrixSize = nb::cast<int>(shape[0]);
      const nb::tuple data       = nb::cast<nb::tuple>(distanceMatrix["data"]);
      const auto      pointer    = nb::cast<std::size_t>(data[0]);
      const auto      matrix =
        nvMolKit::nanobind_bindings::getSpanFromDictElems<double>(reinterpret_cast<void*>(pointer), shape);
      auto result = nvMolKit::butinaFromDistanceMatrix(matrix,
                                                       matrixSize,
                                                       cutoff,
                                                       neighborlistMaxSize,
                                                       returnCentroids,
                                                       reordering,
                                                       getStream(streamPointer));
      return wrapButinaResult(result, matrixSize, returnCentroids);
    },
    "distance_matrix"_a,
    "cutoff"_a,
    "neighborlist_max_size"_a = 64,
    "return_centroids"_a      = false,
    "reordering"_a            = true,
    "stream"_a                = 0);

  module.def(
    "fused_butina",
    [](const nb::dict&      fingerprints,
       const double         cutoff,
       const bool           returnCentroids,
       const std::string&   metric,
       const std::uintptr_t streamPointer) {
      const nb::tuple shape = nb::cast<nb::tuple>(fingerprints["shape"]);
      if (nb::len(shape) != 2) {
        throw std::invalid_argument("fingerprints must be a 2D matrix");
      }
      const int       numItems = nb::cast<int>(shape[0]);
      const int       numWords = nb::cast<int>(shape[1]);
      const nb::tuple data     = nb::cast<nb::tuple>(fingerprints["data"]);
      const auto      pointer  = nb::cast<std::size_t>(data[0]);
      const auto      bits =
        nvMolKit::nanobind_bindings::getSpanFromDictElems<std::uint32_t>(reinterpret_cast<void*>(pointer), shape);

      nvMolKit::FingerprintSimilarityMetric parsedMetric;
      if (metric == "tanimoto") {
        parsedMetric = nvMolKit::FingerprintSimilarityMetric::Tanimoto;
      } else if (metric == "cosine") {
        parsedMetric = nvMolKit::FingerprintSimilarityMetric::Cosine;
      } else {
        throw std::invalid_argument("metric must be one of ['tanimoto', 'cosine']");
      }

      auto result = nvMolKit::fusedButinaGpu(bits,
                                             numItems,
                                             numWords,
                                             cutoff,
                                             parsedMetric,
                                             returnCentroids,
                                             getStream(streamPointer));
      return wrapButinaResult(result, numItems, returnCentroids);
    },
    "fingerprints"_a,
    "cutoff"_a,
    "return_centroids"_a = false,
    "metric"_a           = "tanimoto",
    "stream"_a           = 0);
}
