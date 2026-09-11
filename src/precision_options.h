// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NVMOLKIT_PRECISION_OPTIONS_H
#define NVMOLKIT_PRECISION_OPTIONS_H

#include <stdexcept>
#include <string>

namespace nvMolKit {

//! Precision profiles supported by the 3D minimizers and force fields.
enum class PrecisionMode {
  LEGACY = 0,  //!< Existing full-precision device path.
  SINGLE       //!< Full single precision for device storage, compute, and reductions.
};

struct PrecisionOptions {
  PrecisionMode mode = PrecisionMode::LEGACY;
};

inline const char* precisionModeName(PrecisionMode mode) {
  switch (mode) {
    case PrecisionMode::LEGACY:
      return "LEGACY";
    case PrecisionMode::SINGLE:
      return "SINGLE";
  }
  throw std::invalid_argument("Unknown precision mode value");
}

inline PrecisionMode parsePrecisionMode(const std::string& value) {
  if (value == "LEGACY" || value == "legacy")
    return PrecisionMode::LEGACY;
  if (value == "SINGLE" || value == "single")
    return PrecisionMode::SINGLE;
  throw std::invalid_argument("Unknown precision mode '" + value + "'. Expected 'LEGACY' or 'SINGLE'.");
}

inline bool usesSinglePrecision(const PrecisionOptions& options) {
  return options.mode == PrecisionMode::SINGLE;
}

//! Full FP32 requires the typed batched kernels; fused per-molecule kernels remain legacy-only.
inline bool bfgsPrecisionRequiresBatchedBackend(const PrecisionOptions& options) {
  return usesSinglePrecision(options);
}
inline bool bfgsDistGeomPrecisionRequiresBatchedBackend(const PrecisionOptions& options) {
  return usesSinglePrecision(options);
}
inline bool firePrecisionRequiresBatchedBackend(const PrecisionOptions& options) {
  return usesSinglePrecision(options);
}

}  // namespace nvMolKit

#endif
