// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NVMOLKIT_PRECISION_MODE_H
#define NVMOLKIT_PRECISION_MODE_H

namespace nvMolKit {

//! Precision profiles supported by the 3D minimizers and force fields.
enum class PrecisionMode {
  FULL = 0,  //!< Existing full-precision device path.
  SINGLE     //!< Full single precision for device storage, compute, and reductions.
};

inline const char* precisionModeName(const PrecisionMode precision) {
  return precision == PrecisionMode::SINGLE ? "SINGLE" : "FULL";
}

inline bool usesSinglePrecision(const PrecisionMode precision) {
  return precision == PrecisionMode::SINGLE;
}

//! Full FP32 requires the typed batched kernels; fused per-molecule kernels remain full-precision-only.
inline bool bfgsPrecisionRequiresBatchedBackend(const PrecisionMode precision) {
  return usesSinglePrecision(precision);
}
inline bool bfgsDistGeomPrecisionRequiresBatchedBackend(const PrecisionMode precision) {
  return usesSinglePrecision(precision);
}
inline bool firePrecisionRequiresBatchedBackend(const PrecisionMode precision) {
  return usesSinglePrecision(precision);
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_PRECISION_MODE_H
