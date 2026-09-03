// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NVMOLKIT_PRECISION_OPTIONS_H
#define NVMOLKIT_PRECISION_OPTIONS_H
#include <stdexcept>
#include <string>
namespace nvMolKit {
enum class PrecisionMode {
  LEGACY = 0,
  HESSIAN_F32,
  MINIMIZER_F32,
  FORCEFIELD_F32,
  MIXED,
  SINGLE
};
//! DEFAULT means "inherit this axis from mode". Explicit values override the preset.
enum class PrecisionDType {
  DEFAULT = 0,
  FLOAT32,
  FLOAT64
};
enum class FloatMathMode {
  DEFAULT = 0,
  RELAXED
};
struct PrecisionOptions {
  // Kernel templates use three standard scalar roles:
  //   real     - arithmetic performed by the forcefield or minimizer stage
  //   reduceT  - cross-thread and scalar reduction accumulator arithmetic
  //   storageT - persistent or staged device storage
  // The axes below select those roles independently; DEFAULT inherits the
  // corresponding value from mode.
  PrecisionMode  mode                        = PrecisionMode::LEGACY;
  PrecisionDType forcefieldParameterStorage  = PrecisionDType::DEFAULT;
  PrecisionDType forcefieldCoordinateStorage = PrecisionDType::DEFAULT;
  PrecisionDType forcefieldGradientStorage   = PrecisionDType::DEFAULT;
  PrecisionDType hessianStorage              = PrecisionDType::DEFAULT;
  PrecisionDType minimizerStateStorage       = PrecisionDType::DEFAULT;
  PrecisionDType forcefieldCompute           = PrecisionDType::DEFAULT;
  PrecisionDType minimizerCompute            = PrecisionDType::DEFAULT;
  PrecisionDType reductionCompute            = PrecisionDType::DEFAULT;
  FloatMathMode  floatMath                   = FloatMathMode::DEFAULT;
};
struct ResolvedPrecisionOptions {
  PrecisionDType forcefieldParameterStorage, forcefieldCoordinateStorage, forcefieldGradientStorage;
  PrecisionDType hessianStorage, minimizerStateStorage;
  PrecisionDType forcefieldCompute, minimizerCompute, reductionCompute;
  FloatMathMode  floatMath;
};
inline const char* precisionModeName(PrecisionMode v) {
  switch (v) {
    case PrecisionMode::LEGACY:
      return "LEGACY";
    case PrecisionMode::HESSIAN_F32:
      return "HESSIAN_F32";
    case PrecisionMode::MINIMIZER_F32:
      return "MINIMIZER_F32";
    case PrecisionMode::FORCEFIELD_F32:
      return "FORCEFIELD_F32";
    case PrecisionMode::MIXED:
      return "MIXED";
    case PrecisionMode::SINGLE:
      return "SINGLE";
  }
  throw std::invalid_argument("Unknown precision mode value");
}
inline PrecisionMode parsePrecisionMode(const std::string& v) {
  if (v == "LEGACY" || v == "legacy")
    return PrecisionMode::LEGACY;
  if (v == "HESSIAN_F32" || v == "hessian_f32")
    return PrecisionMode::HESSIAN_F32;
  if (v == "MINIMIZER_F32" || v == "minimizer_f32")
    return PrecisionMode::MINIMIZER_F32;
  if (v == "FORCEFIELD_F32" || v == "forcefield_f32")
    return PrecisionMode::FORCEFIELD_F32;
  if (v == "MIXED" || v == "mixed")
    return PrecisionMode::MIXED;
  if (v == "SINGLE" || v == "single")
    return PrecisionMode::SINGLE;
  throw std::invalid_argument("Unknown precision mode '" + v + "'");
}
inline const char* precisionDTypeName(PrecisionDType v) {
  switch (v) {
    case PrecisionDType::DEFAULT:
      return "DEFAULT";
    case PrecisionDType::FLOAT32:
      return "FLOAT32";
    case PrecisionDType::FLOAT64:
      return "FLOAT64";
  }
  throw std::invalid_argument("Unknown precision dtype value");
}
inline PrecisionDType parsePrecisionDType(const std::string& v) {
  if (v == "DEFAULT" || v == "default")
    return PrecisionDType::DEFAULT;
  if (v == "FLOAT32" || v == "float32" || v == "f32")
    return PrecisionDType::FLOAT32;
  if (v == "FLOAT64" || v == "float64" || v == "f64")
    return PrecisionDType::FLOAT64;
  throw std::invalid_argument("Unknown precision dtype '" + v + "'");
}
inline const char* floatMathModeName(FloatMathMode v) {
  switch (v) {
    case FloatMathMode::DEFAULT:
      return "DEFAULT";
    case FloatMathMode::RELAXED:
      return "RELAXED";
  }
  throw std::invalid_argument("Unknown float math mode value");
}
inline FloatMathMode parseFloatMathMode(const std::string& v) {
  if (v == "DEFAULT" || v == "default")
    return FloatMathMode::DEFAULT;
  if (v == "RELAXED" || v == "relaxed")
    return FloatMathMode::RELAXED;
  throw std::invalid_argument("Unknown float math mode '" + v + "'");
}
inline PrecisionDType overrideOr(PrecisionDType v, PrecisionDType preset) {
  return v == PrecisionDType::DEFAULT ? preset : v;
}
inline ResolvedPrecisionOptions resolvePrecisionOptions(const PrecisionOptions& o) {
  const bool ff32 =
    o.mode == PrecisionMode::FORCEFIELD_F32 || o.mode == PrecisionMode::MIXED || o.mode == PrecisionMode::SINGLE;
  const bool min32 =
    o.mode == PrecisionMode::MINIMIZER_F32 || o.mode == PrecisionMode::MIXED || o.mode == PrecisionMode::SINGLE;
  const bool               h32          = o.mode == PrecisionMode::HESSIAN_F32 || min32;
  const bool               single       = o.mode == PrecisionMode::SINGLE;
  const bool               mixedStorage = o.mode == PrecisionMode::MIXED || single;
  ResolvedPrecisionOptions r{
    overrideOr(o.forcefieldParameterStorage, ff32 ? PrecisionDType::FLOAT32 : PrecisionDType::FLOAT64),
    overrideOr(o.forcefieldCoordinateStorage, mixedStorage ? PrecisionDType::FLOAT32 : PrecisionDType::FLOAT64),
    overrideOr(o.forcefieldGradientStorage, mixedStorage ? PrecisionDType::FLOAT32 : PrecisionDType::FLOAT64),
    overrideOr(o.hessianStorage, h32 ? PrecisionDType::FLOAT32 : PrecisionDType::FLOAT64),
    overrideOr(o.minimizerStateStorage, min32 ? PrecisionDType::FLOAT32 : PrecisionDType::FLOAT64),
    overrideOr(o.forcefieldCompute, ff32 ? PrecisionDType::FLOAT32 : PrecisionDType::FLOAT64),
    overrideOr(o.minimizerCompute, min32 ? PrecisionDType::FLOAT32 : PrecisionDType::FLOAT64),
    overrideOr(o.reductionCompute, single ? PrecisionDType::FLOAT32 : PrecisionDType::FLOAT64),
    o.floatMath == FloatMathMode::DEFAULT ? FloatMathMode::RELAXED : o.floatMath};
  return r;
}
inline bool isFloat32(PrecisionDType v) {
  return v == PrecisionDType::FLOAT32;
}
inline bool usesFloatHessian(const PrecisionOptions& o) {
  return isFloat32(resolvePrecisionOptions(o).hessianStorage);
}
inline bool usesFloatMinimizerState(const PrecisionOptions& o) {
  return isFloat32(resolvePrecisionOptions(o).minimizerStateStorage);
}
inline bool usesFloatForcefield(const PrecisionOptions& o) {
  return isFloat32(resolvePrecisionOptions(o).forcefieldParameterStorage);
}
inline bool usesFloatForcefieldCoordinates(const PrecisionOptions& o) {
  return isFloat32(resolvePrecisionOptions(o).forcefieldCoordinateStorage);
}
inline bool usesFloatForcefieldGradients(const PrecisionOptions& o) {
  return isFloat32(resolvePrecisionOptions(o).forcefieldGradientStorage);
}
inline bool usesFloatForcefieldCompute(const PrecisionOptions& o) {
  return isFloat32(resolvePrecisionOptions(o).forcefieldCompute);
}
inline bool usesFloatMinimizerCompute(const PrecisionOptions& o) {
  return isFloat32(resolvePrecisionOptions(o).minimizerCompute);
}
inline bool usesFloatReduction(const PrecisionOptions& o) {
  return isFloat32(resolvePrecisionOptions(o).reductionCompute);
}
//! The per-molecule BFGS kernels support float force-field parameters and
//! float Hessian storage, but all other storage and arithmetic roles are the
//! legacy double specializations.
inline bool bfgsPrecisionRequiresBatchedBackend(const PrecisionOptions& o) {
  const auto resolved = resolvePrecisionOptions(o);
  const bool explicitUnsupportedAxis =
    o.forcefieldCoordinateStorage != PrecisionDType::DEFAULT ||
    o.forcefieldGradientStorage != PrecisionDType::DEFAULT || o.minimizerStateStorage != PrecisionDType::DEFAULT ||
    o.forcefieldCompute != PrecisionDType::DEFAULT || o.minimizerCompute != PrecisionDType::DEFAULT ||
    o.reductionCompute != PrecisionDType::DEFAULT;
  return explicitUnsupportedAxis || isFloat32(resolved.forcefieldCoordinateStorage) ||
         isFloat32(resolved.forcefieldGradientStorage) || isFloat32(resolved.minimizerStateStorage) ||
         isFloat32(resolved.forcefieldCompute) || isFloat32(resolved.minimizerCompute) ||
         isFloat32(resolved.reductionCompute);
}
//! DG/ETK per-molecule BFGS kernels share the float-Hessian specialization,
//! but unlike MMFF they do not have float force-field parameter buffers.
//! Stage dispatch should use this predicate before entering those kernels.
inline bool bfgsDistGeomPrecisionRequiresBatchedBackend(const PrecisionOptions& o) {
  const auto resolved = resolvePrecisionOptions(o);
  return bfgsPrecisionRequiresBatchedBackend(o) || isFloat32(resolved.forcefieldParameterStorage);
}
//! The per-molecule FIRE kernels additionally support float minimizer-state
//! storage. Compute and reduction roles, and force-field coordinate/gradient
//! storage, require the independently-dispatched batched kernels.
inline bool firePrecisionRequiresBatchedBackend(const PrecisionOptions& o) {
  const auto resolved = resolvePrecisionOptions(o);
  const bool explicitUnsupportedAxis =
    o.forcefieldCoordinateStorage != PrecisionDType::DEFAULT ||
    o.forcefieldGradientStorage != PrecisionDType::DEFAULT || o.forcefieldCompute != PrecisionDType::DEFAULT ||
    o.minimizerCompute != PrecisionDType::DEFAULT || o.reductionCompute != PrecisionDType::DEFAULT;
  return explicitUnsupportedAxis || isFloat32(resolved.forcefieldCoordinateStorage) ||
         isFloat32(resolved.forcefieldGradientStorage) || isFloat32(resolved.forcefieldCompute) ||
         isFloat32(resolved.minimizerCompute) || isFloat32(resolved.reductionCompute);
}
inline PrecisionOptions withMode(PrecisionMode m) {
  PrecisionOptions o;
  o.mode = m;
  return o;
}
inline bool usesFloatHessian(PrecisionMode m) {
  return usesFloatHessian(withMode(m));
}
inline bool usesFloatMinimizerState(PrecisionMode m) {
  return usesFloatMinimizerState(withMode(m));
}
inline bool usesFloatForcefield(PrecisionMode m) {
  return usesFloatForcefield(withMode(m));
}
inline bool usesFloatForcefieldCoordinates(PrecisionMode m) {
  return usesFloatForcefieldCoordinates(withMode(m));
}
inline bool usesFloatForcefieldGradients(PrecisionMode m) {
  return usesFloatForcefieldGradients(withMode(m));
}
}  // namespace nvMolKit
#endif
