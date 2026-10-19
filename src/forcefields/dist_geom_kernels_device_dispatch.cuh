// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NVMOLKIT_DISTGEOM_KERNELS_DEVICE_DISPATCH_CUH
#define NVMOLKIT_DISTGEOM_KERNELS_DEVICE_DISPATCH_CUH

#include <cooperative_groups.h>

#include <cstdint>
#include <type_traits>

#include "src/forcefields/dist_geom_kernels.h"
#include "src/forcefields/kernel_utils.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace nvMolKit::FFKernelUtils;

namespace nvMolKit {
namespace DistGeom {

//! Gradient accumulation shared by every term: grad[k] += v[k] for the consecutive components of one atom.
//! Shared memory: plain atomicAdd (the compiler's float/double add emulation). Global memory on sm_90+: float
//! components go out as native vector reductions (REDG.F32x4 / F32x2), one L2 atomic per aligned pair or quad.
#if defined(__CUDACC_VER_MAJOR__) && \
  (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 2))
#define NVMOLKIT_DG_HAS_VEC_RED 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(NVMOLKIT_DG_HAS_VEC_RED)
//! Relaxed gpu-scope vector reductions on global memory (what atomicAdd(float2/float4) lowers to). Inline PTX on a
//! global-space address: the float2/float4 atomicAdd intrinsics are rejected at compile time when inlining shows a
//! shared-memory pointer, even on the branch excluded by __isGlobal.
__device__ __forceinline__ void redGlobalAddF32x2(float* p, const float a, const float b) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(__cvta_generic_to_global(p)), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ void redGlobalAddF32x4(float*      p,
                                                  const float a,
                                                  const float b,
                                                  const float c,
                                                  const float d) {
  asm volatile(
    "red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(__cvta_generic_to_global(p)), "f"(a), "f"(b), "f"(c), "f"(d)
    : "memory");
}
#endif

template <typename T, int n> __device__ __forceinline__ void gradAtomicAdd(T* grad, const T (&v)[n]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(NVMOLKIT_DG_HAS_VEC_RED)
  if constexpr (std::is_same<T, float>::value && (n == 3 || n == 4)) {
    if (__isGlobal(grad)) {
      const uintptr_t addr = reinterpret_cast<uintptr_t>(grad);
      if constexpr (n == 4) {
        if ((addr & 15u) == 0) {
          redGlobalAddF32x4(grad, v[0], v[1], v[2], v[3]);
          return;
        }
      }
      if ((addr & 7u) == 0) {
        redGlobalAddF32x2(grad, v[0], v[1]);
        if constexpr (n == 4) {
          redGlobalAddF32x2(grad + 2, v[2], v[3]);
        } else {
          atomicAdd(grad + 2, v[2]);
        }
      } else {
        atomicAdd(grad, v[0]);
        redGlobalAddF32x2(grad + 1, v[1], v[2]);
        if constexpr (n == 4) {
          atomicAdd(grad + 3, v[3]);
        }
      }
      return;
    }
  }
#endif
#pragma unroll
  for (int k = 0; k < n; ++k) {
    atomicAdd(grad + k, v[k]);
  }
}

//! Like gradAtomicAdd for three components stored with a 4-value stride: on sm_90+ global memory the padded slot is
//! updated with one float4 reduction whose fourth lane adds -0.0f (an exact identity for every value).
template <typename T> __device__ __forceinline__ void gradAtomicAddPadded3(T* grad, const T (&v)[3]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && defined(NVMOLKIT_DG_HAS_VEC_RED)
  if constexpr (std::is_same<T, float>::value) {
    if (__isGlobal(grad) && (reinterpret_cast<uintptr_t>(grad) & 15u) == 0) {
      redGlobalAddF32x4(grad, v[0], v[1], v[2], -0.0f);
      return;
    }
  }
#endif
  gradAtomicAdd(grad, v);
}

template <typename T> __device__ __forceinline__ void gradAtomicAdd3(T* grad, const T x, const T y, const T z) {
  const T v[3] = {x, y, z};
  gradAtomicAdd(grad, v);
}

//! gradAtomicAdd3 for ETK's 4-value atom stride (xyz components of a padded slot).
template <typename T> __device__ __forceinline__ void etkGradAtomicAdd3(T* grad, const T x, const T y, const T z) {
  const T v[3] = {x, y, z};
  gradAtomicAddPadded3(grad, v);
}

namespace fp64 {
using EnergyForceContribsDevicePtr   = DistGeom::EnergyForceContribsDevicePtr;
using Energy3DForceContribsDevicePtr = DistGeom::Energy3DForceContribsDevicePtr;
constexpr double RAD2DEG             = 180.0 / M_PI;
#define DG_REAL       double
#define DG_PARAM_REAL double
#define DG_RSQRTF     rsqrtf
#define DG_RSQRT      rsqrt
#define DG_SQRTF      sqrtf
#define DG_SQRT       sqrt
#define DG_ACOS       acos
#define DG_FMAX       fmax
#define DG_IS_ZERO    isDoubleZero
#include "src/forcefields/dist_geom_kernels_device.cuh"
#undef DG_REAL
#undef DG_PARAM_REAL
#undef DG_RSQRTF
#undef DG_RSQRT
#undef DG_SQRTF
#undef DG_SQRT
#undef DG_ACOS
#undef DG_FMAX
#undef DG_IS_ZERO
}  // namespace fp64

namespace fp32 {
using EnergyForceContribsDevicePtr   = DistGeom::EnergyForceContribsDevicePtrSingle;
using Energy3DForceContribsDevicePtr = DistGeom::Energy3DForceContribsDevicePtrSingle;
constexpr float RAD2DEG              = 180.0f / static_cast<float>(M_PI);
#define DG_REAL       float
#define DG_PARAM_REAL float
#define DG_RSQRTF     rsqrtf
#define DG_RSQRT      rsqrtf
#define DG_SQRTF      sqrtf
#define DG_SQRT       sqrtf
#define DG_ACOS       acosf
#define DG_FMAX       fmaxf
#define DG_IS_ZERO    isFloatZero
#include "src/forcefields/dist_geom_kernels_device.cuh"
#undef DG_REAL
#undef DG_PARAM_REAL
#undef DG_RSQRTF
#undef DG_RSQRT
#undef DG_SQRTF
#undef DG_SQRT
#undef DG_ACOS
#undef DG_FMAX
#undef DG_IS_ZERO
}  // namespace fp32

using namespace fp64;

}  // namespace DistGeom
}  // namespace nvMolKit

#endif  // NVMOLKIT_DISTGEOM_KERNELS_DEVICE_DISPATCH_CUH
