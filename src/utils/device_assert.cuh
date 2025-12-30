#ifndef NVMOLKIT_DEVICE_ASSERT_CUDA_H
#define NVMOLKIT_DEVICE_ASSERT_CUDA_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__ __forceinline__
#else
#define CUDA_CALLABLE_MEMBER inline
#endif  // __CUDACC__

#include <cassert>

namespace nvMolKit {
// Turn this on for dev build device asserts.
constexpr bool enableDeviceAssert = false;

CUDA_CALLABLE_MEMBER void debugAssert(const bool cond) {
  if constexpr (enableDeviceAssert) {
    assert(cond);
  }
}

} // namespace nvMolKit

#undef CUDA_CALLABLE_MEMBER

#endif // NVMOLKIT_DEVICE_ASSERT_CUDA_H