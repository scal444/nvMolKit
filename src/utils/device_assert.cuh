#ifndef NVMOLKIT_DEVICE_ASSERT_CUDA_H
#define NVMOLKIT_DEVICE_ASSERT_CUDA_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CPU_ONLY_MEMBER      __host__
#define DEVICE_ONLY_MEMBER   __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CPU_ONLY_MEMBER
#define DEVICE_ONLY_MEMBER
#endif  // __CUDACC__

namespace nvMolKit {
// Turn this on for dev build device asserts.
constexpr bool enableDeviceAssert = false;

CUDA_CALLABLE_MEMBER inline void debugAssert(const bool cond) {
  if constexpr (enableDeviceAssert) {
    assert(cond);
  }
}


} // namespace nvMolKit



#endif // NVMOLKIT_DEVICE_ASSERT_CUDA_H