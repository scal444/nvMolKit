// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "src/gpu_scheduler/cpu_fallback_queue.h"
#include "src/gpu_scheduler/exception_aggregator.h"
#include "src/gpu_scheduler/pipeline.h"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device.h"
#include "src/utils/device_vector.h"

namespace {

using nvMolKit::AsyncDeviceVector;
using nvMolKit::checkReturnCode;
using nvMolKit::ScopedCudaEvent;
using nvMolKit::ScopedStream;
using nvMolKit::WithDevice;
using nvMolKit::gpu_scheduler::Config;
using nvMolKit::gpu_scheduler::CpuFallbackQueue;
using nvMolKit::gpu_scheduler::ExceptionAggregator;
using nvMolKit::gpu_scheduler::FallbackProducerGuard;
using nvMolKit::gpu_scheduler::IndexRange;
using nvMolKit::gpu_scheduler::Pipeline;
using nvMolKit::gpu_scheduler::resolve;
using nvMolKit::gpu_scheduler::ResolvedConfig;

// =============================================================================
// Synthetic workload: y = a*x + b on slices of a flat input vector.
//
// Each input "unit" is a single float x[i]. The workload claims a stride of
// units, packs them into a pinned host buffer, dispatches a small kernel that
// computes y[i] = a*x[i] + b on the GPU, copies results back, and writes them
// into a shared output vector. A subset of inputs (those flagged as "fallback"
// by index parity) are routed through a CPU fallback queue instead.
// =============================================================================

__global__ void axpbKernel(const float* x, float* y, int count, float a, float b) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    y[idx] = a * x[idx] + b;
  }
}

// Workload-owned pinned buffer pool: fixed capacity, blocking acquire. Each
// buffer holds up to capacityPerBuffer floats and a matching index vector.
struct PinnedFloatBuffer {
  std::vector<float> hostX;
  std::vector<float> hostY;
  std::vector<int>   hostIndices;
  int                used = 0;

  void resizeTo(int capacity) {
    hostX.assign(static_cast<size_t>(capacity), 0.0f);
    hostY.assign(static_cast<size_t>(capacity), 0.0f);
    hostIndices.assign(static_cast<size_t>(capacity), 0);
  }
};

class PinnedBufferPool {
 public:
  PinnedBufferPool(int poolSize, int capacityPerBuffer) {
    buffers_.resize(static_cast<size_t>(poolSize));
    for (auto& buffer : buffers_) {
      buffer.resizeTo(capacityPerBuffer);
      free_.push_back(&buffer);
    }
  }

  PinnedFloatBuffer* acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !free_.empty() || shutdown_; });
    if (free_.empty()) {
      return nullptr;
    }
    PinnedFloatBuffer* result = free_.back();
    free_.pop_back();
    return result;
  }

  void release(PinnedFloatBuffer* buffer) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      free_.push_back(buffer);
    }
    cv_.notify_one();
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::vector<PinnedFloatBuffer>  buffers_;
  std::vector<PinnedFloatBuffer*> free_;
  std::mutex                      mutex_;
  std::condition_variable         cv_;
  bool                            shutdown_ = false;
};

struct FallbackEntry {
  int index = 0;
};

// One unit of work flowing through the pipeline. Holds a borrowed pinned
// buffer; releases it back to the pool on destruction so abort paths don't
// leak (postprocess clears pinnedBuffer to opt out of the auto-release).
struct AxpbBatch {
  PinnedFloatBuffer* pinnedBuffer = nullptr;
  PinnedBufferPool*  pool         = nullptr;
  int                count        = 0;

  ~AxpbBatch() {
    if (pinnedBuffer != nullptr && pool != nullptr) {
      pool->release(pinnedBuffer);
    }
  }
};

struct AxpbInputs {
  std::vector<float> x;              // total input vector (one element per unit)
  std::vector<int>   outputIndices;  // for sanity testing - normally implicit
  float              a                = 0.0f;
  float              b                = 0.0f;
  int                unitsPerClaim    = 32;
  int                maxBatchCapacity = 64;

  PinnedBufferPool*                pool                = nullptr;
  CpuFallbackQueue<FallbackEntry>* fallbackQueue       = nullptr;
  bool                             fallbackEvenIndices = false;

  // Faults to inject for negative testing.
  bool failInPreprocess = false;
  bool failInDispatch   = false;

  // Output sink (shared, mutex-protected).
  std::vector<float> output;
  std::mutex         outputMutex;

  // Stat counters for assertions.
  std::atomic<int> numBatchesDispatched{0};
  std::atomic<int> numFallbackProcessed{0};
};

struct AxpbPerGpu {
  int deviceId = -1;
  // Sanity check that PerGpuState is constructed with the device active.
};

struct AxpbSlot {
  ScopedStream             stream;
  ScopedCudaEvent          completion;
  AsyncDeviceVector<float> devX;
  AsyncDeviceVector<float> devY;
  std::vector<float>       hostScratchY;  // back-buffer for sync D2H read

  AxpbSlot(int capacity) : devX(static_cast<size_t>(capacity)), devY(static_cast<size_t>(capacity)) {
    devX.setStream(stream.stream());
    devY.setStream(stream.stream());
    hostScratchY.assign(static_cast<size_t>(capacity), 0.0f);
  }

  cudaStream_t primaryStream() const { return stream.stream(); }
  cudaEvent_t  completionEvent() const { return completion.event(); }
};

// Per-thread context: holds a producer guard for the fallback queue so the
// queue closes only after every preprocessor and runner thread has exited.
struct AxpbPreprocCtx {
  FallbackProducerGuard<FallbackEntry> guard{nullptr};
};

struct AxpbRunnerCtx {
  FallbackProducerGuard<FallbackEntry> guard{nullptr};
};

struct AxpbWorkload {
  using Inputs               = AxpbInputs;
  using PreparedBatch        = AxpbBatch;
  using PerGpuState          = AxpbPerGpu;
  using GpuSlotState         = AxpbSlot;
  using PreprocThreadContext = AxpbPreprocCtx;
  using RunnerThreadContext  = AxpbRunnerCtx;

  static int totalUnits(Inputs& inputs) { return static_cast<int>(inputs.x.size()); }
  static int unitsPerPreprocBatch(Inputs& inputs) { return inputs.unitsPerClaim; }

  static PreprocThreadContext makePreprocCtx(Inputs& inputs) {
    PreprocThreadContext ctx;
    if (inputs.fallbackQueue != nullptr) {
      ctx.guard = FallbackProducerGuard<FallbackEntry>(inputs.fallbackQueue);
    }
    return ctx;
  }

  static RunnerThreadContext makeRunnerCtx(Inputs& inputs) {
    RunnerThreadContext ctx;
    if (inputs.fallbackQueue != nullptr) {
      ctx.guard = FallbackProducerGuard<FallbackEntry>(inputs.fallbackQueue);
    }
    return ctx;
  }

  template <class PushFn>
  static void preprocess(Inputs& inputs, IndexRange range, PreprocThreadContext& /*ctx*/, PushFn pushBatch) {
    if (inputs.failInPreprocess) {
      throw std::runtime_error("preprocess fault");
    }

    PinnedFloatBuffer* buffer = inputs.pool->acquire();
    if (buffer == nullptr) {
      return;
    }

    int written = 0;
    for (int i = range.start; i < range.end; ++i) {
      const bool fallback = inputs.fallbackEvenIndices && (i % 2 == 0);
      if (fallback) {
        inputs.fallbackQueue->enqueue(FallbackEntry{i});
        continue;
      }
      buffer->hostX[static_cast<size_t>(written)]       = inputs.x[static_cast<size_t>(i)];
      buffer->hostIndices[static_cast<size_t>(written)] = i;
      ++written;
    }

    if (written == 0) {
      inputs.pool->release(buffer);
      return;
    }

    auto batch          = std::make_unique<AxpbBatch>();
    batch->pinnedBuffer = buffer;
    batch->pool         = inputs.pool;
    batch->count        = written;
    pushBatch(std::move(batch));
  }

  static std::unique_ptr<PerGpuState> makePerGpuState(Inputs& /*inputs*/, int gpuId) {
    int current = -1;
    cudaCheckError(cudaGetDevice(&current));
    EXPECT_EQ(current, gpuId) << "PerGpuState must be constructed with the device active";
    return std::make_unique<AxpbPerGpu>(AxpbPerGpu{gpuId});
  }

  static std::unique_ptr<GpuSlotState> makeSlotState(Inputs& inputs, PerGpuState& pgs, int gpuId) {
    int current = -1;
    cudaCheckError(cudaGetDevice(&current));
    EXPECT_EQ(current, gpuId) << "GpuSlotState must be constructed with the device active";
    EXPECT_EQ(pgs.deviceId, gpuId);
    return std::make_unique<AxpbSlot>(inputs.maxBatchCapacity);
  }

  static void dispatchAndCopyBack(GpuSlotState& slot,
                                  PerGpuState& /*pgs*/,
                                  PreparedBatch& batch,
                                  Inputs&        inputs,
                                  RunnerThreadContext& /*ctx*/) {
    if (inputs.failInDispatch) {
      throw std::runtime_error("dispatch fault");
    }
    inputs.numBatchesDispatched.fetch_add(1, std::memory_order_relaxed);

    slot.devX.copyFromHost(batch.pinnedBuffer->hostX.data(), static_cast<size_t>(batch.count));
    const int blockSize = 64;
    const int gridSize  = (batch.count + blockSize - 1) / blockSize;
    axpbKernel<<<gridSize, blockSize, 0, slot.primaryStream()>>>(slot.devX.data(),
                                                                 slot.devY.data(),
                                                                 batch.count,
                                                                 inputs.a,
                                                                 inputs.b);
    cudaCheckError(cudaGetLastError());
    slot.devY.copyToHost(batch.pinnedBuffer->hostY.data(), static_cast<size_t>(batch.count));
  }

  static void onAbort(Inputs& inputs) {
    if (inputs.pool != nullptr) {
      inputs.pool->shutdown();
    }
  }

  static void postprocess(GpuSlotState& /*slot*/, PreparedBatch& batch, Inputs& inputs, RunnerThreadContext& /*ctx*/) {
    {
      std::lock_guard<std::mutex> lock(inputs.outputMutex);
      for (int i = 0; i < batch.count; ++i) {
        const int outIdx                           = batch.pinnedBuffer->hostIndices[static_cast<size_t>(i)];
        inputs.output[static_cast<size_t>(outIdx)] = batch.pinnedBuffer->hostY[static_cast<size_t>(i)];
      }
    }

    // Opportunistic fallback drain.
    if (inputs.fallbackQueue != nullptr) {
      inputs.fallbackQueue->tryProcessOne();
    }
  }
};

}  // namespace

// =============================================================================
// Tests
// =============================================================================

class GpuSchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    cudaCheckError(cudaGetDeviceCount(&count));
    if (count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
    deviceCount_ = count;
  }

  int deviceCount_ = 0;
};

TEST(GpuSchedulerConfigTest, ResolveAutoselect) {
  Config config;
  config.gpuIds.push_back(0);
  const ResolvedConfig resolved = resolve(config, /*hardwareConcurrency=*/8, /*currentDevice=*/0);
  EXPECT_EQ(resolved.workerThreadsPerGpu, 4);
  EXPECT_EQ(resolved.globalPreprocessingThreads, 8);
  EXPECT_EQ(resolved.slotsPerWorker, 2);
  EXPECT_EQ(resolved.gpuIds.size(), 1u);
  EXPECT_EQ(resolved.gpuIds[0], 0);
}

TEST(GpuSchedulerConfigTest, SingleRunnerGetsThreeSlots) {
  Config config;
  config.workerThreadsPerGpu = 1;
  config.gpuIds.push_back(0);
  const ResolvedConfig resolved = resolve(config, /*hardwareConcurrency=*/8, /*currentDevice=*/0);
  EXPECT_EQ(resolved.slotsPerWorker, 3);
}

TEST(GpuSchedulerConfigTest, EmptyGpuIdsUsesCurrentDevice) {
  Config               config;
  const ResolvedConfig resolved = resolve(config, /*hardwareConcurrency=*/4, /*currentDevice=*/2);
  ASSERT_EQ(resolved.gpuIds.size(), 1u);
  EXPECT_EQ(resolved.gpuIds[0], 2);
}

TEST(GpuSchedulerConfigTest, MultiGpuSplitsHardwareConcurrency) {
  Config config;
  config.gpuIds                 = {0, 1, 2, 3};
  const ResolvedConfig resolved = resolve(config, /*hardwareConcurrency=*/16, /*currentDevice=*/0);
  EXPECT_EQ(resolved.workerThreadsPerGpu, 4);
}

TEST(GpuSchedulerExceptionAggregatorTest, RecordsAndRethrows) {
  ExceptionAggregator agg;
  EXPECT_FALSE(agg.aborted());
  try {
    throw std::runtime_error("boom");
  } catch (...) {
    agg.recordAndAbort(std::current_exception());
  }
  EXPECT_TRUE(agg.aborted());
  EXPECT_EQ(agg.storedCount(), 1u);
  EXPECT_THROW(agg.rethrowIfAny(), std::runtime_error);
}

TEST(GpuSchedulerExceptionAggregatorTest, RethrowFirstWhenMultiple) {
  ExceptionAggregator agg;
  try {
    throw std::runtime_error("first");
  } catch (...) {
    agg.recordAndAbort(std::current_exception());
  }
  try {
    throw std::logic_error("second");
  } catch (...) {
    agg.recordAndAbort(std::current_exception());
  }
  EXPECT_EQ(agg.storedCount(), 2u);
  try {
    agg.rethrowIfAny();
    FAIL() << "expected throw";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "first");
  }
}

TEST(GpuSchedulerCpuFallbackQueueTest, ProducerGuardClosesQueue) {
  int                             processed = 0;
  CpuFallbackQueue<FallbackEntry> queue([&processed](const FallbackEntry&) { ++processed; });
  {
    FallbackProducerGuard<FallbackEntry> guard(&queue);
    queue.enqueue(FallbackEntry{1});
    queue.enqueue(FallbackEntry{2});
    EXPECT_EQ(queue.activeProducers(), 1);
  }
  EXPECT_EQ(queue.activeProducers(), 0);
  while (queue.tryProcessOne()) {
  }
  EXPECT_EQ(processed, 2);
}

TEST_F(GpuSchedulerTest, RunsAxpbCorrectlySingleGpu) {
  constexpr int N = 1024;
  AxpbInputs    inputs;
  inputs.x.resize(N);
  inputs.output.assign(N, 0.0f);
  std::iota(inputs.x.begin(), inputs.x.end(), 1.0f);
  inputs.a                = 2.0f;
  inputs.b                = 3.0f;
  inputs.unitsPerClaim    = 32;
  inputs.maxBatchCapacity = 64;

  PinnedBufferPool pool(/*poolSize=*/8, /*capacityPerBuffer=*/inputs.maxBatchCapacity);
  inputs.pool = &pool;

  Config config;
  config.workerThreadsPerGpu        = 2;
  config.globalPreprocessingThreads = 4;
  config.slotsPerWorker             = 2;
  config.gpuIds.push_back(0);

  Pipeline<AxpbWorkload> pipeline(config, inputs);
  pipeline.run();

  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(inputs.output[static_cast<size_t>(i)], inputs.a * inputs.x[static_cast<size_t>(i)] + inputs.b)
      << "mismatch at i=" << i;
  }
  EXPECT_GT(inputs.numBatchesDispatched.load(), 0);
}

TEST_F(GpuSchedulerTest, BackpressureHonoredWhenPoolSmall) {
  // Pool of 1 forces preprocess() threads to serialize on acquire/release;
  // pipeline must still complete without deadlock.
  constexpr int N = 512;
  AxpbInputs    inputs;
  inputs.x.resize(N);
  inputs.output.assign(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    inputs.x[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  inputs.a                = 1.5f;
  inputs.b                = -0.25f;
  inputs.unitsPerClaim    = 16;
  inputs.maxBatchCapacity = 32;

  PinnedBufferPool pool(/*poolSize=*/1, /*capacityPerBuffer=*/inputs.maxBatchCapacity);
  inputs.pool = &pool;

  Config config;
  config.workerThreadsPerGpu        = 1;
  config.globalPreprocessingThreads = 4;
  config.slotsPerWorker             = 1;
  config.gpuIds.push_back(0);

  Pipeline<AxpbWorkload> pipeline(config, inputs);
  pipeline.run();

  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(inputs.output[static_cast<size_t>(i)], inputs.a * inputs.x[static_cast<size_t>(i)] + inputs.b);
  }
}

TEST_F(GpuSchedulerTest, FallbackQueueDrainedOpportunistically) {
  constexpr int N = 256;
  AxpbInputs    inputs;
  inputs.x.resize(N);
  inputs.output.assign(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    inputs.x[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  inputs.a                   = 4.0f;
  inputs.b                   = 1.0f;
  inputs.unitsPerClaim       = 16;
  inputs.maxBatchCapacity    = 32;
  inputs.fallbackEvenIndices = true;

  PinnedBufferPool pool(/*poolSize=*/4, /*capacityPerBuffer=*/inputs.maxBatchCapacity);
  inputs.pool = &pool;

  std::atomic<int>                fallbackProcessed{0};
  CpuFallbackQueue<FallbackEntry> fallbackQueue([&](const FallbackEntry& entry) {
    const float value = inputs.a * inputs.x[static_cast<size_t>(entry.index)] + inputs.b;
    {
      std::lock_guard<std::mutex> lock(inputs.outputMutex);
      inputs.output[static_cast<size_t>(entry.index)] = value;
    }
    fallbackProcessed.fetch_add(1, std::memory_order_relaxed);
  });
  inputs.fallbackQueue = &fallbackQueue;

  Config config;
  config.workerThreadsPerGpu        = 2;
  config.globalPreprocessingThreads = 4;
  config.slotsPerWorker             = 2;
  config.gpuIds.push_back(0);

  Pipeline<AxpbWorkload> pipeline(config, inputs);
  pipeline.run();

  // After all preprocessors and runners exit, the queue closes. Drain any
  // residual entries from the main thread (the runner's opportunistic drain
  // doesn't guarantee the queue is empty - it only tries one per postprocess).
  while (fallbackQueue.tryProcessOne()) {
  }

  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(inputs.output[static_cast<size_t>(i)], inputs.a * inputs.x[static_cast<size_t>(i)] + inputs.b)
      << "mismatch at i=" << i;
  }
  EXPECT_EQ(fallbackProcessed.load(), N / 2);
}

TEST_F(GpuSchedulerTest, MultipleGpusSplitWork) {
  if (deviceCount_ < 2) {
    GTEST_SKIP() << "Test requires >= 2 CUDA devices, found " << deviceCount_;
  }
  constexpr int N = 2048;
  AxpbInputs    inputs;
  inputs.x.resize(N);
  inputs.output.assign(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    inputs.x[static_cast<size_t>(i)] = static_cast<float>(i) * 0.5f;
  }
  inputs.a                = 2.5f;
  inputs.b                = 7.0f;
  inputs.unitsPerClaim    = 64;
  inputs.maxBatchCapacity = 128;

  PinnedBufferPool pool(/*poolSize=*/16, /*capacityPerBuffer=*/inputs.maxBatchCapacity);
  inputs.pool = &pool;

  Config config;
  config.workerThreadsPerGpu        = 2;
  config.globalPreprocessingThreads = 4;
  config.slotsPerWorker             = 2;
  config.gpuIds                     = {0, 1};

  Pipeline<AxpbWorkload> pipeline(config, inputs);
  pipeline.run();

  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(inputs.output[static_cast<size_t>(i)], inputs.a * inputs.x[static_cast<size_t>(i)] + inputs.b)
      << "mismatch at i=" << i;
  }
}

TEST_F(GpuSchedulerTest, PreprocessExceptionRethrown) {
  AxpbInputs inputs;
  inputs.x.assign(64, 1.0f);
  inputs.output.assign(64, 0.0f);
  inputs.a                = 1.0f;
  inputs.b                = 0.0f;
  inputs.unitsPerClaim    = 8;
  inputs.maxBatchCapacity = 16;
  inputs.failInPreprocess = true;

  PinnedBufferPool pool(/*poolSize=*/2, /*capacityPerBuffer=*/inputs.maxBatchCapacity);
  inputs.pool = &pool;

  Config config;
  config.workerThreadsPerGpu        = 1;
  config.globalPreprocessingThreads = 2;
  config.slotsPerWorker             = 1;
  config.gpuIds.push_back(0);

  Pipeline<AxpbWorkload> pipeline(config, inputs);
  EXPECT_THROW(pipeline.run(), std::runtime_error);
}

TEST_F(GpuSchedulerTest, DispatchExceptionRethrown) {
  AxpbInputs inputs;
  inputs.x.assign(128, 1.0f);
  inputs.output.assign(128, 0.0f);
  inputs.a                = 1.0f;
  inputs.b                = 0.0f;
  inputs.unitsPerClaim    = 16;
  inputs.maxBatchCapacity = 32;
  inputs.failInDispatch   = true;

  PinnedBufferPool pool(/*poolSize=*/4, /*capacityPerBuffer=*/inputs.maxBatchCapacity);
  inputs.pool = &pool;

  Config config;
  config.workerThreadsPerGpu        = 1;
  config.globalPreprocessingThreads = 2;
  config.slotsPerWorker             = 2;
  config.gpuIds.push_back(0);

  Pipeline<AxpbWorkload> pipeline(config, inputs);
  EXPECT_THROW(pipeline.run(), std::runtime_error);
}

TEST_F(GpuSchedulerTest, ZeroInputUnitsIsNoOp) {
  AxpbInputs inputs;
  inputs.unitsPerClaim    = 1;
  inputs.maxBatchCapacity = 1;
  PinnedBufferPool pool(1, 1);
  inputs.pool = &pool;
  Config config;
  config.gpuIds.push_back(0);
  Pipeline<AxpbWorkload> pipeline(config, inputs);
  pipeline.run();  // should return immediately
}
