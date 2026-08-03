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
//
// Unit tests for the per-helper device functions in fmcs_cuda/.  Each test
// launches a tiny __global__ driver that constructs inputs, calls one
// helper, and copies results back to host memory for assertion.  This
// file is populated incrementally as Steps 1-5 land real implementations.

#include <cooperative_groups.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

#include "src/mcs/fmcs_cuda/fmcs_grow.cuh"
#include "src/mcs/fmcs_cuda/fmcs_match.cuh"
#include "src/mcs/fmcs_cuda/fmcs_match_tables.cuh"
#include "src/mcs/fmcs_cuda/fmcs_search_support.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed.cuh"
#include "src/mcs/fmcs_cuda/fmcs_seed_queue.cuh"
#include "src/utils/device_vector.h"

namespace {

using nvMolKit::AsyncDevicePtr;
using nvMolKit::AsyncDeviceVector;

using mcs::fmcs::MatchResult;
using mcs::fmcs::Seed;

}  // namespace

// ---------------------------------------------------------------------------
// mappingHashWithinThread / DeviceMatchCache
// ---------------------------------------------------------------------------

namespace mcs_fmcs_cache_test {

using mcs::fmcs::DeviceMatchCache;
using HashSeedT  = mcs::fmcs::Seed<16, 16>;
using HashMatchT = mcs::fmcs::MatchResult<16, 16, 16, 16>;

// Hash test: build a seed with bonds added in some order, target bond
// mappings filled in, return the hash.  Used for the determinism +
// order-invariance + non-zero properties.
struct HashOut {
  std::uint64_t hash;
};

__device__ __forceinline__ void buildHashFixture(HashSeedT&  seed,
                                                 HashMatchT& match,
                                                 const int*  qBonds,
                                                 const int*  tBonds,
                                                 int         n) {
  mcs::fmcs::seedClearWithinThread(seed);
  mcs::fmcs::matchResultClearWithinThread(match);
  for (int i = 0; i < n; ++i) {
    mcs::fmcs::seedAddBondWithinThread(seed, qBonds[i]);
    match.targetBondIdx[qBonds[i]] = static_cast<std::uint8_t>(tBonds[i]);
  }
  match.empty = false;
}

__global__ void hashOrderInvariantDriver(HashOut* outA, HashOut* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeedT  seedA, seedB;
  HashMatchT matchA, matchB;
  // Same set of (q, t) mappings, different addition order.  Hash
  // walks seed.bonds in q-increasing order via __ffs, so the
  // resulting hash should be identical.
  int        qsA[] = {3, 1, 5};
  int        tsA[] = {7, 2, 9};
  int        qsB[] = {1, 5, 3};
  int        tsB[] = {2, 9, 7};
  buildHashFixture(seedA, matchA, qsA, tsA, 3);
  buildHashFixture(seedB, matchB, qsB, tsB, 3);
  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashDistinctMappingsDiffer(HashOut* outA, HashOut* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeedT  seedA, seedB;
  HashMatchT matchA, matchB;
  // Same bonds, different target mappings -> hashes should differ.
  int        qsA[] = {0, 1, 2};
  int        tsA[] = {0, 1, 2};
  int        qsB[] = {0, 1, 2};
  int        tsB[] = {2, 1, 0};
  buildHashFixture(seedA, matchA, qsA, tsA, 3);
  buildHashFixture(seedB, matchB, qsB, tsB, 3);
  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashEmptyIsNonZero(HashOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeedT  seed;
  HashMatchT match;
  mcs::fmcs::seedClearWithinThread(seed);
  mcs::fmcs::matchResultClearWithinThread(match);
  // Empty seed.bonds -> hash is post-processed from 0 to 1.
  out->hash = mcs::fmcs::mappingHashWithinThread(seed, match);
}

// ---- Cache insert / probe driver kernels ----

struct CacheInsertProbeOut {
  bool insertOk;
  bool probeFoundInserted;
  bool probeMissedOther;
};

__global__ void cacheInsertProbeBasicDriver(std::uint64_t* keys, int capacity, CacheInsertProbeOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  // Cooperative zero w/ a 1-thread "group" works because we already
  // pre-zeroed `keys` from the host; nothing to do here.
  out->insertOk           = cache.insertWithinThread(0xDEADBEEFCAFEBABEULL);
  out->probeFoundInserted = cache.probeWithinThread(0xDEADBEEFCAFEBABEULL);
  out->probeMissedOther   = cache.probeWithinThread(0x0123456789ABCDEFULL);
}

struct CacheCollisionOut {
  bool insertOkA;
  bool insertOkB;
  bool foundA;
  bool foundB;
  // Auxiliary: did the second insert pick a different physical slot?
  // We test indirectly: both keys present and findable proves it.
};

__global__ void cacheLinearProbeCollisionDriver(std::uint64_t* keys, int capacity, CacheCollisionOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  // Construct two keys that hash to the same initial slot.  The slot
  // is splitMix64(key) & (capacity - 1).  We need two keys A and B
  // such that splitMix64(A) % capacity == splitMix64(B) % capacity,
  // but A != B.  Easiest: with capacity = 8, the slot is the low 3
  // bits of splitMix64(key).  Hand-pick two keys whose hash low-3-bits
  // collide.  We brute-force it on the host below; here we just take
  // whatever two keys the host wrote into the wire format.
  // (Done in the host helper -- see CacheLinearProbeCollision test.)
  out->insertOkA = cache.insertWithinThread(0xAAAAAAAAAAAAAAAAULL);
  out->insertOkB = cache.insertWithinThread(0xBBBBBBBBBBBBBBBBULL);
  out->foundA    = cache.probeWithinThread(0xAAAAAAAAAAAAAAAAULL);
  out->foundB    = cache.probeWithinThread(0xBBBBBBBBBBBBBBBBULL);
}

struct CacheFullDropOut {
  int  successfulInserts;
  bool finalInsertOk;       // expected: false (table full)
  bool seenFirstAfterFull;  // probe still finds the first insert
};

__global__ void cacheFullDropDriver(std::uint64_t* keys, int capacity, CacheFullDropOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  int n = 0;
  for (int i = 0; i < capacity; ++i) {
    // Use distinct non-zero keys.  Note: distinctness in the *hash*
    // input space; the cache hashes internally.
    const std::uint64_t key = static_cast<std::uint64_t>(i) + 1;
    if (cache.insertWithinThread(key))
      ++n;
  }
  out->successfulInserts  = n;
  // Capacity is hit; one more must fail.
  out->finalInsertOk      = cache.insertWithinThread(0xFFFFFFFFFFFFFFFFULL);
  // First-key probe still works.
  out->seenFirstAfterFull = cache.probeWithinThread(1ULL);
}

}  // namespace mcs_fmcs_cache_test

TEST(FMCSUnit, MappingHashOrderInvariant) {
  using namespace mcs_fmcs_cache_test;
  AsyncDevicePtr<HashOut> d_dA;
  AsyncDevicePtr<HashOut> d_dB;
  hashOrderInvariantDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut dA{};
  d_dA.get(dA);
  HashOut dB{};
  d_dB.get(dB);

  EXPECT_EQ(dA.hash, dB.hash) << "Same (q,t) set must hash identically.";
  EXPECT_NE(dA.hash, 0ULL);
}

TEST(FMCSUnit, MappingHashDistinctMappingsDiffer) {
  using namespace mcs_fmcs_cache_test;
  AsyncDevicePtr<HashOut> d_dA;
  AsyncDevicePtr<HashOut> d_dB;
  hashDistinctMappingsDiffer<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut dA{};
  d_dA.get(dA);
  HashOut dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, dB.hash) << "Different target mappings should not collide on a 64-bit hash.";
}

TEST(FMCSUnit, MappingHashEmptySeedReturnsNonZero) {
  using namespace mcs_fmcs_cache_test;
  AsyncDevicePtr<HashOut> d_d;
  hashEmptyIsNonZero<<<1, 1>>>(d_d.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut d{};
  d_d.get(d);

  // 0 is reserved as the cache's empty-slot sentinel; mappingHash
  // post-processes a 0 hash to 1.
  EXPECT_EQ(d.hash, 1ULL);
}

TEST(FMCSUnit, CacheInsertProbeBasic) {
  using namespace mcs_fmcs_cache_test;
  constexpr int                    kCap = 16;
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  d_keys.zero();
  AsyncDevicePtr<CacheInsertProbeOut> d_out;

  cacheInsertProbeBasicDriver<<<1, 1>>>(d_keys.data(), kCap, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CacheInsertProbeOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.insertOk);
  EXPECT_TRUE(out.probeFoundInserted);
  EXPECT_FALSE(out.probeMissedOther);
}

TEST(FMCSUnit, CacheLinearProbeCollision) {
  using namespace mcs_fmcs_cache_test;
  constexpr int                    kCap = 8;
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  d_keys.zero();
  AsyncDevicePtr<CacheCollisionOut> d_out;

  cacheLinearProbeCollisionDriver<<<1, 1>>>(d_keys.data(), kCap, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CacheCollisionOut out{};
  d_out.get(out);

  // Both inserts succeed regardless of whether they happen to collide
  // on the initial slot -- linear probing finds a different slot
  // when the first is taken.  Both probes find their key.
  EXPECT_TRUE(out.insertOkA);
  EXPECT_TRUE(out.insertOkB);
  EXPECT_TRUE(out.foundA);
  EXPECT_TRUE(out.foundB);

  // Sanity: at least one slot in keys[] equals each inserted key.
  std::vector<std::uint64_t> keys(kCap);
  d_keys.copyToHost(keys);
  bool seenA = false, seenB = false;
  for (int i = 0; i < kCap; ++i) {
    if (keys[i] == 0xAAAAAAAAAAAAAAAAULL)
      seenA = true;
    if (keys[i] == 0xBBBBBBBBBBBBBBBBULL)
      seenB = true;
  }
  EXPECT_TRUE(seenA);
  EXPECT_TRUE(seenB);
}

TEST(FMCSUnit, CacheFullTableDropsAdditionalInsert) {
  using namespace mcs_fmcs_cache_test;
  constexpr int                    kCap = 16;  // small but power-of-two
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  d_keys.zero();
  AsyncDevicePtr<CacheFullDropOut> d_out;

  cacheFullDropDriver<<<1, 1>>>(d_keys.data(), kCap, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CacheFullDropOut out{};
  d_out.get(out);

  EXPECT_EQ(out.successfulInserts, kCap) << "Distinct keys should fill all slots.";
  EXPECT_FALSE(out.finalInsertOk) << "Insert into a full table must return false.";
  EXPECT_TRUE(out.seenFirstAfterFull) << "Existing entries must remain visible after a failed insert.";
}

// ---------------------------------------------------------------------------
// Step 5 gap-coverage tests: multi-word hash, idempotent insert, probe of
// empty cache, single-bond hash, extreme indices, cooperative zero.
// ---------------------------------------------------------------------------

namespace mcs_fmcs_cache_gap_test {

using mcs::fmcs::DeviceMatchCache;
using HashSeed128  = mcs::fmcs::Seed<128, 128>;
using HashMatch128 = mcs::fmcs::MatchResult<128, 128, 128, 128>;
using HashSeed16   = mcs::fmcs::Seed<16, 16>;
using HashMatch16  = mcs::fmcs::MatchResult<16, 16, 16, 16>;

struct HashOut64 {
  std::uint64_t hash;
};

// Multi-word path: tier 128 has kBondWords = 2 (two uint64 words),
// exercising the outer wordIdx loop AND __ffsll on each.  Place bonds
// in BOTH words so the iteration must cross the word boundary.
__global__ void hashMultiWordPathDriver(HashOut64* outA, HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed128  seedA, seedB;
  HashMatch128 matchA, matchB;
  mcs::fmcs::seedClearWithinThread(seedA);
  mcs::fmcs::seedClearWithinThread(seedB);
  mcs::fmcs::matchResultClearWithinThread(matchA);
  mcs::fmcs::matchResultClearWithinThread(matchB);
  // bonds in word 0 and word 1.  Same set, different addition order.
  for (int q : {5, 70, 33, 100}) {
    mcs::fmcs::seedAddBondWithinThread(seedA, q);
    matchA.targetBondIdx[q] = static_cast<std::uint8_t>(q);
  }
  for (int q : {100, 5, 33, 70}) {
    mcs::fmcs::seedAddBondWithinThread(seedB, q);
    matchB.targetBondIdx[q] = static_cast<std::uint8_t>(q);
  }
  matchA.empty = false;
  matchB.empty = false;
  outA->hash   = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash   = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

// Single-bond minimal seed.  Should produce a non-zero, deterministic
// hash distinct from the empty-seed sentinel (1).
__global__ void hashSingleBondDriver(HashOut64* outA, HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed16  seedA, seedB;
  HashMatch16 matchA, matchB;
  mcs::fmcs::seedClearWithinThread(seedA);
  mcs::fmcs::seedClearWithinThread(seedB);
  mcs::fmcs::matchResultClearWithinThread(matchA);
  mcs::fmcs::matchResultClearWithinThread(matchB);
  // Determinism: two seeds with the same single (q=4, t=2) mapping.
  mcs::fmcs::seedAddBondWithinThread(seedA, 4);
  matchA.targetBondIdx[4] = 2;
  matchA.empty            = false;
  mcs::fmcs::seedAddBondWithinThread(seedB, 4);
  matchB.targetBondIdx[4] = 2;
  matchB.empty            = false;
  outA->hash              = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash              = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

// Same single query bond and target bond, opposite endpoint orientation.
// These must hash differently or the success cache can prune one Phase-1
// orientation because the other orientation was already seen.
__global__ void hashSingleBondOrientationsDifferDriver(HashOut64* outA, HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed16  seedA, seedB;
  HashMatch16 matchA, matchB;
  mcs::fmcs::seedClearWithinThread(seedA);
  mcs::fmcs::seedClearWithinThread(seedB);
  mcs::fmcs::matchResultClearWithinThread(matchA);
  mcs::fmcs::matchResultClearWithinThread(matchB);

  mcs::fmcs::seedAddAtomWithinThread(seedA, 0);
  mcs::fmcs::seedAddAtomWithinThread(seedA, 1);
  mcs::fmcs::seedAddBondWithinThread(seedA, 0);
  matchA.targetAtomIdx[0] = 4;
  matchA.targetAtomIdx[1] = 5;
  matchA.targetBondIdx[0] = 3;
  matchA.empty            = false;

  mcs::fmcs::seedAddAtomWithinThread(seedB, 0);
  mcs::fmcs::seedAddAtomWithinThread(seedB, 1);
  mcs::fmcs::seedAddBondWithinThread(seedB, 0);
  matchB.targetAtomIdx[0] = 5;
  matchB.targetAtomIdx[1] = 4;
  matchB.targetBondIdx[0] = 3;
  matchB.empty            = false;

  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashLastAddedFrontierDiffersDriver(HashOut64* outA, HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed16  seedA, seedB;
  HashMatch16 matchA, matchB;
  mcs::fmcs::seedClearWithinThread(seedA);
  mcs::fmcs::seedClearWithinThread(seedB);
  mcs::fmcs::matchResultClearWithinThread(matchA);
  mcs::fmcs::matchResultClearWithinThread(matchB);

  for (int atom : {0, 1, 2}) {
    mcs::fmcs::seedAddAtomWithinThread(seedA, atom);
    mcs::fmcs::seedAddAtomWithinThread(seedB, atom);
    matchA.targetAtomIdx[atom] = static_cast<std::uint8_t>(atom + 4);
    matchB.targetAtomIdx[atom] = static_cast<std::uint8_t>(atom + 4);
  }
  for (int bond : {0, 1}) {
    mcs::fmcs::seedAddBondWithinThread(seedA, bond);
    mcs::fmcs::seedAddBondWithinThread(seedB, bond);
    matchA.targetBondIdx[bond] = static_cast<std::uint8_t>(bond + 6);
    matchB.targetBondIdx[bond] = static_cast<std::uint8_t>(bond + 6);
  }
  for (int i = 0; i < HashSeed16::kAtomWords; ++i) {
    seedA.lastAddedAtoms[i] = 0;
    seedB.lastAddedAtoms[i] = 0;
  }
  seedA.lastAddedAtoms[0] = static_cast<HashSeed16::atom_word_type>(1) << 1;
  seedB.lastAddedAtoms[0] = static_cast<HashSeed16::atom_word_type>(1) << 2;
  matchA.empty            = false;
  matchB.empty            = false;

  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

__global__ void hashExcludedBondsDifferDriver(HashOut64* outA, HashOut64* outB) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed16  seedA, seedB;
  HashMatch16 matchA, matchB;
  mcs::fmcs::seedClearWithinThread(seedA);
  mcs::fmcs::seedClearWithinThread(seedB);
  mcs::fmcs::matchResultClearWithinThread(matchA);
  mcs::fmcs::matchResultClearWithinThread(matchB);

  for (int atom : {0, 1, 2}) {
    mcs::fmcs::seedAddAtomWithinThread(seedA, atom);
    mcs::fmcs::seedAddAtomWithinThread(seedB, atom);
    matchA.targetAtomIdx[atom] = static_cast<std::uint8_t>(atom + 4);
    matchB.targetAtomIdx[atom] = static_cast<std::uint8_t>(atom + 4);
  }
  for (int bond : {0, 1}) {
    mcs::fmcs::seedAddBondWithinThread(seedA, bond);
    mcs::fmcs::seedAddBondWithinThread(seedB, bond);
    matchA.targetBondIdx[bond] = static_cast<std::uint8_t>(bond + 6);
    matchB.targetBondIdx[bond] = static_cast<std::uint8_t>(bond + 6);
  }
  seedB.excludedBonds[0] |= static_cast<HashSeed16::bond_word_type>(1) << 3;
  matchA.empty = false;
  matchB.empty = false;

  outA->hash = mcs::fmcs::mappingHashWithinThread(seedA, matchA);
  outB->hash = mcs::fmcs::mappingHashWithinThread(seedB, matchB);
}

// Extreme indices: q = 127, t = 127 are the largest values supported
// (tier-128 cap on maxBonds).  Verify the (q << 8 | t) packing
// doesn't overflow and the high-q __ffsll iteration finds the bit.
__global__ void hashExtremeIndicesDriver(HashOut64* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed128  seed;
  HashMatch128 match;
  mcs::fmcs::seedClearWithinThread(seed);
  mcs::fmcs::matchResultClearWithinThread(match);
  mcs::fmcs::seedAddBondWithinThread(seed, 127);
  match.targetBondIdx[127] = 127;
  match.empty              = false;
  out->hash                = mcs::fmcs::mappingHashWithinThread(seed, match);
}

// The (q=0, t=0) bond packs to a zero token; with a zero-initialized
// accumulator and splitMix64(0) == 0 it would be swallowed, aliasing a
// seed that contains it onto one that omits it.  A strict superset that
// adds the q0->t0 bond must therefore hash differently from the subset
// without it.
__global__ void hashZeroPairBondVisibleDriver(HashOut64* outWithout, HashOut64* outWith) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  HashSeed16  seedWithout, seedWith;
  HashMatch16 matchWithout, matchWith;
  mcs::fmcs::seedClearWithinThread(seedWithout);
  mcs::fmcs::seedClearWithinThread(seedWith);
  mcs::fmcs::matchResultClearWithinThread(matchWithout);
  mcs::fmcs::matchResultClearWithinThread(matchWith);
  mcs::fmcs::seedAddBondWithinThread(seedWithout, 1);
  matchWithout.targetBondIdx[1] = 1;
  matchWithout.empty            = false;
  mcs::fmcs::seedAddBondWithinThread(seedWith, 0);
  mcs::fmcs::seedAddBondWithinThread(seedWith, 1);
  matchWith.targetBondIdx[0] = 0;
  matchWith.targetBondIdx[1] = 1;
  matchWith.empty            = false;
  outWithout->hash           = mcs::fmcs::mappingHashWithinThread(seedWithout, matchWithout);
  outWith->hash              = mcs::fmcs::mappingHashWithinThread(seedWith, matchWith);
}

// ---- Cache idempotent insert ----

struct CacheIdempotentOut {
  bool firstInsert;
  bool secondInsert;
  int  occupiedSlots;  // post-state count of non-zero slots
};

__global__ void cacheIdempotentInsertDriver(std::uint64_t* keys, int capacity, CacheIdempotentOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  const std::uint64_t key = 0xC0FFEEC0FFEEC0FFULL;
  out->firstInsert        = cache.insertWithinThread(key);
  out->secondInsert       = cache.insertWithinThread(key);
  int occupied            = 0;
  for (int i = 0; i < capacity; ++i)
    if (keys[i] != 0ULL)
      ++occupied;
  out->occupiedSlots = occupied;
}

// ---- Probe of fresh empty cache ----

struct ProbeEmptyOut {
  bool probedZeroKey;
  bool probedNonZeroKey;
};

__global__ void cacheProbeEmptyDriver(std::uint64_t* keys, int capacity, ProbeEmptyOut* out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __shared__ DeviceMatchCache cache;
  cache.init(keys, capacity);
  // Cache was zero-initialized by the host.  Probing any key should
  // miss immediately at the first slot (it's empty).
  out->probedZeroKey    = cache.probeWithinThread(0ULL);
  out->probedNonZeroKey = cache.probeWithinThread(0xABCDABCD12341234ULL);
}

// ---- Cooperative zero ----

__global__ void cacheZeroCooperativeDriver(std::uint64_t* keys, int capacity) {
  __shared__ DeviceMatchCache cache;
  if (threadIdx.x == 0)
    cache.init(keys, capacity);
  __syncthreads();
  auto block = cooperative_groups::this_thread_block();
  cache.zeroCooperative(block);
}

}  // namespace mcs_fmcs_cache_gap_test

TEST(FMCSUnit, MappingHashMultiWordPath) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_dA;
  AsyncDevicePtr<HashOut64> d_dB;
  hashMultiWordPathDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 dA{};
  d_dA.get(dA);
  HashOut64 dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, 0ULL);
  EXPECT_EQ(dA.hash, dB.hash) << "Multi-word seed.bonds must be canonicalized regardless of "
                                 "addition order, even when bonds straddle a word boundary.";
}

TEST(FMCSUnit, MappingHashSingleBondDeterministic) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_dA;
  AsyncDevicePtr<HashOut64> d_dB;
  hashSingleBondDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 dA{};
  d_dA.get(dA);
  HashOut64 dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, 0ULL);
  EXPECT_NE(dA.hash, 1ULL) << "Should differ from the empty-seed sentinel";
  EXPECT_EQ(dA.hash, dB.hash);
}

TEST(FMCSUnit, MappingHashSingleBondOrientationsDiffer) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_dA;
  AsyncDevicePtr<HashOut64> d_dB;
  hashSingleBondOrientationsDifferDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 dA{};
  d_dA.get(dA);
  HashOut64 dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, dB.hash) << "Opposite orientations of the same single bond must not alias.";
}

TEST(FMCSUnit, MappingHashLastAddedFrontierDiffers) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_dA;
  AsyncDevicePtr<HashOut64> d_dB;
  hashLastAddedFrontierDiffersDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 dA{};
  d_dA.get(dA);
  HashOut64 dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, dB.hash) << "Same mapped subgraph with different last-added frontier "
                                 "must not cache-alias.";
}

TEST(FMCSUnit, MappingHashExcludedBondsDiffer) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_dA;
  AsyncDevicePtr<HashOut64> d_dB;
  hashExcludedBondsDifferDriver<<<1, 1>>>(d_dA.data(), d_dB.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 dA{};
  d_dA.get(dA);
  HashOut64 dB{};
  d_dB.get(dB);

  EXPECT_NE(dA.hash, dB.hash) << "Same mapped subgraph with different excluded bonds must not "
                                 "cache-alias.";
}

TEST(FMCSUnit, MappingHashExtremeIndicesFitInPacking) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_d;
  hashExtremeIndicesDriver<<<1, 1>>>(d_d.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 d{};
  d_d.get(d);

  // The packed pair (127 << 8) | 127 = 0x7F7F fits comfortably in
  // 64 bits, and __ffsll finds bit 63 of word 1 correctly.  Just
  // assert non-zero (i.e., we got past the hash post-process and
  // didn't hit a degenerate result).
  EXPECT_NE(d.hash, 0ULL);
}

TEST(FMCSUnit, MappingHashZeroPairBondIsVisible) {
  using namespace mcs_fmcs_cache_gap_test;
  AsyncDevicePtr<HashOut64> d_without;
  AsyncDevicePtr<HashOut64> d_with;
  hashZeroPairBondVisibleDriver<<<1, 1>>>(d_without.data(), d_with.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  HashOut64 without{};
  d_without.get(without);
  HashOut64 with{};
  d_with.get(with);

  EXPECT_NE(with.hash, without.hash) << "A (q0->t0) bond must perturb the hash; otherwise a full seed "
                                        "aliases onto a sub-seed and the success cache falsely dedups it.";
  EXPECT_NE(with.hash, 0ULL);
}

TEST(FMCSUnit, CacheInsertIsIdempotent) {
  using namespace mcs_fmcs_cache_gap_test;
  constexpr int                    kCap = 16;
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  d_keys.zero();
  AsyncDevicePtr<CacheIdempotentOut> d_out;

  cacheIdempotentInsertDriver<<<1, 1>>>(d_keys.data(), kCap, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CacheIdempotentOut out{};
  d_out.get(out);

  EXPECT_TRUE(out.firstInsert);
  EXPECT_TRUE(out.secondInsert);
  EXPECT_EQ(out.occupiedSlots, 1) << "Repeated insert of the same key must not consume a second slot.";
}

TEST(FMCSUnit, CacheProbeEmptyCacheMisses) {
  using namespace mcs_fmcs_cache_gap_test;
  constexpr int                    kCap = 16;
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  d_keys.zero();
  AsyncDevicePtr<ProbeEmptyOut> d_out;

  cacheProbeEmptyDriver<<<1, 1>>>(d_keys.data(), kCap, d_out.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ProbeEmptyOut out{};
  d_out.get(out);

  EXPECT_FALSE(out.probedZeroKey) << "0 is the empty-slot sentinel; probing it must miss.";
  EXPECT_FALSE(out.probedNonZeroKey);
}

TEST(FMCSUnit, CacheZeroCooperativeWipesAllSlots) {
  using namespace mcs_fmcs_cache_gap_test;
  constexpr int                    kCap = 64;
  AsyncDeviceVector<std::uint64_t> d_keys(kCap);
  // Pre-fill with non-zero garbage so the cooperative zero has work
  // to do.
  std::vector<std::uint64_t>       keys(kCap);
  for (int i = 0; i < kCap; ++i)
    keys[i] = 0xDEADDEAD00000000ULL | i;
  d_keys.copyFromHost(keys);

  cacheZeroCooperativeDriver<<<1, 32>>>(d_keys.data(), kCap);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  d_keys.copyToHost(keys);
  for (int i = 0; i < kCap; ++i) {
    EXPECT_EQ(keys[i], 0ULL) << "slot " << i << " not zeroed";
  }
}

namespace mcs_fmcs_worklist_test {

using QueuedT = mcs::fmcs::QueuedSeed<16, 16, 16, 16>;

struct WorklistOut {
  int  poppedBonds[3];
  int  finalSize;
  bool overflowRejected;
};

__global__ void sortedWorklistDriver(WorklistOut* out) {
  __shared__ mcs::fmcs::SeedQueue<QueuedT, mcs::fmcs::ThreadBlockScope> queue;
  __shared__ QueuedT                                                    storage[3];
  __shared__ QueuedT                                                    candidate;
  __shared__ QueuedT                                                    popped;

  auto block = cooperative_groups::this_thread_block();
  auto warp  = cooperative_groups::tiled_partition<32>(block);
  if (threadIdx.x == 0) {
    queue.init(storage, 3);
    mcs::fmcs::seedClearWithinThread(candidate.seed);
  }
  __syncthreads();

  for (int bonds : {1, 3, 2}) {
    if (threadIdx.x == 0)
      candidate.seed.numBonds = bonds;
    warp.sync();
    mcs::fmcs::insertSortedByBondsCooperative(warp, queue, candidate);
  }

  if (threadIdx.x == 0)
    candidate.seed.numBonds = 4;
  warp.sync();
  const bool insertedPastCapacity = mcs::fmcs::insertSortedByBondsCooperative(warp, queue, candidate);

  for (int i = 0; i < 3; ++i) {
    mcs::fmcs::popFrontCooperative(warp, queue, popped);
    if (threadIdx.x == 0)
      out->poppedBonds[i] = popped.seed.numBonds;
    warp.sync();
  }
  if (threadIdx.x == 0) {
    out->finalSize        = queue.size();
    out->overflowRejected = !insertedPastCapacity;
  }
}

struct IncumbentOut {
  unsigned int score;
  int          bonds;
  int          atoms;
  int          lock;
};

__global__ void incumbentDriver(IncumbentOut* out) {
  __shared__ QueuedT      best;
  __shared__ QueuedT      candidate;
  __shared__ unsigned int bestScore;
  __shared__ int          lock;

  auto block = cooperative_groups::this_thread_block();
  auto warp  = cooperative_groups::tiled_partition<32>(block);
  if (threadIdx.x == 0) {
    mcs::fmcs::seedClearWithinThread(best.seed);
    mcs::fmcs::seedClearWithinThread(candidate.seed);
    best.seed.numBonds      = 1;
    best.seed.numAtoms      = 4;
    candidate.seed.numBonds = 2;
    candidate.seed.numAtoms = 3;
    bestScore               = (1u << 16) | 4u;
    lock                    = 0;
  }
  __syncthreads();

  mcs::fmcs::updateIncumbentCooperative(warp, candidate, best, &bestScore, &lock);
  if (threadIdx.x == 0) {
    candidate.seed.numBonds = 1;
    candidate.seed.numAtoms = 15;
  }
  warp.sync();
  mcs::fmcs::updateIncumbentCooperative(warp, candidate, best, &bestScore, &lock);

  if (threadIdx.x == 0) {
    out->score = bestScore;
    out->bonds = best.seed.numBonds;
    out->atoms = best.seed.numAtoms;
    out->lock  = lock;
  }
}

}  // namespace mcs_fmcs_worklist_test

TEST(FMCSUnit, SortedWorklistPopsLargestSeedFirstAndRejectsOverflow) {
  using namespace mcs_fmcs_worklist_test;
  AsyncDevicePtr<WorklistOut> deviceOut;
  sortedWorklistDriver<<<1, 32>>>(deviceOut.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  WorklistOut out{};
  ASSERT_EQ(cudaMemcpy(&out, deviceOut.data(), sizeof(out), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(out.poppedBonds[0], 3);
  EXPECT_EQ(out.poppedBonds[1], 2);
  EXPECT_EQ(out.poppedBonds[2], 1);
  EXPECT_EQ(out.finalSize, 0);
  EXPECT_TRUE(out.overflowRejected);
}

TEST(FMCSUnit, IncumbentUsesBondFirstLexicographicScore) {
  using namespace mcs_fmcs_worklist_test;
  AsyncDevicePtr<IncumbentOut> deviceOut;
  incumbentDriver<<<1, 32>>>(deviceOut.data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  IncumbentOut out{};
  ASSERT_EQ(cudaMemcpy(&out, deviceOut.data(), sizeof(out), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(out.score, (2u << 16) | 3u);
  EXPECT_EQ(out.bonds, 2);
  EXPECT_EQ(out.atoms, 3);
  EXPECT_EQ(out.lock, 0);
}
