// Optimized nvMolKit fMCS device implementation.
// Same search semantics as the reference (RDKit fMCS seed-grow with exact
// per-seed substructure verification), but with all per-pair invariants hoisted
// into shared memory and every per-node helper made warp-parallel.
#ifndef FMCS_OPT_CUH
#define FMCS_OPT_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace fmcsopt {

constexpr int      kBlockThreads = 512;
constexpr int      kGroupSize    = 32;
constexpr int      kNumGroups    = kBlockThreads / kGroupSize;  // 16
constexpr int      kQueueCap     = 4096;
constexpr int      kMaxCls       = 8;
constexpr int      kMaxDeg       = 8;
constexpr int      kMaxSlots     = 8;  // CSR slots per atom
constexpr uint8_t  kUnmapped     = 0xFFu;
constexpr unsigned kFull         = 0xFFFFFFFFu;

template <int T> struct WordSel {
  using type = uint32_t;
};
template <> struct WordSel<64> {
  using type = uint64_t;
};

__device__ __forceinline__ int popcW(uint32_t w) {
  return __popc(w);
}
__device__ __forceinline__ int popcW(uint64_t w) {
  return __popcll(w);
}
__device__ __forceinline__ int ffsW(uint32_t w) {
  return __ffs(static_cast<int>(w)) - 1;
}
__device__ __forceinline__ int ffsW(uint64_t w) {
  return __ffsll(static_cast<long long>(w)) - 1;
}

// ---------------------------------------------------------------------------
// POD layouts. Sizes are byte-identical to the reference QueuedSeed so the
// supplied 4096-entry queue slab is used exactly as handed in.
// ---------------------------------------------------------------------------
template <int T> struct SeedT {
  using W = typename WordSel<T>::type;
  W        atoms;
  W        bonds;
  W        excl;
  W        lastAdded;
  uint16_t numAtoms;
  uint16_t numBonds;
  uint16_t remA;
  uint16_t remB;
  uint16_t stage;
};

template <int T> struct MatchT {
  using W = typename WordSel<T>::type;
  uint8_t  tAtom[T];
  uint8_t  tBond[T];
  W        visA;
  W        visB;
  uint16_t mAtoms;
  uint16_t mBonds;
  uint8_t  empty;
};

template <int T> struct alignas(16) QS {
  SeedT<T>  seed;
  MatchT<T> match;
};

struct NBond {
  uint8_t bondIdx;
  uint8_t newAtom;
  uint8_t ring;  // 1 = ring closing (both endpoints already in seed)
  uint8_t alive;
};

// Per-pair descriptor resolved on the device straight from pair_meta.
struct PairPtrs {
  const uint32_t* qRow;
  const uint32_t* qCol;
  const uint32_t* qBnd;
  const uint32_t* qEp;
  const uint32_t* tRow;
  const uint32_t* tCol;
  const uint32_t* tBnd;
  const uint32_t* tEp;
  const uint32_t* aTab;
  const uint32_t* bTab;
  int             qNA, qNB, tNA, tNB, aWPR, bWPR;
};

// ---------------------------------------------------------------------------
// Block shared state (allocated dynamically).
// ---------------------------------------------------------------------------
template <int T> struct BlockShared {
  using W                        = typename WordSel<T>::type;
  static constexpr int MA        = T;
  static constexpr int MB        = T;
  static constexpr int MAXE      = 2 * T;
  static constexpr int kAtomRegs = (T + 31) / 32;

  // --- per group hot state (16B aligned first) ---
  QS<T> cur[kNumGroups];
  QS<T> big[kNumGroups];
  QS<T> best;

  // --- pair invariants ---
  W atomRow[MA];              // query atom -> compatible target atoms
  W incBond[MA];              // query atom -> incident query bonds
  W bondRow[MB];              // query bond -> compatible target bonds
  W degAtLeast[kMaxDeg + 1];  // target atoms with degree >= d
  W clsRep[kMaxCls];
  W clsNbr[kMaxCls][MA];      // bond class, target atom -> reachable target atoms
  W p1Matched;

  W depthCand[kNumGroups][MA];

  uint16_t backEdges[kNumGroups][MA][kMaxSlots];
  NBond    nb[kNumGroups][MB];

  uint8_t qRow[MA + 1], tRow[MA + 1];
  uint8_t qCol[MAXE], qBnd[MAXE], tCol[MAXE], tBnd[MAXE];
  uint8_t qEpU[MB], qEpV[MB], tEpU[MB], tEpV[MB];
  uint8_t bondCls[MB];
  uint8_t p1TB[MB], p1AU[MB], p1AV[MB];

  uint8_t order[kNumGroups][MA];
  uint8_t orderPos[kNumGroups][MA];
  uint8_t sdeg[kNumGroups][MA];
  uint8_t tgtForQ[kNumGroups][MA];
  uint8_t beCount[kNumGroups][MA];

  int nbCount[kNumGroups];
  int popped[kNumGroups];
  int found[kNumGroups];

  int qTop;
  int bestScore;
  int bestLock;
  int overflowed;
  int phase2Done;
  int qNA, qNB, tNA, tNB, numCls, p1Count;
};

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void warpCopyBytes(void* dst, const void* src, int bytes, int lane) {
  uint32_t*       d = reinterpret_cast<uint32_t*>(dst);
  const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
  const int       n = bytes >> 2;
  for (int i = lane; i < n; i += kGroupSize)
    d[i] = s[i];
}

__device__ __forceinline__ unsigned warpMaxU(unsigned v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = max(v, __shfl_xor_sync(kFull, v, off));
  return v;
}

template <class W> __device__ __forceinline__ W warpOrW(W v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v |= __shfl_xor_sync(kFull, v, off);
  return v;
}

}  // namespace fmcsopt

#endif  // FMCS_OPT_CUH
