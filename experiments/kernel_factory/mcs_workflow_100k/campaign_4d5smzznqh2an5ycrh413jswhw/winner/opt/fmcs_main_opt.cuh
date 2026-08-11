#ifndef FMCS_MAIN_OPT_CUH
#define FMCS_MAIN_OPT_CUH

#include "opt/fmcs_kernel_opt.cuh"

namespace fmcsopt {

template <int T>
__device__ __forceinline__ void addNewBond(SeedT<T>& s, const NBond& nb) {
  using W       = typename WordSel<T>::type;
  const W bmask = static_cast<W>(1) << nb.bondIdx;
  s.bonds |= bmask;
  s.excl |= bmask;
  s.numBonds += 1;
  if (!nb.ring) {
    const W amask = static_cast<W>(1) << nb.newAtom;
    if ((s.atoms & amask) == 0) {
      s.atoms |= amask;
      s.lastAdded |= amask;
      s.numAtoms += 1;
    }
  }
}

template <int T> __device__ __forceinline__ int readBestScore(BlockShared<T>& S, int lane) {
  int v = 0;
  if (lane == 0)
    v = *(volatile int*)&S.bestScore;
  return __shfl_sync(kFull, v, 0);
}

template <int T> __device__ __forceinline__ int readFlag(int* p, int lane) {
  int v = 0;
  if (lane == 0)
    v = *(volatile int*)p;
  return __shfl_sync(kFull, v, 0);
}

// ===========================================================================
template <int T>
__global__ __launch_bounds__(kBlockThreads) void fmcsOptKernel(const int32_t* __restrict__ meta,
                                                               const uint32_t* __restrict__ qRowB,
                                                               const uint32_t* __restrict__ qColB,
                                                               const uint32_t* __restrict__ qBndB,
                                                               const uint32_t* __restrict__ qEpB,
                                                               const uint32_t* __restrict__ tRowB,
                                                               const uint32_t* __restrict__ tColB,
                                                               const uint32_t* __restrict__ tBndB,
                                                               const uint32_t* __restrict__ tEpB,
                                                               const uint32_t* __restrict__ aTabB,
                                                               const uint32_t* __restrict__ bTabB,
                                                               QS<T>* __restrict__ queueAll,
                                                               int32_t* __restrict__ stats,
                                                               uint8_t* __restrict__ atomMapping,
                                                               uint8_t* __restrict__ bondMapping,
                                                               int numPairs) {
  using W = typename WordSel<T>::type;
  extern __shared__ __align__(16) char smemRaw[];
  BlockShared<T>& S = *reinterpret_cast<BlockShared<T>*>(smemRaw);

  const int pairIdx = blockIdx.x;
  if (pairIdx >= numPairs)
    return;
  const int tid  = threadIdx.x;
  const int g    = tid >> 5;
  const int lane = tid & 31;

  const int32_t* m = meta + static_cast<size_t>(pairIdx) * 16;
  PairPtrs       P;
  P.qNA  = m[0];
  P.qNB  = m[1];
  P.tNA  = m[2];
  P.tNB  = m[3];
  P.qRow = qRowB + m[4];
  P.qCol = qColB + m[5];
  P.qBnd = qBndB + m[5];
  P.qEp  = qEpB + m[6];
  P.tRow = tRowB + m[7];
  P.tCol = tColB + m[8];
  P.tBnd = tBndB + m[8];
  P.tEp  = tEpB + m[9];
  P.aTab = aTabB + m[10];
  P.aWPR = m[11];
  P.bTab = bTabB + m[12];
  P.bWPR = m[13];

  QS<T>* qbase = queueAll + static_cast<size_t>(pairIdx) * kQueueCap;

  loadPair<T>(S, P, tid);

  // clear the incumbent
  if (tid < T) {
    S.best.match.tAtom[tid] = kUnmapped;
    S.best.match.tBond[tid] = kUnmapped;
  }
  if (tid == 0) {
    S.best.seed  = SeedT<T>{0, 0, 0, 0, 0, 0, 0, 0, 0};
    S.best.match.visA   = 0;
    S.best.match.visB   = 0;
    S.best.match.mAtoms = 0;
    S.best.match.mBonds = 0;
    S.best.match.empty  = 1;
  }
  __syncthreads();

  const int qNA = S.qNA, qNB = S.qNB;
  const W   validBonds = (qNB >= static_cast<int>(sizeof(W) * 8)) ? ~static_cast<W>(0)
                                                                 : ((static_cast<W>(1) << qNB) - 1);

  // ---------------- Phase 1: initial one-bond seeds ----------------
  // Whether query bond q embeds depends only on q, so every bond is tested in
  // parallel; the RDKit prefix/failure exclusion bookkeeping is then pure
  // bitmask algebra.
  if (tid < qNB) {
    const int q  = tid;
    const int u  = S.qEpU[q], v = S.qEpV[q];
    const W   au = S.atomRow[u], av = S.atomRow[v];
    W         row = S.bondRow[q];
    int       tb = -1, ta_u = -1, ta_v = -1;
    while (row != 0) {
      const int j = ffsW(row);
      row &= row - 1;
      const int tu = S.tEpU[j], tv = S.tEpV[j];
      if (((au >> tu) & 1) && ((av >> tv) & 1)) {
        tb   = j;
        ta_u = tu;
        ta_v = tv;
        break;
      }
      if (((au >> tv) & 1) && ((av >> tu) & 1)) {
        tb   = j;
        ta_u = tv;
        ta_v = tu;
        break;
      }
    }
    S.p1TB[q] = static_cast<uint8_t>(tb < 0 ? 0xFF : tb);
    S.p1AU[q] = static_cast<uint8_t>(tb < 0 ? 0xFF : ta_u);
    S.p1AV[q] = static_cast<uint8_t>(tb < 0 ? 0xFF : ta_v);
  }
  __syncthreads();

  if (tid == 0) {
    W mm = 0;
    for (int q = 0; q < qNB; ++q)
      if (S.p1TB[q] != 0xFF)
        mm |= static_cast<W>(1) << q;
    S.p1Matched = mm;
    S.p1Count   = popcW(mm);
  }
  __syncthreads();
  const W   matchedMask = S.p1Matched;
  const int numMatched  = S.p1Count;
  const W   notMatched  = validBonds & ~matchedMask;

  for (int mi = g; mi < numMatched; mi += kNumGroups) {
    // mi-th matched query bond
    W   mm = matchedMask;
    int q  = -1;
    {
      W t = mm;
      for (int k = 0; k < mi; ++k)
        t &= t - 1;
      q = ffsW(t);
    }
    const int u = S.qEpU[q], v = S.qEpV[q];
    const W   low = (q + 1 >= static_cast<int>(sizeof(W) * 8)) ? ~static_cast<W>(0)
                                                              : ((static_cast<W>(1) << (q + 1)) - 1);
    QS<T>& c = S.cur[g];
    if (lane == 0) {
      c.seed.atoms     = (static_cast<W>(1) << u) | (static_cast<W>(1) << v);
      c.seed.lastAdded = c.seed.atoms;
      c.seed.bonds     = static_cast<W>(1) << q;
      c.seed.excl      = low;
      c.seed.numAtoms  = (u == v) ? 1 : 2;
      c.seed.numBonds  = 1;
      c.seed.stage     = 0;
      c.seed.remA      = 0;
      c.seed.remB      = 0;
    }
    __syncwarp();
    computeRemaining<T>(S, c.seed, lane);
    // full exclusion set: prefix plus every later bond that failed to match
    if (lane == 0)
      c.seed.excl = low | (notMatched & ~low);
    for (int i = lane; i < T; i += kGroupSize) {
      c.match.tAtom[i] = kUnmapped;
      c.match.tBond[i] = kUnmapped;
    }
    __syncwarp();
    if (lane == 0) {
      const int tb = S.p1TB[q];
      const int au = S.p1AU[q];
      const int av = S.p1AV[q];
      c.match.tAtom[u]  = static_cast<uint8_t>(au);
      c.match.tAtom[v]  = static_cast<uint8_t>(av);
      c.match.tBond[q]  = static_cast<uint8_t>(tb);
      c.match.visA      = (static_cast<W>(1) << au) | (static_cast<W>(1) << av);
      c.match.visB      = static_cast<W>(1) << tb;
      c.match.mAtoms    = c.seed.numAtoms;
      c.match.mBonds    = 1;
      c.match.empty     = 0;
    }
    __syncwarp();
    warpCopyBytes(&qbase[mi], &c, sizeof(QS<T>), lane);
    __syncwarp();
  }
  __syncthreads();
  if (tid == 0) {
    S.qTop = numMatched;
    if (numMatched > 0)
      S.bestScore = (1 << 16) | 2;
  }
  __syncthreads();
  if (numMatched > 0 && tid < kGroupSize)
    warpCopyBytes(&S.best, &qbase[0], sizeof(QS<T>), tid);
  __syncthreads();

  // ---------------- Phase 2: grow ----------------
  while (true) {
    if (tid == 0)
      S.phase2Done = (S.overflowed != 0 || S.qTop == 0) ? 1 : 0;
    __syncthreads();
    if (S.phase2Done)
      break;

    const bool got = popSeed<T>(S, qbase, S.cur[g], lane);
    if (lane == 0)
      S.popped[g] = got ? 1 : 0;
    __syncwarp();
    __syncthreads();

    do {
      if (!S.popped[g])
        break;
      QS<T>& cur = S.cur[g];
      {
        const int bs = readBestScore<T>(S, lane);
        if (!canGrowBigger(cur.seed.numBonds, cur.seed.remB, cur.seed.numAtoms, cur.seed.remA, bs >> 16,
                           bs & 0xFFFF))
          break;
      }
      updateIncumbent<T>(S, cur, lane);

      // ---- fill new bonds ----
      if (lane == 0)
        S.nbCount[g] = 0;
      __syncwarp();
      {
        const W excl = cur.seed.excl;
        const W la   = cur.seed.lastAdded;
        const W at   = cur.seed.atoms;
        for (int q = lane; q < qNB; q += kGroupSize) {
          if ((excl >> q) & 1)
            continue;
          const int  u    = S.qEpU[q], v = S.qEpV[q];
          const bool uNew = ((la >> u) & 1) != 0;
          const bool vNew = ((la >> v) & 1) != 0;
          if (!uNew && !vNew)
            continue;
          const bool uIn = ((at >> u) & 1) != 0;
          const bool vIn = ((at >> v) & 1) != 0;
          NBond      e;
          e.bondIdx = static_cast<uint8_t>(q);
          e.alive   = 1;
          if (uIn && vIn) {
            e.ring    = 1;
            e.newAtom = static_cast<uint8_t>(v);
          } else {
            e.ring    = 0;
            e.newAtom = static_cast<uint8_t>(uIn ? v : u);
          }
          const int slot = atomicAdd(&S.nbCount[g], 1);
          if (slot < T)
            S.nb[g][slot] = e;
        }
      }
      __syncwarp();
      int total = S.nbCount[g];
      if (lane == 0 && total > T)
        S.nbCount[g] = T;
      __syncwarp();
      if (total > T) {
        if (lane == 0)
          S.overflowed = 1;
        break;
      }
      if (total == 0)
        break;

      bool runInner = (cur.seed.stage != 0);

      if (cur.seed.stage == 0) {
        QS<T>& big = S.big[g];
        warpCopyBytes(&big, &cur, sizeof(QS<T>), lane);
        __syncwarp();
        if (lane == 0) {
          big.seed.lastAdded = 0;
          big.seed.stage     = 0;
          for (int i = 0; i < total; ++i)
            addNewBond<T>(big.seed, S.nb[g][i]);
        }
        __syncwarp();
        computeRemaining<T>(S, big.seed, lane);
        {
          const int bs = readBestScore<T>(S, lane);
          if (!canGrowBigger(big.seed.numBonds, big.seed.remB, big.seed.numAtoms, big.seed.remA, bs >> 16,
                             bs & 0xFFFF))
            break;
        }
        const bool ok = checkSeed<T>(S, big, g, lane);
        if (ok) {
          updateIncumbent<T>(S, big, lane);
          if (!pushSeed<T>(S, qbase, big, lane) && lane == 0)
            S.overflowed = 1;
          __syncwarp();
          if (total > 1) {
            if (lane == 0)
              cur.seed.stage = 1;
            __syncwarp();
            if (!pushSeed<T>(S, qbase, cur, lane) && lane == 0)
              S.overflowed = 1;
            __syncwarp();
          }
          break;
        }
        if (total == 1)
          break;
        runInner = true;
      }

      if (!runInner)
        break;

      // ---- stage 1: singleton children ----
      for (int i = 0; i < total; ++i) {
        if (!S.nb[g][i].alive)
          continue;
        QS<T>& big = S.big[g];
        warpCopyBytes(&big, &cur, sizeof(QS<T>), lane);
        __syncwarp();
        if (lane == 0) {
          big.seed.lastAdded = 0;
          big.seed.stage     = 0;
          addNewBond<T>(big.seed, S.nb[g][i]);
        }
        __syncwarp();
        computeRemaining<T>(S, big.seed, lane);
        const int bs = readBestScore<T>(S, lane);
        if (!canGrowBigger(big.seed.numBonds, big.seed.remB, big.seed.numAtoms, big.seed.remA, bs >> 16,
                           bs & 0xFFFF))
          continue;
        const bool ok = checkSeed<T>(S, big, g, lane);
        if (ok) {
          updateIncumbent<T>(S, big, lane);
          if (!pushSeed<T>(S, qbase, big, lane) && lane == 0)
            S.overflowed = 1;
        } else if (lane == 0) {
          S.nb[g][i].alive = 0;
        }
        __syncwarp();
      }

      // ---- stage 2: non-singleton subsets of the surviving bonds ----
      int aliveCount = 0;
      for (int i = 0; i < total; ++i)
        aliveCount += S.nb[g][i].alive ? 1 : 0;
      if (aliveCount > 63) {
        if (lane == 0)
          S.overflowed = 1;
        break;
      }
      if (aliveCount > 1) {
        const int                erased  = total - aliveCount;
        const unsigned long long maxComp = (1ULL << aliveCount) - 1ULL;
        for (unsigned long long comp = maxComp; comp != 0ULL; --comp) {
          if ((comp & (comp - 1ULL)) == 0ULL)
            continue;
          if (erased == 0 && comp == maxComp)
            continue;
          QS<T>& big = S.big[g];
          warpCopyBytes(&big, &cur, sizeof(QS<T>), lane);
          __syncwarp();
          if (lane == 0) {
            big.seed.lastAdded = 0;
            big.seed.stage     = 0;
            int ab             = 0;
            for (int i = 0; i < total; ++i) {
              if (!S.nb[g][i].alive)
                continue;
              if ((comp >> ab) & 1ULL)
                addNewBond<T>(big.seed, S.nb[g][i]);
              ++ab;
            }
          }
          __syncwarp();
          computeRemaining<T>(S, big.seed, lane);
          const int bs = readBestScore<T>(S, lane);
          if (!canGrowBigger(big.seed.numBonds, big.seed.remB, big.seed.numAtoms, big.seed.remA, bs >> 16,
                             bs & 0xFFFF))
            continue;
          const bool ok = checkSeed<T>(S, big, g, lane);
          if (ok) {
            updateIncumbent<T>(S, big, lane);
            if (!pushSeed<T>(S, qbase, big, lane) && lane == 0)
              S.overflowed = 1;
          }
          __syncwarp();
          if (readFlag<T>(&S.overflowed, lane))
            break;
        }
      }
    } while (false);

    __syncthreads();
    if (tid == 0)
      S.phase2Done = (S.overflowed != 0 || S.qTop == 0) ? 1 : 0;
    __syncthreads();
    if (S.phase2Done)
      break;
  }

  // ---------------- Phase 3: writeback ----------------
  __syncthreads();
  const int  nA    = S.best.seed.numAtoms;
  const int  nB    = S.best.seed.numBonds;
  const W    bAtom = S.best.seed.atoms;
  const W    bBond = S.best.seed.bonds;
  if (tid == 0) {
    stats[pairIdx * 4 + 0] = nA;
    stats[pairIdx * 4 + 1] = nB;
    stats[pairIdx * 4 + 2] = 0;
    stats[pairIdx * 4 + 3] = S.overflowed ? 1 : 0;
  }
  for (int a = tid; a < T; a += kBlockThreads) {
    if ((bAtom >> a) & 1) {
      const W   lowm = (static_cast<W>(1) << a) - 1;
      const int idx  = popcW(bAtom & lowm);
      const int off  = (pairIdx * T + idx) * 2;
      atomMapping[off + 0] = static_cast<uint8_t>(a);
      atomMapping[off + 1] = S.best.match.tAtom[a];
    }
    if ((bBond >> a) & 1) {
      const W   lowm = (static_cast<W>(1) << a) - 1;
      const int idx  = popcW(bBond & lowm);
      const int off  = (pairIdx * T + idx) * 2;
      bondMapping[off + 0] = static_cast<uint8_t>(a);
      bondMapping[off + 1] = S.best.match.tBond[a];
    }
  }
  for (int i = tid; i < T; i += kBlockThreads) {
    const int off = (pairIdx * T + i) * 2;
    if (i >= nA) {
      atomMapping[off + 0] = 0xFFu;
      atomMapping[off + 1] = 0xFFu;
    }
    if (i >= nB) {
      bondMapping[off + 0] = 0xFFu;
      bondMapping[off + 1] = 0xFFu;
    }
  }
}

}  // namespace fmcsopt

#endif  // FMCS_MAIN_OPT_CUH
