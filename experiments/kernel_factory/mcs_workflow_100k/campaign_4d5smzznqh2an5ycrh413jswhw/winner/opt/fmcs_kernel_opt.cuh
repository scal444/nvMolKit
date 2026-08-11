#ifndef FMCS_KERNEL_OPT_CUH
#define FMCS_KERNEL_OPT_CUH

#include "opt/fmcs_opt.cuh"

namespace fmcsopt {

// ===========================================================================
// Per-pair shared-memory preload
// ===========================================================================
template <int T>
__device__ void loadPair(BlockShared<T>& S, const PairPtrs& P, int tid) {
  using W                  = typename WordSel<T>::type;
  constexpr int kWordsPerW = sizeof(W) / 4;

  const int qE = static_cast<int>(P.qRow[P.qNA]);
  const int tE = static_cast<int>(P.tRow[P.tNA]);

  if (tid == 0) {
    S.qNA        = P.qNA;
    S.qNB        = P.qNB;
    S.tNA        = P.tNA;
    S.tNB        = P.tNB;
    S.qTop       = 0;
    S.bestScore  = 0;
    S.bestLock   = 0;
    S.overflowed = 0;
    S.phase2Done = 0;
  }

  for (int i = tid; i <= P.qNA; i += kBlockThreads)
    S.qRow[i] = static_cast<uint8_t>(P.qRow[i]);
  for (int i = tid; i <= P.tNA; i += kBlockThreads)
    S.tRow[i] = static_cast<uint8_t>(P.tRow[i]);
  for (int i = tid; i < qE; i += kBlockThreads) {
    S.qCol[i] = static_cast<uint8_t>(P.qCol[i]);
    S.qBnd[i] = static_cast<uint8_t>(P.qBnd[i]);
  }
  for (int i = tid; i < tE; i += kBlockThreads) {
    S.tCol[i] = static_cast<uint8_t>(P.tCol[i]);
    S.tBnd[i] = static_cast<uint8_t>(P.tBnd[i]);
  }
  for (int i = tid; i < P.qNB; i += kBlockThreads) {
    const uint32_t e = P.qEp[i];
    S.qEpU[i]        = static_cast<uint8_t>(e >> 16);
    S.qEpV[i]        = static_cast<uint8_t>(e & 0xFFFFu);
    W r              = 0;
#pragma unroll
    for (int w = 0; w < kWordsPerW; ++w)
      if (w < P.bWPR)
        r |= static_cast<W>(P.bTab[static_cast<size_t>(i) * P.bWPR + w]) << (32 * w);
    S.bondRow[i] = r;
  }
  for (int i = tid; i < P.tNB; i += kBlockThreads) {
    const uint32_t e = P.tEp[i];
    S.tEpU[i]        = static_cast<uint8_t>(e >> 16);
    S.tEpV[i]        = static_cast<uint8_t>(e & 0xFFFFu);
  }
  for (int i = tid; i < P.qNA; i += kBlockThreads) {
    W r = 0;
#pragma unroll
    for (int w = 0; w < kWordsPerW; ++w)
      if (w < P.aWPR)
        r |= static_cast<W>(P.aTab[static_cast<size_t>(i) * P.aWPR + w]) << (32 * w);
    S.atomRow[i] = r;
  }
  __syncthreads();

  // Derived per-atom bitsets.
  for (int a = tid; a < P.qNA; a += kBlockThreads) {
    W         inc = 0;
    const int b0 = S.qRow[a], b1 = S.qRow[a + 1];
    for (int e = b0; e < b1; ++e)
      inc |= static_cast<W>(1) << S.qBnd[e];
    S.incBond[a] = inc;
  }
  if (tid <= kMaxDeg) {
    W m = 0;
    for (int a = 0; a < P.tNA; ++a) {
      const int d = S.tRow[a + 1] - S.tRow[a];
      if (d >= tid)
        m |= static_cast<W>(1) << a;
    }
    S.degAtLeast[tid] = m;
  }
  // Bond classes: query bonds with identical compatibility rows share a class.
  if (tid == 32) {
    W   reps[kMaxCls];
    int nc = 0;
    for (int b = 0; b < P.qNB; ++b) {
      const W r = S.bondRow[b];
      int     c = kMaxCls;
      for (int k = 0; k < nc; ++k)
        if (reps[k] == r) {
          c = k;
          break;
        }
      if (c == kMaxCls && nc < kMaxCls) {
        reps[nc]    = r;
        S.clsRep[nc] = r;
        c           = nc;
        ++nc;
      }
      S.bondCls[b] = static_cast<uint8_t>(c);
    }
    S.numCls = nc;
  }
  __syncthreads();

  // clsNbr[k][a] = target atoms reachable from target atom a over a target bond
  // compatible with bond class k.
  {
    const int nc    = S.numCls;
    const int total = nc * P.tNA;
    for (int idx = tid; idx < total; idx += kBlockThreads) {
      const int k       = idx / P.tNA;
      const int a       = idx - k * P.tNA;
      const W   rowMask = S.clsRep[k];
      W         m  = 0;
      const int b0 = S.tRow[a], b1 = S.tRow[a + 1];
      for (int e = b0; e < b1; ++e)
        if ((rowMask >> S.tBnd[e]) & 1)
          m |= static_cast<W>(1) << S.tCol[e];
      S.clsNbr[k][a] = m;
    }
  }
  __syncthreads();
}

// ===========================================================================
// remaining-size bound (RDKit canGrowBiggerThan support), warp parallel
// ===========================================================================
template <int T>
__device__ __forceinline__ void computeRemaining(BlockShared<T>& S, SeedT<T>& seed, int lane) {
  using W          = typename WordSel<T>::type;
  W         visA   = seed.atoms;
  W         visB   = seed.excl;
  W         front  = seed.lastAdded;
  const int base0  = popcW(visB);
  const int baseA  = popcW(visA);

  constexpr int kAtomRegs = (T + 31) / 32;
  while (front != 0) {
    W bondAcc = 0;
    W candAcc = 0;
#pragma unroll
    for (int r = 0; r < kAtomRegs; ++r) {
      const int a = lane + r * 32;
      if (a < T && ((front >> a) & 1)) {
        const int b0 = S.qRow[a], b1 = S.qRow[a + 1];
        for (int e = b0; e < b1; ++e) {
          const int b = S.qBnd[e];
          if ((visB >> b) & 1)
            continue;
          bondAcc |= static_cast<W>(1) << b;
          candAcc |= static_cast<W>(1) << S.qCol[e];
        }
      }
    }
    bondAcc = warpOrW(bondAcc);
    candAcc = warpOrW(candAcc);
    visB |= bondAcc;
    const W nf = candAcc & ~visA;
    visA |= nf;
    front = nf;
  }
  if (lane == 0) {
    seed.remB = static_cast<uint16_t>(popcW(visB) - base0);
    seed.remA = static_cast<uint16_t>(popcW(visA) - baseA);
  }
  __syncwarp();
}

__device__ __forceinline__ bool canGrowBigger(int nb, int rb, int na, int ra, int bestB, int bestA) {
  const int pb = nb + rb;
  if (pb > bestB)
    return true;
  if (pb < bestB)
    return false;
  return na + ra > bestA;
}

// ===========================================================================
// exact seed-in-target substructure check
// ===========================================================================
template <int T>
__device__ __forceinline__ typename WordSel<T>::type nbrMatchSlow(BlockShared<T>& S, int tAtom, int qBond) {
  using W      = typename WordSel<T>::type;
  const W  row = S.bondRow[qBond];
  W        m   = 0;
  const int b0 = S.tRow[tAtom], b1 = S.tRow[tAtom + 1];
  for (int e = b0; e < b1; ++e)
    if ((row >> S.tBnd[e]) & 1)
      m |= static_cast<W>(1) << S.tCol[e];
  return m;
}

// Build the most-constrained-first search order + back-edge table for `seed`.
// Returns the number of seed atoms, or -1 if the seed provably cannot embed.
template <int T>
__device__ int prepareSearch(BlockShared<T>& S, const SeedT<T>& seed, int g, int lane) {
  using W                = typename WordSel<T>::type;
  constexpr int kAtomRegs = (T + 31) / 32;
  const int     qNA      = S.qNA;
  const W       atoms    = seed.atoms;
  const W       bonds    = seed.bonds;

  int fail = 0;
  W   nsReg[kAtomRegs];
#pragma unroll
  for (int r = 0; r < kAtomRegs; ++r) {
    nsReg[r] = 0;
    const int a = lane + r * 32;
    if (a < qNA) {
      const bool in = ((atoms >> a) & 1) != 0;
      S.orderPos[g][a] = kUnmapped;
      int deg          = 0;
      W   ns           = 0;
      if (in) {
        deg          = popcW(S.incBond[a] & bonds);
        const int b0 = S.qRow[a], b1 = S.qRow[a + 1];
        for (int e = b0; e < b1; ++e)
          if ((bonds >> S.qBnd[e]) & 1)
            ns |= static_cast<W>(1) << S.qCol[e];
        if ((S.atomRow[a] & S.degAtLeast[deg > kMaxDeg ? kMaxDeg : deg]) == 0)
          fail = 1;
      }
      S.sdeg[g][a] = static_cast<uint8_t>(deg);
      nsReg[r]     = ns;
    }
  }
  __syncwarp();
  if (__any_sync(kFull, fail))
    return -1;

  const int n = seed.numAtoms;
  W         unordered = atoms;
  W         ordered   = 0;
  for (int d = 0; d < n; ++d) {
    unsigned bestKey = 0;
#pragma unroll
    for (int r = 0; r < kAtomRegs; ++r) {
      const int a = lane + r * 32;
      if (a < qNA && ((unordered >> a) & 1)) {
        const int mnc = popcW(nsReg[r] & ordered);
        if (d == 0 || mnc != 0) {
          const int      deg = S.sdeg[g][a];
          const int      cc  = popcW(S.atomRow[a] & S.degAtLeast[deg > kMaxDeg ? kMaxDeg : deg]);
          const unsigned key = (static_cast<unsigned>(mnc) << 24) | (static_cast<unsigned>(deg) << 20) |
                               (static_cast<unsigned>(127 - cc) << 8) | static_cast<unsigned>(127 - a) | 0x80u;
          bestKey = max(bestKey, key);
        }
      }
    }
    bestKey = warpMaxU(bestKey);
    int pick;
    if (bestKey == 0) {
      pick = ffsW(unordered);  // disconnected fallback
    } else {
      pick = 127 - static_cast<int>(bestKey & 0x7Fu);
    }
    if (lane == 0)
      S.order[g][d] = static_cast<uint8_t>(pick);
    S.orderPos[g][pick] = static_cast<uint8_t>(d);
    ordered |= static_cast<W>(1) << pick;
    unordered &= ~(static_cast<W>(1) << pick);
  }
  __syncwarp();

  // Per depth: label candidates + back edges.
  for (int d = lane; d < n; d += kGroupSize) {
    const int a  = S.order[g][d];
    const int dg = S.sdeg[g][a];
    S.depthCand[g][d] = S.atomRow[a] & S.degAtLeast[dg > kMaxDeg ? kMaxDeg : dg];
    int       cnt  = 0;
    unsigned  seen = 0;
    const int b0 = S.qRow[a], b1 = S.qRow[a + 1];
    for (int e = b0; e < b1 && cnt < kMaxSlots; ++e) {
      const int b = S.qBnd[e];
      if (!((bonds >> b) & 1))
        continue;
      const int o  = S.qCol[e];
      const int op = S.orderPos[g][o];
      if (op == kUnmapped || op >= d)
        continue;
      if ((seen >> op) & 1)
        continue;
      seen |= 1u << op;
      S.backEdges[g][d][cnt] = static_cast<uint16_t>(op | (b << 8));
      ++cnt;
    }
    S.beCount[g][d] = static_cast<uint8_t>(cnt);
  }
  __syncwarp();
  return n;
}

template <int T>
__device__ __forceinline__ typename WordSel<T>::type candidatesAt(BlockShared<T>&                  S,
                                                                 int                              g,
                                                                 int                              d,
                                                                 const unsigned char*             mapping,
                                                                 typename WordSel<T>::type        used) {
  using W = typename WordSel<T>::type;
  W       c   = S.depthCand[g][d] & ~used;
  const int k = S.beCount[g][d];
  for (int i = 0; i < k && c != 0; ++i) {
    const uint16_t be  = S.backEdges[g][d][i];
    const int      m   = mapping[be & 0xFFu];
    const int      b   = be >> 8;
    const int      cls = S.bondCls[b];
    c &= (cls < kMaxCls) ? S.clsNbr[cls][m] : nbrMatchSlow<T>(S, m, b);
  }
  return c;
}

// Lane-parallel DFS racing for one embedding. Writes tgtForQ on success.
template <int T>
__device__ bool dfsSearch(BlockShared<T>& S, int g, int n, int lane) {
  using W = typename WordSel<T>::type;

  W laneStripe = 0;
#pragma unroll
  for (int r = 0; r < (T + 31) / 32; ++r)
    laneStripe |= static_cast<W>(1) << (lane + r * 32);
  W roots = S.depthCand[g][0] & laneStripe;

  if (n == 1) {
    int low = (roots == 0) ? T : ffsW(roots);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      low = min(low, __shfl_xor_sync(kFull, low, off));
    if (low >= T)
      return false;
    if (lane == 0) {
      S.tgtForQ[g][S.order[g][0]] = static_cast<uint8_t>(low);
      S.found[g]                  = 1;
    }
    __syncwarp();
    return true;
  }

  const int lastDepth = n - 1;
  W         remaining[T];
  unsigned char mapping[T];
  int           poll = 0;

  while (roots != 0) {
    if (*(volatile int*)&S.found[g] != 0)
      break;
    const int rootAtom = ffsW(roots);
    roots &= roots - 1;

    W used = static_cast<W>(1) << rootAtom;
    mapping[0]   = static_cast<unsigned char>(rootAtom);
    int depth    = 1;
    remaining[1] = candidatesAt<T>(S, g, 1, mapping, used);

    while (depth >= 1) {
      if (depth == lastDepth) {
        if (remaining[depth] != 0) {
          if (atomicCAS(&S.found[g], 0, 1) == 0) {
            for (int p = 0; p < lastDepth; ++p)
              S.tgtForQ[g][S.order[g][p]] = mapping[p];
            S.tgtForQ[g][S.order[g][lastDepth]] = static_cast<uint8_t>(ffsW(remaining[depth]));
          }
          return true;
        }
      }
      if (remaining[depth] == 0) {
        --depth;
        if (depth >= 1)
          used &= ~(static_cast<W>(1) << mapping[depth]);
        continue;
      }
      if (((++poll) & 127) == 0 && *(volatile int*)&S.found[g] != 0)
        return false;
      const int cand    = ffsW(remaining[depth]);
      remaining[depth] &= remaining[depth] - 1;
      mapping[depth]    = static_cast<unsigned char>(cand);
      used |= static_cast<W>(1) << cand;
      ++depth;
      remaining[depth] = candidatesAt<T>(S, g, depth, mapping, used);
    }
  }
  return false;
}

template <int T>
__device__ __forceinline__ void clearMatch(MatchT<T>& m, int lane) {
  for (int i = lane; i < T; i += kGroupSize) {
    m.tAtom[i] = kUnmapped;
    m.tBond[i] = kUnmapped;
  }
  if (lane == 0) {
    m.visA   = 0;
    m.visB   = 0;
    m.mAtoms = 0;
    m.mBonds = 0;
    m.empty  = 1;
  }
  __syncwarp();
}

// Rebuild the full (atom + bond) mapping from the embedding recorded in tgtForQ.
template <int T>
__device__ bool rebuildMatch(BlockShared<T>& S, const SeedT<T>& seed, MatchT<T>& m, int g, int lane) {
  using W = typename WordSel<T>::type;
  clearMatch<T>(m, lane);

  const W atoms = seed.atoms;
  const W bonds = seed.bonds;
  W       visA  = 0;
  int     okA   = 1;
  for (int a = lane; a < T; a += kGroupSize) {
    if ((atoms >> a) & 1) {
      const uint8_t t = S.tgtForQ[g][a];
      if (t == kUnmapped)
        okA = 0;
      else {
        m.tAtom[a] = t;
        visA |= static_cast<W>(1) << t;
      }
    }
  }
  visA = warpOrW(visA);
  if (__any_sync(kFull, okA == 0) || popcW(visA) != seed.numAtoms) {
    clearMatch<T>(m, lane);
    return false;
  }

  W   visB = 0;
  int okB  = 1;
  for (int b = lane; b < T; b += kGroupSize) {
    if ((bonds >> b) & 1) {
      const int tu = S.tgtForQ[g][S.qEpU[b]];
      const int tv = S.tgtForQ[g][S.qEpV[b]];
      int       tb = -1;
      if (tu != kUnmapped && tv != kUnmapped) {
        const W   row = S.bondRow[b];
        const int e0 = S.tRow[tu], e1 = S.tRow[tu + 1];
        for (int e = e0; e < e1; ++e)
          if (S.tCol[e] == tv && ((row >> S.tBnd[e]) & 1)) {
            tb = S.tBnd[e];
            break;
          }
      }
      if (tb < 0)
        okB = 0;
      else {
        m.tBond[b] = static_cast<uint8_t>(tb);
        visB |= static_cast<W>(1) << tb;
      }
    }
  }
  visB = warpOrW(visB);
  if (__any_sync(kFull, okB == 0) || popcW(visB) != seed.numBonds) {
    clearMatch<T>(m, lane);
    return false;
  }
  if (lane == 0) {
    m.visA   = visA;
    m.visB   = visB;
    m.mAtoms = seed.numAtoms;
    m.mBonds = seed.numBonds;
    m.empty  = 0;
  }
  __syncwarp();
  return true;
}

template <int T>
__device__ bool matchExact(BlockShared<T>& S, const SeedT<T>& seed, MatchT<T>& m, int g, int lane) {
  if (seed.numAtoms == 0) {
    clearMatch<T>(m, lane);
    return seed.numBonds == 0;
  }
  if (seed.numAtoms > S.tNA || seed.numBonds > S.tNB) {
    clearMatch<T>(m, lane);
    return false;
  }
  if (lane == 0)
    S.found[g] = 0;
  __syncwarp();
  const int n = prepareSearch<T>(S, seed, g, lane);
  if (n < 0) {
    clearMatch<T>(m, lane);
    return false;
  }
  dfsSearch<T>(S, g, n, lane);
  __syncwarp();
  if (*(volatile int*)&S.found[g] == 0) {
    clearMatch<T>(m, lane);
    return false;
  }
  return rebuildMatch<T>(S, seed, m, g, lane);
}

// Greedy incremental extension of a parent's recorded embedding. Returns true
// only when every unmapped seed bond was successfully committed.
template <int T>
__device__ bool matchGreedy(BlockShared<T>& S, const SeedT<T>& seed, MatchT<T>& m, int lane) {
  using W = typename WordSel<T>::type;
  W bits  = seed.bonds;
  int ok  = 1;
  if (lane == 0) {
    while (bits != 0) {
      const int b = ffsW(bits);
      bits &= bits - 1;
      if (m.tBond[b] != kUnmapped)
        continue;
      const int u = S.qEpU[b], v = S.qEpV[b];
      const int tu = m.tAtom[u], tv = m.tAtom[v];
      const W   row = S.bondRow[b];
      int       chosenB = -1, chosenA = -1;
      if (tu != kUnmapped && tv != kUnmapped) {
        const int e0 = S.tRow[tu], e1 = S.tRow[tu + 1];
        for (int e = e0; e < e1; ++e) {
          if (S.tCol[e] != tv)
            continue;
          const int tb = S.tBnd[e];
          if ((m.visB >> tb) & 1)
            continue;
          if (!((row >> tb) & 1))
            continue;
          chosenB = tb;
          break;
        }
      } else if (tu != kUnmapped || tv != kUnmapped) {
        const int src   = (tu != kUnmapped) ? tu : tv;
        const int qFree = (tu != kUnmapped) ? v : u;
        const W   arow  = S.atomRow[qFree];
        const int e0 = S.tRow[src], e1 = S.tRow[src + 1];
        for (int e = e0; e < e1; ++e) {
          const int tb = S.tBnd[e];
          if ((m.visB >> tb) & 1)
            continue;
          const int ta = S.tCol[e];
          if ((m.visA >> ta) & 1)
            continue;
          if (!((row >> tb) & 1))
            continue;
          if (!((arow >> ta) & 1))
            continue;
          chosenB = tb;
          chosenA = ta;
          break;
        }
      } else {
        ok = 0;
        break;
      }
      if (chosenB < 0) {
        ok = 0;
        break;
      }
      m.tBond[b] = static_cast<uint8_t>(chosenB);
      m.visB |= static_cast<W>(1) << chosenB;
      m.mBonds += 1;
      m.empty = 0;
      if (chosenA >= 0) {
        const int qFree     = (tu != kUnmapped) ? v : u;
        m.tAtom[qFree]      = static_cast<uint8_t>(chosenA);
        m.visA |= static_cast<W>(1) << chosenA;
        m.mAtoms += 1;
      }
    }
  }
  ok = __shfl_sync(kFull, ok, 0);
  return ok != 0;
}

template <int T>
__device__ __forceinline__ bool checkSeed(BlockShared<T>& S, QS<T>& c, int g, int lane) {
  if (!c.match.empty) {
    if (matchGreedy<T>(S, c.seed, c.match, lane))
      return true;
  }
  return matchExact<T>(S, c.seed, c.match, g, lane);
}

// ===========================================================================
// incumbent + queue
// ===========================================================================
template <int T>
__device__ __forceinline__ void updateIncumbent(BlockShared<T>& S, const QS<T>& cand, int lane) {
  int doCopy = 0;
  int locked = 0;
  if (lane == 0) {
    const int sc  = (static_cast<int>(cand.seed.numBonds) << 16) | static_cast<int>(cand.seed.numAtoms);
    int       prv = S.bestScore;
    bool      won = false;
    while (sc > prv) {
      const int seen = atomicCAS(&S.bestScore, prv, sc);
      if (seen == prv) {
        won = true;
        break;
      }
      prv = seen;
    }
    if (won) {
      while (atomicCAS(&S.bestLock, 0, 1) != 0) {
      }
      locked = 1;
      doCopy = (sc == S.bestScore) ? 1 : 0;
    }
  }
  doCopy = __shfl_sync(kFull, doCopy, 0);
  locked = __shfl_sync(kFull, locked, 0);
  if (doCopy)
    warpCopyBytes(&S.best, &cand, sizeof(QS<T>), lane);
  __syncwarp();
  if (lane == 0 && locked) {
    __threadfence_block();
    atomicExch(&S.bestLock, 0);
  }
  __syncwarp();
}

__device__ __forceinline__ int adjustTop(int* top, int delta, int cap) {
  int old = *top;
  while (true) {
    const int nw = old + delta;
    if (nw < 0 || nw > cap)
      return -1;
    const int prev = atomicCAS(top, old, nw);
    if (prev == old)
      return old;
    old = prev;
  }
}

template <int T>
__device__ __forceinline__ bool pushSeed(BlockShared<T>& S, QS<T>* qbase, const QS<T>& e, int lane) {
  int slot = -1;
  if (lane == 0)
    slot = adjustTop(&S.qTop, 1, kQueueCap);
  slot = __shfl_sync(kFull, slot, 0);
  if (slot < 0)
    return false;
  warpCopyBytes(&qbase[slot], &e, sizeof(QS<T>), lane);
  __syncwarp();
  return true;
}

template <int T>
__device__ __forceinline__ bool popSeed(BlockShared<T>& S, QS<T>* qbase, QS<T>& out, int lane) {
  int top = -1;
  if (lane == 0)
    top = adjustTop(&S.qTop, -1, kQueueCap);
  top = __shfl_sync(kFull, top, 0);
  if (top < 0)
    return false;
  warpCopyBytes(&out, &qbase[top - 1], sizeof(QS<T>), lane);
  __syncwarp();
  return true;
}

}  // namespace fmcsopt

#endif  // FMCS_KERNEL_OPT_CUH
