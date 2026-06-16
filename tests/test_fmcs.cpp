// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "benchmark_data.h"
#include "fmcs_cuda/fmcs.cuh"
#include "mcs_common/mcs_types.cuh"
#include "src/mcs/mcs_compile_flags.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using mcs::Graph;
using mcs::MCSResult;
using mcs::fmcs::ExecutionStats;
using mcs::fmcs::Parameters;
using mcs::benchmark::MiviaGraphData;

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

Graph g(std::size_t n,
        std::vector<std::pair<std::size_t, std::size_t>> edges) {
  return mcs::buildGraphFromEdges(n, std::move(edges));
}

MCSResult findSingleMCES(const Graph& a, const Graph& b,
                         Parameters params = {},
                         cudaStream_t stream = nullptr) {
  const auto results =
      mcs::fmcs::findMCESfMCSBatch({a}, {b}, params, nullptr, stream);
  EXPECT_EQ(results.size(), 1);
  return results.empty() ? MCSResult{} : results.front();
}

// Path graph: numAtoms vertices in a chain, (i, i+1) edges.
Graph path(int numAtoms) {
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (int i = 0; i + 1 < numAtoms; ++i) {
    edges.emplace_back(i, i + 1);
  }
  return mcs::buildGraphFromEdges(static_cast<std::size_t>(numAtoms),
                                  std::move(edges));
}

// Cycle graph: numAtoms vertices in a ring.
Graph cycle(int numAtoms) {
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (int i = 0; i + 1 < numAtoms; ++i) edges.emplace_back(i, i + 1);
  if (numAtoms >= 3) edges.emplace_back(0, numAtoms - 1);
  return mcs::buildGraphFromEdges(static_cast<std::size_t>(numAtoms),
                                  std::move(edges));
}

// Star K1,n: vertex 0 is the hub, vertices 1..n are leaves.
Graph star(int numLeaves) {
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (int i = 1; i <= numLeaves; ++i) edges.emplace_back(0, i);
  return mcs::buildGraphFromEdges(static_cast<std::size_t>(numLeaves + 1),
                                  std::move(edges));
}

// Standard six-membered ring (used as benzene topology).
Graph benzene() { return cycle(6); }

// Toluene: benzene + a methyl substituent on atom 0 -> atom 6.
Graph toluene() {
  return g(7, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {0, 5}, {0, 6}});
}

// Naphthalene: two fused 6-rings sharing edge (4,5).
//
//   0 - 1
//   |   |
//   5 - 4 - 6
//   |       |
//   ...    ...
//
// Atoms: 0,1,2,3,4,5 (ring A), 5,4,6,7,8,9 (ring B sharing edge 4-5).
// 10 atoms total.
Graph naphthalene() {
  return g(10, {
    {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {0, 5},  // ring A
    {4, 6}, {6, 7}, {7, 8}, {8, 9}, {5, 9},          // ring B (shares 4-5)
  });
}

// Phenanthrene: three angularly fused 6-rings, 14 atoms / 16 bonds.
// Ring A: 0-1-2-3-4-5-0
// Ring B: 4-5-6-7-8-9 (shares edge 4-5)
// Ring C: 8-9-10-11-12-13 (shares edge 8-9)
Graph phenanthrene() {
  return g(14, {
    {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {0, 5},   // ring A
    {4, 6}, {6, 7}, {7, 8}, {8, 9}, {5, 9},           // ring B
    {8, 10}, {10, 11}, {11, 12}, {12, 13}, {9, 13},   // ring C
  });
}

// Biphenyl: two C6 rings linked by a single bond between atom 0 and atom 6.
Graph biphenyl() {
  return g(12, {
    {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {0, 5},     // ring A
    {6, 7}, {7, 8}, {8, 9}, {9, 10}, {10, 11}, {6, 11}, // ring B
    {0, 6},                                              // linker
  });
}

// Build a labelled MiviaGraphData with the given topology, vertex
// labels, and explicit (u, v, label) edge-label triples (symmetrized
// into the dense edgeLabels matrix).
MiviaGraphData buildLabeled(
    int numAtoms,
    std::vector<std::pair<int, int>> edges,
    std::vector<uint16_t> vertexLabels,
    std::vector<std::tuple<int, int, uint16_t>> edgeLabelTriples) {
  MiviaGraphData out;
  std::vector<std::pair<std::size_t, std::size_t>> edgesPair;
  edgesPair.reserve(edges.size());
  for (const auto& e : edges) edgesPair.emplace_back(e.first, e.second);
  out.graph = mcs::buildGraphFromEdges(static_cast<std::size_t>(numAtoms),
                                       std::move(edgesPair));
  out.vertexLabels = std::move(vertexLabels);
  out.edgeLabels.assign(static_cast<std::size_t>(numAtoms) * numAtoms, 0);
  for (const auto& t : edgeLabelTriples) {
    const int u = std::get<0>(t);
    const int v = std::get<1>(t);
    const uint16_t lbl = std::get<2>(t);
    out.edgeLabels[static_cast<std::size_t>(u) * numAtoms + v] = lbl;
    out.edgeLabels[static_cast<std::size_t>(v) * numAtoms + u] = lbl;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Result-checking helpers
// ---------------------------------------------------------------------------

void expectFullSelfPair(const MCSResult& r, const Graph& gph) {
  EXPECT_FALSE(r.timedOut);
  EXPECT_FALSE(r.overflowed);
  EXPECT_EQ(r.numCommonVertices, gph.numVertices);
  EXPECT_EQ(r.numCommonEdges, gph.numEdges);
}

// Verify the returned mappings form a valid subgraph isomorphism:
//   - mappingA / mappingB cover numCommonVertices distinct query / target
//     atoms; the i-th matched vertex pairs (mappingA[i], mappingB[i]).
//   - For each (qBond, tBond) in (edgeMappingA[i], edgeMappingB[i]), the
//     target endpoints are the targets that the query bond's endpoints
//     mapped to.
void expectMappingsConsistent(const MCSResult& r,
                              const Graph& a,
                              const Graph& b) {
  ASSERT_EQ(static_cast<int>(r.mappingA.size()), r.numCommonVertices);
  ASSERT_EQ(static_cast<int>(r.mappingB.size()), r.numCommonVertices);
  ASSERT_EQ(static_cast<int>(r.edgeMappingA.size()), r.numCommonEdges);
  ASSERT_EQ(static_cast<int>(r.edgeMappingB.size()), r.numCommonEdges);

  // mappingA / mappingB are bijections within the matched subset.
  std::set<std::size_t> qAtoms(r.mappingA.begin(), r.mappingA.end());
  std::set<std::size_t> tAtoms(r.mappingB.begin(), r.mappingB.end());
  EXPECT_EQ(qAtoms.size(), r.mappingA.size())
      << "mappingA has duplicate query atoms";
  EXPECT_EQ(tAtoms.size(), r.mappingB.size())
      << "mappingB has duplicate target atoms";

  // Every query atom is in graph a; every target atom is in graph b.
  for (std::size_t qa : qAtoms) {
    EXPECT_LT(qa, static_cast<std::size_t>(a.numVertices));
  }
  for (std::size_t ta : tAtoms) {
    EXPECT_LT(ta, static_cast<std::size_t>(b.numVertices));
  }

  // Build atom-mapping lookup: query atom -> target atom.
  std::vector<std::size_t> qToT(a.numVertices,
                                static_cast<std::size_t>(-1));
  for (int i = 0; i < r.numCommonVertices; ++i) {
    qToT[r.mappingA[i]] = r.mappingB[i];
  }

  // Each matched edge: target endpoints == qToT of query endpoints (in
  // either bond orientation).
  for (int i = 0; i < r.numCommonEdges; ++i) {
    const auto qE = r.edgeMappingA[i];
    const auto tE = r.edgeMappingB[i];
    const auto qU = qE.first, qV = qE.second;
    const auto tU = tE.first, tV = tE.second;
    EXPECT_LT(qU, static_cast<std::size_t>(a.numVertices));
    EXPECT_LT(qV, static_cast<std::size_t>(a.numVertices));
    EXPECT_LT(tU, static_cast<std::size_t>(b.numVertices));
    EXPECT_LT(tV, static_cast<std::size_t>(b.numVertices));
    const auto mappedU = qToT[qU];
    const auto mappedV = qToT[qV];
    const bool fwd = (mappedU == tU && mappedV == tV);
    const bool rev = (mappedU == tV && mappedV == tU);
    EXPECT_TRUE(fwd || rev)
        << "Edge mapping (" << qU << "," << qV << ") -> ("
        << tU << "," << tV << ") inconsistent with atom mapping ("
        << qU << "->" << mappedU << ", " << qV << "->" << mappedV << ")";
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// FMCSDispatch: smoke tests on the host-level entry points.
// ---------------------------------------------------------------------------

TEST(FMCSDispatch, BatchSingleEntrySelfPathReturnsFullPath) {
  const auto a = path(4);
  auto r = findSingleMCES(a, a);
  expectFullSelfPair(r, a);
  expectMappingsConsistent(r, a, a);
}

TEST(FMCSDispatch, BatchOfThreeReturnsExpectedSizes) {
  std::vector<Graph> graphs{path(2), path(3), path(4)};
  auto rs = mcs::fmcs::findMCESfMCSBatch(graphs, graphs);
  ASSERT_EQ(rs.size(), graphs.size());
  for (size_t i = 0; i < graphs.size(); ++i) {
    EXPECT_FALSE(rs[i].overflowed);
    EXPECT_FALSE(rs[i].timedOut);
    EXPECT_EQ(rs[i].numCommonEdges, graphs[i].numEdges);
    EXPECT_EQ(rs[i].numCommonVertices, graphs[i].numVertices);
  }
}

TEST(FMCSDispatch, OptionalPerPairTimingsAreRejected) {
  std::vector<Graph> graphs{path(4), cycle(6)};
  std::vector<float> timesMs;
  EXPECT_THROW(
      (void)mcs::fmcs::findMCESfMCSBatch(
          graphs, graphs, Parameters{}, &timesMs),
      std::runtime_error);
}

TEST(FMCSDispatch, OptionalExecutionStatsAreRejected) {
  std::vector<Graph> graphs{path(4), cycle(6)};
  std::vector<mcs::fmcs::ExecutionStats> stats;
  EXPECT_THROW(
      (void)mcs::fmcs::findMCESfMCSBatch(
          graphs, graphs, Parameters{}, nullptr, nullptr, &stats),
      std::runtime_error);
}

TEST(FMCSDispatch, MismatchedBatchSizesThrows) {
  std::vector<Graph> a{path(2)};
  std::vector<Graph> b{};
  EXPECT_THROW(mcs::fmcs::findMCESfMCSBatch(a, b), std::runtime_error);
}

// ---------------------------------------------------------------------------
// FMCSDegenerate: empty / single-vertex / no-shared-bond inputs.
// ---------------------------------------------------------------------------

TEST(FMCSDegenerate, EmptyGraphs) {
  const auto a = g(0, {});
  const auto b = g(0, {});
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 0);
  EXPECT_EQ(r.numCommonEdges, 0);
}

TEST(FMCSDegenerate, SingleVertex) {
  // No bonds anywhere -> seed-grow lattice is empty (Phase 1 enumerates
  // bond pairs, of which there are zero).  Connected MCES is reported
  // as size zero.
  const auto a = g(1, {});
  const auto b = g(1, {});
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 0);
  EXPECT_EQ(r.numCommonEdges, 0);
}

TEST(FMCSDegenerate, SingleEdge) {
  const auto a = g(2, {{0, 1}});
  const auto b = g(2, {{0, 1}});
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 2);
  EXPECT_EQ(r.numCommonEdges, 1);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSDegenerate, OneSideEmpty) {
  const auto a = g(0, {});
  const auto b = path(3);
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 0);
  EXPECT_EQ(r.numCommonEdges, 0);
}

TEST(FMCSDegenerate, DisjointInputs) {
  // Triangle vs three isolated atoms: target has zero bonds, so no
  // (q_bond, t_bond) pair compatible -> empty MCES.
  const auto a = cycle(3);
  const auto b = g(3, {});
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 0);
  EXPECT_EQ(r.numCommonEdges, 0);
}

// ---------------------------------------------------------------------------
// FMCSBasics: hand-derived expected sizes on small graph topologies.
// ---------------------------------------------------------------------------

TEST(FMCSBasics, PathEqualLength) {
  const auto p = path(4);
  auto r = findSingleMCES(p, p);
  expectFullSelfPair(r, p);
  expectMappingsConsistent(r, p, p);
}

TEST(FMCSBasics, PathShorterVsLonger) {
  const auto a = path(3);
  const auto b = path(6);
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 3);
  EXPECT_EQ(r.numCommonEdges, 2);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSBasics, TreeVsTree) {
  // K1,3 vs K1,4: the smaller star is a connected subgraph of the larger.
  const auto a = star(3);
  const auto b = star(4);
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 4);
  EXPECT_EQ(r.numCommonEdges, 3);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSBasics, HighFanoutStarSelfPair) {
  // Regression for the grow-step boundary scratch space: after seeding
  // one spoke, the hub exposes nine more spokes at once.  A too-small
  // new-bond buffer silently truncated this and missed the full star.
  const auto s = star(10);
  auto r = findSingleMCES(s, s);
  expectFullSelfPair(r, s);
  expectMappingsConsistent(r, s, s);
}

TEST(FMCSBasics, StarKeepsHubFrontierForSiblingSpokes) {
  // Query side after host swapping is K1,4; target is a diamond
  // (K4 missing one edge).  The exact connected MCES is K1,3: center
  // at either degree-3 diamond vertex.  If a singleton grow from one
  // spoke drops the hub from lastAddedAtoms, sibling spokes are never
  // considered and the search gets stuck at 2 bonds.
  const auto diamond = g(4, {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}});
  const auto fourStar = star(4);
  auto r = findSingleMCES(diamond, fourStar);
  EXPECT_EQ(r.numCommonVertices, 4);
  EXPECT_EQ(r.numCommonEdges, 3);
  EXPECT_FALSE(r.overflowed);
  expectMappingsConsistent(r, diamond, fourStar);
}

TEST(FMCSBasics, CycleVsCycleSame) {
  const auto c = cycle(6);
  auto r = findSingleMCES(c, c);
  expectFullSelfPair(r, c);
  expectMappingsConsistent(r, c, c);
}

TEST(FMCSBasics, CycleVsCycleLarger) {
  // C3 (triangle) vs C4 (square).  C4 has no triangle subgraph; the
  // best connected common edge subgraph is a 2-bond path.
  const auto a = cycle(3);
  const auto b = cycle(4);
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonEdges, 2);
  EXPECT_EQ(r.numCommonVertices, 3);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSBasics, TreeVsCycle) {
  const auto a = path(3);
  const auto b = cycle(3);
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonEdges, 2);  // path-3 in C3
  EXPECT_EQ(r.numCommonVertices, 3);
  expectMappingsConsistent(r, a, b);
}

// ---------------------------------------------------------------------------
// FMCSConnected: connected-MCES behaviour on graphs whose unconstrained
// common subgraph would be disconnected.
// ---------------------------------------------------------------------------

TEST(FMCSConnected, DisconnectedCommonGraphReturnsOnlyLargestConnected) {
  // Two disjoint edges on each side: 4 atoms / 2 bonds, no path between
  // the two components.  Connected MCES is therefore at most a single
  // edge (the largest connected subgraph of either component).
  const auto a = g(4, {{0, 1}, {2, 3}});
  const auto b = g(4, {{0, 1}, {2, 3}});
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonEdges, 1);
  EXPECT_EQ(r.numCommonVertices, 2);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSConnected, TwoComponentsEachReturnsOneComponent) {
  // Each side has a path-3 component and an isolated edge; connected
  // MCES is the larger component (path-3 -> 2 bonds).
  const auto a = g(5, {{0, 1}, {1, 2}, {3, 4}});
  const auto b = g(5, {{0, 1}, {1, 2}, {3, 4}});
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonEdges, 2);
  EXPECT_EQ(r.numCommonVertices, 3);
  expectMappingsConsistent(r, a, b);
}

// ---------------------------------------------------------------------------
// FMCSMolecule: hand-encoded molecule pairs with reference values
// computed from rdFMCS.FindMCS(MaximizeBonds=True, Threshold=1.0,
// AtomCompareElements, BondCompareOrder).  Topology-only here -- the
// labelled equivalents live in FMCSLabels below.
// ---------------------------------------------------------------------------

TEST(FMCSMolecule, BenzeneVsBenzene) {
  const auto a = benzene();
  auto r = findSingleMCES(a, a);
  expectFullSelfPair(r, a);
  expectMappingsConsistent(r, a, a);
}

TEST(FMCSMolecule, BenzeneVsToluene) {
  // Benzene ring is a connected subgraph of toluene (NullPolicy =
  // topology only, so aromatic-vs-single bond labelling is ignored).
  const auto a = benzene();
  const auto b = toluene();
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 6);
  EXPECT_EQ(r.numCommonEdges, 6);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSMolecule, BenzeneVsCyclohexaneTopologyOnly) {
  // Same topology (6-cycle), different chemistry; NullPolicy ignores
  // bond labels so both look like C6.
  const auto a = benzene();
  const auto b = cycle(6);  // cyclohexane topology = 6-cycle
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 6);
  EXPECT_EQ(r.numCommonEdges, 6);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSMolecule, NaphthaleneVsPhenanthrene) {
  // Naphthalene is a connected subgraph of phenanthrene (the two
  // terminal rings share an edge, matching naphthalene's topology).
  const auto a = naphthalene();
  const auto b = phenanthrene();
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 10);
  EXPECT_EQ(r.numCommonEdges, 11);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSMolecule, BiphenylVsBiphenyl) {
  const auto a = biphenyl();
  auto r = findSingleMCES(a, a);
  expectFullSelfPair(r, a);
  expectMappingsConsistent(r, a, a);
}

TEST(FMCSRegression, FixturePair00x03UnlabeledTopology) {
  // First non-overflow mismatch from tests/test_fmcs_parity.py:
  // sampled_smiles pair00x03, unlabeled.  RDKit's connected MCES is
  // the 17-bond subgraph of the first molecule that omits edge (2,3).
  const auto a = g(17, {
      {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6},
      {5, 7}, {7, 8}, {8, 9}, {9, 10}, {10, 11}, {11, 12},
      {12, 13}, {12, 14}, {14, 15}, {4, 16}, {1, 16}, {9, 15}});
  const auto b = g(18, {
      {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6},
      {6, 7}, {7, 8}, {8, 9}, {8, 10}, {10, 11}, {11, 12},
      {12, 13}, {13, 14}, {13, 15}, {15, 16}, {3, 17},
      {1, 17}, {3, 6}, {10, 16}});
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 17);
  EXPECT_EQ(r.numCommonEdges, 17);
  EXPECT_FALSE(r.overflowed);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSRegression, FixturePair01x03UnlabeledTopology) {
  // Current first non-overflow mismatch after restoring Stage 1 coverage:
  // sampled_smiles pair01x03, unlabeled.  RDKit's connected MCES is 17/17.
  const auto a = g(19, {
      {0, 1}, {1, 2}, {2, 3}, {2, 4}, {4, 5}, {5, 6},
      {6, 7}, {7, 8}, {8, 9}, {9, 10}, {10, 11}, {11, 12},
      {10, 13}, {13, 14}, {1, 15}, {15, 16}, {16, 17},
      {16, 18}, {1, 18}, {8, 14}});
  const auto b = g(18, {
      {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6},
      {6, 7}, {7, 8}, {8, 9}, {8, 10}, {10, 11}, {11, 12},
      {12, 13}, {13, 14}, {13, 15}, {15, 16}, {3, 17},
      {1, 17}, {3, 6}, {10, 16}});
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 17);
  EXPECT_EQ(r.numCommonEdges, 17);
  EXPECT_FALSE(r.overflowed);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSRegression, FiveNodePathInsideTriangleWithLeaves) {
  // Query contains a 4-edge path 4-0-2-1-3 plus the extra chord (0,1).
  // Target is exactly a 4-edge path 0-3-2-1-4.
  const auto a = g(5, {{0, 1}, {0, 2}, {0, 4}, {1, 2}, {1, 3}});
  const auto b = g(5, {{0, 3}, {1, 2}, {1, 4}, {2, 3}});
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 5);
  EXPECT_EQ(r.numCommonEdges, 4);
  EXPECT_FALSE(r.overflowed);
  expectMappingsConsistent(r, a, b);
}

// ---------------------------------------------------------------------------
// FMCSLabels: vertex / edge label tests via findMCESfMCSBatchLabeled.
//
// The labelled API takes mcs::benchmark::MiviaGraphData with explicit
// uint16 vertex labels and a dense uint16 edge-label matrix
// (edgeLabels[u * N + v]).
// ---------------------------------------------------------------------------

TEST(FMCSLabels, NullLabelTopology) {
  // All-zero vertex labels; edge labels uniform.  Should match
  // identically to the topology-only case (path-3 self-pair).
  auto m = buildLabeled(3,
      {{0, 1}, {1, 2}},
      /*vertexLabels=*/{0, 0, 0},
      /*edges=*/{{0, 1, 1}, {1, 2, 1}});
  std::vector<MiviaGraphData> a{m};
  std::vector<MiviaGraphData> b{m};
  auto rs = mcs::fmcs::findMCESfMCSBatchLabeled(a, b);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
}

TEST(FMCSLabels, VertexLabelMatch) {
  // Two paths with identical vertex labels at corresponding positions.
  auto m = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {7, 8, 9},
      {{0, 1, 1}, {1, 2, 1}});
  std::vector<MiviaGraphData> a{m};
  std::vector<MiviaGraphData> b{m};
  auto rs = mcs::fmcs::findMCESfMCSBatchLabeled(a, b);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
}

TEST(FMCSLabels, VertexLabelMismatch) {
  // Labels {7, 8, 9} vs {7, 8, 10} -- atom 2 doesn't match across.
  // The connected MCES drops the edge (1,2) and ends at the (0,1) edge.
  auto a = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {7, 8, 9},
      {{0, 1, 1}, {1, 2, 1}});
  auto b = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {7, 8, 10},
      {{0, 1, 1}, {1, 2, 1}});
  std::vector<MiviaGraphData> as{a};
  std::vector<MiviaGraphData> bs{b};
  auto rs = mcs::fmcs::findMCESfMCSBatchLabeled(as, bs);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonEdges, 1);
  EXPECT_EQ(rs[0].numCommonVertices, 2);
}

TEST(FMCSLabels, EdgeLabelMatch) {
  // Two cycles with identical edge labels.
  auto m = buildLabeled(3,
      {{0, 1}, {1, 2}, {0, 2}},
      {0, 0, 0},
      {{0, 1, 5}, {1, 2, 5}, {0, 2, 5}});
  std::vector<MiviaGraphData> a{m};
  std::vector<MiviaGraphData> b{m};
  auto rs = mcs::fmcs::findMCESfMCSBatchLabeled(a, b);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
  EXPECT_EQ(rs[0].numCommonEdges, 3);
}

TEST(FMCSLabels, EdgeLabelMismatch) {
  // Path-3, but one bond's label differs.  That edge can't appear in
  // the MCES; only the matching one survives.
  auto a = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {0, 0, 0},
      {{0, 1, 5}, {1, 2, 7}});
  auto b = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {0, 0, 0},
      {{0, 1, 5}, {1, 2, 9}});
  std::vector<MiviaGraphData> as{a};
  std::vector<MiviaGraphData> bs{b};
  auto rs = mcs::fmcs::findMCESfMCSBatchLabeled(as, bs);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonEdges, 1);
  EXPECT_EQ(rs[0].numCommonVertices, 2);
}

TEST(FMCSLabels, BothLabelsPartialOverlap) {
  // Path-4 with labels diverging at vertex 3 AND edge (2,3) on side B.
  // MCES is the agreeing prefix: 3 atoms, 2 bonds.
  auto a = buildLabeled(4,
      {{0, 1}, {1, 2}, {2, 3}},
      {1, 2, 3, 4},
      {{0, 1, 5}, {1, 2, 5}, {2, 3, 5}});
  auto b = buildLabeled(4,
      {{0, 1}, {1, 2}, {2, 3}},
      {1, 2, 3, 9},          // vertex 3 label diverges
      {{0, 1, 5}, {1, 2, 5}, {2, 3, 7}});  // and edge (2,3) too
  std::vector<MiviaGraphData> as{a};
  std::vector<MiviaGraphData> bs{b};
  auto rs = mcs::fmcs::findMCESfMCSBatchLabeled(as, bs);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
}

TEST(FMCSLabels, AtomCompareAnyIgnoresVertexLabels) {
  auto a = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {1, 2, 3},
      {{0, 1, 5}, {1, 2, 5}});
  auto b = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {7, 8, 9},
      {{0, 1, 5}, {1, 2, 5}});

  Parameters params;
  params.matchVertexLabels = false;
  params.matchEdgeLabels = true;
  auto rs = mcs::fmcs::findMCESfMCSBatchLabeled({a}, {b}, params);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
  expectMappingsConsistent(rs[0], a.graph, b.graph);
}

TEST(FMCSLabels, BondCompareAnyIgnoresEdgeLabels) {
  auto a = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {1, 2, 3},
      {{0, 1, 5}, {1, 2, 7}});
  auto b = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {1, 2, 3},
      {{0, 1, 11}, {1, 2, 13}});

  Parameters params;
  params.matchVertexLabels = true;
  params.matchEdgeLabels = false;
  auto rs = mcs::fmcs::findMCESfMCSBatchLabeled({a}, {b}, params);
  ASSERT_EQ(rs.size(), 1u);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[0].numCommonVertices, 3);
  expectMappingsConsistent(rs[0], a.graph, b.graph);
}

TEST(FMCSLabels, RingMembershipEncodedLabelsRestrictMatches) {
  constexpr uint16_t kRingAtom = static_cast<uint16_t>(6 | (1u << 9));
  constexpr uint16_t kRingBond = static_cast<uint16_t>(1 | (1u << 9));
  auto ring = buildLabeled(3,
      {{0, 1}, {1, 2}, {0, 2}},
      {kRingAtom, kRingAtom, kRingAtom},
      {{0, 1, kRingBond}, {1, 2, kRingBond}, {0, 2, kRingBond}});
  auto chain = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {6, 6, 6},
      {{0, 1, 1}, {1, 2, 1}});

  auto strict = mcs::fmcs::findMCESfMCSBatchLabeled({ring}, {chain});
  ASSERT_EQ(strict.size(), 1u);
  EXPECT_EQ(strict[0].numCommonEdges, 0);
  EXPECT_EQ(strict[0].numCommonVertices, 0);

  Parameters compareAny;
  compareAny.matchVertexLabels = false;
  compareAny.matchEdgeLabels = false;
  auto topologyOnly = mcs::fmcs::findMCESfMCSBatchLabeled(
      {ring}, {chain}, compareAny);
  ASSERT_EQ(topologyOnly.size(), 1u);
  EXPECT_EQ(topologyOnly[0].numCommonEdges, 2);
  EXPECT_EQ(topologyOnly[0].numCommonVertices, 3);
  expectMappingsConsistent(topologyOnly[0], ring.graph, chain.graph);
}

// ---------------------------------------------------------------------------
// FMCSTiers: exercise each kernel template instantiation.
// ---------------------------------------------------------------------------

TEST(FMCSTiers, MaxSize16) {
  const auto p = path(16);
  auto r = findSingleMCES(p, p);
  expectFullSelfPair(r, p);
}

TEST(FMCSTiers, MaxSize32) {
  const auto p = path(32);
  auto r = findSingleMCES(p, p);
  expectFullSelfPair(r, p);
}

TEST(FMCSTiers, MaxSize64) {
  const auto p = path(64);
  auto r = findSingleMCES(p, p);
  expectFullSelfPair(r, p);
}

TEST(FMCSTiers, MaxSize128) {
  const auto p = path(128);
  auto r = findSingleMCES(p, p);
  expectFullSelfPair(r, p);
}

TEST(FMCSBlockSize, SupportedSpecializationsMatchDefault) {
  const auto p = path(32);
  for (int blockSize : {128, 512}) {
    Parameters params;
    params.blockSize = blockSize;
    auto r = findSingleMCES(p, p, params);
    expectFullSelfPair(r, p);
  }
}

TEST(FMCSBlockSize, RejectsOneWarpBlockSize) {
  const auto p = path(8);
  for (int blockSize : {32, 64, 256}) {
    Parameters params;
    params.blockSize = blockSize;
    EXPECT_THROW(
        (void)mcs::fmcs::findMCESfMCSBatch({p}, {p}, params),
        std::invalid_argument);
  }
}

TEST(FMCSBlockSize, BlockSize512RejectsTier128) {
  const auto p = path(128);
  Parameters params;
  params.blockSize = 512;
  EXPECT_THROW(
      (void)mcs::fmcs::findMCESfMCSBatch({p}, {p}, params),
      std::invalid_argument);
}

// ---------------------------------------------------------------------------
// FMCSObjective: MaximizeBonds tie-break.
// ---------------------------------------------------------------------------

TEST(FMCSObjective, MaximizeBondsPreferredOverVerticesWhenTie) {
  // A query that contains both a 3-atom path (3 atoms, 2 bonds) and a
  // 3-atom triangle (3 atoms, 3 bonds) as connected subgraphs.  Target
  // is just the triangle.  Both candidate MCSes have 3 atoms; the
  // triangle has more bonds, so MaximizeBonds picks it.
  const auto a = g(4, {{0, 1}, {1, 2}, {0, 2}, {2, 3}});
  // Triangle on a, plus a tail 2-3.
  const auto b = cycle(3);
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(r.numCommonVertices, 3);
  EXPECT_EQ(r.numCommonEdges, 3);
  expectMappingsConsistent(r, a, b);
}

// ---------------------------------------------------------------------------
// FMCSBatch: batch-mode dispatch.
// ---------------------------------------------------------------------------

TEST(FMCSBatch, MixedSizes) {
  // One pair per tier.
  std::vector<Graph> a{path(8), path(24), path(48), path(96)};
  std::vector<Graph> b = a;
  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), 4u);
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_FALSE(rs[i].overflowed) << "pair " << i;
    EXPECT_EQ(rs[i].numCommonEdges, a[i].numEdges) << "pair " << i;
  }
}

TEST(FMCSBatch, EmptyInputReturnsEmpty) {
  std::vector<Graph> a;
  std::vector<Graph> b;
  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  EXPECT_TRUE(rs.empty());
}

TEST(FMCSBatch, CollectTimingsFollowsBuildFlag) {
  const auto p = path(8);
  std::vector<float> timesMs;

  if constexpr (nvMolKit::kMCSCollectTimingsEnabled) {
    auto rs = mcs::fmcs::findMCESfMCSBatch({p}, {p}, {}, &timesMs);
    ASSERT_EQ(rs.size(), 1u);
    ASSERT_EQ(timesMs.size(), 1u);
    EXPECT_GE(timesMs[0], 0.0f);
    expectFullSelfPair(rs[0], p);
  } else {
    EXPECT_THROW(
        (void)mcs::fmcs::findMCESfMCSBatch({p}, {p}, {}, &timesMs),
        std::runtime_error);
  }
}

TEST(FMCSBatch, CollectStatsFollowsBuildFlag) {
  const auto p = path(8);
  std::vector<ExecutionStats> stats;

  if constexpr (nvMolKit::kMCSCollectStatsEnabled) {
    auto rs = mcs::fmcs::findMCESfMCSBatch(
        {p}, {p}, {}, nullptr, nullptr, &stats);
    ASSERT_EQ(rs.size(), 1u);
    ASSERT_EQ(stats.size(), 1u);
    EXPECT_GT(stats[0].totalClocks, 0u);
    expectFullSelfPair(rs[0], p);
  } else {
    EXPECT_THROW(
        (void)mcs::fmcs::findMCESfMCSBatch(
            {p}, {p}, {}, nullptr, nullptr, &stats),
        std::runtime_error);
  }
}

TEST(FMCSBatch, TwoPairsDifferentAnswersSameTier) {
  // Same launch, same tier, different known answers.  A per-block slab
  // indexing bug tends to show up as pair 0 receiving pair 1's answer,
  // or vice versa.
  std::vector<Graph> a{path(3), star(3)};
  std::vector<Graph> b{path(6), star(4)};
  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), 2u);

  EXPECT_EQ(rs[0].numCommonVertices, 3);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  expectMappingsConsistent(rs[0], a[0], b[0]);

  EXPECT_EQ(rs[1].numCommonVertices, 4);
  EXPECT_EQ(rs[1].numCommonEdges, 3);
  expectMappingsConsistent(rs[1], a[1], b[1]);
}

TEST(FMCSBatch, NoCommonPairAdjacentToFullMatch) {
  // The zero-bond pairs should not leave stale mappings/results that
  // contaminate the full-match pair in the middle.
  std::vector<Graph> a{g(3, {}), path(5), path(2)};
  std::vector<Graph> b{path(4), path(5), g(2, {})};
  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), 3u);

  EXPECT_EQ(rs[0].numCommonVertices, 0);
  EXPECT_EQ(rs[0].numCommonEdges, 0);
  EXPECT_FALSE(rs[0].overflowed);

  expectFullSelfPair(rs[1], a[1]);
  expectMappingsConsistent(rs[1], a[1], b[1]);

  EXPECT_EQ(rs[2].numCommonVertices, 0);
  EXPECT_EQ(rs[2].numCommonEdges, 0);
  EXPECT_FALSE(rs[2].overflowed);
}

TEST(FMCSBatch, EmptyPairInBatch) {
  // Middle pair empty; flanking pairs should still produce results.
  std::vector<Graph> a{path(3), g(0, {}), path(5)};
  std::vector<Graph> b = a;
  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), 3u);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[1].numCommonEdges, 0);
  EXPECT_EQ(rs[2].numCommonEdges, 4);
}

TEST(FMCSBatch, HonorsBatchSizeChunksResults) {
  std::vector<Graph> a{path(3), star(3), cycle(4), path(5), g(4, {})};
  std::vector<Graph> b{path(6), star(4), cycle(4), path(7), path(3)};
  Parameters params;
  params.batchSize = 2;

  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b, params);
  ASSERT_EQ(rs.size(), a.size());
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[1].numCommonEdges, 3);
  EXPECT_EQ(rs[2].numCommonEdges, 4);
  EXPECT_EQ(rs[3].numCommonEdges, 4);
  EXPECT_EQ(rs[4].numCommonEdges, 0);
  for (size_t i = 0; i < rs.size(); ++i) {
    EXPECT_FALSE(rs[i].overflowed) << "pair " << i;
    expectMappingsConsistent(rs[i], a[i], b[i]);
  }
}

TEST(FMCSBatch, MultiExecutorChunksResults) {
  std::vector<Graph> a{path(3), star(3), cycle(4), path(5), g(4, {})};
  std::vector<Graph> b{path(6), star(4), cycle(4), path(7), path(3)};
  Parameters params;
  params.batchSize          = 1;
  params.executorsPerRunner = 2;

  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b, params);
  ASSERT_EQ(rs.size(), a.size());
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[1].numCommonEdges, 3);
  EXPECT_EQ(rs[2].numCommonEdges, 4);
  EXPECT_EQ(rs[3].numCommonEdges, 4);
  EXPECT_EQ(rs[4].numCommonEdges, 0);
  for (size_t i = 0; i < rs.size(); ++i) {
    EXPECT_FALSE(rs[i].overflowed) << "pair " << i;
    expectMappingsConsistent(rs[i], a[i], b[i]);
  }
}

TEST(FMCSBatch, ManyTinyPairsReusePerBlockSlabs) {
  // This is intentionally moderate while kFmcsQueueCapacity is still
  // over-provisioned per pair.  It still creates far more blocks than
  // the small correctness batches and exercises repeated queue/cache
  // slab slices within one tier launch.
  constexpr int kNumPairs = 512;
  std::vector<Graph> a;
  std::vector<Graph> b;
  a.reserve(kNumPairs);
  b.reserve(kNumPairs);
  for (int i = 0; i < kNumPairs; ++i) {
    const int n = (i % 2 == 0) ? 3 : 4;
    a.push_back(path(n));
    b.push_back(path(n));
  }

  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), static_cast<size_t>(kNumPairs));
  for (int i = 0; i < kNumPairs; ++i) {
    EXPECT_FALSE(rs[i].overflowed) << "pair " << i;
    EXPECT_EQ(rs[i].numCommonVertices, a[i].numVertices) << "pair " << i;
    EXPECT_EQ(rs[i].numCommonEdges, a[i].numEdges) << "pair " << i;
  }
}

TEST(FMCSBatch, LabeledMixed) {
  // Two labelled pairs in one batch.
  auto m1 = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {1, 2, 3},
      {{0, 1, 7}, {1, 2, 7}});
  auto m2 = buildLabeled(4,
      {{0, 1}, {1, 2}, {2, 3}},
      {1, 2, 3, 4},
      {{0, 1, 7}, {1, 2, 7}, {2, 3, 7}});
  std::vector<MiviaGraphData> as{m1, m2};
  std::vector<MiviaGraphData> bs{m1, m2};
  auto rs = mcs::fmcs::findMCESfMCSBatchLabeled(as, bs);
  ASSERT_EQ(rs.size(), 2u);
  EXPECT_EQ(rs[0].numCommonEdges, 2);
  EXPECT_EQ(rs[1].numCommonEdges, 3);
}

TEST(FMCSBatch, LabeledTwoPairsDifferentAnswersSameTier) {
  auto full = buildLabeled(4,
      {{0, 1}, {1, 2}, {2, 3}},
      {1, 2, 3, 4},
      {{0, 1, 7}, {1, 2, 7}, {2, 3, 7}});
  auto partialA = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {1, 2, 3},
      {{0, 1, 5}, {1, 2, 7}});
  auto partialB = buildLabeled(3,
      {{0, 1}, {1, 2}},
      {1, 2, 9},
      {{0, 1, 5}, {1, 2, 7}});

  std::vector<MiviaGraphData> as{full, partialA};
  std::vector<MiviaGraphData> bs{full, partialB};
  auto rs = mcs::fmcs::findMCESfMCSBatchLabeled(as, bs);
  ASSERT_EQ(rs.size(), 2u);
  EXPECT_EQ(rs[0].numCommonVertices, 4);
  EXPECT_EQ(rs[0].numCommonEdges, 3);
  EXPECT_EQ(rs[1].numCommonVertices, 2);
  EXPECT_EQ(rs[1].numCommonEdges, 1);
}

TEST(FMCSBatch, TargetLargerThanQueryUsesTargetTier) {
  // MatchResult target-side bitsets are tier-sized too.  This pair
  // should dispatch to the 64 tier even though the query itself fits
  // in tier 16.
  const auto small = path(4);
  const auto large = path(64);

  auto forward = findSingleMCES(small, large);
  EXPECT_EQ(forward.numCommonVertices, small.numVertices);
  EXPECT_EQ(forward.numCommonEdges, small.numEdges);
  EXPECT_FALSE(forward.overflowed);
  expectMappingsConsistent(forward, small, large);

  auto reversed = findSingleMCES(large, small);
  EXPECT_EQ(reversed.numCommonVertices, small.numVertices);
  EXPECT_EQ(reversed.numCommonEdges, small.numEdges);
  EXPECT_FALSE(reversed.overflowed);
  expectMappingsConsistent(reversed, large, small);
}

TEST(FMCSBatch, OverflowPairDoesNotBlockNeighbors) {
  std::vector<Graph> a{path(4), path(200), path(3)};
  std::vector<Graph> b = a;
  auto rs = mcs::fmcs::findMCESfMCSBatch(a, b);
  ASSERT_EQ(rs.size(), 3u);

  EXPECT_EQ(rs[0].numCommonEdges, 3);
  EXPECT_FALSE(rs[0].overflowed);

  EXPECT_TRUE(rs[1].overflowed);
  EXPECT_EQ(rs[1].numCommonVertices, 0);
  EXPECT_EQ(rs[1].numCommonEdges, 0);

  EXPECT_EQ(rs[2].numCommonEdges, 2);
  EXPECT_FALSE(rs[2].overflowed);
}

// ---------------------------------------------------------------------------
// FMCSOverflow: graph too large for the largest tier.
// ---------------------------------------------------------------------------

TEST(FMCSOverflow, GraphTooLargeFlagSet) {
  // Tier cap is 128 atoms / 128 bonds.  Build a path with 200 atoms
  // (199 bonds) -- exceeds tier 128 in atoms AND bonds.
  const auto p = path(200);
  auto r = findSingleMCES(p, p);
  EXPECT_TRUE(r.overflowed);
  EXPECT_EQ(r.numCommonEdges, 0);
  EXPECT_EQ(r.numCommonVertices, 0);
}

TEST(FMCSOverflow, TargetTooLargeFlagSetEvenWhenQueryFits) {
  const auto small = path(4);
  const auto large = path(200);
  auto r = findSingleMCES(small, large);
  EXPECT_TRUE(r.overflowed);
  EXPECT_EQ(r.numCommonEdges, 0);
  EXPECT_EQ(r.numCommonVertices, 0);
}

TEST(FMCSTimeout, PartialResultReturned) {
  const auto p = path(128);
  Parameters params;
  params.timeoutMs = 0.0001f;

  auto r = findSingleMCES(p, p, params);
  EXPECT_TRUE(r.timedOut);
  EXPECT_FALSE(r.overflowed);
  EXPECT_LT(r.numCommonEdges, p.numEdges);
  EXPECT_LT(r.numCommonVertices, p.numVertices);
}

// ---------------------------------------------------------------------------
// FMCSMappingConsistency: dedicated mapping-correctness tests.
//
// expectMappingsConsistent already checks the bond-mapping structure;
// these tests pin specific known properties on top of that.
// ---------------------------------------------------------------------------

TEST(FMCSMappingConsistency, SelfPairMappingsAreBijection) {
  const auto c = cycle(6);
  auto r = findSingleMCES(c, c);
  expectFullSelfPair(r, c);

  // Bijection: every query atom mapped exactly once, every target atom
  // mapped exactly once.
  std::set<std::size_t> qSeen(r.mappingA.begin(), r.mappingA.end());
  std::set<std::size_t> tSeen(r.mappingB.begin(), r.mappingB.end());
  ASSERT_EQ(qSeen.size(), 6u);
  ASSERT_EQ(tSeen.size(), 6u);
  for (int i = 0; i < 6; ++i) {
    EXPECT_TRUE(qSeen.count(i)) << "missing query atom " << i;
    EXPECT_TRUE(tSeen.count(i)) << "missing target atom " << i;
  }
}

TEST(FMCSMappingConsistency, BondMappingsConnectMatchedAtoms) {
  const auto a = path(5);
  const auto b = path(5);
  auto r = findSingleMCES(a, b);
  expectFullSelfPair(r, a);
  // expectMappingsConsistent does the structural check; we additionally
  // assert that every query bond appears in edgeMappingA exactly once.
  std::set<std::pair<std::size_t, std::size_t>> qBonds;
  for (const auto& e : r.edgeMappingA) {
    auto canon = std::make_pair(std::min(e.first, e.second),
                                std::max(e.first, e.second));
    EXPECT_TRUE(qBonds.insert(canon).second)
        << "duplicate query bond (" << e.first << "," << e.second << ")";
  }
  EXPECT_EQ(static_cast<int>(qBonds.size()), r.numCommonEdges);
  expectMappingsConsistent(r, a, b);
}

TEST(FMCSMappingConsistency, EdgeMappingArrayLengths) {
  // numCommonEdges must equal both edgeMappingA.size() and
  // edgeMappingB.size().  numCommonVertices must equal both
  // mappingA.size() and mappingB.size().
  const auto a = naphthalene();
  const auto b = phenanthrene();
  auto r = findSingleMCES(a, b);
  EXPECT_EQ(static_cast<int>(r.mappingA.size()),
            r.numCommonVertices);
  EXPECT_EQ(static_cast<int>(r.mappingB.size()),
            r.numCommonVertices);
  EXPECT_EQ(static_cast<int>(r.edgeMappingA.size()),
            r.numCommonEdges);
  EXPECT_EQ(static_cast<int>(r.edgeMappingB.size()),
            r.numCommonEdges);
  expectMappingsConsistent(r, a, b);
}
