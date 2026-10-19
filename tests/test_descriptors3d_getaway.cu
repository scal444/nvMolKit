// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "src/descriptors3d_getaway.cuh"
#include "src/utils/device_vector.h"

using namespace nvMolKit::descriptors3d_detail;

namespace {

constexpr int kConformersPerMolecule = 2;

//! Bitwise NaN check: the project's -ffast-math host flags fold std::isnan to false.
bool isNanBits(const double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint64_t kExponentMask = 0x7ff0000000000000ULL;
  constexpr uint64_t kMantissaMask = 0x000fffffffffffffULL;
  return (bits & kExponentMask) == kExponentMask && (bits & kMantissaMask) != 0;
}

/**
 * Linear chains of the given sizes on jittered helices (about 1.5 A between bonded atoms), two conformers each,
 * with arbitrary positive atom-property weights. Conformer @p invalidConformer gets a molecule index past the
 * batch, so it is invalid.
 */
struct ChainBatch {
  std::vector<int32_t> moleculeAtomStarts{0};
  std::vector<int32_t> neighborStarts{0};
  std::vector<int32_t> neighbors;
  std::vector<double>  weights;
  std::vector<uint8_t> heavyFlags;
  std::vector<int32_t> defaultRows;
  std::vector<double>  positions;
  std::vector<int32_t> atomStarts{0};
  std::vector<int32_t> molIndices;
  int                  maxAtoms = 0;
  int64_t              pairs    = 0;

  ChainBatch(const std::vector<int>& sizes, const int invalidConformer) {
    std::mt19937                     rng(17);
    std::normal_distribution<double> jitter(0.0, 0.05);
    for (size_t molIdx = 0; molIdx < sizes.size(); ++molIdx) {
      const int size = sizes[molIdx];
      for (int atom = 0; atom < size; ++atom) {
        if (atom > 0) {
          neighbors.push_back(atom - 1);
        }
        if (atom + 1 < size) {
          neighbors.push_back(atom + 1);
        }
        neighborStarts.push_back(static_cast<int32_t>(neighbors.size()));
        heavyFlags.push_back(1);
      }
      moleculeAtomStarts.push_back(moleculeAtomStarts.back() + size);
      maxAtoms = std::max(maxAtoms, size);
      pairs += static_cast<int64_t>(size) * (size - 1) / 2;
      defaultRows.push_back(static_cast<int32_t>(molIndices.size()));
      for (int conf = 0; conf < kConformersPerMolecule; ++conf) {
        const double radius = 3.0 + 0.2 * conf;
        for (int atom = 0; atom < size; ++atom) {
          const double angle = 0.5 * atom;
          positions.push_back(radius * std::cos(angle) + jitter(rng));
          positions.push_back(radius * std::sin(angle) + jitter(rng));
          positions.push_back(0.3 * atom + jitter(rng));
        }
        atomStarts.push_back(atomStarts.back() + size);
        const int confIdx = static_cast<int>(molIndices.size());
        molIndices.push_back(confIdx == invalidConformer ? static_cast<int32_t>(sizes.size()) :
                                                           static_cast<int32_t>(molIdx));
      }
    }
    const int totalAtoms = moleculeAtomStarts.back();
    for (int channel = 0; channel < kNumAtomPropertyChannels; ++channel) {
      for (int atom = 0; atom < totalAtoms; ++atom) {
        weights.push_back(0.5 + 0.1 * ((7 * atom + 3 * channel) % 11));
      }
    }
  }
};

//! How the batch's bond-distance table is limited.
enum class TableMode {
  kFull,       //!< every molecule has table entries
  kTruncated,  //!< molecules from @p firstUncovered on lie past the table's capacity
  kNone,       //!< no table at all
};

/**
 * @brief GETAWAY for every conformer of @p batch through launchGetawayProperties(), with the bond-distance table
 *        limited as @p mode says.
 */
std::vector<double> getaway(const ChainBatch& batch, const TableMode mode, const int firstUncovered = 0) {
  nvMolKit::AsyncDeviceVector<int32_t> moleculeAtomStarts(batch.moleculeAtomStarts.size());
  nvMolKit::AsyncDeviceVector<int32_t> neighborStarts(batch.neighborStarts.size());
  nvMolKit::AsyncDeviceVector<int32_t> neighbors(batch.neighbors.size());
  nvMolKit::AsyncDeviceVector<double>  weights(batch.weights.size());
  nvMolKit::AsyncDeviceVector<uint8_t> heavyFlags(batch.heavyFlags.size());
  nvMolKit::AsyncDeviceVector<int32_t> defaultRows(batch.defaultRows.size());
  nvMolKit::AsyncDeviceVector<double>  positions(batch.positions.size());
  nvMolKit::AsyncDeviceVector<int32_t> atomStarts(batch.atomStarts.size());
  nvMolKit::AsyncDeviceVector<int32_t> molIndices(batch.molIndices.size());
  moleculeAtomStarts.copyFromHost(batch.moleculeAtomStarts);
  neighborStarts.copyFromHost(batch.neighborStarts);
  neighbors.copyFromHost(batch.neighbors);
  weights.copyFromHost(batch.weights);
  heavyFlags.copyFromHost(batch.heavyFlags);
  defaultRows.copyFromHost(batch.defaultRows);
  positions.copyFromHost(batch.positions);
  atomStarts.copyFromHost(batch.atomStarts);
  molIndices.copyFromHost(batch.molIndices);

  nvMolKit::DeviceCoordView coordinates;
  coordinates.positions     = positions.data();
  coordinates.atomStarts    = atomStarts.data();
  coordinates.molIndices    = molIndices.data();
  coordinates.numConformers = static_cast<int>(batch.molIndices.size());
  coordinates.nMols         = static_cast<int>(batch.defaultRows.size());
  coordinates.numAtoms      = static_cast<int64_t>(batch.positions.size() / 3);

  nvMolKit::Property3DDeviceInputs inputs;
  inputs.moleculeAtomStarts   = moleculeAtomStarts.data();
  inputs.atomPropertyWeights  = weights.data();
  inputs.bondNeighborStarts   = neighborStarts.data();
  inputs.bondNeighbors        = neighbors.data();
  inputs.heavyAtomFlags       = heavyFlags.data();
  inputs.defaultConformerRows = defaultRows.data();
  inputs.maxMoleculeAtoms     = batch.maxAtoms;
  inputs.moleculeAtomPairs    = batch.pairs;

  BondDistanceTableStorage storage;
  BondDistanceTable        table;
  if (mode != TableMode::kNone) {
    buildBondDistanceTable(coordinates, inputs, storage, nullptr);
    table = storage.table;
  }
  if (mode == TableMode::kTruncated) {
    // As when the batch's pairs exceed the table budget: the table ends where molecule firstUncovered starts.
    int64_t capacity = 0;
    for (int molIdx = 0; molIdx < firstUncovered; ++molIdx) {
      const int64_t size = batch.moleculeAtomStarts[molIdx + 1] - batch.moleculeAtomStarts[molIdx];
      capacity += size * (size - 1) / 2;
    }
    table.capacity = capacity;
  }

  nvMolKit::AsyncDeviceVector<double> output(batch.molIndices.size() * nvMolKit::kNumGetawayProperties);
  launchGetawayProperties(coordinates, inputs, nvMolKit::GetawayOptions{}, table, output.data(), nullptr);
  std::vector<double> host(output.size());
  output.copyToHost(host);
  EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
  return host;
}

void expectBitwiseEqual(const std::vector<double>& actual, const std::vector<double>& expected, const char* label) {
  ASSERT_EQ(actual.size(), expected.size()) << label;
  for (size_t idx = 0; idx < actual.size(); ++idx) {
    EXPECT_EQ(std::memcmp(&actual[idx], &expected[idx], sizeof(double)), 0)
      << label << ": conformer " << idx / nvMolKit::kNumGetawayProperties << ", value "
      << idx % nvMolKit::kNumGetawayProperties << ": " << actual[idx] << " vs " << expected[idx];
  }
}

}  // namespace

TEST(GetawayBondDistanceTable, CoverageDoesNotChangeResults) {
  // Molecules of 12 to 400 atoms (the largest past every device's shared memory) with one invalid conformer, which
  // belongs to the 40-atom molecule. Whether a molecule's bond distances come from the table or from per-row
  // searches must not change any value.
  const std::vector<int> sizes = {12, 132, 40, 400, 20};
  const ChainBatch       batch(sizes, /*invalidConformer=*/4);

  const auto searched = getaway(batch, TableMode::kNone);
  for (int value = 0; value < nvMolKit::kNumGetawayProperties; ++value) {
    EXPECT_TRUE(isNanBits(searched[4 * nvMolKit::kNumGetawayProperties + value])) << "value " << value;
    EXPECT_FALSE(isNanBits(searched[5 * nvMolKit::kNumGetawayProperties + value])) << "value " << value;
  }
  expectBitwiseEqual(getaway(batch, TableMode::kFull), searched, "full table");
  // As when the batch's pairs exceed the table budget: the last three molecules (including the invalid
  // conformer's) have no table entries while the first two do.
  expectBitwiseEqual(getaway(batch, TableMode::kTruncated, 2), searched, "truncated table");
}
