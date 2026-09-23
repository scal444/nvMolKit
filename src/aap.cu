// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/ROMol.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/aap.h"
#include "src/diversity_picker_algorithms.cuh"
#include "src/utils/cuda_error_check.h"
#include "src/utils/device_vector.h"
#include "src/utils/host_vector.h"

namespace nvMolKit {
namespace {

// TODO: Support molecules with more than 64 atoms.
constexpr int           kMaxAtoms        = 64;
constexpr int           kSharedStride    = kMaxAtoms + 1;
constexpr int           kThreads         = 256;
constexpr std::uint64_t kFnvOffset       = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime        = 1099511628211ULL;
constexpr std::uint64_t kSignedInt64Mask = (1ULL << 63) - 1;

struct AapHostDescriptors {
  std::vector<std::int64_t> moleculeAtomOffsets{0};
  std::vector<std::int64_t> atomBinOffsets{0};
  std::vector<std::int64_t> atomPathLengths;
  std::vector<std::int16_t> atomNumbers;
  std::vector<std::uint8_t> aromatic;
  std::vector<std::int16_t> binIds;
  std::vector<std::int32_t> binCounts;
};

struct AapDeviceDescriptors {
  explicit AapDeviceDescriptors(const AapHostDescriptors& host, cudaStream_t stream)
      : moleculeAtomOffsets(host.moleculeAtomOffsets.size(), stream),
        atomBinOffsets(host.atomBinOffsets.size(), stream),
        atomPathLengths(host.atomPathLengths.size(), stream),
        atomNumbers(host.atomNumbers.size(), stream),
        aromatic(host.aromatic.size(), stream),
        binIds(host.binIds.size(), stream),
        binCounts(host.binCounts.size(), stream) {
    moleculeAtomOffsets.copyFromHost(host.moleculeAtomOffsets);
    atomBinOffsets.copyFromHost(host.atomBinOffsets);
    atomPathLengths.copyFromHost(host.atomPathLengths);
    atomNumbers.copyFromHost(host.atomNumbers);
    aromatic.copyFromHost(host.aromatic);
    binIds.copyFromHost(host.binIds);
    binCounts.copyFromHost(host.binCounts);
  }

  AsyncDeviceVector<std::int64_t> moleculeAtomOffsets;
  AsyncDeviceVector<std::int64_t> atomBinOffsets;
  AsyncDeviceVector<std::int64_t> atomPathLengths;
  AsyncDeviceVector<std::int16_t> atomNumbers;
  AsyncDeviceVector<std::uint8_t> aromatic;
  AsyncDeviceVector<std::int16_t> binIds;
  AsyncDeviceVector<std::int32_t> binCounts;
};

int aapBondCode(const RDKit::Bond& bond) {
  switch (bond.getBondType()) {
    case RDKit::Bond::SINGLE:
      return 1;
    case RDKit::Bond::DOUBLE:
      return 2;
    case RDKit::Bond::TRIPLE:
      return 3;
    case RDKit::Bond::AROMATIC:
      return 4;
    default:
      throw std::invalid_argument("AAP supports only single, double, triple, and aromatic bonds");
  }
}

void collectRootedPathBins(const RDKit::ROMol&                   mol,
                           const unsigned int                    currentAtom,
                           const int                             depth,
                           const int                             maxDepth,
                           const int                             histogramBins,
                           const std::uint64_t                   pathCode,
                           std::vector<std::uint8_t>&            visited,
                           std::map<std::int16_t, std::int32_t>& counts,
                           std::int64_t&                         pathCount) {
  if (depth == maxDepth) {
    return;
  }

  const auto* atom       = mol.getAtomWithIdx(currentAtom);
  auto [bondIt, bondEnd] = mol.getAtomBonds(atom);
  while (bondIt != bondEnd) {
    const auto* bond      = mol[*bondIt];
    const auto  otherAtom = bond->getOtherAtomIdx(currentAtom);
    ++bondIt;
    if (visited[otherAtom]) {
      continue;
    }

    const auto* neighbor = mol.getAtomWithIdx(otherAtom);
    const auto  atomCode = neighbor->getAtomicNum() + (neighbor->getIsAromatic() ? 108 : 0);
    const auto  token    = static_cast<std::uint64_t>(aapBondCode(*bond) * 256 + atomCode);
    const auto  nextCode = ((pathCode ^ token) * kFnvPrime) & kSignedInt64Mask;
    const auto  bin      = static_cast<std::int16_t>(nextCode % static_cast<std::uint64_t>(histogramBins));
    ++counts[bin];
    ++pathCount;

    visited[otherAtom] = 1;
    collectRootedPathBins(mol, otherAtom, depth + 1, maxDepth, histogramBins, nextCode, visited, counts, pathCount);
    visited[otherAtom] = 0;
  }
}

void validateOptions(const AapOptions& options) {
  if (options.maxPathLength <= 0) {
    throw std::invalid_argument("maxPathLength must be positive");
  }
  if (options.histogramBins <= 0 || options.histogramBins > std::numeric_limits<std::int16_t>::max()) {
    throw std::invalid_argument("histogramBins must be between 1 and 32767");
  }
  if (options.sinkhornIterations <= 0) {
    throw std::invalid_argument("sinkhornIterations must be positive");
  }
  if (!(options.sinkhornTemperature > 0.0F) || !std::isfinite(options.sinkhornTemperature)) {
    throw std::invalid_argument("sinkhornTemperature must be finite and positive");
  }
  if (options.sinkhornTemperature < std::numeric_limits<float>::min()) {
    throw std::invalid_argument("sinkhornTemperature must be at least the smallest positive normal float");
  }
}

AapHostDescriptors buildDescriptors(const std::vector<const RDKit::ROMol*>& molecules, const AapOptions& options) {
  AapHostDescriptors result;
  for (std::size_t moleculeIdx = 0; moleculeIdx < molecules.size(); ++moleculeIdx) {
    const auto* mol = molecules[moleculeIdx];
    if (mol == nullptr) {
      throw std::invalid_argument("Invalid molecule at index " + std::to_string(moleculeIdx));
    }
    const auto numAtoms = mol->getNumAtoms();
    if (numAtoms == 0) {
      throw std::invalid_argument("AAP does not support empty molecules");
    }
    if (numAtoms > kMaxAtoms) {
      throw std::invalid_argument("AAP currently supports at most " + std::to_string(kMaxAtoms) +
                                  " atoms per molecule");
    }

    std::vector<std::uint8_t> visited(numAtoms, 0);
    for (unsigned int atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
      const auto* atom = mol->getAtomWithIdx(atomIdx);
      result.atomNumbers.push_back(static_cast<std::int16_t>(atom->getAtomicNum()));
      result.aromatic.push_back(static_cast<std::uint8_t>(atom->getIsAromatic()));

      std::map<std::int16_t, std::int32_t> counts;
      std::int64_t                         pathCount = 0;
      visited[atomIdx]                               = 1;
      collectRootedPathBins(*mol,
                            atomIdx,
                            0,
                            options.maxPathLength,
                            options.histogramBins,
                            kFnvOffset,
                            visited,
                            counts,
                            pathCount);
      visited[atomIdx] = 0;

      result.atomPathLengths.push_back(pathCount);
      for (const auto& [bin, count] : counts) {
        result.binIds.push_back(bin);
        result.binCounts.push_back(count);
      }
      result.atomBinOffsets.push_back(static_cast<std::int64_t>(result.binIds.size()));
    }
    result.moleculeAtomOffsets.push_back(static_cast<std::int64_t>(result.atomNumbers.size()));
  }
  return result;
}

bool sameAtomDescriptor(const AapHostDescriptors& descriptors, const std::int64_t left, const std::int64_t right) {
  if (descriptors.atomNumbers[left] != descriptors.atomNumbers[right] ||
      descriptors.aromatic[left] != descriptors.aromatic[right] ||
      descriptors.atomPathLengths[left] != descriptors.atomPathLengths[right]) {
    return false;
  }
  const auto leftBinStart  = descriptors.atomBinOffsets[left];
  const auto leftBinEnd    = descriptors.atomBinOffsets[left + 1];
  const auto rightBinStart = descriptors.atomBinOffsets[right];
  const auto rightBinEnd   = descriptors.atomBinOffsets[right + 1];
  if (leftBinEnd - leftBinStart != rightBinEnd - rightBinStart) {
    return false;
  }
  for (std::int64_t binOffset = 0; binOffset < leftBinEnd - leftBinStart; ++binOffset) {
    if (descriptors.binIds[leftBinStart + binOffset] != descriptors.binIds[rightBinStart + binOffset] ||
        descriptors.binCounts[leftBinStart + binOffset] != descriptors.binCounts[rightBinStart + binOffset]) {
      return false;
    }
  }
  return true;
}

bool sameMoleculeDescriptor(const AapHostDescriptors& descriptors, const int left, const int right) {
  const auto leftAtomStart  = descriptors.moleculeAtomOffsets[left];
  const auto leftAtomEnd    = descriptors.moleculeAtomOffsets[left + 1];
  const auto rightAtomStart = descriptors.moleculeAtomOffsets[right];
  const auto rightAtomEnd   = descriptors.moleculeAtomOffsets[right + 1];
  if (leftAtomEnd - leftAtomStart != rightAtomEnd - rightAtomStart) {
    return false;
  }

  std::vector<std::uint8_t> matched(rightAtomEnd - rightAtomStart, 0);
  for (std::int64_t atomOffset = 0; atomOffset < leftAtomEnd - leftAtomStart; ++atomOffset) {
    bool found = false;
    for (std::int64_t rightOffset = 0; rightOffset < rightAtomEnd - rightAtomStart; ++rightOffset) {
      if (!matched[rightOffset] &&
          sameAtomDescriptor(descriptors, leftAtomStart + atomOffset, rightAtomStart + rightOffset)) {
        matched[rightOffset] = 1;
        found                = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ float rowLogSumExp(const float* values, const int row, const int size) {
  float maximum = -INFINITY;
  for (int column = 0; column < size; ++column) {
    maximum = fmaxf(maximum, values[row * kSharedStride + column]);
  }
  float sum = 0.0F;
  for (int column = 0; column < size; ++column) {
    sum += expf(values[row * kSharedStride + column] - maximum);
  }
  return maximum + logf(sum);
}

__device__ __forceinline__ float columnLogSumExp(const float* values, const int column, const int size) {
  float maximum = -INFINITY;
  for (int row = 0; row < size; ++row) {
    maximum = fmaxf(maximum, values[row * kSharedStride + column]);
  }
  float sum = 0.0F;
  for (int row = 0; row < size; ++row) {
    sum += expf(values[row * kSharedStride + column] - maximum);
  }
  return maximum + logf(sum);
}

//! Identical descriptor multisets score exactly 1 regardless of Sinkhorn convergence. Groups are keyed by their first
//! member so every pair lookup on the device is a single comparison.
std::vector<int> descriptorGroups(const AapHostDescriptors& descriptors) {
  const auto mix = [](std::uint64_t hash, const std::uint64_t value) { return (hash ^ value) * kFnvPrime; };

  const int        numMolecules = static_cast<int>(descriptors.moleculeAtomOffsets.size()) - 1;
  std::vector<int> groups(numMolecules);
  std::unordered_map<std::uint64_t, std::vector<int>> representatives;
  std::vector<std::uint64_t>                          atomHashes;
  for (int molecule = 0; molecule < numMolecules; ++molecule) {
    atomHashes.clear();
    for (auto atom = descriptors.moleculeAtomOffsets[molecule]; atom < descriptors.moleculeAtomOffsets[molecule + 1];
         ++atom) {
      std::uint64_t hash = kFnvOffset;
      hash               = mix(hash, static_cast<std::uint64_t>(descriptors.atomNumbers[atom]));
      hash               = mix(hash, descriptors.aromatic[atom]);
      hash               = mix(hash, static_cast<std::uint64_t>(descriptors.atomPathLengths[atom]));
      for (auto bin = descriptors.atomBinOffsets[atom]; bin < descriptors.atomBinOffsets[atom + 1]; ++bin) {
        hash = mix(hash, static_cast<std::uint64_t>(descriptors.binIds[bin]));
        hash = mix(hash, static_cast<std::uint64_t>(descriptors.binCounts[bin]));
      }
      atomHashes.push_back(hash);
    }
    std::sort(atomHashes.begin(), atomHashes.end());
    std::uint64_t moleculeHash = kFnvOffset;
    for (const auto atomHash : atomHashes) {
      moleculeHash = mix(moleculeHash, atomHash);
    }

    auto& candidates = representatives[moleculeHash];
    groups[molecule] = molecule;
    const auto match = std::find_if(candidates.begin(), candidates.end(), [&](const int representative) {
      return sameMoleculeDescriptor(descriptors, representative, molecule);
    });
    if (match != candidates.end()) {
      groups[molecule] = *match;
    } else {
      candidates.push_back(molecule);
    }
  }
  return groups;
}

struct AapDeviceView {
  const std::int64_t* moleculeAtomOffsets;
  const std::int64_t* atomBinOffsets;
  const std::int64_t* atomPathLengths;
  const std::int16_t* atomNumbers;
  const std::uint8_t* aromatic;
  const std::int16_t* binIds;
  const std::int32_t* binCounts;
  const int*          groups;
};

template <typename Op>
__global__ void compactCandidatesKernel(const int  numItems,
                                        const int* sourcePtr,
                                        const Op   op,
                                        int*       candidates,
                                        int*       count) {
  const int source    = *sourcePtr;
  const int candidate = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (source < 0 || source >= numItems || candidate >= numItems || op.skip(candidate)) {
    return;
  }
  candidates[atomicAdd(count, 1)] = candidate;
}

template <typename Op>
__global__ void aapDistanceKernel(const AapDeviceView view,
                                  const int*          candidates,
                                  const int*          candidateCount,
                                  const int*          sourcePtr,
                                  const float         temperature,
                                  const int           iterations,
                                  const Op            op) {
  __shared__ float affinity[kMaxAtoms * kSharedStride];
  __shared__ float transport[kMaxAtoms * kSharedStride];

  const int count  = *candidateCount;
  const int source = count > 0 ? *sourcePtr : 0;
  for (int position = static_cast<int>(blockIdx.x); position < count; position += static_cast<int>(gridDim.x)) {
    const int candidate = candidates[position];
    if (view.groups[candidate] == view.groups[source]) {
      if (threadIdx.x == 0) {
        op.apply(candidate, 0.0);
      }
      continue;
    }

    const int leftStart      = static_cast<int>(view.moleculeAtomOffsets[source]);
    const int rightStart     = static_cast<int>(view.moleculeAtomOffsets[candidate]);
    const int leftCount      = static_cast<int>(view.moleculeAtomOffsets[source + 1]) - leftStart;
    const int candidateAtoms = static_cast<int>(view.moleculeAtomOffsets[candidate + 1]) - rightStart;
    const int squareSize     = max(leftCount, candidateAtoms);

    for (int linear = static_cast<int>(threadIdx.x); linear < squareSize * squareSize; linear += blockDim.x) {
      const int row    = linear / squareSize;
      const int column = linear % squareSize;
      float     score  = 0.0F;
      if (row < leftCount && column < candidateAtoms &&
          view.atomNumbers[leftStart + row] == view.atomNumbers[rightStart + column] &&
          view.aromatic[leftStart + row] == view.aromatic[rightStart + column]) {
        std::int64_t leftPosition  = view.atomBinOffsets[leftStart + row];
        const auto   leftEnd       = view.atomBinOffsets[leftStart + row + 1];
        std::int64_t rightPosition = view.atomBinOffsets[rightStart + column];
        const auto   rightEnd      = view.atomBinOffsets[rightStart + column + 1];
        std::int64_t overlap       = 0;
        while (leftPosition < leftEnd && rightPosition < rightEnd) {
          const auto leftBin  = view.binIds[leftPosition];
          const auto rightBin = view.binIds[rightPosition];
          if (leftBin == rightBin) {
            overlap += min(view.binCounts[leftPosition], view.binCounts[rightPosition]);
            ++leftPosition;
            ++rightPosition;
          } else if (leftBin < rightBin) {
            ++leftPosition;
          } else {
            ++rightPosition;
          }
        }
        const auto largestPathCount =
          max(view.atomPathLengths[leftStart + row], view.atomPathLengths[rightStart + column]);
        score = static_cast<float>(overlap + 1) / static_cast<float>(2 * largestPathCount - overlap + 1);
      }
      affinity[row * kSharedStride + column]  = score;
      transport[row * kSharedStride + column] = score / temperature;
    }
    __syncthreads();

    for (int iteration = 0; iteration < iterations; ++iteration) {
      if (threadIdx.x < squareSize) {
        const int   row        = static_cast<int>(threadIdx.x);
        const float normalizer = rowLogSumExp(transport, row, squareSize);
        for (int column = 0; column < squareSize; ++column) {
          transport[row * kSharedStride + column] -= normalizer;
        }
      }
      __syncthreads();
      if (threadIdx.x < squareSize) {
        const int   column     = static_cast<int>(threadIdx.x);
        const float normalizer = columnLogSumExp(transport, column, squareSize);
        for (int row = 0; row < squareSize; ++row) {
          transport[row * kSharedStride + column] -= normalizer;
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      float matched = 0.0F;
      for (int row = 0; row < leftCount; ++row) {
        for (int column = 0; column < candidateAtoms; ++column) {
          matched += expf(transport[row * kSharedStride + column]) * affinity[row * kSharedStride + column];
        }
      }
      const float similarity = matched / (2.0F * static_cast<float>(leftCount) - matched + 1e-10F);
      op.apply(candidate, 1.0 - static_cast<double>(similarity));
    }
    // The next candidate overwrites shared memory that thread 0 just read.
    __syncthreads();
  }
}

class AapDistanceProvider {
 public:
  AapDistanceProvider(const std::vector<const RDKit::ROMol*>& molecules, const AapOptions& options, cudaStream_t stream)
      : options_(options),
        numItems_(static_cast<int>(molecules.size())),
        hostDescriptors_(buildDescriptors(molecules, options)),
        descriptors_(hostDescriptors_, stream),
        groups_(numItems_, stream),
        candidates_(numItems_, stream),
        candidateCount_(0, stream) {
    groups_.copyFromHost(descriptorGroups(hostDescriptors_));
    int device = 0;
    int numSms = 0;
    cudaCheckError(cudaGetDevice(&device));
    cudaCheckError(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));
    gridSize_ = std::max(1, std::min(numItems_, numSms * kBlocksPerSm));
  }

  int size() const { return numItems_; }

  template <typename Op> void forEachDistance(const int* source, const Op& op, cudaStream_t stream) {
    if (numItems_ == 0) {
      return;
    }
    cudaCheckError(cudaMemsetAsync(candidateCount_.data(), 0, sizeof(int), stream));
    compactCandidatesKernel<<<(numItems_ + kThreads - 1) / kThreads, kThreads, 0, stream>>>(numItems_,
                                                                                            source,
                                                                                            op,
                                                                                            candidates_.data(),
                                                                                            candidateCount_.data());
    cudaCheckError(cudaGetLastError());
    const AapDeviceView view{descriptors_.moleculeAtomOffsets.data(),
                             descriptors_.atomBinOffsets.data(),
                             descriptors_.atomPathLengths.data(),
                             descriptors_.atomNumbers.data(),
                             descriptors_.aromatic.data(),
                             descriptors_.binIds.data(),
                             descriptors_.binCounts.data(),
                             groups_.data()};
    aapDistanceKernel<<<gridSize_, kThreads, 0, stream>>>(view,
                                                          candidates_.data(),
                                                          candidateCount_.data(),
                                                          source,
                                                          options_.sinkhornTemperature,
                                                          options_.sinkhornIterations,
                                                          op);
    cudaCheckError(cudaGetLastError());
  }

 private:
  static constexpr int kBlocksPerSm = 4;

  AapOptions             options_;
  int                    numItems_;
  AapHostDescriptors     hostDescriptors_;
  AapDeviceDescriptors   descriptors_;
  AsyncDeviceVector<int> groups_;
  AsyncDeviceVector<int> candidates_;
  AsyncDevicePtr<int>    candidateCount_;
  int                    gridSize_ = 1;
};

//! Stores the similarity of one candidate to the source.
struct PairSimilarityOp {
  float* similarity;
  int    target;

  __device__ bool skip(const int candidate) const { return candidate != target; }
  __device__ void apply(const int /*candidate*/, const double distance) const {
    *similarity = static_cast<float>(1.0 - distance);
  }
};

}  // namespace

float aapSimilarityGpu(const RDKit::ROMol& left,
                       const RDKit::ROMol& right,
                       const AapOptions&   options,
                       cudaStream_t        stream) {
  validateOptions(options);
  AapDistanceProvider     provider({&left, &right}, options, stream);
  AsyncDevicePtr<int>     source(0, stream);
  AsyncDevicePtr<float>   similarity(0.0F, stream);
  PinnedHostVector<float> similarityHost(1);
  provider.forEachDistance(source.data(), PairSimilarityOp{similarity.data(), 1}, stream);
  cudaCheckError(
    cudaMemcpyAsync(similarityHost.data(), similarity.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaCheckError(cudaStreamSynchronize(stream));
  return similarityHost[0];
}

PickerResult aapLeader(const std::vector<const RDKit::ROMol*>& molecules,
                       const double                            cutoff,
                       const AapOptions&                       options,
                       const int                               pickSize,
                       const std::vector<int>&                 firstPicks,
                       cudaStream_t                            stream) {
  validateOptions(options);
  detail::validateUnitCutoff(cutoff);
  AapDistanceProvider provider(molecules, options, stream);
  return detail::leaderPick(provider, cutoff, pickSize, firstPicks, nullptr, stream);
}

PickerResult aapMaxMin(const std::vector<const RDKit::ROMol*>& molecules,
                       const int                               pickSize,
                       const AapOptions&                       options,
                       const std::vector<int>&                 firstPicks,
                       const int                               seed,
                       const double                            threshold,
                       cudaStream_t                            stream) {
  validateOptions(options);
  detail::validateMaxMinThreshold(threshold, 1.0);
  AapDistanceProvider provider(molecules, options, stream);
  return detail::maxMinPick(provider, pickSize, firstPicks, seed, threshold, stream);
}

ClusteringResult aapDise(const std::vector<const RDKit::ROMol*>& molecules,
                         const double                            cutoff,
                         const AapOptions&                       options,
                         const bool                              nearestAssignment,
                         cudaStream_t                            stream) {
  validateOptions(options);
  detail::validateUnitCutoff(cutoff);
  AapDistanceProvider provider(molecules, options, stream);
  return detail::diseCluster(provider, cutoff, nearestAssignment, stream);
}

}  // namespace nvMolKit
