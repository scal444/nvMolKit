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

#include <GraphMol/Conformer.h>
#include <GraphMol/ROMol.h>

#include <bit>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "src/coulomb_fields.h"
#include "src/utils/host_vector.h"

namespace nvMolKit {
namespace {

bool finiteValue(const double value) {
  // The host compiler enables fast math, so classify the exponent directly.
  return (std::bit_cast<uint64_t>(value) & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

void validateOptions(const CoulombFieldOptions& options) {
  if (!finiteValue(options.probeCharge) || !finiteValue(options.alpha) || options.alpha < 0 ||
      !finiteValue(options.cutoff) || options.cutoff < 0 || !finiteValue(options.cutoff * options.cutoff) ||
      !finiteValue(options.threshold) || (options.threshold < 0 && options.threshold != -1) ||
      !finiteValue(options.threshold * options.threshold) || !finiteValue(options.epsilon) || options.epsilon <= 0 ||
      !finiteValue(options.xi) || options.xi <= 0 || !finiteValue(options.xi + options.epsilon))
    throw std::invalid_argument("Invalid Coulomb field options");
}

}  // namespace

AsyncDeviceVector<double> coulombFieldsMol(const std::vector<const RDKit::ROMol*>& molecules,
                                           cuda::std::span<const double>           points,
                                           const std::vector<int>&                 conformerIds,
                                           const std::string&                      chargeProperty,
                                           const CoulombFieldOptions&              options,
                                           cudaStream_t                            stream) {
  validateOptions(options);
  if (molecules.size() != conformerIds.size())
    throw std::invalid_argument("Expected one conformer ID per molecule");
  if (points.size() % 3 != 0 || chargeProperty.empty())
    throw std::invalid_argument("Expected xyz query points and a nonempty charge property");
  const size_t pointCount = points.size() / 3;
  if (molecules.size() > INT_MAX ||
      (pointCount != 0 && molecules.size() > std::numeric_limits<size_t>::max() / sizeof(double) / pointCount))
    throw std::overflow_error("Coulomb field output exceeds addressable size");
  std::vector<size_t> offsets{0};
  for (const auto* molecule : molecules) {
    if (!molecule)
      throw std::invalid_argument("Null molecule in Coulomb field batch");
    const size_t count = molecule->getNumAtoms();
    if (count > std::numeric_limits<size_t>::max() / sizeof(double) / 3 - offsets.back())
      throw std::overflow_error("Coulomb atom coordinates exceed addressable size");
    offsets.push_back(offsets.back() + count);
  }
  PinnedHostVector<double> positions(3 * offsets.back());
  PinnedHostVector<double> charges(offsets.back());
  for (size_t item = 0; item < molecules.size(); ++item) {
    if (conformerIds[item] < -1)
      throw std::invalid_argument("Conformer IDs must be -1 or nonnegative");
    const auto& conformer = molecules[item]->getConformer(conformerIds[item]);
    for (unsigned int atom = 0; atom < molecules[item]->getNumAtoms(); ++atom) {
      const size_t packed = offsets[item] + atom;
      const double charge = molecules[item]->getAtomWithIdx(atom)->getProp<double>(chargeProperty);
      if (!finiteValue(charge))
        throw std::invalid_argument("Partial charges must be finite");
      charges[packed]      = charge;
      const auto& position = conformer.getAtomPos(atom);
      for (int axis = 0; axis < 3; ++axis) {
        if (!finiteValue(position[axis]))
          throw std::invalid_argument("Atom coordinates must be finite");
        positions[3 * packed + axis] = position[axis];
      }
    }
  }
  AsyncDeviceVector<double> result(molecules.size() * pointCount, stream);
  if (result.size() == 0)
    return result;
  AsyncDeviceVector<double> devicePositions(positions.size(), stream);
  AsyncDeviceVector<double> deviceCharges(charges.size(), stream);
  AsyncDeviceVector<size_t> deviceOffsets(offsets.size(), stream);
  AsyncDeviceVector<double> burial(options.dielectric ? charges.size() : 0, stream);
  if (positions.size() != 0) {
    devicePositions.copyFromHost(positions.data(), positions.size());
    deviceCharges.copyFromHost(charges.data(), charges.size());
  }
  deviceOffsets.copyFromHost(offsets);
  // Pinned host packing storage must outlive the asynchronous copies.
  cudaCheckError(cudaStreamSynchronize(stream));
  coulombFieldsGpu(toSpan(devicePositions),
                   toSpan(deviceCharges),
                   toSpan(deviceOffsets),
                   points,
                   toSpan(burial),
                   toSpan(result),
                   options,
                   molecules.size(),
                   stream);
  return result;
}

}  // namespace nvMolKit
