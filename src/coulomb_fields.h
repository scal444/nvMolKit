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

#ifndef NVMOLKIT_COULOMB_FIELDS_H
#define NVMOLKIT_COULOMB_FIELDS_H

#include <string>
#include <vector>

#include "src/utils/device_vector.h"

namespace RDKit {
class ROMol;
}

namespace nvMolKit {

/** @brief RDKit Coulomb field options; distances are Angstrom, charges are e. */
struct CoulombFieldOptions {
  bool   dielectric    = false;
  bool   absoluteValue = false;
  double probeCharge   = 1.0;
  double alpha         = 0.0;
  double cutoff        = 1.0;
  double threshold     = -1.0;
  double epsilon       = 80.0;
  double xi            = 4.0;
};

/** @brief Evaluate vacuum or distance-dependent dielectric Coulomb energies.
 * Output is molecule-major, with one kJ/mol value per shared query point.
 * Points contain contiguous xyz triplets on device. Each molecule supplies
 * one selected conformer and finite atom charges in chargeProperty.
 * -1 selects the first conformer; threshold -1 includes all distances.
 * Empty molecules with a conformer contribute zero. Host buffers finish
 * uploading before return; field evaluation remains asynchronous on stream.
 */
AsyncDeviceVector<double> coulombFieldsMol(const std::vector<const RDKit::ROMol*>& molecules,
                                           cuda::std::span<const double>           points,
                                           const std::vector<int>&                 conformerIds,
                                           const std::string&                      chargeProperty,
                                           const CoulombFieldOptions&              options,
                                           cudaStream_t                            stream);

/** @brief Evaluate validated packed atoms and queries on stream.
 * Offsets contain moleculeCount+1 atom offsets. Dielectric burial scratch
 * has one entry per atom. Output has moleculeCount*pointCount entries.
 */
void coulombFieldsGpu(cuda::std::span<const double> positions,
                      cuda::std::span<const double> charges,
                      cuda::std::span<const size_t> offsets,
                      cuda::std::span<const double> points,
                      cuda::std::span<double>       burial,
                      cuda::std::span<double>       output,
                      const CoulombFieldOptions&    options,
                      int                           moleculeCount,
                      cudaStream_t                  stream);

}  // namespace nvMolKit
#endif
