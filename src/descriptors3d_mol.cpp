// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "src/descriptors3d_mol.h"

#include <GraphMol/Conformer.h>
#include <GraphMol/Descriptors/MolData3Ddescriptors.h>
#include <GraphMol/ROMol.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/conformer/conformer_coord_upload.h"

namespace nvMolKit {

namespace {

struct DeviceDescriptorInputs {
  AsyncDeviceVector<double>  momentWeights;
  AsyncDeviceVector<double>  whimWeights;
  AsyncDeviceVector<int8_t>  conformerIs3D;
  AsyncDeviceVector<int32_t> moleculeAtomStarts;
};

DeviceDescriptorInputs uploadDescriptorInputs(const std::vector<const RDKit::ROMol*>& mols,
                                              const bool                              includeMomentWeights,
                                              const bool                              includeWhimWeights,
                                              const bool                              includeConformerFlags,
                                              cudaStream_t                            stream) {
  const int            numMols = static_cast<int>(mols.size());
  std::vector<int32_t> atomStarts(numMols + 1, 0);
  int64_t              totalAtoms = 0;
  for (int molIdx = 0; molIdx < numMols; ++molIdx) {
    atomStarts[molIdx] = static_cast<int32_t>(totalAtoms);
    totalAtoms += mols[molIdx]->getNumAtoms();
    if (totalAtoms > std::numeric_limits<int32_t>::max()) {
      throw std::overflow_error("Total molecule atom count exceeds int32 range");
    }
  }
  atomStarts[numMols] = static_cast<int32_t>(totalAtoms);

  std::vector<double> weights(includeMomentWeights ? static_cast<size_t>(totalAtoms) : 0);
  std::vector<double> whimWeights(includeWhimWeights ? static_cast<size_t>(totalAtoms) * 6 : 0);
  std::vector<int8_t> conformerIs3D;
  if (includeConformerFlags) {
    for (const RDKit::ROMol* mol : mols) {
      for (auto conformer = mol->beginConformers(); conformer != mol->endConformers(); ++conformer) {
        conformerIs3D.push_back((*conformer)->is3D());
      }
    }
  }
  if (includeMomentWeights) {
#pragma omp parallel for schedule(dynamic)
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      size_t atomOffset = static_cast<size_t>(atomStarts[molIdx]);
      for (const auto* atom : mols[molIdx]->atoms()) {
        weights[atomOffset++] = atom->getMass();
      }
    }
  }
  if (includeWhimWeights) {
#pragma omp parallel for schedule(dynamic)
    for (int molIdx = 0; molIdx < numMols; ++molIdx) {
      MolData3Ddescriptors                     descriptorData;
      const std::array<std::vector<double>, 6> moleculeWeights = {
        descriptorData.GetRelativeMW(*mols[molIdx]),
        descriptorData.GetRelativeVdW(*mols[molIdx]),
        descriptorData.GetRelativeENeg(*mols[molIdx]),
        descriptorData.GetRelativePol(*mols[molIdx]),
        descriptorData.GetRelativeIonPol(*mols[molIdx]),
        descriptorData.GetIState(*mols[molIdx]),
      };
      const size_t atomStart = static_cast<size_t>(atomStarts[molIdx]);
      for (size_t channel = 0; channel < moleculeWeights.size(); ++channel) {
        std::copy(moleculeWeights[channel].begin(),
                  moleculeWeights[channel].end(),
                  whimWeights.begin() + static_cast<size_t>(totalAtoms) * channel + atomStart);
      }
    }
  }

  DeviceDescriptorInputs result{AsyncDeviceVector<double>(weights.size(), stream),
                                AsyncDeviceVector<double>(whimWeights.size(), stream),
                                AsyncDeviceVector<int8_t>(conformerIs3D.size(), stream),
                                AsyncDeviceVector<int32_t>(atomStarts.size(), stream)};
  if (!weights.empty()) {
    result.momentWeights.copyFromHost(weights);
  }
  if (!whimWeights.empty()) {
    result.whimWeights.copyFromHost(whimWeights);
  }
  if (!conformerIs3D.empty()) {
    result.conformerIs3D.copyFromHost(conformerIs3D);
  }
  result.moleculeAtomStarts.copyFromHost(atomStarts);
  return result;
}

}  // namespace

template <typename Real>
Property3DBatchResult<Real> calc3DProperties(const std::vector<const RDKit::ROMol*>& mols,
                                             const std::vector<Property3D>&          properties,
                                             const bool                              useAtomicMasses,
                                             cudaStream_t                            stream,
                                             const DeviceCoordView*                  coordinates,
                                             const double                            whimThreshold) {
  for (size_t molIdx = 0; molIdx < mols.size(); ++molIdx) {
    if (mols[molIdx] == nullptr) {
      throw std::invalid_argument("Null molecule at index " + std::to_string(molIdx));
    }
  }
  if (coordinates != nullptr && coordinates->nMols != static_cast<int>(mols.size())) {
    throw std::invalid_argument("Device coordinates describe " + std::to_string(coordinates->nMols) +
                                " molecules, but " + std::to_string(mols.size()) + " molecules were provided");
  }

  // Uploaded buffers outlive the kernel launch below; their stream-ordered frees run after it.
  DeviceCoordResult uploaded;
  DeviceCoordView   view;
  if (coordinates != nullptr) {
    view = *coordinates;
  } else {
    uploaded = uploadConformerCoordinates(mols, stream);
    view     = makeDeviceCoordView(uploaded);
  }
  const bool includeWhimWeights = std::find(properties.begin(), properties.end(), Property3D::WHIM) != properties.end();
  bool       includeMomentWeights = false;
  for (const Property3D property : properties) {
    const int propertyValue = static_cast<int>(property);
    includeMomentWeights |= useAtomicMasses && property != Property3D::SpherocityIndex && propertyValue >= 0 &&
                            propertyValue < static_cast<int>(Property3D::PBF);
  }
  const bool includeConformerFlags =
    coordinates == nullptr && std::find(properties.begin(), properties.end(), Property3D::PBF) != properties.end();
  const DeviceDescriptorInputs inputs =
    uploadDescriptorInputs(mols, includeMomentWeights, includeWhimWeights, includeConformerFlags, stream);

  Property3DBatchResult<Real> result;
  const double*               atomWeights = inputs.momentWeights.data();
  result.properties                       = calc3DPropertiesGpu<Real>(view,
                                                atomWeights,
                                                inputs.whimWeights.data(),
                                                inputs.conformerIs3D.data(),
                                                inputs.moleculeAtomStarts.data(),
                                                properties,
                                                whimThreshold,
                                                stream);
  result.molIndices                       = std::move(uploaded.molIndices);
  result.confIndices                      = std::move(uploaded.confIndices);
  return result;
}

template Property3DBatchResult<float>  calc3DProperties<float>(const std::vector<const RDKit::ROMol*>&,
                                                              const std::vector<Property3D>&,
                                                              bool,
                                                              cudaStream_t,
                                                              const DeviceCoordView*,
                                                              double);
template Property3DBatchResult<double> calc3DProperties<double>(const std::vector<const RDKit::ROMol*>&,
                                                                const std::vector<Property3D>&,
                                                                bool,
                                                                cudaStream_t,
                                                                const DeviceCoordView*,
                                                                double);

}  // namespace nvMolKit
