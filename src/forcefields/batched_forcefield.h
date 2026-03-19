#ifndef NVMOLKIT_BATCHED_FORCEFIELD_H
#define NVMOLKIT_BATCHED_FORCEFIELD_H

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "minimizer/bfgs_types.h"

namespace nvMolKit {

class BatchedForcefield {
 public:
  virtual ~BatchedForcefield() = default;

  virtual cudaError_t computeEnergy(double*        energyOuts,
                                    const double*  positions,
                                    const uint8_t* activeSystemMask = nullptr,
                                    cudaStream_t   stream           = nullptr) = 0;

  virtual cudaError_t computeGradients(double*        grad,
                                       const double*  positions,
                                       const uint8_t* activeSystemMask = nullptr,
                                       cudaStream_t   stream           = nullptr) = 0;

  int                      numMolecules() const { return numMolecules_; }
  int                      dataDim() const { return dataDim_; }
  int                      totalPositions() const { return totalPositions_; }
  const std::vector<int>&  atomStartsHost() const { return atomStartsHost_; }
  const int*               atomStartsDevice() const { return atomStartsDevice_; }
  ForceFieldType           type() const { return type_; }

 protected:
  BatchedForcefield(ForceFieldType        type,
                    int                   dataDim,
                    std::vector<int>      atomStartsHost,
                    const int*            atomStartsDevice)
      : numMolecules_(static_cast<int>(atomStartsHost.size()) - 1),
        dataDim_(dataDim),
        totalPositions_(atomStartsHost.empty() ? 0 : atomStartsHost.back() * dataDim),
        atomStartsHost_(std::move(atomStartsHost)),
        atomStartsDevice_(atomStartsDevice),
        type_(type) {}

  void setAtomStartsDevice(const int* atomStartsDevice) { atomStartsDevice_ = atomStartsDevice; }

 private:
  int              numMolecules_   = 0;
  int              dataDim_        = 0;
  int              totalPositions_ = 0;
  std::vector<int> atomStartsHost_;
  const int*       atomStartsDevice_ = nullptr;
  ForceFieldType   type_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_BATCHED_FORCEFIELD_H
