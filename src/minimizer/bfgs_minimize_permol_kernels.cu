#include "bfgs_minimize_permol_kernels.h"
#include "mmff_kernels.h"
#include "mmff_kernels_device.cuh"
#include "device_vector.h"

#include <cub/cub.cuh>

namespace nvMolKit {

namespace {
constexpr int BLOCK_SIZE = 128;
constexpr int MAX_LINESEARCH_ITERS = 1000;
constexpr double FUNCTOL = 1e-4;
constexpr double MOVETOL = 1e-7;
constexpr double TOLX = 4. * 3e-8;

__device__ void setMaxStep(const double* pos, const int numTerms, double* maxStepOut,
                           typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  double sumSquaredPos = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    double dx2 = pos[i] * pos[i];
    sumSquaredPos += dx2;
  }
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;

  const double squaredSum = BlockReduce(tempStorage).Sum(sumSquaredPos);
  if (threadIdx.x == 0) {
    constexpr double maxStepFactor = 100.0;
    *maxStepOut               = maxStepFactor * max(sqrt(squaredSum), static_cast<double>(numTerms));
  }
}

__device__ void lineSearchSetup(const int numTerms, const double* posStart, const double* gradStart, const double maxStep, double* dirStart, double& slope,  double& lambdaMin,
                                typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {

  const int idxInSys = threadIdx.x;
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
  __shared__ double dirSum[1];

  // ---------------------------------
  //  Scale direction vector if needed
  // ---------------------------------
  double sumSquaredLocal = 0.0;
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    double dx2 = dirStart[i] * dirStart[i];
    sumSquaredLocal += dx2;
  }
  double blockSum = BlockReduce(tempStorage).Sum(sumSquaredLocal);
  if (idxInSys == 0) {
    dirSum[0] = sqrt(blockSum);
  }
  __syncthreads();
  if (dirSum[0] > maxStep) {
    for (int i = idxInSys; i < numTerms; i += blockDim.x) {
      dirStart[i] *= maxStep / dirSum[0];
    }
  }
  __syncthreads();

  // -------------------------
  // Set slope, check validity
  // -------------------------
  double localSum = 0.0;
  double localGradSum = 0.0;
  double localDirSum = 0.0;
  // Each thread computes its partial sum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    localSum += dirStart[i] * gradStart[i];
    localGradSum += gradStart[i] * gradStart[i];
    localDirSum += dirStart[i] * dirStart[i];
  }

  // Perform block-wide reduction to compute the total sum
  blockSum = BlockReduce(tempStorage).Sum(localSum);
  __syncthreads();
  
  // The first thread in the block writes the result
  if (idxInSys == 0) {
    slope = blockSum;
  }
  __syncthreads();

  // ----------------------
  // Compute initial lambda
  // ----------------------
  double localMax = 0.0;
  // Each thread computes its local maximum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    double temp = fabs(dirStart[i]) / fmax(fabs(posStart[i]), 1.0);
    if (temp > localMax) {
      localMax = temp;
    }
  }
  // Perform block-wide reduction to find the maximum
  double blockMax = BlockReduce(tempStorage).Reduce(localMax, cub::Max());

  // The first thread in the block writes the result
  if (threadIdx.x == 0) {
    lambdaMin = MOVETOL / blockMax;
  }
}

__device__ void lineSearchPerturb(const int numTerms, 
                                  const double* refPos,
                                  const double* dirStart,
                                  const double lambda,
                                  double* scratchPos) {
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    scratchPos[i] = refPos[i] + lambda * dirStart[i];
  }
  __syncthreads();
}

__device__ bool lineSearchPostEnergy(const bool isFirstIter,
                                     const double prevE,
                                     const double newE,
                                     const double slope,
                                     const double lambda,
                                     const double lambdaMin,
                                     double& lambda2,
                                     double& eScratch,
                                     double& lambdaOut) {
  bool converged = false;
  
  if (threadIdx.x == 0) {
    double eDiff = newE - prevE;
    double threshold = FUNCTOL * lambda * slope;
    if (lambda < lambdaMin) {
      converged = true;
    } else if (eDiff <= threshold) {
      converged = true;
    } else {
      // Need to backtrack
      double tmpLambda;
      if (isFirstIter) {
        tmpLambda = -slope / (2.0 * (newE - prevE - slope));
      } else {
        double rhs1 = newE - prevE - lambda * slope;
        double rhs2 = eScratch - prevE - lambda2 * slope;
        double a = (rhs1 / (lambda * lambda) - rhs2 / (lambda2 * lambda2)) / (lambda - lambda2);
        double b = (-lambda2 * rhs1 / (lambda * lambda) + lambda * rhs2 / (lambda2 * lambda2)) / (lambda - lambda2);
        if (a == 0.0) {
          tmpLambda = -slope / (2.0 * b);
        } else {
          double disc = b * b - 3 * a * slope;
          if (disc < 0.0) {
            tmpLambda = 0.5 * lambda;
          } else if (b <= 0.0) {
            tmpLambda = (-b + sqrt(disc)) / (3.0 * a);
          } else {
            tmpLambda = -slope / (b + sqrt(disc));
          }
        }
        if (tmpLambda > 0.5 * lambda) {
          tmpLambda = 0.5 * lambda;
        }
      }
      lambda2 = lambda;
      eScratch = newE;
      lambdaOut = max(tmpLambda, 0.1 * lambda);
    }
  }
  __syncthreads();
  return converged;
}

__device__ void setDirection(const int numTerms,
                             const double* posFromLineSearch,
                             const double* pos,
                             double* xi,
                             double* dGrad,
                             const double* grad,
                             bool& converged,
                             typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  double localMax = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    xi[i] = posFromLineSearch[i] - pos[i];
    dGrad[i] = grad[i];
    
    double temp = fabs(xi[i]) / fmax(fabs(posFromLineSearch[i]), 1.0);
    if (temp > localMax) {
      localMax = temp;
    }
  }
  
  double blockMax = cub::BlockReduce<double, BLOCK_SIZE>(tempStorage).Reduce(localMax, cub::Max());
  
  if (threadIdx.x == 0 && blockMax < TOLX) {
    converged = true;
  }
  __syncthreads();
}

template <bool scaleGrads>
__device__ void scaleGrad(const int numTerms, double* grad, double& gradScale,
                          typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  gradScale = scaleGrads ? 0.1 : 1.0;
  
  double maxGrad = -1e8;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    grad[i] *= gradScale;
    if (grad[i] > maxGrad) {
      maxGrad = grad[i];
    }
  }
  
  double blockMax = cub::BlockReduce<double, BLOCK_SIZE>(tempStorage).Reduce(maxGrad, cub::Max());
  
  __shared__ double distributedMax[1];
  if (threadIdx.x == 0) {
    distributedMax[0] = blockMax;
  }
  __syncthreads();
  
  maxGrad = distributedMax[0];
  
  if (scaleGrads && maxGrad > 10.0) {
    while (maxGrad * gradScale > 10.0) {
      gradScale *= 0.5;
    }
    for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
      grad[i] *= gradScale;
    }
  }
  __syncthreads();
}

__device__ void updateDGrad(const int numTerms,
                           const double gradTol,
                           const double energy,
                           const double gradScale,
                           const double* grad,
                           const double* pos,
                           double* dGrad,
                           bool& converged,
                           typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  double localMax = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    dGrad[i] = grad[i] - dGrad[i];
    double temp = fabs(grad[i]) * fmax(fabs(pos[i]), 1.0);
    if (temp > localMax) {
      localMax = temp;
    }
  }
  
  double blockMax = cub::BlockReduce<double, BLOCK_SIZE>(tempStorage).Reduce(localMax, cub::Max());
  
  if (threadIdx.x == 0) {
    const double term = max(energy * gradScale, 1.0);
    blockMax /= term;
    if (blockMax < gradTol) {
      converged = true;
    }
  }
  __syncthreads();
}

__device__ void updateInverseHessian(const int numTerms,
                                     double* invHessian,
                                     double* dGrad,
                                     double* xi,
                                     double* hessDGrad,
                                     double* grad,
                                     typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
  
  // Compute hessDGrad = invHessian * dGrad
  for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
    double dotProduct = 0.0;
    for (int col = 0; col < numTerms; col++) {
      dotProduct += invHessian[row * numTerms + col] * dGrad[col];
    }
    hessDGrad[row] = dotProduct;
  }
  __syncthreads();
  
  // Compute BFGS sums
  __shared__ double fac, fae, fad, sumDGrad, sumXi;
  __shared__ bool needUpdate;
  
  double sumFac = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumFac += dGrad[i] * xi[i];
  }
  double facReduced = BlockReduce(tempStorage).Sum(sumFac);
  if (threadIdx.x == 0) fac = facReduced;
  __syncthreads();
  
  double sumFae = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumFae += dGrad[i] * hessDGrad[i];
  }
  double faeReduced = BlockReduce(tempStorage).Sum(sumFae);
  if (threadIdx.x == 0) fae = faeReduced;
  __syncthreads();
  
  double sumDGradSq = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumDGradSq += dGrad[i] * dGrad[i];
  }
  double sumDGradReduced = BlockReduce(tempStorage).Sum(sumDGradSq);
  if (threadIdx.x == 0) sumDGrad = sumDGradReduced;
  __syncthreads();
  
  double sumXiSq = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumXiSq += xi[i] * xi[i];
  }
  double sumXiReduced = BlockReduce(tempStorage).Sum(sumXiSq);
  if (threadIdx.x == 0) sumXi = sumXiReduced;
  __syncthreads();
  
  if (threadIdx.x == 0) {
    constexpr double EPS = 3e-8;
    needUpdate = fac > sqrt(EPS * sumDGrad * sumXi);
    
    if (needUpdate) {
      fac = 1.0 / fac;
      fad = 1.0 / fae;
    }
  }
  __syncthreads();
  
  if (needUpdate) {
    // Update dGrad for Hessian update
    for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
      dGrad[i] = fac * xi[i] - fad * hessDGrad[i];
    }
    __syncthreads();
    
    // Update inverse Hessian and compute new direction
    for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
      double pxi = fac * xi[row];
      double hdgi = fad * hessDGrad[row];
      double dgi = fae * dGrad[row];
      
      for (int col = 0; col < numTerms; col++) {
        double pxj = xi[col];
        double hdgj = hessDGrad[col];
        double dgj = dGrad[col];
        double update = pxi * pxj - hdgi * hdgj + dgi * dgj;
        invHessian[row * numTerms + col] += update;
      }
    }
    __syncthreads();
  }
  
  // Update xi = -invHessian * grad
  for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
    double dotProduct = 0.0;
    for (int col = 0; col < numTerms; col++) {
      dotProduct += invHessian[row * numTerms + col] * grad[col];
    }
    xi[row] = -dotProduct;
  }
  __syncthreads();
}


}  // namespace

__global__ void bfgsMinimizeKernel(const int numIters,
                                   const double gradTol,
                                   const bool scaleGrads,
                                   const MMFF::EnergyForceContribsDevicePtr* terms,
                                   const MMFF::BatchedIndicesDevicePtr* systemIndices,
                                   double* positions,
                                   double* energyOuts,
                                   double* invHessians,
                                   const int DIM) {
  const int molIdx = blockIdx.x;
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  
  const int atomStart = systemIndices->atomStarts[molIdx];
  const int atomEnd = systemIndices->atomStarts[molIdx + 1];
  const int numAtoms = atomEnd - atomStart;
  const int numTerms = DIM * numAtoms;
  
  // Shared memory for local molecule data (for small molecules)
  constexpr int maxAtomSize = 256;
  constexpr int maxTerms = maxAtomSize * 3;
  __shared__ double localPos[maxTerms];
  __shared__ double localGrad[maxTerms];
  __shared__ double localDir[maxTerms];
  __shared__ double scratchPos[maxTerms];
  __shared__ double dGrad[maxTerms];
  __shared__ double hessDGrad[maxTerms];
  __shared__ double oldPos[maxTerms];

  // Shared scalars
  __shared__ double maxStep;
  __shared__ double prevE;
  __shared__ double currE;
  __shared__ double slope;
  __shared__ double lambda;
  __shared__ double lambdaMin;
  __shared__ double lambda2;
  __shared__ double eScratch;
  __shared__ double gradScale;
  __shared__ bool converged;
  __shared__ bool lineSearchConverged;
  
  if (numTerms > maxTerms) {
    // Molecule too large for this kernel
    if (tid == 0) {
      energyOuts[molIdx] = -1.0;  // Error flag
    }
    return;
  }
  
  // Inverse Hessian in global memory (O(n^2), too large for shared)
  double* invHessian = invHessians + molIdx * maxTerms * maxTerms;
  
  // Initialize positions from global memory
  double* globalPos = positions + atomStart * DIM;
  for (int i = tid; i < numTerms; i += stride) {
    localPos[i] = globalPos[i];
  }
  __syncthreads();
  
  // Initialize inverse Hessian to identity
  const int hessianSize = numTerms * numTerms;
  for (int i = tid; i < hessianSize; i += stride) {
    const int row = i / numTerms;
    const int col = i % numTerms;
    invHessian[i] = (row == col) ? 1.0 : 0.0;
  }
  
  if (tid == 0) {
    converged = false;
  }
  __syncthreads();
  
  // Shared temp storage for all BlockReduce operations
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  
  // Compute initial energy
  const double threadEnergy = MMFF::molEnergy(*terms, *systemIndices, positions, molIdx, tid, stride);
  const double blockEnergy = BlockReduce(tempStorage).Sum(threadEnergy);
  
  if (tid == 0) {
    prevE = blockEnergy;
    energyOuts[molIdx] = blockEnergy;
  }
  __syncthreads();
  
  // Compute initial gradient  
  for (int i = tid; i < numTerms; i += stride) {
    localGrad[i] = 0.0;
  }
  __syncthreads();
  
  MMFF::molGrad(*terms, *systemIndices, positions, localGrad, molIdx, tid, stride);
  __syncthreads();
  
  // Scale gradients
  if (scaleGrads) {
    scaleGrad<true>(numTerms, localGrad, gradScale, tempStorage);
  } else {
    scaleGrad<false>(numTerms, localGrad, gradScale, tempStorage);
  }
  
  // Set initial direction as negative gradient
  for (int i = tid; i < numTerms; i += stride) {
    localDir[i] = -localGrad[i];
  }
  __syncthreads();
  
  // Set max step
  setMaxStep(localPos, numTerms, &maxStep, tempStorage);
  __syncthreads();
  
  // Main BFGS loop
  __shared__ int currIter;
  if (tid == 0) {
    currIter = 0;
  }
  __syncthreads();
  
  while (!converged && currIter < numIters) {
    // Save current position before line search
    for (int i = tid; i < numTerms; i += stride) {
      oldPos[i] = localPos[i];
    }
    __syncthreads();
    
    // Line search setup
    if (tid == 0) {
      lineSearchConverged = false;
      lambda = 1.0;
    }
    __syncthreads();
    
    lineSearchSetup(numTerms, localPos, localGrad, maxStep, localDir, slope, lambdaMin, tempStorage);
    __syncthreads();
    
    // Line search loop
    __shared__ int lineSearchIter;
    if (tid == 0) {
      lineSearchIter = 0;
    }
    __syncthreads();
    
    while (!lineSearchConverged && lineSearchIter < MAX_LINESEARCH_ITERS) {
      // Perturb positions
      lineSearchPerturb(numTerms, localPos, localDir, lambda, scratchPos);
      
      // Copy to global for energy calculation
      for (int i = tid; i < numTerms; i += stride) {
        globalPos[i] = scratchPos[i];
      }
      __syncthreads();
      
      // Compute energy at perturbed position
      const double lsThreadEnergy = MMFF::molEnergy(*terms, *systemIndices, positions, molIdx, tid, stride);
      const double lsBlockEnergy = BlockReduce(tempStorage).Sum(lsThreadEnergy);
      
      if (tid == 0) {
        currE = lsBlockEnergy;
      }
      __syncthreads();
      
      // Check convergence and update lambda
      lineSearchConverged = lineSearchPostEnergy(lineSearchIter == 0, prevE, currE, slope, lambda, lambdaMin, lambda2, eScratch, lambda);
      __syncthreads();
      
      if (tid == 0) {
        lineSearchIter++;
      }
      __syncthreads();
    }
    
    // Update positions with final line search result and compute direction
    for (int i = tid; i < numTerms; i += stride) {
      localPos[i] = scratchPos[i];
      globalPos[i] = scratchPos[i];
    }
    __syncthreads();
    
    // Set direction (compute xi = new - old)
    setDirection(numTerms, scratchPos, oldPos, localDir, dGrad, localGrad, converged, tempStorage);
    if (converged) break;
    
    // Update stored energy for next iteration
    if (tid == 0) {
      prevE = currE;
    }
    __syncthreads();
    
    // Compute gradients at new position
    for (int i = tid; i < numTerms; i += stride) {
      localGrad[i] = 0.0;
    }
    __syncthreads();
    
    MMFF::molGrad(*terms, *systemIndices, positions, localGrad, molIdx, tid, stride);
    __syncthreads();
    
    // Scale gradients
    if (scaleGrads) {
      scaleGrad<true>(numTerms, localGrad, gradScale, tempStorage);
    } else {
      scaleGrad<false>(numTerms, localGrad, gradScale, tempStorage);
    }
    
    // Update dGrad and check convergence
    updateDGrad(numTerms, gradTol, currE, gradScale, localGrad, localPos, dGrad, converged, tempStorage);
    if (converged) break;
    
    // Update Hessian and compute new direction
    updateInverseHessian(numTerms, invHessian, dGrad, localDir, hessDGrad, localGrad, tempStorage);
    
    if (tid == 0) {
      currIter++;
    }
    __syncthreads();
  }
  
  // Write final energy
  if (tid == 0) {
    energyOuts[molIdx] = prevE;
  }
}

cudaError_t launchBfgsMinimizePerMolKernel(int numMols,
                                           int numIters,
                                           double gradTol,
                                           bool scaleGrads,
                                           const MMFF::EnergyForceContribsDevicePtr& terms,
                                           const MMFF::BatchedIndicesDevicePtr& systemIndices,
                                           double* positions,
                                           double* energyOuts,
                                           int dataDim,
                                           cudaStream_t stream) {
  constexpr int maxAtoms = 256;
  constexpr int maxTerms = maxAtoms * 3;
  
  // Allocate global memory for inverse Hessians (one per molecule, size maxTerms x maxTerms)
  const size_t hessianSize = static_cast<size_t>(numMols) * maxTerms * maxTerms * sizeof(double);
  double* invHessians = nullptr;
  cudaError_t err = cudaMallocAsync(&invHessians, hessianSize, stream);
  if (err != cudaSuccess) {
    return err;
  }
  
  const AsyncDevicePtr<MMFF::EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<MMFF::BatchedIndicesDevicePtr> devSysIdx(systemIndices, stream);
  
  bfgsMinimizeKernel<<<numMols, BLOCK_SIZE, 0, stream>>>(
    numIters,
    gradTol,
    scaleGrads,
    devTerms.data(),
    devSysIdx.data(),
    positions,
    energyOuts,
    invHessians,
    dataDim);
  
  err = cudaGetLastError();
  
  // Free the inverse Hessian memory
  cudaFreeAsync(invHessians, stream);
  
  return err;
}

}  // namespace nvMolKit