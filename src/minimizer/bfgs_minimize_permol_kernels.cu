#include "bfgs_minimize_permol_kernels.h"
#include "mmff_kernels.h"

#include <cub/cub.cuh>

namespace {
constexpr int MAX_LINESEARCH_ITERS = 1000;
__device__ void initialize();

__device__ void setMaxStep(const double* pos, const int numTerms, double* maxStepOut) {
  double sumSquaredPos = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    double dx2 = pos[i] * pos[i];
    sumSquaredPos += dx2;
  }
  using BlockReduce = cub::BlockReduce<double, 128>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  const double squaredSum = BlockReduce(tempStorage).Sum(sumSquaredPos);
  if (threadIdx.x == 0) {
    constexpr double maxStepFactor = 100.0;
    *maxStepOut               = maxStepFactor * max(sqrt(squaredSum), static_cast<double>(numTerms));
  }
}

__device__ void lineSearchSetup(const int numTerms, const double* posStart, const double* gradStart, const double maxStep, double* dirStart, double& slope,  double& lambdaMin) {

  const int idxInSys = threadIdx.x;
  using BlockReduce = cub::BlockReduce<double, 128>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ double                            dirSum[1];

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

  // -------------------------
  // Set slope, check validity
  // -------------------------
  double localSum = 0.0;
  // Each thread computes its partial sum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    localSum += dirStart[i] * gradStart[i];
  }

  // Perform block-wide reduction to compute the total sum
  blockSum = BlockReduce(tempStorage).Sum(localSum);
  __syncthreads();
  // The first thread in the block writes the result

  if (idxInSys == 0) {
    slope = blockSum;
  }

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
    constexpr double MOVETOL = 1e-7;  //!< Default tolerance for x changes in the minimizer
    lambdaMin = MOVETOL / blockMax;
  }
}
}

__device__ void lineSearchPerturb();

__device__ void energy();

__device__ bool lineSearchPostEnergy();

__device__ void setDirection();

__device__ void grad();

__device__ void scaleGrad();

__device__ void updateDGrad();

__device__ void updateHessian();


__global__ void bfgsMinimize(const int numIters, const double* energies, double* positions, double* grad, double* dir, const int* atomStarts, const int DIM) {
  const int sysIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int numTerms = DIM * (atomStarts[sysIdx + 1] - atomStarts[sysIdx]);
  const int startIdx = atomStarts[sysIdx] * DIM;
  // TODO: Move what we can to shared mem.
  double* localPos = &positions[startIdx];
  double* localGrad = &grad[startIdx];
  double* localDir = &dir[startIdx];

  initialize();
  bool converged = false;

  __shared__ double lambda;
  __shared__ double lambdaScratch;
  __shared__ double currE;
  __shared__ double slope;
  __shared__ double prevE;
  __shared__ double maxStep;

  setMaxStep(localPos, numTerms, &maxStep);
  if (threadIdx.x == 0) {
    prevE = energies[sysIdx];
  }
  __syncthreads();

  int currIter = 0;
  while (!converged && currIter < numIters) {
    // Line search setup
    int lineSearchIter = 0;
    bool lineSearchConverged = false;
    double lambda = 1.0;
    double lambdaMin = 0.0;
    currE = prevE;
    lineSearchSetup(numTerms, localPos, localGrad,maxStep, localDir,  slope, lambdaMin);

    while (!lineSearchConverged && lineSearchIter < MAX_LINESEARCH_ITERS)  {

      lineSearchPerturb();

      energy();

      lineSearchPostEnergy();
    }

    setDirection();

    grad();

    scaleGrad();

    updateDGrad();

    updateHessian();
    currIter++;
  }


}