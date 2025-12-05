#include "npp.h"
#include <cuda_runtime.h>

// ==============================================================================
// GPU Kernels for Set Operations
// ==============================================================================

__global__ void nppsSet_8u_kernel_impl(Npp8u nValue, Npp8u *pDst, size_t nLength) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nLength) {
    pDst[idx] = nValue;
  }
}

__global__ void nppsSet_32f_kernel_impl(Npp32f nValue, Npp32f *pDst, size_t nLength) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nLength) {
    pDst[idx] = nValue;
  }
}

__global__ void nppsSet_32fc_kernel_impl(Npp32fc nValue, Npp32fc *pDst, size_t nLength) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nLength) {
    pDst[idx].re = nValue.re;
    pDst[idx].im = nValue.im;
  }
}

// ==============================================================================
// Kernel Launch Functions
// ==============================================================================

extern "C" {

cudaError_t nppsSet_8u_kernel(Npp8u nValue, Npp8u *pDst, size_t nLength, cudaStream_t stream) {
  const int blockSize = 256;
  const size_t gridSize = (nLength + blockSize - 1) / blockSize;

  nppsSet_8u_kernel_impl<<<gridSize, blockSize, 0, stream>>>(nValue, pDst, nLength);

  return cudaGetLastError();
}

cudaError_t nppsSet_32f_kernel(Npp32f nValue, Npp32f *pDst, size_t nLength, cudaStream_t stream) {
  const int blockSize = 256;
  const size_t gridSize = (nLength + blockSize - 1) / blockSize;

  nppsSet_32f_kernel_impl<<<gridSize, blockSize, 0, stream>>>(nValue, pDst, nLength);

  return cudaGetLastError();
}

cudaError_t nppsSet_32fc_kernel(Npp32fc nValue, Npp32fc *pDst, size_t nLength, cudaStream_t stream) {
  const int blockSize = 256;
  const size_t gridSize = (nLength + blockSize - 1) / blockSize;

  nppsSet_32fc_kernel_impl<<<gridSize, blockSize, 0, stream>>>(nValue, pDst, nLength);

  return cudaGetLastError();
}
}
