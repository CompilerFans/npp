#include "npp.h"
#include <cuda_runtime.h>



// ==============================================================================
// GPU Kernels for Add Operations
// ==============================================================================


__global__ void nppsAdd_32f_kernel_impl(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst, size_t nLength) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nLength) {
    pDst[idx] = pSrc1[idx] + pSrc2[idx];
  }
}


__global__ void nppsAdd_16s_kernel_impl(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst, size_t nLength) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nLength) {
    int result = static_cast<int>(pSrc1[idx]) + static_cast<int>(pSrc2[idx]);
    // Saturate to 16-bit signed range
    if (result > 32767)
      result = 32767;
    else if (result < -32768)
      result = -32768;
    pDst[idx] = static_cast<Npp16s>(result);
  }
}


__global__ void nppsAdd_32fc_kernel_impl(const Npp32fc *pSrc1, const Npp32fc *pSrc2, Npp32fc *pDst, size_t nLength) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nLength) {
    pDst[idx].re = pSrc1[idx].re + pSrc2[idx].re;
    pDst[idx].im = pSrc1[idx].im + pSrc2[idx].im;
  }
}

// ==============================================================================
// Kernel Launch Functions
// ==============================================================================

extern "C" {

cudaError_t nppsAdd_32f_kernel(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst, size_t nLength,
                               cudaStream_t stream) {
  const int blockSize = 256;
  const int gridSize = (nLength + blockSize - 1) / blockSize;

  nppsAdd_32f_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc1, pSrc2, pDst, nLength);

  return cudaGetLastError();
}

cudaError_t nppsAdd_16s_kernel(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst, size_t nLength,
                               cudaStream_t stream) {
  const int blockSize = 256;
  const int gridSize = (nLength + blockSize - 1) / blockSize;

  nppsAdd_16s_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc1, pSrc2, pDst, nLength);

  return cudaGetLastError();
}

cudaError_t nppsAdd_32fc_kernel(const Npp32fc *pSrc1, const Npp32fc *pSrc2, Npp32fc *pDst, size_t nLength,
                                cudaStream_t stream) {
  const int blockSize = 256;
  const int gridSize = (nLength + blockSize - 1) / blockSize;

  nppsAdd_32fc_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc1, pSrc2, pDst, nLength);

  return cudaGetLastError();
}

} // extern "C"
