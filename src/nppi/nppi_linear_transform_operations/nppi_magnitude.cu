#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void magnitude_32fc32f_kernel(const Npp32fc *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                         int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32fc *srcPtr = (const Npp32fc *)((const char *)pSrc + y * nSrcStep);
    Npp32f *dstPtr = (Npp32f *)((char *)pDst + y * nDstStep);

    Npp32fc srcValue = srcPtr[x];
    float real = srcValue.re;
    float imag = srcValue.im;

    // Compute magnitude: sqrt(real^2 + imag^2)
    dstPtr[x] = sqrtf(real * real + imag * imag);
  }
}
__global__ void magnitude_sqr_32fc32f_kernel(const Npp32fc *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                             int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32fc *srcPtr = (const Npp32fc *)((const char *)pSrc + y * nSrcStep);
    Npp32f *dstPtr = (Npp32f *)((char *)pDst + y * nDstStep);

    Npp32fc srcValue = srcPtr[x];
    float real = srcValue.re;
    float imag = srcValue.im;

    // Compute squared magnitude: real^2 + imag^2
    dstPtr[x] = real * real + imag * imag;
  }
}

extern "C" {

// Magnitude function implementations
NppStatus nppiMagnitude_32fc32f_C1R_Ctx_impl(const Npp32fc *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                             NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  magnitude_32fc32f_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                             oSizeROI.width, oSizeROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiMagnitudeSqr_32fc32f_C1R_Ctx_impl(const Npp32fc *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  magnitude_sqr_32fc32f_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                 oSizeROI.width, oSizeROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}
}
