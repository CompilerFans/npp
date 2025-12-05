#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void sumWindowRowKernel_8u32f(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                         int height, int nMaskSize, int nAnchor) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *pSrcRow = (const Npp8u *)((char *)pSrc + y * nSrcStep);
  Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);

  // window start
  int start = x - nAnchor;
  // window end
  int end = start + nMaskSize;

  Npp32f sum = 0.0f;

  for (int k = start; k < end; k++) {
    if (k >= 0 && k < width) {
      sum += static_cast<Npp32f>(pSrcRow[k]);
    }
  }

  pDstRow[x] = sum;
}

__global__ void sumWindowColumnKernel_8u32f(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                            int height, int nMaskSize, int nAnchor) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);

  int start = y - nAnchor;
  int end = start + nMaskSize;

  Npp32f sum = 0.0f;

  for (int k = start; k < end; k++) {
    if (k >= 0 && k < height) {
      const Npp8u *pSrcRow = (const Npp8u *)((char *)pSrc + k * nSrcStep);
      sum += static_cast<Npp32f>(pSrcRow[x]);
    }
  }

  pDstRow[x] = sum;
}

extern "C" {

NppStatus sumWindowRow_8u32f_Impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oROI,
                                  Npp32s nMaskSize, Npp32s nAnchor, cudaStream_t hStream) {

  dim3 blockSize(16, 16);
  dim3 gridSize((oROI.width + blockSize.x - 1) / blockSize.x, (oROI.height + blockSize.y - 1) / blockSize.y);

  sumWindowRowKernel_8u32f<<<gridSize, blockSize, 0, hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height,
                                                                nMaskSize, nAnchor);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus sumWindowColumn_8u32f_Impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oROI,
                                     Npp32s nMaskSize, Npp32s nAnchor, cudaStream_t hStream) {

  dim3 blockSize(16, 16);
  dim3 gridSize((oROI.width + blockSize.x - 1) / blockSize.x, (oROI.height + blockSize.y - 1) / blockSize.y);

  sumWindowColumnKernel_8u32f<<<gridSize, blockSize, 0, hStream>>>(pSrc, nSrcStep, pDst, nDstStep, oROI.width,
                                                                   oROI.height, nMaskSize, nAnchor);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}