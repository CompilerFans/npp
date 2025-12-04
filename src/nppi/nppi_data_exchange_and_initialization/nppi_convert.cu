#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void convert_8u32f_C1R_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                         int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *srcRow = (const Npp8u *)(((const char *)pSrc) + y * nSrcStep);
  Npp32f *dstRow = (Npp32f *)(((char *)pDst) + y * nDstStep);

  // Convert 8-bit unsigned to 32-bit float (0-255 -> 0.0-255.0)
  dstRow[x] = (Npp32f)srcRow[x];
}

__global__ void convert_8u32f_C3R_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                         int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *srcRow = (const Npp8u *)(((const char *)pSrc) + y * nSrcStep);
  Npp32f *dstRow = (Npp32f *)(((char *)pDst) + y * nDstStep);

  // Convert all three channels: 8-bit unsigned to 32-bit float (0-255 -> 0.0-255.0)
  dstRow[x * 3 + 0] = (Npp32f)srcRow[x * 3 + 0]; // Channel 0 (R/B)
  dstRow[x * 3 + 1] = (Npp32f)srcRow[x * 3 + 1]; // Channel 1 (G)
  dstRow[x * 3 + 2] = (Npp32f)srcRow[x * 3 + 2]; // Channel 2 (B/R)
}

__global__ void convert_8u16u_C1R_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp16u *__restrict__ pDst,
                                         int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *srcRow = (const Npp8u *)(((const char *)pSrc) + y * nSrcStep);
  Npp16u *dstRow = (Npp16u *)(((char *)pDst) + y * nDstStep);

  dstRow[x] = (Npp16u)srcRow[x];
}

__global__ void convert_32f8u_C1R_kernel(const Npp32f *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                         int nDstStep, int width, int height, NppRoundMode eRoundMode) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32f *srcRow = (const Npp32f *)(((const char *)pSrc) + y * nSrcStep);
  Npp8u *dstRow = (Npp8u *)(((char *)pDst) + y * nDstStep);

  Npp32f val = srcRow[x];
  Npp32f rounded;
  if (eRoundMode == NPP_RND_NEAR) {
    rounded = roundf(val);
  } else if (eRoundMode == NPP_RND_FINANCIAL) {
    Npp32f floor_val = floorf(val);
    Npp32f diff = val - floor_val;
    if (diff > 0.5f) {
      rounded = floor_val + 1.0f;
    } else if (diff < 0.5f) {
      rounded = floor_val;
    } else {
      int int_floor = (int)floor_val;
      rounded = (int_floor % 2 == 0) ? floor_val : floor_val + 1.0f;
    }
  } else {
    rounded = val;
  }

  if (rounded < 0.0f)
    rounded = 0.0f;
  if (rounded > 255.0f)
    rounded = 255.0f;

  dstRow[x] = (Npp8u)rounded;
}

__global__ void convert_16u32f_C1R_kernel(const Npp16u *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                          int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp16u *srcRow = (const Npp16u *)(((const char *)pSrc) + y * nSrcStep);
  Npp32f *dstRow = (Npp32f *)(((char *)pDst) + y * nDstStep);

  dstRow[x] = (Npp32f)srcRow[x];
}

extern "C" {
NppStatus nppiConvert_8u32f_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {

  // Setup kernel launch parameters
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  convert_8u32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                             oSizeROI.width, oSizeROI.height);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiConvert_8u32f_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {

  // Parameter validation
  if (!pSrc || !pDst)
    return NPP_NULL_POINTER_ERROR;
  if (nDstStep <= 0)
    return NPP_STEP_ERROR;
  if (nSrcStep <= 0)
    return NPP_NO_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0)
    return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0)
    return NPP_NO_ERROR;

  // Setup kernel launch parameters
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  convert_8u32f_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                             oSizeROI.width, oSizeROI.height);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiConvert_8u16u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  convert_8u16u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                             oSizeROI.width, oSizeROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiConvert_32f8u_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppRoundMode eRoundMode, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  convert_32f8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, eRoundMode);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiConvert_16u32f_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  convert_16u32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                              oSizeROI.width, oSizeROI.height);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}
}
