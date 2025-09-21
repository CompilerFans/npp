#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * kernels for MPP Image Multiply Constant operations
 */

/**
 * kernel for multiplying 8-bit unsigned 1-channel image by constant
 */
__global__ void mulC_8u_C1RSfs_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u nConstant,
                                      Npp8u *__restrict__ pDst, int nDstStep, int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Calculate memory addresses
  const Npp8u *srcPixel = pSrc + y * nSrcStep + x;
  Npp8u *dstPixel = pDst + y * nDstStep + x;

  // Load source pixel
  Npp8u srcValue = *srcPixel;

  // Multiply by constant and scale
  int result = static_cast<int>(srcValue) * static_cast<int>(nConstant);
  result = result >> nScaleFactor;

  // Clamp to 8-bit range
  result = max(0, min(255, result));

  // Store result
  *dstPixel = static_cast<Npp8u>(result);
}

/**
 * kernel for multiplying 16-bit unsigned 1-channel image by constant
 */
__global__ void mulC_16u_C1RSfs_kernel(const Npp16u *__restrict__ pSrc, int nSrcStep, Npp16u nConstant,
                                       Npp16u *__restrict__ pDst, int nDstStep, int width, int height,
                                       int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Calculate memory addresses (16-bit addressing)
  const Npp16u *srcPixel = (const Npp16u *)((const char *)pSrc + y * nSrcStep) + x;
  Npp16u *dstPixel = (Npp16u *)((char *)pDst + y * nDstStep) + x;

  // Load source pixel
  Npp16u srcValue = *srcPixel;

  // Multiply by constant and scale
  int result = static_cast<int>(srcValue) * static_cast<int>(nConstant);
  result = result >> nScaleFactor;

  // Clamp to 16-bit unsigned range
  result = max(0, min(65535, result));

  // Store result
  *dstPixel = static_cast<Npp16u>(result);
}

/**
 * kernel for multiplying 16-bit signed 1-channel image by constant
 */
__global__ void mulC_16s_C1RSfs_kernel(const Npp16s *__restrict__ pSrc, int nSrcStep, Npp16s nConstant,
                                       Npp16s *__restrict__ pDst, int nDstStep, int width, int height,
                                       int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Calculate memory addresses (16-bit addressing)
  const Npp16s *srcPixel = (const Npp16s *)((const char *)pSrc + y * nSrcStep) + x;
  Npp16s *dstPixel = (Npp16s *)((char *)pDst + y * nDstStep) + x;

  // Load source pixel
  Npp16s srcValue = *srcPixel;

  // Multiply by constant and scale
  int result = static_cast<int>(srcValue) * static_cast<int>(nConstant);
  result = result >> nScaleFactor;

  // Clamp to 16-bit signed range
  result = max(-32768, min(32767, result));

  // Store result
  *dstPixel = static_cast<Npp16s>(result);
}

/**
 * kernel for multiplying 32-bit float 1-channel image by constant
 */
__global__ void mulC_32f_C1R_kernel(const Npp32f *__restrict__ pSrc, int nSrcStep, Npp32f nConstant,
                                    Npp32f *__restrict__ pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  // Calculate memory addresses (32-bit addressing)
  const Npp32f *srcPixel = (const Npp32f *)((const char *)pSrc + y * nSrcStep) + x;
  Npp32f *dstPixel = (Npp32f *)((char *)pDst + y * nDstStep) + x;

  // Load source pixel
  Npp32f srcValue = *srcPixel;

  // Multiply by constant (no scaling for float)
  Npp32f result = srcValue * nConstant;

  // Store result
  *dstPixel = result;
}

extern "C" {

/**
 * CUDA implementation of nppiMulC_8u_C1RSfs_Ctx
 */
NppStatus nppiMulC_8u_C1RSfs_Ctx_cuda(const Npp8u *pSrc1, int nSrc1Step, const Npp8u nConstant, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  // Set up CUDA grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified CUDA stream
  mulC_8u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

/**
 * CUDA implementation of nppiMulC_16u_C1RSfs_Ctx
 */
NppStatus nppiMulC_16u_C1RSfs_Ctx_cuda(const Npp16u *pSrc1, int nSrc1Step, const Npp16u nConstant, Npp16u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  // Set up CUDA grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified CUDA stream
  mulC_16u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

/**
 * CUDA implementation of nppiMulC_16s_C1RSfs_Ctx
 */
NppStatus nppiMulC_16s_C1RSfs_Ctx_cuda(const Npp16s *pSrc1, int nSrc1Step, const Npp16s nConstant, Npp16s *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  // Set up CUDA grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified CUDA stream
  mulC_16s_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

/**
 * CUDA implementation of nppiMulC_32f_C1R_Ctx
 */
NppStatus nppiMulC_32f_C1R_Ctx_cuda(const Npp32f *pSrc1, int nSrc1Step, const Npp32f nConstant, Npp32f *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Set up CUDA grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified CUDA stream
  mulC_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc1, nSrc1Step, nConstant, pDst, nDstStep,
                                                                        oSizeROI.width, oSizeROI.height);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

} // extern "C"