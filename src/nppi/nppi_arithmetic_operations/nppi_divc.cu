#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__global__ void divC_8u_C1RSfs_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u nConstant,
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

  // Divide by constant and scale
  if (nConstant == 0) {
    *dstPixel = 255; // Handle division by zero
    return;
  }

  int result = (static_cast<int>(srcValue) / static_cast<int>(nConstant)) >> nScaleFactor;

  // Clamp to 8-bit range
  result = max(0, min(255, result));

  // Store result
  *dstPixel = static_cast<Npp8u>(result);
}
__global__ void divC_16u_C1RSfs_kernel(const Npp16u *__restrict__ pSrc, int nSrcStep, Npp16u nConstant,
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

  // Divide by constant and scale
  if (nConstant == 0) {
    *dstPixel = 65535; // Handle division by zero
    return;
  }

  int result = (static_cast<int>(srcValue) / static_cast<int>(nConstant)) >> nScaleFactor;

  // Clamp to 16-bit unsigned range
  result = max(0, min(65535, result));

  // Store result
  *dstPixel = static_cast<Npp16u>(result);
}
__global__ void divC_16s_C1RSfs_kernel(const Npp16s *__restrict__ pSrc, int nSrcStep, Npp16s nConstant,
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

  // Divide by constant and scale
  if (nConstant == 0) {
    *dstPixel = (srcValue >= 0) ? 32767 : -32768; // Handle division by zero
    return;
  }

  int result = (static_cast<int>(srcValue) / static_cast<int>(nConstant)) >> nScaleFactor;

  // Clamp to 16-bit signed range
  result = max(-32768, min(32767, result));

  // Store result
  *dstPixel = static_cast<Npp16s>(result);
}
__global__ void divC_32f_C1R_kernel(const Npp32f *__restrict__ pSrc, int nSrcStep, Npp32f nConstant,
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

  // Divide by constant (no scaling for float)
  if (nConstant == 0.0f) {
    // Handle division by zero for float
    if (srcValue > 0.0f) {
      *dstPixel = __int_as_float(0x7f800000); // +inf
    } else if (srcValue < 0.0f) {
      *dstPixel = __int_as_float(0xff800000); // -inf
    } else {
      *dstPixel = __int_as_float(0x7fc00000); // nan
    }
    return;
  }

  Npp32f result = srcValue / nConstant;

  // Store result
  *dstPixel = result;
}

extern "C" {
NppStatus nppiDivC_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u nConstant, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  // Set up GPU grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  divC_8u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}
NppStatus nppiDivC_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, const Npp16u nConstant, Npp16u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  // Set up GPU grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  divC_16u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}
NppStatus nppiDivC_16s_C1RSfs_Ctx_impl(const Npp16s *pSrc1, int nSrc1Step, const Npp16s nConstant, Npp16s *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  // Set up GPU grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  divC_16s_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}
NppStatus nppiDivC_32f_C1R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f nConstant, Npp32f *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Set up GPU grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  divC_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc1, nSrc1Step, nConstant, pDst, nDstStep,
                                                                        oSizeROI.width, oSizeROI.height);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}
__global__ void divC_8u_C3RSfs_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep,
                                      const Npp8u *__restrict__ aConstants, Npp8u *__restrict__ pDst, int nDstStep,
                                      int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *srcRow = pSrc + y * nSrcStep;
  Npp8u *dstRow = pDst + y * nDstStep;

  int srcIdx = x * 3;
  int dstIdx = x * 3;

  // Process three channels
  for (int c = 0; c < 3; c++) {
    int result = (srcRow[srcIdx + c] / aConstants[c]) >> nScaleFactor;
    dstRow[dstIdx + c] = (Npp8u)min(255, max(0, result));
  }
}
NppStatus nppiDivC_8u_C3RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u aConstants[3], Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {

  // Copy constants to device memory
  Npp8u *d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp8u));
  cudaMemcpy(d_constants, aConstants, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // Setup kernel launch parameters
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  divC_8u_C3RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, d_constants, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  // Clean up device memory
  cudaFree(d_constants);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}
}
