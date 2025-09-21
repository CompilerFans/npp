#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>




__global__ void subC_8u_C1RSfs_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u nConstant,
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

  // Subtract constant and scale
  int result = static_cast<int>(srcValue) - static_cast<int>(nConstant);
  result = result >> nScaleFactor;

  // Clamp to 8-bit range
  result = max(0, min(255, result));

  // Store result
  *dstPixel = static_cast<Npp8u>(result);
}


__global__ void subC_16u_C1RSfs_kernel(const Npp16u *__restrict__ pSrc, int nSrcStep, Npp16u nConstant,
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

  // Subtract constant and scale
  int result = static_cast<int>(srcValue) - static_cast<int>(nConstant);
  result = result >> nScaleFactor;

  // Clamp to 16-bit unsigned range
  result = max(0, min(65535, result));

  // Store result
  *dstPixel = static_cast<Npp16u>(result);
}


__global__ void subC_16s_C1RSfs_kernel(const Npp16s *__restrict__ pSrc, int nSrcStep, Npp16s nConstant,
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

  // Subtract constant and scale
  int result = static_cast<int>(srcValue) - static_cast<int>(nConstant);
  result = result >> nScaleFactor;

  // Clamp to 16-bit signed range
  result = max(-32768, min(32767, result));

  // Store result
  *dstPixel = static_cast<Npp16s>(result);
}


__global__ void subC_32f_C1R_kernel(const Npp32f *__restrict__ pSrc, int nSrcStep, Npp32f nConstant,
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

  // Subtract constant (no scaling for float)
  Npp32f result = srcValue - nConstant;

  // Store result
  *dstPixel = result;
}

extern "C" {


NppStatus nppiSubC_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u nConstant, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  // Set up GPU grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  subC_8u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}


NppStatus nppiSubC_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, const Npp16u nConstant, Npp16u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  // Set up GPU grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  subC_16u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}


NppStatus nppiSubC_16s_C1RSfs_Ctx_impl(const Npp16s *pSrc1, int nSrc1Step, const Npp16s nConstant, Npp16s *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  // Set up GPU grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  subC_16s_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}


NppStatus nppiSubC_32f_C1R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f nConstant, Npp32f *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Set up GPU grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  subC_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc1, nSrc1Step, nConstant, pDst, nDstStep,
                                                                        oSizeROI.width, oSizeROI.height);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}


__global__ void subC_32s_C1RSfs_kernel(const Npp32s *__restrict__ pSrc, int nSrcStep, Npp32s nConstant,
                                       Npp32s *__restrict__ pDst, int nDstStep, int width, int height,
                                       int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32s *srcPixel = reinterpret_cast<const Npp32s *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep) + x;
  Npp32s *dstPixel = reinterpret_cast<Npp32s *>(reinterpret_cast<char *>(pDst) + y * nDstStep) + x;

  // Perform operation with 64-bit intermediate to avoid overflow
  long long result = static_cast<long long>(*srcPixel) - static_cast<long long>(nConstant);
  result = result >> nScaleFactor;

  // Clamp to 32-bit signed range
  if (result > INT_MAX)
    result = INT_MAX;
  if (result < INT_MIN)
    result = INT_MIN;

  *dstPixel = static_cast<Npp32s>(result);
}


__global__ void subC_8u_C3RSfs_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep,
                                      const Npp8u *__restrict__ aConstants, Npp8u *__restrict__ pDst, int nDstStep,
                                      int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *srcPixel = pSrc + y * nSrcStep + x * 3;
  Npp8u *dstPixel = pDst + y * nDstStep + x * 3;

// Process each channel
#pragma unroll
  for (int c = 0; c < 3; c++) {
    int result = static_cast<int>(srcPixel[c]) - static_cast<int>(aConstants[c]);
    result = result >> nScaleFactor;
    result = max(0, min(255, result));
    dstPixel[c] = static_cast<Npp8u>(result);
  }
}


__global__ void subC_16u_C3RSfs_kernel(const Npp16u *__restrict__ pSrc, int nSrcStep,
                                       const Npp16u *__restrict__ aConstants, Npp16u *__restrict__ pDst, int nDstStep,
                                       int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp16u *srcPixel =
      reinterpret_cast<const Npp16u *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep) + x * 3;
  Npp16u *dstPixel = reinterpret_cast<Npp16u *>(reinterpret_cast<char *>(pDst) + y * nDstStep) + x * 3;

#pragma unroll
  for (int c = 0; c < 3; c++) {
    int result = static_cast<int>(srcPixel[c]) - static_cast<int>(aConstants[c]);
    result = result >> nScaleFactor;
    result = max(0, min(65535, result));
    dstPixel[c] = static_cast<Npp16u>(result);
  }
}


__global__ void subC_32f_C3R_kernel(const Npp32f *__restrict__ pSrc, int nSrcStep,
                                    const Npp32f *__restrict__ aConstants, Npp32f *__restrict__ pDst, int nDstStep,
                                    int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32f *srcPixel =
      reinterpret_cast<const Npp32f *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep) + x * 3;
  Npp32f *dstPixel = reinterpret_cast<Npp32f *>(reinterpret_cast<char *>(pDst) + y * nDstStep) + x * 3;

#pragma unroll
  for (int c = 0; c < 3; c++) {
    dstPixel[c] = srcPixel[c] - aConstants[c];
  }
}


__global__ void subC_8u_C4RSfs_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep,
                                      const Npp8u *__restrict__ aConstants, Npp8u *__restrict__ pDst, int nDstStep,
                                      int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *srcPixel = pSrc + y * nSrcStep + x * 4;
  Npp8u *dstPixel = pDst + y * nDstStep + x * 4;

#pragma unroll
  for (int c = 0; c < 4; c++) {
    int result = static_cast<int>(srcPixel[c]) - static_cast<int>(aConstants[c]);
    result = result >> nScaleFactor;
    result = max(0, min(255, result));
    dstPixel[c] = static_cast<Npp8u>(result);
  }
}


__global__ void subC_8u_C1IRSfs_kernel(Npp8u nConstant, Npp8u *__restrict__ pSrcDst, int nSrcDstStep, int width,
                                       int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  Npp8u *pixel = pSrcDst + y * nSrcDstStep + x;

  int result = static_cast<int>(*pixel) - static_cast<int>(nConstant);
  result = result >> nScaleFactor;
  result = max(0, min(255, result));

  *pixel = static_cast<Npp8u>(result);
}

// GPU function implementations
NppStatus nppiSubC_32s_C1RSfs_Ctx_impl(const Npp32s *pSrc1, int nSrc1Step, const Npp32s nConstant, Npp32s *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  subC_32s_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, nConstant, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiSubC_8u_C3RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u aConstants[3], Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  // Copy constants to device
  Npp8u *d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp8u));
  cudaMemcpyAsync(d_constants, aConstants, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  subC_8u_C3RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, d_constants, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  cudaFree(d_constants);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiSubC_16u_C3RSfs_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, const Npp16u aConstants[3], Npp16u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  Npp16u *d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16u));
  cudaMemcpyAsync(d_constants, aConstants, 3 * sizeof(Npp16u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  subC_16u_C3RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, d_constants, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  cudaFree(d_constants);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiSubC_32f_C3R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f aConstants[3], Npp32f *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  Npp32f *d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp32f));
  cudaMemcpyAsync(d_constants, aConstants, 3 * sizeof(Npp32f), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  subC_32f_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc1, nSrc1Step, d_constants, pDst, nDstStep,
                                                                        oSizeROI.width, oSizeROI.height);

  cudaFree(d_constants);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiSubC_8u_C4RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u aConstants[4], Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  Npp8u *d_constants;
  cudaMalloc(&d_constants, 4 * sizeof(Npp8u));
  cudaMemcpyAsync(d_constants, aConstants, 4 * sizeof(Npp8u), cudaMemcpyHostToDevice, nppStreamCtx.hStream);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  subC_8u_C4RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, d_constants, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  cudaFree(d_constants);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiSubC_8u_C1IRSfs_Ctx_impl(const Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                       int nScaleFactor, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  subC_8u_C1IRSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      nConstant, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

} // extern "C"
