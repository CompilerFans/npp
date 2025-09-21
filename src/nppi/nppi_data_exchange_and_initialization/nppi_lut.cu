#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * kernels for MPP Linear LUT Functions
 */

// Device function for linear interpolation between two points
__device__ inline int linearInterpolate(int input, int level0, int level1, int value0, int value1) {
  if (input <= level0)
    return value0;
  if (input >= level1)
    return value1;

  // Linear interpolation: y = y0 + (y1-y0) * (x-x0) / (x1-x0)
  float ratio = (float)(input - level0) / (float)(level1 - level0);
  return (int)(value0 + ratio * (value1 - value0));
}

// Device function to find the appropriate LUT segment for interpolation
__device__ inline int findLUTSegment(int input, const Npp32s *pLevels, int nLevels) {
  // Binary search to find the segment
  int left = 0, right = nLevels - 1;

  if (input <= pLevels[0])
    return 0;
  if (input >= pLevels[nLevels - 1])
    return nLevels - 2;

  while (left < right - 1) {
    int mid = (left + right) / 2;
    if (input <= pLevels[mid]) {
      right = mid;
    } else {
      left = mid;
    }
  }

  return left;
}

// Kernel for 8-bit unsigned single channel linear LUT
__global__ void nppiLUT_Linear_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                             int height, const Npp32s *pValues, const Npp32s *pLevels, int nLevels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    int input = src_row[x];

    // Find the segment for interpolation
    int segment = findLUTSegment(input, pLevels, nLevels);

    // Perform linear interpolation
    int result =
        linearInterpolate(input, pLevels[segment], pLevels[segment + 1], pValues[segment], pValues[segment + 1]);

    // Clamp to valid range for 8-bit unsigned
    result = max(0, min(255, result));
    dst_row[x] = (Npp8u)result;
  }
}

// Kernel for 8-bit unsigned three channel linear LUT
__global__ void nppiLUT_Linear_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                             int height, const Npp32s *const *pValues, const Npp32s *const *pLevels,
                                             const int *nLevels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    int src_idx = x * 3;
    int dst_idx = x * 3;

    // Process each channel independently
    for (int c = 0; c < 3; c++) {
      int input = src_row[src_idx + c];

      // Find the segment for interpolation
      int segment = findLUTSegment(input, pLevels[c], nLevels[c]);

      // Perform linear interpolation
      int result = linearInterpolate(input, pLevels[c][segment], pLevels[c][segment + 1], pValues[c][segment],
                                     pValues[c][segment + 1]);

      // Clamp to valid range for 8-bit unsigned
      result = max(0, min(255, result));
      dst_row[dst_idx + c] = (Npp8u)result;
    }
  }
}

// Kernel for 16-bit unsigned single channel linear LUT
__global__ void nppiLUT_Linear_16u_C1R_kernel(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, int width,
                                              int height, const Npp32s *pValues, const Npp32s *pLevels, int nLevels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16u *src_row = (const Npp16u *)((const char *)pSrc + y * nSrcStep);
    Npp16u *dst_row = (Npp16u *)((char *)pDst + y * nDstStep);

    int input = src_row[x];

    // Find the segment for interpolation
    int segment = findLUTSegment(input, pLevels, nLevels);

    // Perform linear interpolation
    int result =
        linearInterpolate(input, pLevels[segment], pLevels[segment + 1], pValues[segment], pValues[segment + 1]);

    // Clamp to valid range for 16-bit unsigned
    result = max(0, min(65535, result));
    dst_row[x] = (Npp16u)result;
  }
}

extern "C" {

// 8-bit unsigned single channel implementation
NppStatus nppiLUT_Linear_8u_C1R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                         NppStreamContext nppStreamCtx) {
  // Copy LUT data to device memory
  Npp32s *d_pValues, *d_pLevels;
  size_t lutSize = nLevels * sizeof(Npp32s);

  cudaMalloc(&d_pValues, lutSize);
  cudaMalloc(&d_pLevels, lutSize);

  cudaMemcpy(d_pValues, pValues, lutSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pLevels, pLevels, lutSize, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiLUT_Linear_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_pValues, d_pLevels, nLevels);

  cudaError_t cudaStatus = cudaGetLastError();

  // Cleanup device memory
  cudaFree(d_pValues);
  cudaFree(d_pLevels);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned three channel implementation
NppStatus nppiLUT_Linear_8u_C3R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                         NppStreamContext nppStreamCtx) {
  // Copy LUT data to device memory for each channel
  Npp32s *d_pValues[3], *d_pLevels[3];
  Npp32s **d_pValuesPtr, **d_pLevelsPtr;
  int *d_nLevels;

  // Allocate device memory for pointers
  cudaMalloc(&d_pValuesPtr, 3 * sizeof(Npp32s *));
  cudaMalloc(&d_pLevelsPtr, 3 * sizeof(Npp32s *));
  cudaMalloc(&d_nLevels, 3 * sizeof(int));

  // Copy nLevels array to device
  cudaMemcpy(d_nLevels, nLevels, 3 * sizeof(int), cudaMemcpyHostToDevice);

  // Allocate and copy LUT data for each channel
  for (int c = 0; c < 3; c++) {
    size_t lutSize = nLevels[c] * sizeof(Npp32s);

    cudaMalloc(&d_pValues[c], lutSize);
    cudaMalloc(&d_pLevels[c], lutSize);

    cudaMemcpy(d_pValues[c], pValues[c], lutSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pLevels[c], pLevels[c], lutSize, cudaMemcpyHostToDevice);
  }

  // Copy pointer arrays to device
  cudaMemcpy(d_pValuesPtr, d_pValues, 3 * sizeof(Npp32s *), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pLevelsPtr, d_pLevels, 3 * sizeof(Npp32s *), cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiLUT_Linear_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_pValuesPtr, d_pLevelsPtr, d_nLevels);

  cudaError_t cudaStatus = cudaGetLastError();

  // Cleanup device memory
  for (int c = 0; c < 3; c++) {
    cudaFree(d_pValues[c]);
    cudaFree(d_pLevels[c]);
  }
  cudaFree(d_pValuesPtr);
  cudaFree(d_pLevelsPtr);
  cudaFree(d_nLevels);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 16-bit unsigned single channel implementation
NppStatus nppiLUT_Linear_16u_C1R_Ctx_cuda(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                          NppiSize oSizeROI, const Npp32s *pValues, const Npp32s *pLevels, int nLevels,
                                          NppStreamContext nppStreamCtx) {
  // Copy LUT data to device memory
  Npp32s *d_pValues, *d_pLevels;
  size_t lutSize = nLevels * sizeof(Npp32s);

  cudaMalloc(&d_pValues, lutSize);
  cudaMalloc(&d_pLevels, lutSize);

  cudaMemcpy(d_pValues, pValues, lutSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pLevels, pLevels, lutSize, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiLUT_Linear_16u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_pValues, d_pLevels, nLevels);

  cudaError_t cudaStatus = cudaGetLastError();

  // Cleanup device memory
  cudaFree(d_pValues);
  cudaFree(d_pLevels);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"