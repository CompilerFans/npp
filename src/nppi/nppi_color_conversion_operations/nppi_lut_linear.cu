#include "npp.h"
#include <cuda_runtime.h>

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
                                             int height, const Npp32s *pValues0, const Npp32s *pLevels0,
                                             const Npp32s *pValues1, const Npp32s *pLevels1,
                                             const Npp32s *pValues2, const Npp32s *pLevels2, int nLevels0,
                                             int nLevels1, int nLevels2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;

    // Channel 0
    int seg0 = findLUTSegment(src_row[idx], pLevels0, nLevels0);
    int val0 = linearInterpolate(src_row[idx], pLevels0[seg0], pLevels0[seg0 + 1], pValues0[seg0], pValues0[seg0 + 1]);
    dst_row[idx] = (Npp8u)max(0, min(255, val0));

    // Channel 1
    int seg1 = findLUTSegment(src_row[idx + 1], pLevels1, nLevels1);
    int val1 = linearInterpolate(src_row[idx + 1], pLevels1[seg1], pLevels1[seg1 + 1], pValues1[seg1], pValues1[seg1 + 1]);
    dst_row[idx + 1] = (Npp8u)max(0, min(255, val1));

    // Channel 2
    int seg2 = findLUTSegment(src_row[idx + 2], pLevels2, nLevels2);
    int val2 = linearInterpolate(src_row[idx + 2], pLevels2[seg2], pLevels2[seg2 + 1], pValues2[seg2], pValues2[seg2 + 1]);
    dst_row[idx + 2] = (Npp8u)max(0, min(255, val2));
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

// ============================================================================
// 8-bit unsigned implementations
// ============================================================================

// 8-bit unsigned single channel implementation
NppStatus nppiLUT_Linear_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
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
  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_pValues);
  cudaFree(d_pLevels);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// 8-bit unsigned 3-channel implementation
NppStatus nppiLUT_Linear_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         const Npp32s *pValues[3], const Npp32s *pLevels[3], int nLevels[3],
                                         NppStreamContext nppStreamCtx) {
  Npp32s *d_pValues[3], *d_pLevels[3];
  for (int c = 0; c < 3; c++) {
    size_t lutSize = nLevels[c] * sizeof(Npp32s);
    cudaMalloc(&d_pValues[c], lutSize);
    cudaMalloc(&d_pLevels[c], lutSize);
    cudaMemcpy(d_pValues[c], pValues[c], lutSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pLevels[c], pLevels[c], lutSize, cudaMemcpyHostToDevice);
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiLUT_Linear_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height,
      d_pValues[0], d_pLevels[0], d_pValues[1], d_pLevels[1], d_pValues[2], d_pLevels[2],
      nLevels[0], nLevels[1], nLevels[2]);

  cudaError_t cudaStatus = cudaGetLastError();
  cudaStreamSynchronize(nppStreamCtx.hStream);
  for (int c = 0; c < 3; c++) {
    cudaFree(d_pValues[c]);
    cudaFree(d_pLevels[c]);
  }

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// ============================================================================
// 16-bit unsigned implementations
// ============================================================================

NppStatus nppiLUT_Linear_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
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

  nppiLUT_Linear_16u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_pValues, d_pLevels, nLevels);

  cudaError_t cudaStatus = cudaGetLastError();
  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_pValues);
  cudaFree(d_pLevels);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}
