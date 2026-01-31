#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Device function to find the appropriate LUT index (no interpolation)
__device__ inline int findLUTIndex(int input, const Npp32s *pLevels, int nLevels) {
  int index = 0;
  for (int i = 1; i < nLevels; ++i) {
    if (input >= pLevels[i]) {
      index = i;
    } else {
      break;
    }
  }
  return index;
}

// Kernel for 8-bit unsigned single channel LUT (no interpolation)
__global__ void nppiLUT_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                       int width, int height, const Npp32s *pValues,
                                       const Npp32s *pLevels, int nLevels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    int input = src_row[x];
    int index = findLUTIndex(input, pLevels, nLevels);
    int result = pValues[index];

    // Clamp to valid range for 8-bit unsigned
    result = max(0, min(255, result));
    dst_row[x] = (Npp8u)result;
  }
}

// Kernel for 8-bit unsigned 3-channel LUT (no interpolation)
__global__ void nppiLUT_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                       int width, int height,
                                       const Npp32s *pValues0, const Npp32s *pLevels0, int nLevels0,
                                       const Npp32s *pValues1, const Npp32s *pLevels1, int nLevels1,
                                       const Npp32s *pValues2, const Npp32s *pLevels2, int nLevels2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;

    // Channel 0
    int input0 = src_row[idx];
    int index0 = findLUTIndex(input0, pLevels0, nLevels0);
    int result0 = max(0, min(255, pValues0[index0]));

    // Channel 1
    int input1 = src_row[idx + 1];
    int index1 = findLUTIndex(input1, pLevels1, nLevels1);
    int result1 = max(0, min(255, pValues1[index1]));

    // Channel 2
    int input2 = src_row[idx + 2];
    int index2 = findLUTIndex(input2, pLevels2, nLevels2);
    int result2 = max(0, min(255, pValues2[index2]));

    dst_row[idx] = (Npp8u)result0;
    dst_row[idx + 1] = (Npp8u)result1;
    dst_row[idx + 2] = (Npp8u)result2;
  }
}

// Kernel for 8-bit unsigned 4-channel LUT (no interpolation)
__global__ void nppiLUT_8u_C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                       int width, int height,
                                       const Npp32s *pValues0, const Npp32s *pLevels0, int nLevels0,
                                       const Npp32s *pValues1, const Npp32s *pLevels1, int nLevels1,
                                       const Npp32s *pValues2, const Npp32s *pLevels2, int nLevels2,
                                       const Npp32s *pValues3, const Npp32s *pLevels3, int nLevels3) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;

    // Channel 0
    int input0 = src_row[idx];
    int index0 = findLUTIndex(input0, pLevels0, nLevels0);
    int result0 = max(0, min(255, pValues0[index0]));

    // Channel 1
    int input1 = src_row[idx + 1];
    int index1 = findLUTIndex(input1, pLevels1, nLevels1);
    int result1 = max(0, min(255, pValues1[index1]));

    // Channel 2
    int input2 = src_row[idx + 2];
    int index2 = findLUTIndex(input2, pLevels2, nLevels2);
    int result2 = max(0, min(255, pValues2[index2]));

    // Channel 3
    int input3 = src_row[idx + 3];
    int index3 = findLUTIndex(input3, pLevels3, nLevels3);
    int result3 = max(0, min(255, pValues3[index3]));

    dst_row[idx] = (Npp8u)result0;
    dst_row[idx + 1] = (Npp8u)result1;
    dst_row[idx + 2] = (Npp8u)result2;
    dst_row[idx + 3] = (Npp8u)result3;
  }
}

// Kernel for 16-bit unsigned single channel LUT (no interpolation)
__global__ void nppiLUT_16u_C1R_kernel(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                        int width, int height, const Npp32s *pValues,
                                        const Npp32s *pLevels, int nLevels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16u *src_row = (const Npp16u *)((const char *)pSrc + y * nSrcStep);
    Npp16u *dst_row = (Npp16u *)((char *)pDst + y * nDstStep);

    int input = src_row[x];
    int index = findLUTIndex(input, pLevels, nLevels);
    int result = pValues[index];

    // Clamp to valid range for 16-bit unsigned
    result = max(0, min(65535, result));
    dst_row[x] = (Npp16u)result;
  }
}

extern "C" {

// 8u C1R implementation
NppStatus nppiLUT_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, const Npp32s *pValues, const Npp32s *pLevels,
                                   int nLevels, NppStreamContext nppStreamCtx) {
  Npp32s *d_pValues, *d_pLevels;
  size_t lutSize = nLevels * sizeof(Npp32s);

  cudaMalloc(&d_pValues, lutSize);
  cudaMalloc(&d_pLevels, lutSize);

  cudaMemcpy(d_pValues, pValues, lutSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pLevels, pLevels, lutSize, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiLUT_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_pValues, d_pLevels, nLevels);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_pValues);
    cudaFree(d_pLevels);
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  cudaStreamSynchronize(nppStreamCtx.hStream);

  cudaFree(d_pValues);
  cudaFree(d_pLevels);

  return NPP_SUCCESS;
}

// 8u C3R implementation
NppStatus nppiLUT_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, const Npp32s *pValues[3], const Npp32s *pLevels[3],
                                   int nLevels[3], NppStreamContext nppStreamCtx) {
  Npp32s *d_pValues[3], *d_pLevels[3];

  for (int i = 0; i < 3; ++i) {
    size_t lutSize = nLevels[i] * sizeof(Npp32s);
    cudaMalloc(&d_pValues[i], lutSize);
    cudaMalloc(&d_pLevels[i], lutSize);
    cudaMemcpy(d_pValues[i], pValues[i], lutSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pLevels[i], pLevels[i], lutSize, cudaMemcpyHostToDevice);
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiLUT_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height,
      d_pValues[0], d_pLevels[0], nLevels[0],
      d_pValues[1], d_pLevels[1], nLevels[1],
      d_pValues[2], d_pLevels[2], nLevels[2]);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    for (int i = 0; i < 3; ++i) {
      cudaFree(d_pValues[i]);
      cudaFree(d_pLevels[i]);
    }
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  cudaStreamSynchronize(nppStreamCtx.hStream);

  for (int i = 0; i < 3; ++i) {
    cudaFree(d_pValues[i]);
    cudaFree(d_pLevels[i]);
  }

  return NPP_SUCCESS;
}

// 8u C4R implementation
NppStatus nppiLUT_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI, const Npp32s *pValues[4], const Npp32s *pLevels[4],
                                   int nLevels[4], NppStreamContext nppStreamCtx) {
  Npp32s *d_pValues[4], *d_pLevels[4];

  for (int i = 0; i < 4; ++i) {
    size_t lutSize = nLevels[i] * sizeof(Npp32s);
    cudaMalloc(&d_pValues[i], lutSize);
    cudaMalloc(&d_pLevels[i], lutSize);
    cudaMemcpy(d_pValues[i], pValues[i], lutSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pLevels[i], pLevels[i], lutSize, cudaMemcpyHostToDevice);
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiLUT_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height,
      d_pValues[0], d_pLevels[0], nLevels[0],
      d_pValues[1], d_pLevels[1], nLevels[1],
      d_pValues[2], d_pLevels[2], nLevels[2],
      d_pValues[3], d_pLevels[3], nLevels[3]);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    for (int i = 0; i < 4; ++i) {
      cudaFree(d_pValues[i]);
      cudaFree(d_pLevels[i]);
    }
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  cudaStreamSynchronize(nppStreamCtx.hStream);

  for (int i = 0; i < 4; ++i) {
    cudaFree(d_pValues[i]);
    cudaFree(d_pLevels[i]);
  }

  return NPP_SUCCESS;
}

// 16u C1R implementation
NppStatus nppiLUT_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues, const Npp32s *pLevels,
                                    int nLevels, NppStreamContext nppStreamCtx) {
  Npp32s *d_pValues, *d_pLevels;
  size_t lutSize = nLevels * sizeof(Npp32s);

  cudaMalloc(&d_pValues, lutSize);
  cudaMalloc(&d_pLevels, lutSize);

  cudaMemcpy(d_pValues, pValues, lutSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pLevels, pLevels, lutSize, cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiLUT_16u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_pValues, d_pLevels, nLevels);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_pValues);
    cudaFree(d_pLevels);
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  cudaStreamSynchronize(nppStreamCtx.hStream);

  cudaFree(d_pValues);
  cudaFree(d_pLevels);

  return NPP_SUCCESS;
}

}
