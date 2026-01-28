#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Template kernel for channel swapping with generic data types and channel counts
template<typename T, int SRC_CHANNELS, int DST_CHANNELS>
__global__ void nppiSwapChannels_kernel(const T *pSrc, int nSrcStep,
                                         T *pDst, int nDstStep,
                                         int width, int height,
                                         const int *aDstOrder,
                                         T fillValue = 0) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);
  T *dst_row = (T *)((char *)pDst + y * nDstStep);

  int order[4];
  for (int i = 0; i < DST_CHANNELS; i++) {
    order[i] = aDstOrder[i];
  }

  for (int c = 0; c < DST_CHANNELS; c++) {
    int src_ch = order[c];
    if (src_ch >= 0 && src_ch < SRC_CHANNELS) {
      dst_row[x * DST_CHANNELS + c] = src_row[x * SRC_CHANNELS + src_ch];
    } else {
      dst_row[x * DST_CHANNELS + c] = fillValue;
    }
  }
}

// AC4R specialized kernel that skips alpha channel (channel 3)
template<typename T>
__global__ void nppiSwapChannels_AC4_kernel(const T *pSrc, int nSrcStep,
                                             T *pDst, int nDstStep,
                                             int width, int height,
                                             const int *aDstOrder) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);
  T *dst_row = (T *)((char *)pDst + y * nDstStep);

  int order[3];
  order[0] = aDstOrder[0];
  order[1] = aDstOrder[1];
  order[2] = aDstOrder[2];

  // Swap first 3 channels only
  for (int c = 0; c < 3; c++) {
    dst_row[x * 4 + c] = src_row[x * 4 + order[c]];
  }

  // Set alpha channel to 0 (NVIDIA NPP behavior)
  dst_row[x * 4 + 3] = 0;
}

// Template implementation function
template<typename T, int SRC_CHANNELS, int DST_CHANNELS>
NppStatus nppiSwapChannels_impl(const T *pSrc, int nSrcStep,
                                 T *pDst, int nDstStep,
                                 NppiSize oSizeROI,
                                 const int *aDstOrder,
                                 T fillValue,
                                 NppStreamContext nppStreamCtx) {
  int *d_aDstOrder;
  cudaError_t err = cudaMalloc(&d_aDstOrder, DST_CHANNELS * sizeof(int));
  if (err != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  err = cudaMemcpyAsync(d_aDstOrder, aDstOrder, DST_CHANNELS * sizeof(int),
                        cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  if (err != cudaSuccess) {
    cudaFree(d_aDstOrder);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSwapChannels_kernel<T, SRC_CHANNELS, DST_CHANNELS>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height,
      d_aDstOrder, fillValue);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_aDstOrder);
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_aDstOrder);

  return NPP_SUCCESS;
}

// AC4R specialized implementation
template<typename T>
NppStatus nppiSwapChannels_AC4R_impl(const T *pSrc, int nSrcStep,
                                      T *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const int aDstOrder[3],
                                      NppStreamContext nppStreamCtx) {
  int *d_aDstOrder;
  cudaError_t err = cudaMalloc(&d_aDstOrder, 3 * sizeof(int));
  if (err != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  err = cudaMemcpyAsync(d_aDstOrder, aDstOrder, 3 * sizeof(int),
                        cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  if (err != cudaSuccess) {
    cudaFree(d_aDstOrder);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSwapChannels_AC4_kernel<T>
      <<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_aDstOrder);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_aDstOrder);
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_aDstOrder);

  return NPP_SUCCESS;
}

// Explicit template instantiations for export
#define INSTANTIATE_SWAP_CHANNELS(T) \
  template NppStatus nppiSwapChannels_impl<T, 3, 3>(const T*, int, T*, int, NppiSize, const int*, T, NppStreamContext); \
  template NppStatus nppiSwapChannels_impl<T, 4, 4>(const T*, int, T*, int, NppiSize, const int*, T, NppStreamContext); \
  template NppStatus nppiSwapChannels_impl<T, 4, 3>(const T*, int, T*, int, NppiSize, const int*, T, NppStreamContext); \
  template NppStatus nppiSwapChannels_impl<T, 3, 4>(const T*, int, T*, int, NppiSize, const int*, T, NppStreamContext); \
  template NppStatus nppiSwapChannels_AC4R_impl<T>(const T*, int, T*, int, NppiSize, const int[3], NppStreamContext);

INSTANTIATE_SWAP_CHANNELS(Npp8u)
INSTANTIATE_SWAP_CHANNELS(Npp16u)
INSTANTIATE_SWAP_CHANNELS(Npp16s)
INSTANTIATE_SWAP_CHANNELS(Npp32s)
INSTANTIATE_SWAP_CHANNELS(Npp32f)

#undef INSTANTIATE_SWAP_CHANNELS
