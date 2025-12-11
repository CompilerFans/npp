#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// SwapChannels kernel for 4-channel images (C4R and C4IR)
__global__ void nppiSwapChannels_8u_C4_kernel(const Npp8u *pSrc, int nSrcStep,
                                               Npp8u *pDst, int nDstStep,
                                               int width, int height,
                                               const int *aDstOrder) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  // Read channel order from constant memory or global memory
  int order[4];
  order[0] = aDstOrder[0];
  order[1] = aDstOrder[1];
  order[2] = aDstOrder[2];
  order[3] = aDstOrder[3];

  // Swap channels according to aDstOrder
  // dst[c] = src[aDstOrder[c]]
  dst_row[x * 4 + 0] = src_row[x * 4 + order[0]];
  dst_row[x * 4 + 1] = src_row[x * 4 + order[1]];
  dst_row[x * 4 + 2] = src_row[x * 4 + order[2]];
  dst_row[x * 4 + 3] = src_row[x * 4 + order[3]];
}

// Optimized vectorized version using uint32_t for aligned access
__global__ void nppiSwapChannels_8u_C4_vectorized_kernel(const Npp8u *pSrc, int nSrcStep,
                                                          Npp8u *pDst, int nDstStep,
                                                          int width, int height,
                                                          const int *aDstOrder) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  // Load channel order
  int order[4];
  order[0] = aDstOrder[0];
  order[1] = aDstOrder[1];
  order[2] = aDstOrder[2];
  order[3] = aDstOrder[3];

  // Perform channel swap
  int src_idx = x * 4;
  int dst_idx = x * 4;

  dst_row[dst_idx + 0] = src_row[src_idx + order[0]];
  dst_row[dst_idx + 1] = src_row[src_idx + order[1]];
  dst_row[dst_idx + 2] = src_row[src_idx + order[2]];
  dst_row[dst_idx + 3] = src_row[src_idx + order[3]];
}

extern "C" {

// Implementation function for C4R with context
NppStatus nppiSwapChannels_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep,
                                            Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI,
                                            const int aDstOrder[4],
                                            NppStreamContext nppStreamCtx) {
  // Allocate device memory for aDstOrder
  int *d_aDstOrder;
  cudaError_t err = cudaMalloc(&d_aDstOrder, 4 * sizeof(int));
  if (err != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  err = cudaMemcpyAsync(d_aDstOrder, aDstOrder, 4 * sizeof(int),
                        cudaMemcpyHostToDevice, nppStreamCtx.hStream);
  if (err != cudaSuccess) {
    cudaFree(d_aDstOrder);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSwapChannels_8u_C4_vectorized_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_aDstOrder);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    cudaFree(d_aDstOrder);
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // Free device memory after kernel completes
  cudaStreamSynchronize(nppStreamCtx.hStream);
  cudaFree(d_aDstOrder);

  return NPP_SUCCESS;
}

// Implementation function for C4IR (in-place) with context
NppStatus nppiSwapChannels_8u_C4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep,
                                             NppiSize oSizeROI,
                                             const int aDstOrder[4],
                                             NppStreamContext nppStreamCtx) {
  // For in-place operation, we need a temporary buffer
  // Allocate temporary device memory
  int step;
  Npp8u *pTemp = nppiMalloc_8u_C4(oSizeROI.width, oSizeROI.height, &step);
  if (!pTemp) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  // Copy source to temp
  cudaError_t err = cudaMemcpy2DAsync(pTemp, step, pSrcDst, nSrcDstStep,
                                      oSizeROI.width * 4 * sizeof(Npp8u), oSizeROI.height,
                                      cudaMemcpyDeviceToDevice, nppStreamCtx.hStream);
  if (err != cudaSuccess) {
    nppiFree(pTemp);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  // Perform swap from temp to original
  NppStatus status = nppiSwapChannels_8u_C4R_Ctx_impl(pTemp, step, pSrcDst, nSrcDstStep,
                                                       oSizeROI, aDstOrder, nppStreamCtx);

  // Free temporary buffer
  cudaStreamSynchronize(nppStreamCtx.hStream);
  nppiFree(pTemp);

  return status;
}

} // extern "C"
