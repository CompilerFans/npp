#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CopyConstBorder kernel for single channel 8u images
__global__ void nppiCopyConstBorder_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep,
                                                   int srcWidth, int srcHeight,
                                                   Npp8u *pDst, int nDstStep,
                                                   int dstWidth, int dstHeight,
                                                   int nTopBorderHeight, int nLeftBorderWidth,
                                                   Npp8u nValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dstWidth || y >= dstHeight)
    return;

  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

  // Calculate source coordinates
  int srcX = x - nLeftBorderWidth;
  int srcY = y - nTopBorderHeight;

  // Check if we're in the border region
  if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight) {
    // Border region: use constant value
    dst_row[x] = nValue;
  } else {
    // Inside region: copy from source
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + srcY * nSrcStep);
    dst_row[x] = src_row[srcX];
  }
}

// Optimized version with coalesced memory access
__global__ void nppiCopyConstBorder_8u_C1R_optimized_kernel(const Npp8u *pSrc, int nSrcStep,
                                                             int srcWidth, int srcHeight,
                                                             Npp8u *pDst, int nDstStep,
                                                             int dstWidth, int dstHeight,
                                                             int nTopBorderHeight, int nLeftBorderWidth,
                                                             Npp8u nValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dstWidth || y >= dstHeight)
    return;

  // Calculate source coordinates
  int srcX = x - nLeftBorderWidth;
  int srcY = y - nTopBorderHeight;

  Npp8u value;

  // Check if we're in the border region
  if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight) {
    // Border region: use constant value
    value = nValue;
  } else {
    // Inside region: copy from source
    const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + srcY * nSrcStep);
    value = src_row[srcX];
  }

  // Write to destination
  Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
  dst_row[x] = value;
}

extern "C" {

// Implementation function for C1R with context
NppStatus nppiCopyConstBorder_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep,
                                               NppiSize oSrcSizeROI,
                                               Npp8u *pDst, int nDstStep,
                                               NppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth,
                                               Npp8u nValue,
                                               NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopyConstBorder_8u_C1R_optimized_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, oSrcSizeROI.width, oSrcSizeROI.height,
      pDst, nDstStep, oDstSizeROI.width, oDstSizeROI.height,
      nTopBorderHeight, nLeftBorderWidth, nValue);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"
