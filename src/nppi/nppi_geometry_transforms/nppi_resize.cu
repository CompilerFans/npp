#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Bilinear interpolation helper
__device__ inline float lerp(float a, float b, float t) { return a + t * (b - a); }

// Nearest neighbor resize kernel for 8-bit single channel
__global__ void nppiResize_8u_C1R_nearest_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                                 NppiRect oSrcRectROI, Npp8u *pDst, int nDstStep, NppiSize oDstSize,
                                                 NppiRect oDstRectROI) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;

  // TODO
}

// Bilinear resize kernel for 8-bit single channel
__global__ void nppiResize_8u_C1R_linear_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                                NppiRect oSrcRectROI, Npp8u *pDst, int nDstStep, NppiSize oDstSize,
                                                NppiRect oDstRectROI) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  // TODO
}

// Nearest neighbor resize kernel for 8-bit three channel
__global__ void nppiResize_8u_C3R_nearest_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                                 NppiRect oSrcRectROI, Npp8u *pDst, int nDstStep, NppiSize oDstSize,
                                                 NppiRect oDstRectROI) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;

  // TODO
}

// Bilinear resize kernel for 8-bit three channel
__global__ void nppiResize_8u_C3R_linear_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                                                NppiRect oSrcRectROI, Npp8u *pDst, int nDstStep, NppiSize oDstSize,
                                                NppiRect oDstRectROI) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;

  if (dx >= oDstRectROI.width || dy >= oDstRectROI.height) {
    return;
  }

  float scaleX = (float)oSrcRectROI.width / (float)oDstRectROI.width;
  float scaleY = (float)oSrcRectROI.height / (float)oDstRectROI.height;

  float fx = dx * scaleX + oSrcRectROI.x;
  float fy = dy * scaleY + oSrcRectROI.y;

  int x1 = (int)fx;
  int y1 = (int)fy;
  int x2 = min(x1 + 1, oSrcRectROI.x + oSrcRectROI.width - 1);
  int y2 = min(y1 + 1, oSrcRectROI.y + oSrcRectROI.height - 1);

  x1 = max(x1, oSrcRectROI.x);
  y1 = max(y1, oSrcRectROI.y);

  float tx = fx - x1;
  float ty = fy - y1;

  const Npp8u *src_row1 = (const Npp8u *)((const char *)pSrc + y1 * nSrcStep);
  const Npp8u *src_row2 = (const Npp8u *)((const char *)pSrc + y2 * nSrcStep);

  Npp8u *dst_row = (Npp8u *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

  for (int c = 0; c < 3; c++) {
    float p11 = src_row1[x1 * 3 + c];
    float p12 = src_row1[x2 * 3 + c];
    float p21 = src_row2[x1 * 3 + c];
    float p22 = src_row2[x2 * 3 + c];

    float result = lerp(lerp(p11, p12, tx), lerp(p21, p22, tx), ty);
    dst_row[(dx + oDstRectROI.x) * 3 + c] = (Npp8u)(result + 0.5f);
  }
}

extern "C" {

// 8-bit unsigned three channel implementation
NppStatus nppiResize_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstRectROI.width + blockSize.x - 1) / blockSize.x,
                (oDstRectROI.height + blockSize.y - 1) / blockSize.y);

#if 1
  nppiResize_8u_C3R_linear_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
#else
  // Select kernel based on interpolation mode
  switch (eInterpolation) {
  case 1: // NPPI_INTER_NN
    // nppiResize_8u_C3R_nearest_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
    //    pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    return NPP_BAD_ARGUMENT_ERROR;
    break;
  case 2: // NPPI_INTER_LINEAR
    nppiResize_8u_C3R_linear_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    break;
  case 8: // NPPI_INTER_SUPER
    nppiResize_8u_C3R_linear_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    // nppiResize_8u_C3R_super_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
    //     pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    return NPP_BAD_ARGUMENT_ERROR;
    break;
  default:
    // Default to linear for unsupported modes
    // nppiResize_8u_C3R_linear_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
    //    pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI);
    return NPP_BAD_ARGUMENT_ERROR;
    break;
  }
#endif

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}
