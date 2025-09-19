#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * CUDA kernels for NPP Image Remap Functions
 * Simplified implementation using nearest neighbor interpolation
 */

// Generic remap kernel template for different data types
template <typename T>
__global__ void nppiRemap_nearest_kernel(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         const float *pXMap, int nXMapStep, const float *pYMap, int nYMapStep, T *pDst,
                                         int nDstStep, int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  // Get mapping coordinates from lookup tables
  const float *xmap_row = (const float *)((const char *)pXMap + y * nXMapStep);
  const float *ymap_row = (const float *)((const char *)pYMap + y * nYMapStep);

  float src_x = xmap_row[x];
  float src_y = ymap_row[x];

  // Convert to integer coordinates (nearest neighbor)
  int ix = (int)(src_x + 0.5f);
  int iy = (int)(src_y + 0.5f);

  // Clamp to source ROI bounds
  ix = max(oSrcROI.x, min(ix, oSrcROI.x + oSrcROI.width - 1));
  iy = max(oSrcROI.y, min(iy, oSrcROI.y + oSrcROI.height - 1));

  // Copy pixel(s)
  const T *src_row = (const T *)((const char *)pSrc + iy * nSrcStep);
  T *dst_row = (T *)((char *)pDst + y * nDstStep);

  if (channels == 1) {
    dst_row[x] = src_row[ix];
  } else if (channels == 3) {
    int src_idx = ix * 3;
    int dst_idx = x * 3;
    dst_row[dst_idx + 0] = src_row[src_idx + 0];
    dst_row[dst_idx + 1] = src_row[src_idx + 1];
    dst_row[dst_idx + 2] = src_row[src_idx + 2];
  }
}

extern "C" {

// 8-bit unsigned single channel implementation
NppStatus nppiRemap_8u_C1R_Ctx_cuda(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                                    int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                    NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRemap_nearest_kernel<Npp8u><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep, oDstSizeROI.width,
      oDstSizeROI.height, 1);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned three channel implementation
NppStatus nppiRemap_8u_C3R_Ctx_cuda(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                                    int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                    NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRemap_nearest_kernel<Npp8u><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep, oDstSizeROI.width,
      oDstSizeROI.height, 3);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 16-bit unsigned implementations (reuse 8u kernel with casting)
NppStatus nppiRemap_16u_C1R_Ctx_cuda(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRemap_nearest_kernel<Npp16u><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep, oDstSizeROI.width,
      oDstSizeROI.height, 1);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus != cudaSuccess) ? NPP_CUDA_KERNEL_EXECUTION_ERROR : NPP_SUCCESS;
}

NppStatus nppiRemap_16u_C3R_Ctx_cuda(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRemap_nearest_kernel<Npp16u><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep, oDstSizeROI.width,
      oDstSizeROI.height, 3);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus != cudaSuccess) ? NPP_CUDA_KERNEL_EXECUTION_ERROR : NPP_SUCCESS;
}

// 16-bit signed implementations
NppStatus nppiRemap_16s_C1R_Ctx_cuda(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRemap_nearest_kernel<Npp16s><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep, oDstSizeROI.width,
      oDstSizeROI.height, 1);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus != cudaSuccess) ? NPP_CUDA_KERNEL_EXECUTION_ERROR : NPP_SUCCESS;
}

NppStatus nppiRemap_16s_C3R_Ctx_cuda(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRemap_nearest_kernel<Npp16s><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep, oDstSizeROI.width,
      oDstSizeROI.height, 3);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus != cudaSuccess) ? NPP_CUDA_KERNEL_EXECUTION_ERROR : NPP_SUCCESS;
}

// 32-bit float implementations
NppStatus nppiRemap_32f_C1R_Ctx_cuda(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRemap_nearest_kernel<Npp32f><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep, oDstSizeROI.width,
      oDstSizeROI.height, 1);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus != cudaSuccess) ? NPP_CUDA_KERNEL_EXECUTION_ERROR : NPP_SUCCESS;
}

NppStatus nppiRemap_32f_C3R_Ctx_cuda(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRemap_nearest_kernel<Npp32f><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep, oDstSizeROI.width,
      oDstSizeROI.height, 3);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus != cudaSuccess) ? NPP_CUDA_KERNEL_EXECUTION_ERROR : NPP_SUCCESS;
}

// 64-bit double implementations
NppStatus nppiRemap_64f_C1R_Ctx_cuda(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRemap_nearest_kernel<Npp64f><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep, oDstSizeROI.width,
      oDstSizeROI.height, 1);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus != cudaSuccess) ? NPP_CUDA_KERNEL_EXECUTION_ERROR : NPP_SUCCESS;
}

NppStatus nppiRemap_64f_C3R_Ctx_cuda(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRemap_nearest_kernel<Npp64f><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep, oDstSizeROI.width,
      oDstSizeROI.height, 3);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus != cudaSuccess) ? NPP_CUDA_KERNEL_EXECUTION_ERROR : NPP_SUCCESS;
}
}