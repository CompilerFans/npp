#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Implementation file

// Kernel for 8-bit unsigned single channel copy with constant border
__global__ void nppiCopyConstBorder_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp8u *pDst,
                                                  int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
                                                  int nLeftBorderWidth, Npp8u nValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstSizeROI.width && y < oDstSizeROI.height) {
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    // Check if we're in the source region
    if (x >= nLeftBorderWidth && x < nLeftBorderWidth + oSrcSizeROI.width && y >= nTopBorderHeight &&
        y < nTopBorderHeight + oSrcSizeROI.height) {
      // Copy from source
      int src_x = x - nLeftBorderWidth;
      int src_y = y - nTopBorderHeight;
      const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + src_y * nSrcStep);
      dst_row[x] = src_row[src_x];
    } else {
      // Fill with constant value (border region)
      dst_row[x] = nValue;
    }
  }
}

// Kernel for 8-bit unsigned three channel copy with constant border
__global__ void nppiCopyConstBorder_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp8u *pDst,
                                                  int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
                                                  int nLeftBorderWidth, const Npp8u *aValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstSizeROI.width && y < oDstSizeROI.height) {
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    // Check if we're in the source region
    if (x >= nLeftBorderWidth && x < nLeftBorderWidth + oSrcSizeROI.width && y >= nTopBorderHeight &&
        y < nTopBorderHeight + oSrcSizeROI.height) {
      // Copy from source
      int src_x = x - nLeftBorderWidth;
      int src_y = y - nTopBorderHeight;
      const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + src_y * nSrcStep);

      int dst_idx = x * 3;
      int src_idx = src_x * 3;
      dst_row[dst_idx] = src_row[src_idx];
      dst_row[dst_idx + 1] = src_row[src_idx + 1];
      dst_row[dst_idx + 2] = src_row[src_idx + 2];
    } else {
      // Fill with constant values (border region)
      int dst_idx = x * 3;
      dst_row[dst_idx] = aValue[0];
      dst_row[dst_idx + 1] = aValue[1];
      dst_row[dst_idx + 2] = aValue[2];
    }
  }
}

// Kernel for 16-bit signed single channel copy with constant border
__global__ void nppiCopyConstBorder_16s_C1R_kernel(const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp16s *pDst,
                                                   int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
                                                   int nLeftBorderWidth, Npp16s nValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstSizeROI.width && y < oDstSizeROI.height) {
    Npp16s *dst_row = (Npp16s *)((char *)pDst + y * nDstStep);

    // Check if we're in the source region
    if (x >= nLeftBorderWidth && x < nLeftBorderWidth + oSrcSizeROI.width && y >= nTopBorderHeight &&
        y < nTopBorderHeight + oSrcSizeROI.height) {
      // Copy from source
      int src_x = x - nLeftBorderWidth;
      int src_y = y - nTopBorderHeight;
      const Npp16s *src_row = (const Npp16s *)((const char *)pSrc + src_y * nSrcStep);
      dst_row[x] = src_row[src_x];
    } else {
      // Fill with constant value (border region)
      dst_row[x] = nValue;
    }
  }
}

// Kernel for 32-bit float single channel copy with constant border
__global__ void nppiCopyConstBorder_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp32f *pDst,
                                                   int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
                                                   int nLeftBorderWidth, Npp32f nValue) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < oDstSizeROI.width && y < oDstSizeROI.height) {
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

    // Check if we're in the source region
    if (x >= nLeftBorderWidth && x < nLeftBorderWidth + oSrcSizeROI.width && y >= nTopBorderHeight &&
        y < nTopBorderHeight + oSrcSizeROI.height) {
      // Copy from source
      int src_x = x - nLeftBorderWidth;
      int src_y = y - nTopBorderHeight;
      const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + src_y * nSrcStep);
      dst_row[x] = src_row[src_x];
    } else {
      // Fill with constant value (border region)
      dst_row[x] = nValue;
    }
  }
}

extern "C" {

// 8-bit unsigned single channel implementation
NppStatus nppiCopyConstBorder_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp8u *pDst,
                                              int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
                                              int nLeftBorderWidth, Npp8u nValue, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopyConstBorder_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI, nTopBorderHeight, nLeftBorderWidth, nValue);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned three channel implementation
NppStatus nppiCopyConstBorder_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp8u *pDst,
                                              int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
                                              int nLeftBorderWidth, const Npp8u aValue[3],
                                              NppStreamContext nppStreamCtx) {
  // Copy constant values to device memory
  Npp8u *d_aValue;
  cudaMalloc(&d_aValue, 3 * sizeof(Npp8u));
  cudaMemcpy(d_aValue, aValue, 3 * sizeof(Npp8u), cudaMemcpyHostToDevice);

  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopyConstBorder_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI, nTopBorderHeight, nLeftBorderWidth, d_aValue);

  cudaError_t cudaStatus = cudaGetLastError();
  cudaFree(d_aValue);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 16-bit signed single channel implementation
NppStatus nppiCopyConstBorder_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp16s *pDst,
                                               int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
                                               int nLeftBorderWidth, Npp16s nValue, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopyConstBorder_16s_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI, nTopBorderHeight, nLeftBorderWidth, nValue);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel implementation
NppStatus nppiCopyConstBorder_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp32f *pDst,
                                               int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
                                               int nLeftBorderWidth, Npp32f nValue, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiCopyConstBorder_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI, nTopBorderHeight, nLeftBorderWidth, nValue);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"
