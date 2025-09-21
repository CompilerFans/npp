#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * CUDA Kernels for NPP Image Add Functions
 */

// ============================================================================
// Device kernels
// ============================================================================

// 8-bit unsigned add with scaling, single channel
__global__ void nppiAdd_8u_C1RSfs_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                         Npp8u *pDst, int nDstStep, int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src1Row = (const Npp8u *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp8u *src2Row = (const Npp8u *)((const char *)pSrc2 + y * nSrc2Step);
    Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    // Add with scaling
    int result = (int)src1Row[x] + (int)src2Row[x];

    // Apply scale factor (right shift)
    if (nScaleFactor > 0) {
      result = (result + (1 << (nScaleFactor - 1))) >> nScaleFactor;
    }

    // Saturate to 8-bit range
    dstRow[x] = (Npp8u)(result < 0 ? 0 : (result > 255 ? 255 : result));
  }
}

// 8-bit unsigned add with scaling, 3 channels
__global__ void nppiAdd_8u_C3RSfs_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                         Npp8u *pDst, int nDstStep, int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src1Row = (const Npp8u *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp8u *src2Row = (const Npp8u *)((const char *)pSrc2 + y * nSrc2Step);
    Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;

    // Process 3 channels
    for (int c = 0; c < 3; c++) {
      int result = (int)src1Row[idx + c] + (int)src2Row[idx + c];

      // Apply scale factor
      if (nScaleFactor > 0) {
        result = (result + (1 << (nScaleFactor - 1))) >> nScaleFactor;
      }

      // Saturate to 8-bit range
      dstRow[idx + c] = (Npp8u)(result < 0 ? 0 : (result > 255 ? 255 : result));
    }
  }
}

// 8-bit unsigned add with scaling, 4 channels
__global__ void nppiAdd_8u_C4RSfs_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                         Npp8u *pDst, int nDstStep, int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src1Row = (const Npp8u *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp8u *src2Row = (const Npp8u *)((const char *)pSrc2 + y * nSrc2Step);
    Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;

    // Process 4 channels
    for (int c = 0; c < 4; c++) {
      int result = (int)src1Row[idx + c] + (int)src2Row[idx + c];

      // Apply scale factor
      if (nScaleFactor > 0) {
        result = (result + (1 << (nScaleFactor - 1))) >> nScaleFactor;
      }

      // Saturate to 8-bit range
      dstRow[idx + c] = (Npp8u)(result < 0 ? 0 : (result > 255 ? 255 : result));
    }
  }
}

// 16-bit unsigned add with scaling
__global__ void nppiAdd_16u_C1RSfs_kernel(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                          Npp16u *pDst, int nDstStep, int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16u *src1Row = (const Npp16u *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp16u *src2Row = (const Npp16u *)((const char *)pSrc2 + y * nSrc2Step);
    Npp16u *dstRow = (Npp16u *)((char *)pDst + y * nDstStep);

    // Add with scaling
    unsigned int result = (unsigned int)src1Row[x] + (unsigned int)src2Row[x];

    // Apply scale factor
    if (nScaleFactor > 0) {
      result = (result + (1U << (nScaleFactor - 1))) >> nScaleFactor;
    }

    // Saturate to 16-bit unsigned range
    dstRow[x] = (Npp16u)(result > 65535 ? 65535 : result);
  }
}

// 16-bit signed add with scaling
__global__ void nppiAdd_16s_C1RSfs_kernel(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step,
                                          Npp16s *pDst, int nDstStep, int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16s *src1Row = (const Npp16s *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp16s *src2Row = (const Npp16s *)((const char *)pSrc2 + y * nSrc2Step);
    Npp16s *dstRow = (Npp16s *)((char *)pDst + y * nDstStep);

    // Add with scaling
    int result = (int)src1Row[x] + (int)src2Row[x];

    // Apply scale factor with rounding
    if (nScaleFactor > 0) {
      int round = 1 << (nScaleFactor - 1);
      if (result >= 0) {
        result = (result + round) >> nScaleFactor;
      } else {
        result = (result - round + 1) >> nScaleFactor;
      }
    }

    // Saturate to 16-bit signed range
    if (result < -32768)
      result = -32768;
    else if (result > 32767)
      result = 32767;

    dstRow[x] = (Npp16s)result;
  }
}

// 32-bit float add (no scaling), single channel
__global__ void nppiAdd_32f_C1R_kernel(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       Npp32f *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32f *src1Row = (const Npp32f *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp32f *src2Row = (const Npp32f *)((const char *)pSrc2 + y * nSrc2Step);
    Npp32f *dstRow = (Npp32f *)((char *)pDst + y * nDstStep);

    // Simple float addition
    dstRow[x] = src1Row[x] + src2Row[x];
  }
}

// 32-bit float add (no scaling), 3 channels
__global__ void nppiAdd_32f_C3R_kernel(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       Npp32f *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32f *src1Row = (const Npp32f *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp32f *src2Row = (const Npp32f *)((const char *)pSrc2 + y * nSrc2Step);
    Npp32f *dstRow = (Npp32f *)((char *)pDst + y * nDstStep);

    int idx = x * 3;

    // Process 3 channels
    dstRow[idx] = src1Row[idx] + src2Row[idx];
    dstRow[idx + 1] = src1Row[idx + 1] + src2Row[idx + 1];
    dstRow[idx + 2] = src1Row[idx + 2] + src2Row[idx + 2];
  }
}

// ============================================================================
// Host functions
// ============================================================================

extern "C" {

NppStatus nppiAdd_8u_C1RSfs_Ctx_cuda(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                     int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  dim3 blockSize(32, 8);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAdd_8u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  return NPP_NO_ERROR;
}

NppStatus nppiAdd_8u_C3RSfs_Ctx_cuda(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                     int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  dim3 blockSize(32, 8);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAdd_8u_C3RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  return NPP_NO_ERROR;
}

NppStatus nppiAdd_8u_C4RSfs_Ctx_cuda(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                     int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  dim3 blockSize(32, 8);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAdd_8u_C4RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  return NPP_NO_ERROR;
}

NppStatus nppiAdd_16u_C1RSfs_Ctx_cuda(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                      Npp16u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  dim3 blockSize(32, 8);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAdd_16u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  return NPP_NO_ERROR;
}

NppStatus nppiAdd_16s_C1RSfs_Ctx_cuda(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step,
                                      Npp16s *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  dim3 blockSize(32, 8);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAdd_16s_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  return NPP_NO_ERROR;
}

NppStatus nppiAdd_32f_C1R_Ctx_cuda(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Early return for zero-size ROI to avoid invalid kernel configurations
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_NO_ERROR;
  }

  dim3 blockSize(32, 8);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAdd_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst,
                                                                           nDstStep, oSizeROI.width, oSizeROI.height);

  return NPP_NO_ERROR;
}

NppStatus nppiAdd_32f_C3R_Ctx_cuda(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(32, 8);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiAdd_32f_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst,
                                                                           nDstStep, oSizeROI.width, oSizeROI.height);

  return NPP_NO_ERROR;
}

} // extern "C"