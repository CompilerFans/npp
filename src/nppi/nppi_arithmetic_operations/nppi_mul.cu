#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



// ============================================================================
// Device kernels
// ============================================================================

// 8-bit unsigned mul with scaling, single channel
__global__ void nppiMul_8u_C1RSfs_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                         Npp8u *pDst, int nDstStep, int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *src1Row = (const Npp8u *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp8u *src2Row = (const Npp8u *)((const char *)pSrc2 + y * nSrc2Step);
    Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    // Multiply with scaling (pSrc1 * pSrc2)
    int result = (int)src1Row[x] * (int)src2Row[x];

    // Apply scale factor (right shift)
    if (nScaleFactor > 0) {
      result = (result + (1 << (nScaleFactor - 1))) >> nScaleFactor;
    }

    // Saturate to 8-bit range (clamp to 0-255)
    dstRow[x] = (Npp8u)(result < 0 ? 0 : (result > 255 ? 255 : result));
  }
}

// 8-bit unsigned mul with scaling, 3 channels
__global__ void nppiMul_8u_C3RSfs_kernel(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
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
      int result = (int)src1Row[idx + c] * (int)src2Row[idx + c];

      // Apply scale factor
      if (nScaleFactor > 0) {
        result = (result + (1 << (nScaleFactor - 1))) >> nScaleFactor;
      }

      // Saturate to 8-bit range
      dstRow[idx + c] = (Npp8u)(result < 0 ? 0 : (result > 255 ? 255 : result));
    }
  }
}

// 16-bit unsigned mul with scaling, single channel
__global__ void nppiMul_16u_C1RSfs_kernel(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                          Npp16u *pDst, int nDstStep, int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16u *src1Row = (const Npp16u *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp16u *src2Row = (const Npp16u *)((const char *)pSrc2 + y * nSrc2Step);
    Npp16u *dstRow = (Npp16u *)((char *)pDst + y * nDstStep);

    // Multiply with scaling - use larger intermediate type
    long long result = (long long)src1Row[x] * (long long)src2Row[x];

    // Apply scale factor
    if (nScaleFactor > 0) {
      result = (result + (1LL << (nScaleFactor - 1))) >> nScaleFactor;
    }

    // Saturate to 16-bit range
    dstRow[x] = (Npp16u)(result < 0 ? 0 : (result > 65535 ? 65535 : result));
  }
}

// 16-bit signed mul with scaling, single channel
__global__ void nppiMul_16s_C1RSfs_kernel(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step,
                                          Npp16s *pDst, int nDstStep, int width, int height, int nScaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16s *src1Row = (const Npp16s *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp16s *src2Row = (const Npp16s *)((const char *)pSrc2 + y * nSrc2Step);
    Npp16s *dstRow = (Npp16s *)((char *)pDst + y * nDstStep);

    // Multiply with scaling - use larger intermediate type
    long long result = (long long)src1Row[x] * (long long)src2Row[x];

    // Apply scale factor
    if (nScaleFactor > 0) {
      result = (result + (1LL << (nScaleFactor - 1))) >> nScaleFactor;
    }

    // Saturate to 16-bit signed range
    dstRow[x] = (Npp16s)(result < -32768 ? -32768 : (result > 32767 ? 32767 : result));
  }
}

// 32-bit float mul, single channel (no scaling)
__global__ void nppiMul_32f_C1R_kernel(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       Npp32f *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32f *src1Row = (const Npp32f *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp32f *src2Row = (const Npp32f *)((const char *)pSrc2 + y * nSrc2Step);
    Npp32f *dstRow = (Npp32f *)((char *)pDst + y * nDstStep);

    // Simple float multiplication
    dstRow[x] = src1Row[x] * src2Row[x];
  }
}

// 32-bit float mul, 3 channels (no scaling)
__global__ void nppiMul_32f_C3R_kernel(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       Npp32f *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp32f *src1Row = (const Npp32f *)((const char *)pSrc1 + y * nSrc1Step);
    const Npp32f *src2Row = (const Npp32f *)((const char *)pSrc2 + y * nSrc2Step);
    Npp32f *dstRow = (Npp32f *)((char *)pDst + y * nDstStep);

    int idx = x * 3;

    // Process 3 channels
    dstRow[idx] = src1Row[idx] * src2Row[idx];             // R
    dstRow[idx + 1] = src1Row[idx + 1] * src2Row[idx + 1]; // G
    dstRow[idx + 2] = src1Row[idx + 2] * src2Row[idx + 2]; // B
  }
}

// ============================================================================
// Host functions
// ============================================================================

extern "C" {

// 8-bit unsigned
NppStatus nppiMul_8u_C1RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                     int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  dim3 block(16, 16);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);

  nppiMul_8u_C1RSfs_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                                                                     oSizeROI.width, oSizeROI.height, nScaleFactor);

  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMul_8u_C3RSfs_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                     int nDstStep, NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
  dim3 block(16, 16);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);

  nppiMul_8u_C3RSfs_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                                                                     oSizeROI.width, oSizeROI.height, nScaleFactor);

  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// 16-bit unsigned
NppStatus nppiMul_16u_C1RSfs_Ctx_impl(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                      Npp16u *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  dim3 block(16, 16);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);

  nppiMul_16u_C1RSfs_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// 16-bit signed
NppStatus nppiMul_16s_C1RSfs_Ctx_impl(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pSrc2, int nSrc2Step,
                                      Npp16s *pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx) {
  dim3 block(16, 16);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);

  nppiMul_16s_C1RSfs_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(
      pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);

  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// 32-bit float
NppStatus nppiMul_32f_C1R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 block(16, 16);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);

  nppiMul_32f_C1R_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                                                                   oSizeROI.width, oSizeROI.height);

  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiMul_32f_C3R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 block(16, 16);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);

  nppiMul_32f_C3R_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                                                                   oSizeROI.width, oSizeROI.height);

  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}
}
