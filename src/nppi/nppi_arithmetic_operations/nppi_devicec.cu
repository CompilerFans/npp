#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// Simple unified DeviceC kernel template
template <typename T, int Channels, typename OpType>
__global__ void deviceConstKernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height,
                                  const T *pConstants, int scaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    if constexpr (Channels == 1) {
      OpType op(pConstants[0]);
      dstRow[x] = op(srcRow[x], T(0), scaleFactor);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        OpType op(pConstants[c]);
        dstRow[idx + c] = op(srcRow[idx + c], T(0), scaleFactor);
      }
    }
  }
}

// Simple execution function template
template <typename T, int Channels, typename OpType>
NppStatus executeDeviceConst(const T *pSrc1, int nSrc1Step, T *pDst, int nDstStep, NppiSize oSizeROI, int scaleFactor,
                             cudaStream_t stream, const T *pDeviceConstants) {
  // Parameter validation
  NppStatus status = validateUnaryParameters(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, Channels);
  if (status != NPP_NO_ERROR)
    return status;
  if (oSizeROI.width == 0 || oSizeROI.height == 0)
    return NPP_NO_ERROR;

  if (!pDeviceConstants) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Launch kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  deviceConstKernel<T, Channels, OpType><<<gridSize, blockSize, 0, stream>>>(
      pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pDeviceConstants, scaleFactor);

  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// ============================================================================
// AddDeviceC Implementation
// ============================================================================

extern "C" {

NppStatus nppiAddDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstant, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 1, AddConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstant);
}

NppStatus nppiAddDeviceC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 3, AddConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiAddDeviceC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 4, AddConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiAddDeviceC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 4, AddConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiAddDeviceC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pConstant, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp16u, 1, AddConstOp<Npp16u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiAddDeviceC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pConstant, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp16s, 1, AddConstOp<Npp16s>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiAddDeviceC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pConstant, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp32s, 1, AddConstOp<Npp32s>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiAddDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstant, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 1, AddConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiAddDeviceC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 3, AddConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

NppStatus nppiAddDeviceC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 4, AddConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

NppStatus nppiAddDeviceC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 4, AddConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

// ============================================================================
// SubDeviceC Implementation
// ============================================================================

NppStatus nppiSubDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstant, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 1, SubConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstant);
}

NppStatus nppiSubDeviceC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 3, SubConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiSubDeviceC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 4, SubConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiSubDeviceC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 4, SubConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiSubDeviceC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pConstant, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp16u, 1, SubConstOp<Npp16u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiSubDeviceC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pConstant, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp16s, 1, SubConstOp<Npp16s>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiSubDeviceC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pConstant, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp32s, 1, SubConstOp<Npp32s>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiSubDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstant, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 1, SubConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiSubDeviceC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 3, SubConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

NppStatus nppiSubDeviceC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 4, SubConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

NppStatus nppiSubDeviceC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 4, SubConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

// ============================================================================
// MulDeviceC Implementation
// ============================================================================

NppStatus nppiMulDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstant, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 1, MulConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstant);
}

NppStatus nppiMulDeviceC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 3, MulConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiMulDeviceC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 4, MulConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiMulDeviceC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 4, MulConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiMulDeviceC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pConstant, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp16u, 1, MulConstOp<Npp16u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiMulDeviceC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pConstant, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp16s, 1, MulConstOp<Npp16s>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiMulDeviceC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pConstant, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp32s, 1, MulConstOp<Npp32s>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiMulDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstant, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 1, MulConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiMulDeviceC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 3, MulConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

NppStatus nppiMulDeviceC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 4, MulConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

NppStatus nppiMulDeviceC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 4, MulConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

// ============================================================================
// DivDeviceC Implementation
// ============================================================================

NppStatus nppiDivDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstant, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 1, DivConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstant);
}

NppStatus nppiDivDeviceC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 3, DivConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiDivDeviceC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 4, DivConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiDivDeviceC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 4, DivConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                         nppStreamCtx.hStream, pConstants);
}

NppStatus nppiDivDeviceC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pConstant, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp16u, 1, DivConstOp<Npp16u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiDivDeviceC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pConstant, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp16s, 1, DivConstOp<Npp16s>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiDivDeviceC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pConstant, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                        NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp32s, 1, DivConstOp<Npp32s>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiDivDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstant, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 1, DivConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstant);
}

NppStatus nppiDivDeviceC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 3, DivConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

NppStatus nppiDivDeviceC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 4, DivConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

NppStatus nppiDivDeviceC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 4, DivConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                           nppStreamCtx.hStream, pConstants);
}

// ============================================================================
// AbsDiffDeviceC Implementation
// ============================================================================

NppStatus nppiAbsDiffDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstant, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                           NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 1, AbsDiffConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                             nppStreamCtx.hStream, pConstant);
}

NppStatus nppiAbsDiffDeviceC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                           NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 3, AbsDiffConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                             nppStreamCtx.hStream, pConstants);
}

NppStatus nppiAbsDiffDeviceC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                            NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 4, AbsDiffConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                             nppStreamCtx.hStream, pConstants);
}

NppStatus nppiAbsDiffDeviceC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstants, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                           NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp8u, 4, AbsDiffConstOp<Npp8u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                             nppStreamCtx.hStream, pConstants);
}

NppStatus nppiAbsDiffDeviceC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pConstant, Npp16u *pDst,
                                            int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                            NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp16u, 1, AbsDiffConstOp<Npp16u>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                               nppStreamCtx.hStream, pConstant);
}

NppStatus nppiAbsDiffDeviceC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step, const Npp16s *pConstant, Npp16s *pDst,
                                            int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                            NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp16s, 1, AbsDiffConstOp<Npp16s>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                               nppStreamCtx.hStream, pConstant);
}

NppStatus nppiAbsDiffDeviceC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step, const Npp32s *pConstant, Npp32s *pDst,
                                            int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                            NppStreamContext nppStreamCtx) {
  if (nScaleFactor < 0 || nScaleFactor > 31)
    return NPP_BAD_ARGUMENT_ERROR;
  return executeDeviceConst<Npp32s, 1, AbsDiffConstOp<Npp32s>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, nScaleFactor,
                                                               nppStreamCtx.hStream, pConstant);
}

NppStatus nppiAbsDiffDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI, Npp32f *pConstant, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 1, AbsDiffConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                               nppStreamCtx.hStream, pConstant);
}

NppStatus nppiAbsDiffDeviceC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                         int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 3, AbsDiffConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                               nppStreamCtx.hStream, pConstants);
}

NppStatus nppiAbsDiffDeviceC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                          int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 4, AbsDiffConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                               nppStreamCtx.hStream, pConstants);
}

NppStatus nppiAbsDiffDeviceC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstants, Npp32f *pDst,
                                         int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp32f, 4, AbsDiffConstOp<Npp32f>>(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, 0,
                                                               nppStreamCtx.hStream, pConstants);
}

// ============================================================================
// MulDeviceCScale Implementation
// ============================================================================

// 8u C1
NppStatus nppiMulDeviceCScale_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pConstant, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp8u, 1, MulCScaleOp<Npp8u>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                          nppStreamCtx.hStream, pConstant);
}

NppStatus nppiMulDeviceCScale_8u_C1IR_Ctx(const Npp8u *pConstant, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp8u, 1, MulCScaleOp<Npp8u>>(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0,
                                                          nppStreamCtx.hStream, pConstant);
}

// 8u C3
NppStatus nppiMulDeviceCScale_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pConstants, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp8u, 3, MulCScaleOp<Npp8u>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                          nppStreamCtx.hStream, pConstants);
}

NppStatus nppiMulDeviceCScale_8u_C3IR_Ctx(const Npp8u *pConstants, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp8u, 3, MulCScaleOp<Npp8u>>(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0,
                                                          nppStreamCtx.hStream, pConstants);
}

// 8u AC4
NppStatus nppiMulDeviceCScale_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pConstants, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp8u, 4, MulCScaleOp<Npp8u>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                          nppStreamCtx.hStream, pConstants);
}

NppStatus nppiMulDeviceCScale_8u_AC4IR_Ctx(const Npp8u *pConstants, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                           NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp8u, 4, MulCScaleOp<Npp8u>>(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0,
                                                          nppStreamCtx.hStream, pConstants);
}

// 8u C4
NppStatus nppiMulDeviceCScale_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, const Npp8u *pConstants, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp8u, 4, MulCScaleOp<Npp8u>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                          nppStreamCtx.hStream, pConstants);
}

NppStatus nppiMulDeviceCScale_8u_C4IR_Ctx(const Npp8u *pConstants, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp8u, 4, MulCScaleOp<Npp8u>>(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0,
                                                          nppStreamCtx.hStream, pConstants);
}

// 16u C1
NppStatus nppiMulDeviceCScale_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u *pConstant, Npp16u *pDst,
                                          int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp16u, 1, MulCScaleOp<Npp16u>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                            nppStreamCtx.hStream, pConstant);
}

NppStatus nppiMulDeviceCScale_16u_C1IR_Ctx(const Npp16u *pConstant, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                           NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp16u, 1, MulCScaleOp<Npp16u>>(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0,
                                                            nppStreamCtx.hStream, pConstant);
}

// 16u C3
NppStatus nppiMulDeviceCScale_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u *pConstants, Npp16u *pDst,
                                          int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp16u, 3, MulCScaleOp<Npp16u>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                            nppStreamCtx.hStream, pConstants);
}

NppStatus nppiMulDeviceCScale_16u_C3IR_Ctx(const Npp16u *pConstants, Npp16u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp16u, 3, MulCScaleOp<Npp16u>>(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0,
                                                            nppStreamCtx.hStream, pConstants);
}

// 16u AC4
NppStatus nppiMulDeviceCScale_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u *pConstants, Npp16u *pDst,
                                           int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp16u, 4, MulCScaleOp<Npp16u>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                            nppStreamCtx.hStream, pConstants);
}

NppStatus nppiMulDeviceCScale_16u_AC4IR_Ctx(const Npp16u *pConstants, Npp16u *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp16u, 4, MulCScaleOp<Npp16u>>(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0,
                                                            nppStreamCtx.hStream, pConstants);
}

// 16u C4
NppStatus nppiMulDeviceCScale_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, const Npp16u *pConstants, Npp16u *pDst,
                                          int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp16u, 4, MulCScaleOp<Npp16u>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 0,
                                                            nppStreamCtx.hStream, pConstants);
}

NppStatus nppiMulDeviceCScale_16u_C4IR_Ctx(const Npp16u *pConstants, Npp16u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return executeDeviceConst<Npp16u, 4, MulCScaleOp<Npp16u>>(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, 0,
                                                            nppStreamCtx.hStream, pConstants);
}

} // extern "C"