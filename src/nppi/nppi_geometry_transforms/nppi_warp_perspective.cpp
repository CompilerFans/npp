#include "npp.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations for mpp host func implementations

extern "C" {
NppStatus nppiWarpPerspective_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_16u_C1R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_16u_C3R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_16u_C4R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_32f_C3R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_32f_C4R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_32s_C1R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_32s_C3R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_32s_C4R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_8u_C4R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_16f_C1R_Ctx_impl(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_16f_C3R_Ctx_impl(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_16f_C4R_Ctx_impl(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiMergeAlpha_8u_C4R(const Npp8u *pSrcWarped, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
NppStatus nppiMergeAlpha_16u_C4R(const Npp16u *pSrcWarped, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
NppStatus nppiMergeAlpha_32f_C4R(const Npp32f *pSrcWarped, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
NppStatus nppiMergeAlpha_32s_C4R(const Npp32s *pSrcWarped, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
}

static inline NppStatus validateWarpPerspectiveInputs(const void *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                      NppiRect oSrcROI, void *pDst, int nDstStep, NppiRect oDstROI,
                                                      const double aCoeffs[3][3], int eInterpolation) {
  // Check pointers
  if (!pSrc || !pDst || !aCoeffs) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Check step sizes
  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Check source size
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Check ROI sizes
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0 || oDstROI.width <= 0 || oDstROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Check ROI bounds
  if (oSrcROI.x < 0 || oSrcROI.y < 0 || oSrcROI.x + oSrcROI.width > oSrcSize.width ||
      oSrcROI.y + oSrcROI.height > oSrcSize.height) {
    return NPP_RECTANGLE_ERROR;
  }

  // Check interpolation mode
  if (eInterpolation != NPPI_INTER_NN && eInterpolation != NPPI_INTER_LINEAR && eInterpolation != NPPI_INTER_CUBIC) {
    return NPP_INTERPOLATION_ERROR;
  }

  // Check perspective matrix - the bottom row should not make w=0
  // We check if the matrix would cause division by zero for any destination pixel
  // This is a simplified check - a more thorough check would validate for all pixels
  if (fabs(aCoeffs[2][2]) < 1e-10) {
    return NPP_COEFFICIENT_ERROR;
  }

  return NPP_SUCCESS;
}

static inline NppStreamContext getDefaultStreamContextZero() {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = 0;
  return nppStreamCtx;
}

template <typename T, typename WarpFn, typename MergeFn>
static NppStatus warpPerspectiveAC4Common(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, T *pDst,
                                          int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation,
                                          NppStreamContext nppStreamCtx, WarpFn warpFn, MergeFn mergeFn) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  T *pTmp = nullptr;
  size_t tmpBytes = static_cast<size_t>(nDstStep) * oDstROI.height;
  cudaError_t cudaStatus = cudaMalloc(&pTmp, tmpBytes);
  if (cudaStatus != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  status = warpFn(pSrc, oSrcSize, nSrcStep, oSrcROI, pTmp, nDstStep, oDstROI, aCoeffs, eInterpolation, nppStreamCtx);
  if (status == NPP_SUCCESS) {
    status = mergeFn(pTmp, pDst, nDstStep, {oDstROI.width, oDstROI.height}, nppStreamCtx);
  }

  cudaFree(pTmp);
  return status;
}

template <typename T, typename WarpFn>
static NppStatus warpPerspectivePlanarCommon(const T *const *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             T **pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                             int eInterpolation, int planes, NppStreamContext nppStreamCtx,
                                             WarpFn warpFn) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  for (int i = 0; i < planes; ++i) {
    if (!pSrc[i] || !pDst[i]) {
      return NPP_NULL_POINTER_ERROR;
    }
  }
  NppStatus status =
      validateWarpPerspectiveInputs(pSrc[0], oSrcSize, nSrcStep, oSrcROI, pDst[0], nDstStep, oDstROI, aCoeffs, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }
  for (int i = 0; i < planes; ++i) {
    status = warpFn(pSrc[i], oSrcSize, nSrcStep, oSrcROI, pDst[i], nDstStep, oDstROI, aCoeffs, eInterpolation,
                    nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }
  return NPP_SUCCESS;
}

static bool solveLinearSystem8x8(double a[8][9], double x[8]) {
  for (int col = 0; col < 8; ++col) {
    int pivot = col;
    double maxAbs = std::fabs(a[col][col]);
    for (int row = col + 1; row < 8; ++row) {
      double value = std::fabs(a[row][col]);
      if (value > maxAbs) {
        maxAbs = value;
        pivot = row;
      }
    }
    if (maxAbs < 1e-12) {
      return false;
    }
    if (pivot != col) {
      for (int k = col; k < 9; ++k) {
        std::swap(a[col][k], a[pivot][k]);
      }
    }
    double divisor = a[col][col];
    for (int k = col; k < 9; ++k) {
      a[col][k] /= divisor;
    }
    for (int row = 0; row < 8; ++row) {
      if (row == col) {
        continue;
      }
      double factor = a[row][col];
      if (factor == 0.0) {
        continue;
      }
      for (int k = col; k < 9; ++k) {
        a[row][k] -= factor * a[col][k];
      }
    }
  }
  for (int i = 0; i < 8; ++i) {
    x[i] = a[i][8];
  }
  return true;
}

static NppStatus computePerspectiveCoeffsFromQuads(const double srcQuad[4][2], const double dstQuad[4][2],
                                                   double aCoeffs[3][3]) {
  if (!srcQuad || !dstQuad || !aCoeffs) {
    return NPP_NULL_POINTER_ERROR;
  }
  double augmented[8][9] = {};
  for (int i = 0; i < 4; ++i) {
    double x = dstQuad[i][0];
    double y = dstQuad[i][1];
    double u = srcQuad[i][0];
    double v = srcQuad[i][1];

    augmented[2 * i][0] = x;
    augmented[2 * i][1] = y;
    augmented[2 * i][2] = 1.0;
    augmented[2 * i][6] = -x * u;
    augmented[2 * i][7] = -y * u;
    augmented[2 * i][8] = u;

    augmented[2 * i + 1][3] = x;
    augmented[2 * i + 1][4] = y;
    augmented[2 * i + 1][5] = 1.0;
    augmented[2 * i + 1][6] = -x * v;
    augmented[2 * i + 1][7] = -y * v;
    augmented[2 * i + 1][8] = v;
  }

  double solution[8] = {};
  if (!solveLinearSystem8x8(augmented, solution)) {
    return NPP_COEFFICIENT_ERROR;
  }

  aCoeffs[0][0] = solution[0];
  aCoeffs[0][1] = solution[1];
  aCoeffs[0][2] = solution[2];
  aCoeffs[1][0] = solution[3];
  aCoeffs[1][1] = solution[4];
  aCoeffs[1][2] = solution[5];
  aCoeffs[2][0] = solution[6];
  aCoeffs[2][1] = solution[7];
  aCoeffs[2][2] = 1.0;
  return NPP_SUCCESS;
}

static NppStatus validateWarpPerspectiveBatchInputs(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
                                                    int eInterpolation, const NppiWarpPerspectiveBatchCXR *pBatchList,
                                                    unsigned int nBatchSize) {
  if (!pBatchList) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nBatchSize <= 1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }
  if (oSmallestSrcSize.width <= 0 || oSmallestSrcSize.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (oSrcRectROI.width <= 0 || oSrcRectROI.height <= 0 || oDstRectROI.width <= 0 || oDstRectROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (oSrcRectROI.x < 0 || oSrcRectROI.y < 0 || oSrcRectROI.x + oSrcRectROI.width > oSmallestSrcSize.width ||
      oSrcRectROI.y + oSrcRectROI.height > oSmallestSrcSize.height) {
    return NPP_RECTANGLE_ERROR;
  }
  if (eInterpolation != NPPI_INTER_NN && eInterpolation != NPPI_INTER_LINEAR && eInterpolation != NPPI_INTER_CUBIC) {
    return NPP_INTERPOLATION_ERROR;
  }
  return NPP_SUCCESS;
}

static NppStatus copyWarpPerspectiveBatchListFromDevice(std::vector<NppiWarpPerspectiveBatchCXR> &batchHost,
                                                        const NppiWarpPerspectiveBatchCXR *pBatchList,
                                                        unsigned int nBatchSize) {
  batchHost.resize(nBatchSize);
  cudaError_t cudaStatus =
      cudaMemcpy(batchHost.data(), pBatchList, sizeof(NppiWarpPerspectiveBatchCXR) * nBatchSize, cudaMemcpyDeviceToHost);
  return cudaStatus == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

static NppStatus copyWarpPerspectiveBatchListToDevice(const std::vector<NppiWarpPerspectiveBatchCXR> &batchHost,
                                                      NppiWarpPerspectiveBatchCXR *pBatchList) {
  cudaError_t cudaStatus =
      cudaMemcpy(pBatchList, batchHost.data(), sizeof(NppiWarpPerspectiveBatchCXR) * batchHost.size(), cudaMemcpyHostToDevice);
  return cudaStatus == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <typename WarpFn>
static NppStatus warpPerspectiveBatchCommon(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
                                            int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
                                            unsigned int nBatchSize, NppStreamContext nppStreamCtx, WarpFn warpFn) {
  NppStatus status =
      validateWarpPerspectiveBatchInputs(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize);
  if (status != NPP_SUCCESS) {
    return status;
  }

  std::vector<NppiWarpPerspectiveBatchCXR> batchHost;
  status = copyWarpPerspectiveBatchListFromDevice(batchHost, pBatchList, nBatchSize);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (const auto &entry : batchHost) {
    double aCoeffs[3][3];
    std::memcpy(aCoeffs, entry.aTransformedCoeffs, sizeof(aCoeffs));
    status = warpFn(entry.pSrc, oSmallestSrcSize, entry.nSrcStep, oSrcRectROI, entry.pDst, entry.nDstStep, oDstRectROI,
                    aCoeffs, eInterpolation, nppStreamCtx);
    if (status != NPP_SUCCESS) {
      return status;
    }
  }

  return NPP_SUCCESS;
}

NppStatus nppiWarpPerspective_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_8u_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_8u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_8u_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_32f_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_32f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_16u_C1R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_16u_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_16u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_16u_C3R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_16u_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_16u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_16u_C4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_16u_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_16u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, nppStreamCtx);
}

// 32f C3R
NppStatus nppiWarpPerspective_32f_C3R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_32f_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_32f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32f_C4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_32f_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_32f_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32s_C1R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_32s_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32s_C1R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_32s_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32s_C3R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_32s_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32s_C3R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_32s_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32s_C4R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_32s_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32s_C4R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_32s_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_8u_C4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_8u_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_8u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

// ============================================================================
// nppiWarpPerspectiveBack API
// ============================================================================
extern "C" {

NppStatus nppiWarpPerspectiveBack_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp8u *pDst, int nDstStep, NppiRect oDstROI,
                                                  const double aCoeffs[3][3], int eInterpolation,
                                                  NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspectiveBack_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp8u *pDst, int nDstStep, NppiRect oDstROI,
                                                  const double aCoeffs[3][3], int eInterpolation,
                                                  NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspectiveBack_8u_C4R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                  Npp8u *pDst, int nDstStep, NppiRect oDstROI,
                                                  const double aCoeffs[3][3], int eInterpolation,
                                                  NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspectiveBack_16u_C1R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspectiveBack_16u_C3R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspectiveBack_16u_C4R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp16u *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspectiveBack_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspectiveBack_32f_C3R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspectiveBack_32f_C4R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32f *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspectiveBack_32s_C1R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspectiveBack_32s_C3R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspectiveBack_32s_C4R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                   NppiRect oSrcROI, Npp32s *pDst, int nDstStep, NppiRect oDstROI,
                                                   const double aCoeffs[3][3], int eInterpolation,
                                                   NppStreamContext nppStreamCtx);
}

// 8u C1R
NppStatus nppiWarpPerspectiveBack_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                             int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_8u_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                 eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_8u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                            eInterpolation, nppStreamCtx);
}

// 8u C3R
NppStatus nppiWarpPerspectiveBack_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                             int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_8u_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                 eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                            eInterpolation, nppStreamCtx);
}

// 8u C4R
NppStatus nppiWarpPerspectiveBack_8u_C4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                             int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_8u_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                 eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_8u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                            eInterpolation, nppStreamCtx);
}

// 16u C1R
NppStatus nppiWarpPerspectiveBack_16u_C1R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_16u_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                  eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_16u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

// 16u C3R
NppStatus nppiWarpPerspectiveBack_16u_C3R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_16u_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                  eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_16u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

// 16u C4R
NppStatus nppiWarpPerspectiveBack_16u_C4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_16u_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                  eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_16u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

// 32f C1R
NppStatus nppiWarpPerspectiveBack_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_32f_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                  eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_32f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

// 32f C3R
NppStatus nppiWarpPerspectiveBack_32f_C3R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_32f_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                  eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_32f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

// 32f C4R
NppStatus nppiWarpPerspectiveBack_32f_C4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_32f_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                  eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_32f_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

// 32s C1R
NppStatus nppiWarpPerspectiveBack_32s_C1R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_32s_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                  eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_32s_C1R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_32s_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

// 32s C3R
NppStatus nppiWarpPerspectiveBack_32s_C3R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_32s_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                  eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_32s_C3R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_32s_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

// 32s C4R
NppStatus nppiWarpPerspectiveBack_32s_C4R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspectiveBack_32s_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                  eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspectiveBack_32s_C4R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspectiveBack_32s_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_16f_C1R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiWarpPerspective_16f_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_16f_C1R(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return nppiWarpPerspective_16f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, getDefaultStreamContextZero());
}

NppStatus nppiWarpPerspective_16f_C3R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiWarpPerspective_16f_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_16f_C3R(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return nppiWarpPerspective_16f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, getDefaultStreamContextZero());
}

NppStatus nppiWarpPerspective_16f_C4R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiWarpPerspective_16f_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_16f_C4R(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return nppiWarpPerspective_16f_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, getDefaultStreamContextZero());
}

#define DEFINE_WARP_PERSPECTIVE_AC4(TYPE, SUFFIX, MERGE_FN)                                                                     \
  NppStatus nppiWarpPerspective_##SUFFIX##_AC4R_Ctx(const TYPE *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,      \
                                                    TYPE *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],    \
                                                    int eInterpolation, NppStreamContext nppStreamCtx) {                       \
    return warpPerspectiveAC4Common(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs, eInterpolation,     \
                                    nppStreamCtx, nppiWarpPerspective_##SUFFIX##_C4R_Ctx, MERGE_FN);                          \
  }                                                                                                                            \
  NppStatus nppiWarpPerspective_##SUFFIX##_AC4R(const TYPE *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,         \
                                                TYPE *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],       \
                                                int eInterpolation) {                                                          \
    return nppiWarpPerspective_##SUFFIX##_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,      \
                                                   eInterpolation, getDefaultStreamContextZero());                             \
  }

#define DEFINE_WARP_PERSPECTIVE_PLANAR(TYPE, SUFFIX, PLANES)                                                                    \
  NppStatus nppiWarpPerspective_##SUFFIX##_P##PLANES##R_Ctx(const TYPE *pSrc[PLANES], NppiSize oSrcSize, int nSrcStep,       \
                                                            NppiRect oSrcROI, TYPE *pDst[PLANES], int nDstStep,                \
                                                            NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation,  \
                                                            NppStreamContext nppStreamCtx) {                                    \
    return warpPerspectivePlanarCommon(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,                  \
                                       eInterpolation, PLANES, nppStreamCtx, nppiWarpPerspective_##SUFFIX##_C1R_Ctx);         \
  }                                                                                                                             \
  NppStatus nppiWarpPerspective_##SUFFIX##_P##PLANES##R(const TYPE *pSrc[PLANES], NppiSize oSrcSize, int nSrcStep,           \
                                                        NppiRect oSrcROI, TYPE *pDst[PLANES], int nDstStep,                    \
                                                        NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation) {    \
    return nppiWarpPerspective_##SUFFIX##_P##PLANES##R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI,       \
                                                           aCoeffs, eInterpolation, getDefaultStreamContextZero());            \
  }

DEFINE_WARP_PERSPECTIVE_AC4(Npp8u, 8u, nppiMergeAlpha_8u_C4R)
DEFINE_WARP_PERSPECTIVE_AC4(Npp16u, 16u, nppiMergeAlpha_16u_C4R)
DEFINE_WARP_PERSPECTIVE_AC4(Npp32f, 32f, nppiMergeAlpha_32f_C4R)
DEFINE_WARP_PERSPECTIVE_AC4(Npp32s, 32s, nppiMergeAlpha_32s_C4R)
DEFINE_WARP_PERSPECTIVE_PLANAR(Npp8u, 8u, 3)
DEFINE_WARP_PERSPECTIVE_PLANAR(Npp8u, 8u, 4)
DEFINE_WARP_PERSPECTIVE_PLANAR(Npp16u, 16u, 3)
DEFINE_WARP_PERSPECTIVE_PLANAR(Npp16u, 16u, 4)
DEFINE_WARP_PERSPECTIVE_PLANAR(Npp32f, 32f, 3)
DEFINE_WARP_PERSPECTIVE_PLANAR(Npp32f, 32f, 4)
DEFINE_WARP_PERSPECTIVE_PLANAR(Npp32s, 32s, 3)
DEFINE_WARP_PERSPECTIVE_PLANAR(Npp32s, 32s, 4)

#undef DEFINE_WARP_PERSPECTIVE_AC4
#undef DEFINE_WARP_PERSPECTIVE_PLANAR

#define DEFINE_WARP_PERSPECTIVE_BACK_AC4(TYPE, SUFFIX, MERGE_FN)                                                                \
  NppStatus nppiWarpPerspectiveBack_##SUFFIX##_AC4R_Ctx(const TYPE *pSrc, NppiSize oSrcSize, int nSrcStep,                    \
                                                        NppiRect oSrcROI, TYPE *pDst, int nDstStep, NppiRect oDstROI,          \
                                                        const double aCoeffs[3][3], int eInterpolation,                        \
                                                        NppStreamContext nppStreamCtx) {                                        \
    return warpPerspectiveAC4Common(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs, eInterpolation,     \
                                    nppStreamCtx, nppiWarpPerspectiveBack_##SUFFIX##_C4R_Ctx, MERGE_FN);                      \
  }                                                                                                                             \
  NppStatus nppiWarpPerspectiveBack_##SUFFIX##_AC4R(const TYPE *pSrc, NppiSize oSrcSize, int nSrcStep,                        \
                                                    NppiRect oSrcROI, TYPE *pDst, int nDstStep, NppiRect oDstROI,             \
                                                    const double aCoeffs[3][3], int eInterpolation) {                          \
    return nppiWarpPerspectiveBack_##SUFFIX##_AC4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,  \
                                                       eInterpolation, getDefaultStreamContextZero());                         \
  }

#define DEFINE_WARP_PERSPECTIVE_BACK_PLANAR(TYPE, SUFFIX, PLANES)                                                               \
  NppStatus nppiWarpPerspectiveBack_##SUFFIX##_P##PLANES##R_Ctx(const TYPE *pSrc[PLANES], NppiSize oSrcSize, int nSrcStep,    \
                                                                NppiRect oSrcROI, TYPE *pDst[PLANES], int nDstStep,            \
                                                                NppiRect oDstROI, const double aCoeffs[3][3],                  \
                                                                int eInterpolation, NppStreamContext nppStreamCtx) {           \
    return warpPerspectivePlanarCommon(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,                  \
                                       eInterpolation, PLANES, nppStreamCtx, nppiWarpPerspectiveBack_##SUFFIX##_C1R_Ctx);     \
  }                                                                                                                             \
  NppStatus nppiWarpPerspectiveBack_##SUFFIX##_P##PLANES##R(const TYPE *pSrc[PLANES], NppiSize oSrcSize, int nSrcStep,        \
                                                            NppiRect oSrcROI, TYPE *pDst[PLANES], int nDstStep,                \
                                                            NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation) {\
    return nppiWarpPerspectiveBack_##SUFFIX##_P##PLANES##R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI,   \
                                                               aCoeffs, eInterpolation, getDefaultStreamContextZero());        \
  }

DEFINE_WARP_PERSPECTIVE_BACK_AC4(Npp8u, 8u, nppiMergeAlpha_8u_C4R)
DEFINE_WARP_PERSPECTIVE_BACK_AC4(Npp16u, 16u, nppiMergeAlpha_16u_C4R)
DEFINE_WARP_PERSPECTIVE_BACK_AC4(Npp32f, 32f, nppiMergeAlpha_32f_C4R)
DEFINE_WARP_PERSPECTIVE_BACK_AC4(Npp32s, 32s, nppiMergeAlpha_32s_C4R)
DEFINE_WARP_PERSPECTIVE_BACK_PLANAR(Npp8u, 8u, 3)
DEFINE_WARP_PERSPECTIVE_BACK_PLANAR(Npp8u, 8u, 4)
DEFINE_WARP_PERSPECTIVE_BACK_PLANAR(Npp16u, 16u, 3)
DEFINE_WARP_PERSPECTIVE_BACK_PLANAR(Npp16u, 16u, 4)
DEFINE_WARP_PERSPECTIVE_BACK_PLANAR(Npp32f, 32f, 3)
DEFINE_WARP_PERSPECTIVE_BACK_PLANAR(Npp32f, 32f, 4)
DEFINE_WARP_PERSPECTIVE_BACK_PLANAR(Npp32s, 32s, 3)
DEFINE_WARP_PERSPECTIVE_BACK_PLANAR(Npp32s, 32s, 4)

#undef DEFINE_WARP_PERSPECTIVE_BACK_AC4
#undef DEFINE_WARP_PERSPECTIVE_BACK_PLANAR

#define DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(TYPE, SUFFIX, VARIANT, BASE_FN)                                                    \
  NppStatus nppiWarpPerspectiveQuad_##SUFFIX##_##VARIANT##_Ctx(const TYPE *pSrc, NppiSize oSrcSize, int nSrcStep,            \
                                                               NppiRect oSrcROI, const double aSrcQuad[4][2], TYPE *pDst,      \
                                                               int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2],     \
                                                               int eInterpolation, NppStreamContext nppStreamCtx) {             \
    double aCoeffs[3][3];                                                                                                       \
    NppStatus status = computePerspectiveCoeffsFromQuads(aSrcQuad, aDstQuad, aCoeffs);                                        \
    if (status != NPP_SUCCESS) {                                                                                                \
      return status;                                                                                                            \
    }                                                                                                                           \
    return BASE_FN(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs, eInterpolation, nppStreamCtx);       \
  }                                                                                                                             \
  NppStatus nppiWarpPerspectiveQuad_##SUFFIX##_##VARIANT(const TYPE *pSrc, NppiSize oSrcSize, int nSrcStep,                  \
                                                         NppiRect oSrcROI, const double aSrcQuad[4][2], TYPE *pDst,            \
                                                         int nDstStep, NppiRect oDstROI, const double aDstQuad[4][2],          \
                                                         int eInterpolation) {                                                  \
    return nppiWarpPerspectiveQuad_##SUFFIX##_##VARIANT##_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep,   \
                                                              oDstROI, aDstQuad, eInterpolation, getDefaultStreamContextZero());\
  }

#define DEFINE_WARP_PERSPECTIVE_QUAD_PLANAR(TYPE, SUFFIX, PLANES)                                                               \
  NppStatus nppiWarpPerspectiveQuad_##SUFFIX##_P##PLANES##R_Ctx(const TYPE *pSrc[PLANES], NppiSize oSrcSize, int nSrcStep,   \
                                                                NppiRect oSrcROI, const double aSrcQuad[4][2],                 \
                                                                TYPE *pDst[PLANES], int nDstStep, NppiRect oDstROI,            \
                                                                const double aDstQuad[4][2], int eInterpolation,               \
                                                                NppStreamContext nppStreamCtx) {                               \
    double aCoeffs[3][3];                                                                                                       \
    NppStatus status = computePerspectiveCoeffsFromQuads(aSrcQuad, aDstQuad, aCoeffs);                                        \
    if (status != NPP_SUCCESS) {                                                                                                \
      return status;                                                                                                            \
    }                                                                                                                           \
    return nppiWarpPerspective_##SUFFIX##_P##PLANES##R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI,       \
                                                           aCoeffs, eInterpolation, nppStreamCtx);                            \
  }                                                                                                                             \
  NppStatus nppiWarpPerspectiveQuad_##SUFFIX##_P##PLANES##R(const TYPE *pSrc[PLANES], NppiSize oSrcSize, int nSrcStep,       \
                                                            NppiRect oSrcROI, const double aSrcQuad[4][2],                    \
                                                            TYPE *pDst[PLANES], int nDstStep, NppiRect oDstROI,               \
                                                            const double aDstQuad[4][2], int eInterpolation) {                \
    return nppiWarpPerspectiveQuad_##SUFFIX##_P##PLANES##R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, aSrcQuad, pDst, nDstStep,  \
                                                               oDstROI, aDstQuad, eInterpolation, getDefaultStreamContextZero());\
  }

DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp8u, 8u, C1R, nppiWarpPerspective_8u_C1R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp8u, 8u, C3R, nppiWarpPerspective_8u_C3R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp8u, 8u, C4R, nppiWarpPerspective_8u_C4R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp8u, 8u, AC4R, nppiWarpPerspective_8u_AC4R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PLANAR(Npp8u, 8u, 3)
DEFINE_WARP_PERSPECTIVE_QUAD_PLANAR(Npp8u, 8u, 4)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp16u, 16u, C1R, nppiWarpPerspective_16u_C1R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp16u, 16u, C3R, nppiWarpPerspective_16u_C3R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp16u, 16u, C4R, nppiWarpPerspective_16u_C4R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp16u, 16u, AC4R, nppiWarpPerspective_16u_AC4R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PLANAR(Npp16u, 16u, 3)
DEFINE_WARP_PERSPECTIVE_QUAD_PLANAR(Npp16u, 16u, 4)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp32s, 32s, C1R, nppiWarpPerspective_32s_C1R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp32s, 32s, C3R, nppiWarpPerspective_32s_C3R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp32s, 32s, C4R, nppiWarpPerspective_32s_C4R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp32s, 32s, AC4R, nppiWarpPerspective_32s_AC4R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PLANAR(Npp32s, 32s, 3)
DEFINE_WARP_PERSPECTIVE_QUAD_PLANAR(Npp32s, 32s, 4)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp32f, 32f, C1R, nppiWarpPerspective_32f_C1R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp32f, 32f, C3R, nppiWarpPerspective_32f_C3R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp32f, 32f, C4R, nppiWarpPerspective_32f_C4R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PACKED(Npp32f, 32f, AC4R, nppiWarpPerspective_32f_AC4R_Ctx)
DEFINE_WARP_PERSPECTIVE_QUAD_PLANAR(Npp32f, 32f, 3)
DEFINE_WARP_PERSPECTIVE_QUAD_PLANAR(Npp32f, 32f, 4)

#undef DEFINE_WARP_PERSPECTIVE_QUAD_PACKED
#undef DEFINE_WARP_PERSPECTIVE_QUAD_PLANAR

NppStatus nppiWarpPerspectiveBatchInit_Ctx(NppiWarpPerspectiveBatchCXR *pBatchList, unsigned int nBatchSize,
                                           NppStreamContext) {
  if (!pBatchList) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nBatchSize <= 1) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  std::vector<NppiWarpPerspectiveBatchCXR> batchHost;
  NppStatus status = copyWarpPerspectiveBatchListFromDevice(batchHost, pBatchList, nBatchSize);
  if (status != NPP_SUCCESS) {
    return status;
  }

  for (auto &entry : batchHost) {
    if (!entry.pCoeffs) {
      return NPP_NULL_POINTER_ERROR;
    }
    cudaError_t cudaStatus =
        cudaMemcpy(entry.aTransformedCoeffs, entry.pCoeffs, sizeof(entry.aTransformedCoeffs), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
  }

  return copyWarpPerspectiveBatchListToDevice(batchHost, pBatchList);
}

NppStatus nppiWarpPerspectiveBatchInit(NppiWarpPerspectiveBatchCXR *pBatchList, unsigned int nBatchSize) {
  return nppiWarpPerspectiveBatchInit_Ctx(pBatchList, nBatchSize, getDefaultStreamContextZero());
}

#define DEFINE_WARP_PERSPECTIVE_BATCH(TYPE, SUFFIX, VARIANT, FN)                                                                \
  NppStatus nppiWarpPerspectiveBatch_##SUFFIX##_##VARIANT##_Ctx(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI,              \
                                                                NppiRect oDstRectROI, int eInterpolation,                      \
                                                                NppiWarpPerspectiveBatchCXR *pBatchList,                       \
                                                                unsigned int nBatchSize, NppStreamContext nppStreamCtx) {      \
    auto typedFn = [](const void *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, void *pDst, int nDstStep,          \
                      NppiRect oDstROI, const double aCoeffs[3][3], int eInterp, NppStreamContext ctx) -> NppStatus {         \
      return FN(reinterpret_cast<const TYPE *>(pSrc), oSrcSize, nSrcStep, oSrcROI, reinterpret_cast<TYPE *>(pDst), nDstStep,  \
                oDstROI, aCoeffs, eInterp, ctx);                                                                               \
    };                                                                                                                          \
    return warpPerspectiveBatchCommon(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation, pBatchList, nBatchSize,     \
                                      nppStreamCtx, typedFn);                                                                   \
  }                                                                                                                             \
  NppStatus nppiWarpPerspectiveBatch_##SUFFIX##_##VARIANT(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI,                    \
                                                          NppiRect oDstRectROI, int eInterpolation,                            \
                                                          NppiWarpPerspectiveBatchCXR *pBatchList, unsigned int nBatchSize) {   \
    return nppiWarpPerspectiveBatch_##SUFFIX##_##VARIANT##_Ctx(oSmallestSrcSize, oSrcRectROI, oDstRectROI, eInterpolation,    \
                                                               pBatchList, nBatchSize, getDefaultStreamContextZero());         \
  }

DEFINE_WARP_PERSPECTIVE_BATCH(Npp8u, 8u, C1R, nppiWarpPerspective_8u_C1R_Ctx)
DEFINE_WARP_PERSPECTIVE_BATCH(Npp8u, 8u, C3R, nppiWarpPerspective_8u_C3R_Ctx)
DEFINE_WARP_PERSPECTIVE_BATCH(Npp8u, 8u, C4R, nppiWarpPerspective_8u_C4R_Ctx)
DEFINE_WARP_PERSPECTIVE_BATCH(Npp8u, 8u, AC4R, nppiWarpPerspective_8u_AC4R_Ctx)
DEFINE_WARP_PERSPECTIVE_BATCH(Npp16f, 16f, C1R, nppiWarpPerspective_16f_C1R_Ctx)
DEFINE_WARP_PERSPECTIVE_BATCH(Npp16f, 16f, C3R, nppiWarpPerspective_16f_C3R_Ctx)
DEFINE_WARP_PERSPECTIVE_BATCH(Npp16f, 16f, C4R, nppiWarpPerspective_16f_C4R_Ctx)
DEFINE_WARP_PERSPECTIVE_BATCH(Npp32f, 32f, C1R, nppiWarpPerspective_32f_C1R_Ctx)
DEFINE_WARP_PERSPECTIVE_BATCH(Npp32f, 32f, C3R, nppiWarpPerspective_32f_C3R_Ctx)
DEFINE_WARP_PERSPECTIVE_BATCH(Npp32f, 32f, C4R, nppiWarpPerspective_32f_C4R_Ctx)
DEFINE_WARP_PERSPECTIVE_BATCH(Npp32f, 32f, AC4R, nppiWarpPerspective_32f_AC4R_Ctx)

#undef DEFINE_WARP_PERSPECTIVE_BATCH
