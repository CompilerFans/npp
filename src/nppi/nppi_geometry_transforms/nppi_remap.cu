#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Generic remap kernel template for different data types
template <typename T, typename MapT>
__global__ void nppiRemap_nearest_kernel(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         const MapT *pXMap, int nXMapStep, const MapT *pYMap, int nYMapStep, T *pDst,
                                         int nDstStep, int width, int height, int channels, bool write_alpha) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  // Get mapping coordinates from lookup tables
  const MapT *xmap_row = (const MapT *)((const char *)pXMap + y * nXMapStep);
  const MapT *ymap_row = (const MapT *)((const char *)pYMap + y * nYMapStep);

  double src_x = static_cast<double>(xmap_row[x]);
  double src_y = static_cast<double>(ymap_row[x]);

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
  } else if (channels == 4) {
    int src_idx = ix * 4;
    int dst_idx = x * 4;
    dst_row[dst_idx + 0] = src_row[src_idx + 0];
    dst_row[dst_idx + 1] = src_row[src_idx + 1];
    dst_row[dst_idx + 2] = src_row[src_idx + 2];
    if (write_alpha) {
      dst_row[dst_idx + 3] = src_row[src_idx + 3];
    }
  }
}

template <typename T, typename MapT>
static NppStatus nppiRemap_CnR_Ctx_impl(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                        const MapT *pXMap, int nXMapStep, const MapT *pYMap, int nYMapStep, T *pDst,
                                        int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                        NppStreamContext nppStreamCtx, int channels, bool write_alpha) {
  (void)eInterpolation;
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRemap_nearest_kernel<T, MapT><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep, pDst, nDstStep, oDstSizeROI.width,
      oDstSizeROI.height, channels, write_alpha);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

extern "C" {

// 8-bit unsigned single channel implementation
NppStatus nppiRemap_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                                    int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                    NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp8u, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                              pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 1, true);
}

// 8-bit unsigned three channel implementation
NppStatus nppiRemap_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                                    int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                    NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp8u, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                              pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 3, true);
}

// 8-bit unsigned four channel implementation
NppStatus nppiRemap_8u_C4R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                                    int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                    NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp8u, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                              pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 4, true);
}

// 8-bit unsigned AC4 implementation (same as C4 for nearest neighbor)
NppStatus nppiRemap_8u_AC4R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp8u, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                              pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 4, false);
}

// 16-bit unsigned implementations
NppStatus nppiRemap_16u_C1R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp16u, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 1, true);
}

NppStatus nppiRemap_16u_C3R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp16u, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 3, true);
}

NppStatus nppiRemap_16u_C4R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp16u, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 4, true);
}

NppStatus nppiRemap_16u_AC4R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                      Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                      NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp16u, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 4, false);
}

// 16-bit signed implementations
NppStatus nppiRemap_16s_C1R_Ctx_impl(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp16s, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 1, true);
}

NppStatus nppiRemap_16s_C3R_Ctx_impl(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp16s, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 3, true);
}

NppStatus nppiRemap_16s_C4R_Ctx_impl(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp16s, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 4, true);
}

NppStatus nppiRemap_16s_AC4R_Ctx_impl(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                      Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                      NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp16s, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 4, false);
}

NppStatus nppiRemap_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp32f, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 1, true);
}

NppStatus nppiRemap_32f_C3R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp32f, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 3, true);
}

NppStatus nppiRemap_32f_C4R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                     Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp32f, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 4, true);
}

NppStatus nppiRemap_32f_AC4R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                                      Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                      NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp32f, Npp32f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 4, false);
}

// 64-bit double implementations
NppStatus nppiRemap_64f_C1R_Ctx_impl(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                     Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp64f, Npp64f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 1, true);
}

NppStatus nppiRemap_64f_C3R_Ctx_impl(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                     Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp64f, Npp64f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 3, true);
}

NppStatus nppiRemap_64f_C4R_Ctx_impl(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                     Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp64f, Npp64f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 4, true);
}

NppStatus nppiRemap_64f_AC4R_Ctx_impl(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      const Npp64f *pXMap, int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                                      Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                      NppStreamContext nppStreamCtx) {
  return nppiRemap_CnR_Ctx_impl<Npp64f, Npp64f>(pSrc, oSrcSize, nSrcStep, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
                                               pDst, nDstStep, oDstSizeROI, eInterpolation, nppStreamCtx, 4, false);
}
}
