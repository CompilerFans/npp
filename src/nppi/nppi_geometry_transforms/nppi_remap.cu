#include "npp.h"
#include "nppi_warp_sampling.cuh"
#include <cuda_runtime.h>

// Remap kernel for all supported layouts.  The measured NPP 13.4 contract:
//  * the map arrays are device memory of Npp32f (Npp64f for 64f images) and
//    supply the source coordinate of each destination pixel;
//  * nearest-neighbour rounds with floor(u + 0.5);
//  * linear/cubic use the shared warp sampler (taps clamped to the image,
//    float loads, double accumulation, integer round-half-away + saturate);
//  * source coordinates outside [roi.x - 0.5, roi.x + roi.w) x
//    [roi.y - 0.5, roi.y + roi.h) leave the destination pixel untouched;
//  * AC4 copies only the three colour channels and preserves dst alpha.
template <typename T, typename MapT>
__global__ void nppiRemap_kernel(const T *pSrc, int nSrcStepEl, int srcW, int srcH, NppiRect oSrcROI,
                                 const MapT *pXMap, int nXMapStep, const MapT *pYMap, int nYMapStep, T *pDst,
                                 int nDstStepEl, int width, int height, int channels, bool write_alpha,
                                 int interp) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const MapT *xmap_row = (const MapT *)((const char *)pXMap + (size_t)y * nXMapStep);
  const MapT *ymap_row = (const MapT *)((const char *)pYMap + (size_t)y * nYMapStep);
  const double u = (double)xmap_row[x];
  const double v = (double)ymap_row[x];
  if (!(u >= (double)oSrcROI.x - 0.5 && u < (double)(oSrcROI.x + oSrcROI.width) &&
        v >= (double)oSrcROI.y - 0.5 && v < (double)(oSrcROI.y + oSrcROI.height)))
    return;

  T *dstRow = (T *)((char *)pDst + (size_t)y * nDstStepEl * (int)sizeof(T));
  const int writeCh = write_alpha ? channels : (channels > 1 ? channels - 1 : 1);
  for (int c = 0; c < writeCh; ++c) {
    dstRow[(size_t)x * channels + c] =
        nppi::warp::sampleChannel<T>(pSrc + c, nSrcStepEl, channels, srcW, srcH, u, v, interp);
  }
}

template <typename T, typename MapT>
static NppStatus nppiRemap_CnR_Ctx_impl(const T *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                        const MapT *pXMap, int nXMapStep, const MapT *pYMap, int nYMapStep, T *pDst,
                                        int nDstStep, NppiSize oDstSizeROI, int eInterpolation,
                                        NppStreamContext nppStreamCtx, int channels, bool write_alpha) {
  if (pSrc == nullptr || pDst == nullptr || pXMap == nullptr || pYMap == nullptr) return NPP_NULL_POINTER_ERROR;
  if (eInterpolation != NPPI_INTER_NN && eInterpolation != NPPI_INTER_LINEAR &&
      eInterpolation != NPPI_INTER_CUBIC)
    return NPP_INTERPOLATION_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0 || oDstSizeROI.width <= 0 || oDstSizeROI.height <= 0)
    return NPP_SIZE_ERROR;
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0) return NPP_SIZE_ERROR;
  if (nSrcStep < oSrcSize.width * channels * (int)sizeof(T)) return NPP_STEP_ERROR;
  if (nDstStep < oDstSizeROI.width * channels * (int)sizeof(T)) return NPP_STEP_ERROR;

  dim3 blockSize(16, 16);
  dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);
  nppiRemap_kernel<T, MapT><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep / (int)sizeof(T), oSrcSize.width, oSrcSize.height, oSrcROI, pXMap, nXMapStep, pYMap, nYMapStep,
      pDst, nDstStep / (int)sizeof(T), oDstSizeROI.width, oDstSizeROI.height, channels, write_alpha, eInterpolation);
  return cudaGetLastError() == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
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
