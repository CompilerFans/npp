#include "npp.h"
#include <cstdio>
#include <cuda_runtime.h>































/**
 * Three-channel 8-bit unsigned HistogramEven.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                       int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                       NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3], int nLevels[3],
                                   Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 8-bit unsigned HistogramEven ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                        int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}
NppStatus nppiHistogramEven_8u_AC4R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                    int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}


/**
 * Three-channel 16-bit unsigned HistogramEven.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                        int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_16u_C3R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                    int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit unsigned HistogramEven ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_16u_AC4R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 16-bit signed HistogramEven.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                        int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_16s_C3R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                    int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit signed HistogramEven ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_16s_AC4R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/** @} */

/** @} image_histogrameven */

/** @defgroup image_histogramrange HistogramRange
 * Primitives for computing the histogram of an image within specified ranges.
 * @{
 *
 */

/** @name HistogramEven
 * The histogram is computed according to the ranges provided in \f$pLevels\f$.
 * The histogram \f$pHist[k]\f$ is defined as the total number of pixels that fall into the range:
 * \f$pLevels[k] <= pSrc(j, i) < pLevels[k+1]\f$. The number of the histogram bins is \f$nLevel - 1\f$.
 * The functions require additional scratch buffer for computations.
 * <h3><a name="CommonHistogramEvenParameters">Common parameters for nppiHistogramEven functions include:</a></h3>
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Pointer to array that receives the computed histogram.
 *      The array must be of size nLevels-1.
 * \param pLevels Pointer to array containing the level sizes of the bins.
        The array must be of size nLevels.
 * \param nLevels Number of levels in histogram.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_XX_XXX)
 *      scratch buffer.
 * \param nppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 * @{
 */

/**
 * Three-channel 8-bit unsigned HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                        const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                    const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}


/**
 * Four-channel 8-bit unsigned HistogramRange ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_8u_AC4R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 16-bit unsigned HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16u_C3R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit unsigned HistogramRange ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                          const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                          NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16u_AC4R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                      const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}


/**
 * Three-channel 16-bit signed HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16s_C3R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit signed HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                          const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                          NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16s_AC4R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                      const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 32-bit floating point HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         const Npp32f *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_32f_C3R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     const Npp32f *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}


/**
 * Four-channel 32-bit floating point HistogramRange ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                          const Npp32f *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                          NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_32f_AC4R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                      const Npp32f *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}
