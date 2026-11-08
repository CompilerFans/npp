#ifndef NPP_API_EXTRA_12_2_H
#define NPP_API_EXTRA_12_2_H

/* Superset declarations for the API-12.2 header set. */
#include <npp.h>
#include "npp_api_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

NppStatus nppiCFAToRGBA_32u_C1AC4R_Ctx(const Npp32u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI, Npp32u * pDst, int nDstStep, NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation, Npp32u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiCFAToRGB_32u_C1C3R_Ctx(const Npp32u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI, Npp32u * pDst, int nDstStep, NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiCircularRadialProfile_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, NppiPoint32f oCenter, NppiProfileData *pProfileData, Npp16u nProfileSamples, NppStreamContext nppStreamCtx);
NppStatus nppiEllipticalRadialProfile_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, NppiPoint32f oCenter, Npp32f nAstigmatismRatio, Npp32f nOrientationRadians, NppiProfileData *pProfileData, Npp16u nProfileSamples, NppStreamContext nppStreamCtx);
NppStatus nppiFilterCannyBorder_8u_C3C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize, Npp16s nLowThreshold, Npp16s nHighThreshold, NppiNorm eNorm, NppiBorderType eBorderType, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);
NppStatus nppiI444_8u_ToUYVP_10u_P3C3R_Ctx(const Npp8u* const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToBGR_601_8u_P2C3R_Ctx(const Npp8u * const pSrc[2], int rSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToBGR_8u_P2C4R_Ctx(const Npp8u* const pSrc[2], int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp8u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToBGR_BT2020_8u_P2C3R_Ctx(const Npp8u* const pSrc[2], int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx(const Npp16u * const pSrc[2], int aSrcStep[2], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToRGB_601_8u_P2C3R_Ctx(const Npp8u * const pSrc[2], int rSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(const Npp8u * const pSrc[2], int aSrcStep[2], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToRGB_8u_P2C4R_Ctx(const Npp8u* const pSrc[2], int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp8u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToRGB_BT2020_8u_P2C3R_Ctx(const Npp8u* const pSrc[2], int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiNV12_8u_ToUYVP_10u_P2C3R_Ctx(const Npp8u* const pSrc[2], const int nSrcStep[2], NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiNV16_8u_ToUYVP_10u_P2C3R_Ctx(const Npp8u* const pSrc[2], const int nSrcStep[2], NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiP010ToRGB_16u_P2C3R_Ctx(const Npp16u* const pSrc[2], int nSrcStep, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiP010ToRGB_16u_P2C4R_Ctx(const Npp16u* const pSrc[2], int nSrcStep, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp16u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiP010ToRGB_8u_P2C3R_Ctx(const Npp16u* const pSrc[2], int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiP010ToRGB_8u_P2C4R_Ctx(const Npp16u* const pSrc[2], int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp8u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiP010_16u_ToUYVP_10u_P3C3R_Ctx(const Npp16u* const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiP016ToRGB_16u_P2C3R_Ctx(const Npp16u* const pSrc[2], int nSrcStep, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiP016ToRGB_16u_P2C4R_Ctx(const Npp16u* const pSrc[2], int nSrcStep, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp16u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiP208_8u_ToUYVP_10u_C3C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiP210_16u_ToUYVP_10u_P2C3R_Ctx(const Npp16u* const pSrc[2], const int nSrcStep[2], NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiRGBA_16f_ToUYVP_10u_AC4C3R_Ctx(const Npp16f* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGBA_16u_ToUYVP_10u_AC4C3R_Ctx(const Npp16u* const pSrc, const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGBA_16u_ToUYVY16_16u_AC4C3R_Ctx(const Npp16u* const pSrc, const int nSrcStep, NppiPoint oSrcOffset, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGBA_8u_ToUYVP_10u_AC4C3R_Ctx(const Npp8u* const pSrc, const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGBA_8u_ToUYVY_8u_AC4C3R_Ctx(const Npp8u* const pSrc, const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGBToNV12_16u_ColorTwist32f_C3P2R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst[2], int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToNV12_16u_ColorTwist32f_P3P2R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst[2], int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToNV12_8u_C3P2R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst[2], int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[2], int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToNV12_8u_ColorTwist32f_P3P2R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst[2], int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToNV12_8u_P3P2R_Ctx(const Npp8u* const pSrc[3], int nSrcStep, Npp8u* pDst[2], int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGBToYUV420_16u_ColorTwist32f_C3P3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToYUV420_16u_ColorTwist32f_P3R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToYUV420_8u_ColorTwist32f_C3P3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToYUV420_8u_ColorTwist32f_P3R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToYUV422_16u_ColorTwist32f_C3C2R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToYUV422_16u_ColorTwist32f_C3P3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToYUV422_16u_ColorTwist32f_P3R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToYUV422_8u_ColorTwist32f_C3C2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToYUV422_8u_ColorTwist32f_C3P3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGBToYUV422_8u_ColorTwist32f_P3R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiRGB_16f_ToUYVP_10u_C3C3R_Ctx(const Npp16f* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGB_16f_ToUYVP_10u_P3C3R_Ctx(const Npp16f* const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGB_16u_ToUYVP_10u_C3C3R_Ctx(const Npp16u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGB_16u_ToUYVP_10u_P3C3R_Ctx(const Npp16u* const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGB_16u_ToUYVY16_16u_C3C3R_Ctx(const Npp16u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGB_16u_ToUYVY16_16u_P3C3R_Ctx(const Npp16u* const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGB_8u_ToUYVP_10u_C3C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGB_8u_ToUYVP_10u_C3P3R_Ctx(const Npp8u* const pSrc, const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst[3], int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGB_8u_ToUYVP_10u_P3C3R_Ctx(const Npp8u* const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGB_8u_ToUYVY_8u_C3C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGB_8u_ToUYVY_8u_P3C3R_Ctx(const Npp8u* const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToI444_8u_C3P3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* const pDst[3], const int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToNV16_8u_C3P2R_Ctx(const Npp8u* const pSrc, const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst[2], const int nDstStep[2], NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToP208_8u_C3C3R_Ctx(const Npp8u* pSrc, int srcStep, NppiPoint oSrcOffset, Npp8u* pDst, int dstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToRGBA_16f_C3AC4R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16f* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp32f nAlphaValue, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToRGB_16f_C3C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16f* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToRGB_16f_C3P3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16f* const pDst[3], const int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToRGB_16u_C3AC4R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp16u alpha, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToRGB_16u_C3C3R_Ctx(const Npp8u* pSrc, int srcStep, NppiPoint oSrcOffset, Npp16u* pDst, int dstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToRGB_16u_C3P3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u* pDst[3], const int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToRGB_8u_C3AC4R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp8u alpha, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToRGB_8u_C3C3R_Ctx(const Npp8u* pSrc, int srcStep, NppiPoint oSrcOffset, Npp8u* pDst, int dstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToRGB_8u_C3P3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst[3], const int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToV210_10u_C3C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToY410_10u_C3C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiUYVP_10u_ToYUV444_8u_C3C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiUYVY16_16u_ToRGB_16u_C3AC4R_Ctx(const Npp16u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp16u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiUYVY16_16u_ToRGB_16u_C3C3R_Ctx(const Npp16u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiUYVY16_16u_ToRGB_16u_C3P3R_Ctx(const Npp16u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u* pDst[3], const int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiUYVY_8u_ToRGB_8u_C3AC4R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp8u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiUYVY_8u_ToRGB_8u_C3C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiUYVY_8u_ToRGB_8u_C3P3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst[3], const int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiV210_10u_ToUYVP_10u_C3C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiWMSSSIMBatchGetBufferHostSize_8u_C4R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiWMSSSIMBatch_8u_C4R_Advanced_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oMaxSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx);
NppStatus nppiWMSSSIMBatch_8u_C4R_Ctx(const NppiImageDescriptor * pSrc1BatchList, const NppiImageDescriptor * pSrc2BatchList, int nBatchSize, NppiSize oSizeROI, Npp32f * pWMSSSIM, NppiBufferDescriptor * pDeviceBufferList, NppStreamContext nppStreamCtx);
NppStatus nppiWMSSSIMGetBufferHostSize_8u_C4R_Ctx(NppiSize oSizeROI, size_t * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiWMSSSIM_8u_C4R_Ctx(const Npp8u * pSrc1, int nSrc1Step, const Npp8u * pSrc2, int nSrc2Step, NppiSize oSizeROI, Npp32f * pMSSSIM, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);
NppStatus nppiY410_10u_ToUYVP_10u_C3C3R_Ctx(const Npp16u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiYUV420ToNV12_8u_P3P2R_Ctx(const Npp8u* const pSrc[3], const int aSrcStep[3], Npp8u* pDst[2], int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3AC4R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], const Npp16u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3C3R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3C4R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3AC4R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], const Npp8u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3C3R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3C4R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV422ToRGB_16u_ColorTwist32f_C2C3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV422ToRGB_16u_ColorTwist32f_P3AC4R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], const Npp16u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiYUV422ToRGB_16u_ColorTwist32f_P3C3R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV422ToRGB_16u_ColorTwist32f_P3R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV422ToRGB_8u_ColorTwist32f_C2C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV422ToRGB_8u_ColorTwist32f_P3AC4R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], const Npp8u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiYUV422ToRGB_8u_ColorTwist32f_P3C3R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV422ToRGB_8u_ColorTwist32f_P3R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx);
NppStatus nppiYUV444_8u_ToUYVP_10u_C3C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

NppStatus nppiCompressedMarkerLabelsUFContoursOutputGeometryLists_C1R(NppiContourPixelGeometryInfo * pContoursPixelGeometryListsHost, NppiCompressedMarkerLabelsInfo * pMarkerLabelsInfoListHost, Npp8u * pContoursGeometryImageHost, Npp32s nContoursGeometryImageStep, Npp32u * pContoursPixelCountsListHost, Npp32u * pContoursPixelsFoundListHost, Npp32u * pContoursPixelsStartingOffsetHost, Npp32u nMaxMarkerLabelID, Npp32u nFirstContourGeometryListID, Npp32u nLastContourGeometryListID, Npp32u bOutputInCounterclockwiseOrder, NppiSize oSizeROI);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(MPP_EXPORTS)
/* C++ forwarders for entry points whose selected header has no stream-
 * context parameter. */
#endif

#if defined(__cplusplus) && !defined(MPP_EXPORTS)
/* NPP 13.4 contour entry points and legacy overloads kept available for the
 * shared test suite. */
NppStatus nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(Npp32u * pCompressedMarkerLabels, Npp32s nCompressedMarkerLabelsStep, NppiSize oSizeROI, unsigned int nMaxMarkerLabelID, NppiCompressedMarkerLabelsInfo * pMarkerLabelsInfoList, Npp8u * pContoursImage, Npp32s nContoursImageStep, NppiContourPixelDirectionInfo * pContoursDirectionImage, Npp32s nContoursDirectionImageStep, NppiContourTotalsInfo * pContoursTotalsInfoHost, Npp32u * pContoursPixelCountsListDev, Npp32u * pContoursPixelCountsListHost, Npp32u * pContoursPixelStartingOffsetHost, NppStreamContext nppStreamCtx);
NppStatus nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(NppiCompressedMarkerLabelsInfo * pMarkerLabelsInfoListDev, NppiCompressedMarkerLabelsInfo * pMarkerLabelsInfoListHost, NppiContourPixelDirectionInfo * pContoursDirectionImageDev, Npp32s nContoursDirectionImageStep, NppiContourPixelGeometryInfo * pContoursPixelGeometryListsHost, Npp32u * pContoursPixelCountsListHost, Npp32u * pContoursPixelsFoundListHost, Npp32u * pContoursPixelsStartingOffsetHost, Npp32u nMaxMarkerLabelID, Npp32u nFirstContourGeometryListID, Npp32u nLastContourGeometryListID, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
#endif

#endif
