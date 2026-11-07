#ifndef NPP_API_EXTRA_12_8_H
#define NPP_API_EXTRA_12_8_H

/* Superset declarations for the API-12.8 header set. */
#include <npp.h>
#include "npp_api_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

NppStatus nppiFilterCannyBorder_8u_C3C1R_Ctx(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize, Npp16s nLowThreshold, Npp16s nHighThreshold, NppiNorm eNorm, NppiBorderType eBorderType, Npp8u * pDeviceBuffer, NppStreamContext nppStreamCtx);
NppStatus nppiI444_8u_ToUYVP_10u_P3C3R_Ctx(const Npp8u* const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToBGR_601_8u_P2C3R_Ctx(const Npp8u * const pSrc[2], int rSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToBGR_8u_P2C4R_Ctx(const Npp8u* const pSrc[2], int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp8u nAlpha, NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToBGR_BT2020_8u_P2C3R_Ctx(const Npp8u* const pSrc[2], int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiNV12ToRGB_601_8u_P2C3R_Ctx(const Npp8u * const pSrc[2], int rSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
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
NppStatus nppiRGBToNV12_8u_C3P2R_Ctx(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst[2], int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
NppStatus nppiRGBToNV12_8u_P3P2R_Ctx(const Npp8u* const pSrc[3], int nSrcStep, Npp8u* pDst[2], int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx);
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
NppStatus nppiYUV444_8u_ToUYVP_10u_C3C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(MPP_EXPORTS)
/* C++ forwarders for entry points whose selected header has no stream-
 * context parameter. */
#endif

#endif
