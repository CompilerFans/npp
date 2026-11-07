#ifndef NPP_API_EXTRA_13_4_H
#define NPP_API_EXTRA_13_4_H

/* Superset declarations for the API-13.4 header set. */
#include <npp.h>
#include "npp_api_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

NppStatus nppiCompressMarkerLabelsUF_32u_C1IR(Npp32u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, int nStartingNumber, int * pNewNumber, Npp8u * pBuffer);
NppStatus nppiEvenLevelsHost_32s_Ctx(Npp32s * hpLevels, int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, NppStreamContext nppStreamCtx);
NppStatus nppiFilterCannyBorder_8u_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppiDifferentialKernel eFilterType, NppiMaskSize eMaskSize, Npp16s nLowThreshold, Npp16s nHighThreshold, NppiNorm eNorm, NppiBorderType eBorderType, Npp8u * pDeviceBuffer);
NppStatus nppiFilterHoughLineRegion_8u32f_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, NppPointPolar nDelta, int nThreshold, NppPointPolar * pDeviceLines, NppPointPolar oDstROI[2], int nMaxLineCount, int * pDeviceLineCount, Npp8u * pDeviceBuffer);
NppStatus nppiFilterHoughLine_8u32f_C1R(const Npp8u * pSrc, int nSrcStep, NppiSize oSizeROI, NppPointPolar nDelta, int nThreshold, NppPointPolar * pDeviceLines, int nMaxLineCount, int * pDeviceLineCount, Npp8u * pDeviceBuffer);
NppStatus nppiFilter_64f_C1R(const Npp64f * pSrc, Npp32s nSrcStep, Npp64f * pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp64f * pKernel, NppiSize oKernelSize, NppiPoint oAnchor);
NppStatus nppiYCCKToCMYK_JPEG_601_8u_P4R(const Npp8u * pSrc[4], int nSrcStep, Npp8u * pDst[4], int nDstStep, NppiSize oSizeROI);
NppStatus nppsDiv_32s16s_Sfs(const Npp16s * pSrc1, const Npp32s * pSrc2, Npp16s * pDst, NppSignalLength nLength, int nScaleFactor);
NppStatus nppsIntegral_32s(const Npp32s * pSrc, Npp32s * pDst, NppSignalLength nLength, Npp8u * pDeviceBuffer);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(MPP_EXPORTS)
/* C++ forwarders for entry points whose selected header has no stream-
 * context parameter. */
#endif

#endif
