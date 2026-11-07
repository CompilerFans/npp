// Generated API wrappers -- do not edit by hand.
// family: nppsThreshold_GT


#include "npp.h"
#include "npps_conversion_executor.h"

NppStatus nppsThreshold_GT_16s_Ctx(const Npp16s * pSrc, Npp16s * pDst, NppSignalLength nLength, Npp16s nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrc, pDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_16s_I_Ctx(Npp16s * pSrcDst, NppSignalLength nLength, Npp16s nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrcDst, pSrcDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_16sc_Ctx(const Npp16sc * pSrc, Npp16sc * pDst, NppSignalLength nLength, Npp16s nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrc, pDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_16sc_I_Ctx(Npp16sc * pSrcDst, NppSignalLength nLength, Npp16s nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrcDst, pSrcDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_32f_Ctx(const Npp32f * pSrc, Npp32f * pDst, NppSignalLength nLength, Npp32f nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrc, pDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_32f_I_Ctx(Npp32f * pSrcDst, NppSignalLength nLength, Npp32f nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrcDst, pSrcDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_32fc_Ctx(const Npp32fc * pSrc, Npp32fc * pDst, NppSignalLength nLength, Npp32f nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrc, pDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_32fc_I_Ctx(Npp32fc * pSrcDst, NppSignalLength nLength, Npp32f nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrcDst, pSrcDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_64f_Ctx(const Npp64f * pSrc, Npp64f * pDst, NppSignalLength nLength, Npp64f nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrc, pDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_64f_I_Ctx(Npp64f * pSrcDst, NppSignalLength nLength, Npp64f nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrcDst, pSrcDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_64fc_Ctx(const Npp64fc * pSrc, Npp64fc * pDst, NppSignalLength nLength, Npp64f nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrc, pDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_64fc_I_Ctx(Npp64fc * pSrcDst, NppSignalLength nLength, Npp64f nLevel, NppStreamContext nppStreamCtx) {
  return npps::threshold::thresholdGT(pSrcDst, pSrcDst, nLength, nLevel, nppStreamCtx.hStream);
}

NppStatus nppsThreshold_GT_16s(const Npp16s * pSrc, Npp16s * pDst, NppSignalLength nLength, Npp16s nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_16s_Ctx(pSrc, pDst, nLength, nLevel, nppStreamCtx);
}

NppStatus nppsThreshold_GT_16s_I(Npp16s * pSrcDst, NppSignalLength nLength, Npp16s nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_16s_I_Ctx(pSrcDst, nLength, nLevel, nppStreamCtx);
}

NppStatus nppsThreshold_GT_16sc(const Npp16sc * pSrc, Npp16sc * pDst, NppSignalLength nLength, Npp16s nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_16sc_Ctx(pSrc, pDst, nLength, nLevel, nppStreamCtx);
}

NppStatus nppsThreshold_GT_16sc_I(Npp16sc * pSrcDst, NppSignalLength nLength, Npp16s nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_16sc_I_Ctx(pSrcDst, nLength, nLevel, nppStreamCtx);
}

NppStatus nppsThreshold_GT_32f(const Npp32f * pSrc, Npp32f * pDst, NppSignalLength nLength, Npp32f nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_32f_Ctx(pSrc, pDst, nLength, nLevel, nppStreamCtx);
}

NppStatus nppsThreshold_GT_32f_I(Npp32f * pSrcDst, NppSignalLength nLength, Npp32f nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_32f_I_Ctx(pSrcDst, nLength, nLevel, nppStreamCtx);
}

NppStatus nppsThreshold_GT_32fc(const Npp32fc * pSrc, Npp32fc * pDst, NppSignalLength nLength, Npp32f nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_32fc_Ctx(pSrc, pDst, nLength, nLevel, nppStreamCtx);
}

NppStatus nppsThreshold_GT_32fc_I(Npp32fc * pSrcDst, NppSignalLength nLength, Npp32f nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_32fc_I_Ctx(pSrcDst, nLength, nLevel, nppStreamCtx);
}

NppStatus nppsThreshold_GT_64f(const Npp64f * pSrc, Npp64f * pDst, NppSignalLength nLength, Npp64f nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_64f_Ctx(pSrc, pDst, nLength, nLevel, nppStreamCtx);
}

NppStatus nppsThreshold_GT_64f_I(Npp64f * pSrcDst, NppSignalLength nLength, Npp64f nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_64f_I_Ctx(pSrcDst, nLength, nLevel, nppStreamCtx);
}

NppStatus nppsThreshold_GT_64fc(const Npp64fc * pSrc, Npp64fc * pDst, NppSignalLength nLength, Npp64f nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_64fc_Ctx(pSrc, pDst, nLength, nLevel, nppStreamCtx);
}

NppStatus nppsThreshold_GT_64fc_I(Npp64fc * pSrcDst, NppSignalLength nLength, Npp64f nLevel) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsThreshold_GT_64fc_I_Ctx(pSrcDst, nLength, nLevel, nppStreamCtx);
}
