// Generated API wrappers -- do not edit by hand.
// family: nppsZero


#include "npp.h"
#include "npps_initialization_executor.h"

NppStatus nppsZero_16s_Ctx(Npp16s * pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::zero<Npp16s>(pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsZero_16sc_Ctx(Npp16sc * pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::zero<Npp16sc>(pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsZero_32fc_Ctx(Npp32fc * pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::zero<Npp32fc>(pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsZero_32s_Ctx(Npp32s * pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::zero<Npp32s>(pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsZero_32sc_Ctx(Npp32sc * pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::zero<Npp32sc>(pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsZero_64f_Ctx(Npp64f * pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::zero<Npp64f>(pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsZero_64fc_Ctx(Npp64fc * pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::zero<Npp64fc>(pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsZero_64s_Ctx(Npp64s * pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::zero<Npp64s>(pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsZero_64sc_Ctx(Npp64sc * pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::zero<Npp64sc>(pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsZero_16s(Npp16s * pDst, NppSignalLength nLength) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZero_16s_Ctx(pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_16sc(Npp16sc * pDst, NppSignalLength nLength) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZero_16sc_Ctx(pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_32fc(Npp32fc * pDst, NppSignalLength nLength) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZero_32fc_Ctx(pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_32s(Npp32s * pDst, NppSignalLength nLength) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZero_32s_Ctx(pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_32sc(Npp32sc * pDst, NppSignalLength nLength) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZero_32sc_Ctx(pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_64f(Npp64f * pDst, NppSignalLength nLength) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZero_64f_Ctx(pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_64fc(Npp64fc * pDst, NppSignalLength nLength) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZero_64fc_Ctx(pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_64s(Npp64s * pDst, NppSignalLength nLength) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZero_64s_Ctx(pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_64sc(Npp64sc * pDst, NppSignalLength nLength) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsZero_64sc_Ctx(pDst, nLength, nppStreamCtx);
}
