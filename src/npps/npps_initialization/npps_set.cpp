#include "npp.h"
#include "npp_version_compat.h"
#include "npps_initialization_executor.h"

// The 8u/32f/32fc entry points used to have dedicated scalar kernels here.
// Funnel them through the shared initializer so they take the same 16-byte
// packet path (and the same validation) as the generated types.

// ==============================================================================
// Set Operations - Initialize signal elements with constant values
// ==============================================================================

NppStatus nppsSet_8u_Ctx(Npp8u nValue, Npp8u *pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::set<Npp8u>(nValue, pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsSet_8u(Npp8u nValue, Npp8u *pDst, NppSignalLength nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSet_8u_Ctx(nValue, pDst, nLength, defaultContext);
}

NppStatus nppsSet_32f_Ctx(Npp32f nValue, Npp32f *pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::set<Npp32f>(nValue, pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsSet_32f(Npp32f nValue, Npp32f *pDst, NppSignalLength nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSet_32f_Ctx(nValue, pDst, nLength, defaultContext);
}

NppStatus nppsSet_32fc_Ctx(Npp32fc nValue, Npp32fc *pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return npps::init::set<Npp32fc>(nValue, pDst, nLength, nppStreamCtx.hStream);
}

NppStatus nppsSet_32fc(Npp32fc nValue, Npp32fc *pDst, NppSignalLength nLength) {
  NppStreamContext defaultContext;
  NppStatus status = nppGetStreamContext(&defaultContext);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppsSet_32fc_Ctx(nValue, pDst, nLength, defaultContext);
}

NppStatus nppsZero_8u_Ctx(Npp8u *pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return nppsSet_8u_Ctx(0, pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_8u(Npp8u *pDst, NppSignalLength nLength) {
  return nppsSet_8u(0, pDst, nLength);
}

NppStatus nppsZero_32f_Ctx(Npp32f *pDst, NppSignalLength nLength, NppStreamContext nppStreamCtx) {
  return nppsSet_32f_Ctx(0.0f, pDst, nLength, nppStreamCtx);
}

NppStatus nppsZero_32f(Npp32f *pDst, NppSignalLength nLength) {
  return nppsSet_32f(0.0f, pDst, nLength);
}
