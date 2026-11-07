// Generated API wrappers -- do not edit by hand.
// family: nppiCircularRadialProfile


#include "npp.h"
#include "nppi_radial_profile_executor.h"

NppStatus nppiCircularRadialProfile_32f_C1R_Ctx(const Npp32f * pSrc, int nSrcStep, NppiSize oSizeROI, NppiPoint32f oCenter, NppiProfileData *pProfileData, Npp16u nProfileSamples, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runCircularRadialProfile(pSrc, nSrcStep, oSizeROI, oCenter, pProfileData, nProfileSamples, nppStreamCtx.hStream);
}
