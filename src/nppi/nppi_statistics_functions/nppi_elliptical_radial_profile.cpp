// nppiEllipticalRadialProfile_32f_C1R_Ctx.  The executor reproduces the NPP
// 13.4 kernel decoded from its sm_120 SASS: dense band membership plus the
// per-ring inner-radius scan crop.  See nppi_radial_profile_executor.h.
#include "npp.h"

#include "nppi_radial_profile_executor.h"

NppStatus nppiEllipticalRadialProfile_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI,
                                                  NppiPoint32f oCenter, Npp32f nAstigmatismRatio,
                                                  Npp32f nOrientationRadians, NppiProfileData *pProfileData,
                                                  Npp16u nProfileSamples, NppStreamContext nppStreamCtx) {
  return nppi::statistics::runEllipticalRadialProfile(pSrc, nSrcStep, oSizeROI, oCenter, nAstigmatismRatio,
                                                      nOrientationRadians, pProfileData, nProfileSamples,
                                                      nppStreamCtx.hStream);
}
