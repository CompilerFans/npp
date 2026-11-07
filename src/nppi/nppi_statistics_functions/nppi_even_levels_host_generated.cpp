// Generated API wrappers -- do not edit by hand.
// family: nppiEvenLevelsHost


#include "npp.h"
#include "nppi_quality_metrics_executor.h"

NppStatus nppiEvenLevelsHost_32s_Ctx(Npp32s * hpLevels, int nLevels, Npp32s nLowerLevel, Npp32s nUpperLevel, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::statistics::evenLevelsHost(hpLevels, nLevels, nLowerLevel, nUpperLevel);
}
