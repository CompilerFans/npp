// nppiCrossCorr{Full,Same,Valid}_NormLevel_GetAdvancedScratchBufferSize
// (package p1.8).  Hand-written (no gen_family spec): the API-unified header
// declares an extern "C" int* entry point plus a C++ size_t* overload per
// mode; NVIDIA NPP 13.4 only exports the size_t* form, so the differential
// tests exercise the size_t* overloads and the int* forms forward to the same
// core.
//
// 5090-measured contract (NVIDIA NPP 13.4):
//   * size = 16 + 2 * dstArea * nSizeofDstData * nSrcChannels where dstArea is
//     (Ws + tw - 1) * (Hs + th - 1) for Full and Ws * Hs for Same AND Valid
//     (Same/Valid ignore the template ROI completely, even negative dims).
//   * Zero-sized src dims are not an error: the size is still computed (e.g.
//     src 0x0, tpl 2x2, Full -> 24).  Zero-sized tpl dims flow through the
//     arithmetic as well (src 4x3, tpl 0x0, Full -> 64).
//   * Validation order: null hpBufferSize -> NPP_NULL_POINTER_ERROR (first);
//     any negative src dim -> NPP_SIZE_ERROR; [Full only] any negative tpl
//     dim -> NPP_SIZE_ERROR (matching the kernels; unlike the kernels, zero
//     tpl dims are NOT a no-op here - the size is still computed);
//     nSizeofDstData <= 0 -> NPP_SIZE_ERROR; nSrcChannels outside 1..4 ->
//     NPP_SIZE_ERROR.  Nothing is written on error.
#include "npp.h"
#include "nppi_crosscorr_executor.h"

namespace {

using nppi::statistics::CrossCorrMode;

NppStatus crossCorrAdvancedScratchBufferSize(CrossCorrMode mode, NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                             int nSizeofDstData, int nSrcChannels, long long *value) {
  if (value == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcRoiSize.width < 0 || oSrcRoiSize.height < 0) return NPP_SIZE_ERROR;
  if (mode == CrossCorrMode::Full && (oTplRoiSize.width < 0 || oTplRoiSize.height < 0))
    return NPP_SIZE_ERROR;
  if (nSizeofDstData <= 0) return NPP_SIZE_ERROR;
  if (nSrcChannels < 1 || nSrcChannels > 4) return NPP_SIZE_ERROR;
  long long area;
  if (mode == CrossCorrMode::Full) {
    area = (long long)(oSrcRoiSize.width + oTplRoiSize.width - 1) * (oSrcRoiSize.height + oTplRoiSize.height - 1);
  } else {
    area = (long long)oSrcRoiSize.width * oSrcRoiSize.height;
  }
  *value = 16 + 2 * area * nSizeofDstData * nSrcChannels;
  return NPP_NO_ERROR;
}

}  // namespace

extern "C" {

NppStatus nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                   int nSizeofDstData, int nSrcChannels,
                                                                   int *hpBufferSize) {
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Full, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (int)value;
  return st;
}

NppStatus nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                   int nSizeofDstData, int nSrcChannels,
                                                                   int *hpBufferSize) {
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Same, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (int)value;
  return st;
}

NppStatus nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                    int nSizeofDstData, int nSrcChannels,
                                                                    int *hpBufferSize) {
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Valid, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (int)value;
  return st;
}

}  // extern "C"

NppStatus nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                   int nSizeofDstData, int nSrcChannels,
                                                                   size_t *hpBufferSize) {
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Full, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (size_t)value;
  return st;
}

NppStatus nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                   int nSizeofDstData, int nSrcChannels,
                                                                   size_t *hpBufferSize) {
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Same, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (size_t)value;
  return st;
}

NppStatus nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                    int nSizeofDstData, int nSrcChannels,
                                                                    size_t *hpBufferSize) {
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Valid, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (size_t)value;
  return st;
}
