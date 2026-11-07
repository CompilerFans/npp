// nppiCrossCorr{Full,Same,Valid}_NormLevel_GetAdvancedScratchBufferSize.
// Hand-written: the API-unified header
// declares an extern "C" int* entry point plus a C++ size_t* overload per
// mode; NVIDIA NPP 13.4 only exports the size_t* form, so the reference comparison
// tests exercise the size_t* overloads and the int* forms forward to the same
// core.
//
// reference-measured contract (NVIDIA NPP 13.4):
//   * size = 16 + 2 * dstArea * nSizeofDstData * nSrcChannels where dstArea is
//     (Ws + tw - 1) * (Hs + th - 1) for Full and Ws * Hs for Same AND Valid
//     (Same/Valid ignore the template ROI completely, even negative dims).
//   * Full does not reject negative template dims directly: it requires the
//     resulting dst extents to be positive (src 4x3 + tpl 2x-2 -> dstH 0 ->
//     NPP_SIZE_ERROR, while src 6x5 + tpl 2x-2 -> dstH 2 -> NO_ERROR).
//     Zero-sized src dims are allowed when the Full dst extents stay positive
//     (src 0x0, tpl 2x2 -> 24; src 4x3, tpl 0x0 -> 64).
//   * Same/Valid reject zero or negative src dims (measured: src 0x0 -> -6) and
//     never look at the template.
//   * Validation order: null hpBufferSize -> NPP_NULL_POINTER_ERROR (first);
//     src negative -> NPP_SIZE_ERROR; [Full] non-positive dst extents ->
//     NPP_SIZE_ERROR; [Same/Valid] zero src dim -> NPP_SIZE_ERROR;
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
  long long area;
  if (mode == CrossCorrMode::Full) {
    const long long dstW = (long long)oSrcRoiSize.width + oTplRoiSize.width - 1;
    const long long dstH = (long long)oSrcRoiSize.height + oTplRoiSize.height - 1;
    if (dstW <= 0 || dstH <= 0) return NPP_SIZE_ERROR;
    area = dstW * dstH;
  } else {
    // Same/Valid reject zero-sized source ROIs and ignore the template.
    if (oSrcRoiSize.width == 0 || oSrcRoiSize.height == 0) return NPP_SIZE_ERROR;
    area = (long long)oSrcRoiSize.width * oSrcRoiSize.height;
  }
  if (nSizeofDstData <= 0) return NPP_SIZE_ERROR;
  if (nSrcChannels < 1 || nSrcChannels > 4) return NPP_SIZE_ERROR;
  *value = 16 + 2 * area * nSizeofDstData * nSrcChannels;
  return NPP_NO_ERROR;
}

}  // namespace

extern "C" {

NppStatus nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                   int nSizeofDstData, int nSrcChannels,
                                                                   NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  if (hpBufferSize == nullptr) return NPP_NULL_POINTER_ERROR;
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Full, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (int)value;
  return st;
}

NppStatus nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                   int nSizeofDstData, int nSrcChannels,
                                                                   NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  if (hpBufferSize == nullptr) return NPP_NULL_POINTER_ERROR;
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Same, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (int)value;
  return st;
}

NppStatus nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                    int nSizeofDstData, int nSrcChannels,
                                                                    NPP_BUFFER_SIZE_TYPE *hpBufferSize) {
  if (hpBufferSize == nullptr) return NPP_NULL_POINTER_ERROR;
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Valid, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (int)value;
  return st;
}

}  // extern "C"

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                   int nSizeofDstData, int nSrcChannels,
                                                                   size_t *hpBufferSize) {
  if (hpBufferSize == nullptr) return NPP_NULL_POINTER_ERROR;
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Full, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (size_t)value;
  return st;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                   int nSizeofDstData, int nSrcChannels,
                                                                   size_t *hpBufferSize) {
  if (hpBufferSize == nullptr) return NPP_NULL_POINTER_ERROR;
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Same, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (size_t)value;
  return st;
}
#endif

#if !NPP_BUFFER_SIZE_IS_SIZE_T
NppStatus nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize(NppiSize oSrcRoiSize, NppiSize oTplRoiSize,
                                                                    int nSizeofDstData, int nSrcChannels,
                                                                    size_t *hpBufferSize) {
  if (hpBufferSize == nullptr) return NPP_NULL_POINTER_ERROR;
  long long value = 0;
  const NppStatus st = crossCorrAdvancedScratchBufferSize(nppi::statistics::CrossCorrMode::Valid, oSrcRoiSize,
                                                          oTplRoiSize, nSizeofDstData, nSrcChannels, &value);
  if (st == NPP_NO_ERROR && hpBufferSize != nullptr) *hpBufferSize = (size_t)value;
  return st;
}
#endif
