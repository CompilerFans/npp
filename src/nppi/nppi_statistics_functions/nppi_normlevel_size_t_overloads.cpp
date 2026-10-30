// C++ size_t* overloads for the NormLevel buffer-size helpers.
//
// The public headers declare both an extern "C" int* entry point (implemented
// by the generated wrappers) and a C++ size_t* overload.  NVIDIA's headers
// expose only the size_t* form and the differential tests use it, so provide
// thin forwarders locally to keep both header variants link-compatible.
#include "npp.h"

namespace {
inline NppStatus store_size(int value, size_t *out) {
  if (out != nullptr) {
    *out = static_cast<size_t>(value);
  }
  return NPP_NO_ERROR;
}
} // namespace

NppStatus nppiFullNormLevelGetBufferHostSize_32f_AC4R_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                          NppStreamContext nppStreamCtx) {
  if (hpBufferSize == nullptr) return NPP_NULL_POINTER_ERROR;
  int value = 0;
  NppStatus st = nppiFullNormLevelGetBufferHostSize_32f_AC4R_Ctx(oSizeROI, &value, nppStreamCtx);
  return st == NPP_NO_ERROR ? store_size(value, hpBufferSize) : st;
}

NppStatus nppiSameNormLevelGetBufferHostSize_32f_AC4R_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                          NppStreamContext nppStreamCtx) {
  if (hpBufferSize == nullptr) return NPP_NULL_POINTER_ERROR;
  int value = 0;
  NppStatus st = nppiSameNormLevelGetBufferHostSize_32f_AC4R_Ctx(oSizeROI, &value, nppStreamCtx);
  return st == NPP_NO_ERROR ? store_size(value, hpBufferSize) : st;
}

NppStatus nppiValidNormLevelGetBufferHostSize_32f_AC4R_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                           NppStreamContext nppStreamCtx) {
  if (hpBufferSize == nullptr) return NPP_NULL_POINTER_ERROR;
  int value = 0;
  NppStatus st = nppiValidNormLevelGetBufferHostSize_32f_AC4R_Ctx(oSizeROI, &value, nppStreamCtx);
  return st == NPP_NO_ERROR ? store_size(value, hpBufferSize) : st;
}
