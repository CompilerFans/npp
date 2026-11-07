// C ABI wrappers for header sets whose buffer-size entry points do not take
// an NppStreamContext parameter.  The four-argument C++ implementations are
// generated/defined elsewhere; these wrappers provide the classic C symbols.

#include <npp.h>

#if defined(NPP_HEADER_MAJOR) && NPP_HEADER_MAJOR < 12

NppStatus nppiHistogramEvenGetBufferSize_8u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEvenGetBufferSize_8u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEvenGetBufferSize_8u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEvenGetBufferSize_16u_C1R_Ctx(NppiSize oSizeROI, int nLevels, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEvenGetBufferSize_16u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEvenGetBufferSize_16u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEvenGetBufferSize_16s_C1R_Ctx(NppiSize oSizeROI, int nLevels, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEvenGetBufferSize_16s_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramEvenGetBufferSize_16s_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_8u_C1R_Ctx(NppiSize oSizeROI, int nLevels, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_8u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_8u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_8u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_16u_C1R_Ctx(NppiSize oSizeROI, int nLevels, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_16u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_16u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_16u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_16s_C1R_Ctx(NppiSize oSizeROI, int nLevels, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_16s_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_16s_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_16s_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_32f_C1R_Ctx(NppiSize oSizeROI, int nLevels, NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_32f_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_32f_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);
NppStatus nppiHistogramRangeGetBufferSize_32f_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], NPP_BUFFER_SIZE_TYPE * hpBufferSize, NppStreamContext nppStreamCtx);

extern "C" NppStatus nppiHistogramEvenGetBufferSize_8u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramEvenGetBufferSize_8u_C3R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramEvenGetBufferSize_8u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramEvenGetBufferSize_8u_C4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramEvenGetBufferSize_8u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramEvenGetBufferSize_8u_AC4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramEvenGetBufferSize_16u_C1R_Ctx(NppiSize oSizeROI, int nLevels, int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramEvenGetBufferSize_16u_C1R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramEvenGetBufferSize_16u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramEvenGetBufferSize_16u_C3R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramEvenGetBufferSize_16u_C4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramEvenGetBufferSize_16u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramEvenGetBufferSize_16u_AC4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramEvenGetBufferSize_16s_C1R_Ctx(NppiSize oSizeROI, int nLevels, int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramEvenGetBufferSize_16s_C1R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramEvenGetBufferSize_16s_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramEvenGetBufferSize_16s_C3R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramEvenGetBufferSize_16s_C4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramEvenGetBufferSize_16s_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramEvenGetBufferSize_16s_AC4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_8u_C1R_Ctx(NppiSize oSizeROI, int nLevels, int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_8u_C1R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_8u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_8u_C3R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_8u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_8u_C4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_8u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_8u_AC4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_16u_C1R_Ctx(NppiSize oSizeROI, int nLevels, int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_16u_C1R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_16u_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_16u_C3R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_16u_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_16u_C4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_16u_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_16u_AC4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_16s_C1R_Ctx(NppiSize oSizeROI, int nLevels, int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_16s_C1R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_16s_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_16s_C3R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_16s_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_16s_C4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_16s_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_16s_AC4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_32f_C1R_Ctx(NppiSize oSizeROI, int nLevels, int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_32f_C1R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_32f_C3R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_32f_C3R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_32f_C4R_Ctx(NppiSize oSizeROI, int nLevels[4], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_32f_C4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

extern "C" NppStatus nppiHistogramRangeGetBufferSize_32f_AC4R_Ctx(NppiSize oSizeROI, int nLevels[3], int * hpBufferSize) {
  NppStreamContext nppCtx{};
  NppStatus st = nppGetStreamContext(&nppCtx);
  if (st != NPP_NO_ERROR) return st;
  return nppiHistogramRangeGetBufferSize_32f_AC4R_Ctx(oSizeROI, nLevels, hpBufferSize, nppCtx);
}

#endif
