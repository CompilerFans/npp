#include "../../npp_internal.h"
#include "npp.h"
#include "nppi_filter_executor.h"
#include <cstring>
#include <cuda_runtime.h>

// Legacy 8u32f sum-window implementations are replaced by the package-f12
// executor, whose clipping/anchor semantics are pinned by the differential
// goldens (same as the generated sum-window wrappers).

NppStatus nppiSumWindowRow_8u32f_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                         NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor,
                                         NppStreamContext nppStreamCtx) {
  return nppi::filter::sumWindowRow<Npp8u, 1>(pSrc, nSrcStep, pDst, nDstStep, oROI, nMaskSize, nAnchor,
                                              nppStreamCtx.hStream);
}

NppStatus nppiSumWindowRow_8u32f_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oROI,
                                     Npp32s nMaskSize, Npp32s nAnchor) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream(); // NPP managed stream (nppSetStream)

  return nppiSumWindowRow_8u32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, nMaskSize, nAnchor, nppStreamCtx);
}

NppStatus nppiSumWindowColumn_8u32f_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                            NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor,
                                            NppStreamContext nppStreamCtx) {
  return nppi::filter::sumWindowColumn<Npp8u, 1>(pSrc, nSrcStep, pDst, nDstStep, oROI, nMaskSize, nAnchor,
                                                 nppStreamCtx.hStream);
}

NppStatus nppiSumWindowColumn_8u32f_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                        NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();

  return nppiSumWindowColumn_8u32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, nMaskSize, nAnchor, nppStreamCtx);
}
