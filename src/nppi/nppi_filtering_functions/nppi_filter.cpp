#include "npp.h"
#include "nppi_filter_executor.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// The legacy 8u/32f implementations used their own validation order and skip
// rules, which the package-f12 differential tests showed to differ from NPP
// 13.4.  All variants now go through the shared executor, whose semantics are
// pinned by the recorded goldens (same validation/arithmetic as the generated
// wrappers for the other dtypes).

// 8-bit unsigned single channel 2D convolution
NppStatus nppiFilter_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                                NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp8u, Npp8u, 1, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                         pKernel, oKernelSize, oAnchor, nDivisor,
                                                         nppStreamCtx.hStream);
}

NppStatus nppiFilter_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiFilter_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor,
                               nppStreamCtx);
}

// 8-bit unsigned three channel 2D convolution
NppStatus nppiFilter_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                                NppStreamContext nppStreamCtx) {
  return nppi::filter::filterDiv<Npp8u, Npp8u, 3, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                         pKernel, oKernelSize, oAnchor, nDivisor,
                                                         nppStreamCtx.hStream);
}

NppStatus nppiFilter_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiFilter_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor,
                               nppStreamCtx);
}

// 32-bit float single channel 2D convolution
NppStatus nppiFilter_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 const Npp32f *pKernel, NppiSize oKernelSize, NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx) {
  return nppi::filter::filter32f<Npp32f, Npp32f, 1, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                           pKernel, oKernelSize, oAnchor,
                                                           nppStreamCtx.hStream);
}

NppStatus nppiFilter_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                             const Npp32f *pKernel, NppiSize oKernelSize, NppiPoint oAnchor) {
  NppStreamContext nppStreamCtx{};
  nppStreamCtx.hStream = nppGetStream();
  return nppiFilter_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
}
