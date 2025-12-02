#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// ============================================================================
// Forward Declarations for CUDA implementations (_Ctx versions)
// ============================================================================

extern "C" {
NppStatus nppiGammaFwd_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
NppStatus nppiGammaFwd_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiGammaInv_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
NppStatus nppiGammaInv_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiGammaFwd_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiGammaFwd_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiGammaInv_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
NppStatus nppiGammaInv_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

// ============================================================================
// Helper: Create default NppStreamContext
// ============================================================================

static inline NppStreamContext createDefaultStreamContext() {
  NppStreamContext ctx;
  // FIX: Zero-initialize entire structure to avoid uninitialized fields
  memset(&ctx, 0, sizeof(NppStreamContext));
  ctx.hStream = 0; // Use default CUDA stream
  return ctx;
}

// ============================================================================
// Non-Ctx Versions (wrappers that call Ctx versions with default stream)
// IMPORTANT: Parameter validation is done in .cu executor functions
// ============================================================================

NppStatus nppiGammaFwd_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx = createDefaultStreamContext();
  return nppiGammaFwd_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiGammaFwd_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx = createDefaultStreamContext();
  return nppiGammaFwd_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, ctx);
}

NppStatus nppiGammaInv_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx = createDefaultStreamContext();
  return nppiGammaInv_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiGammaInv_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx = createDefaultStreamContext();
  return nppiGammaInv_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, ctx);
}

NppStatus nppiGammaFwd_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx = createDefaultStreamContext();
  return nppiGammaFwd_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiGammaFwd_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx = createDefaultStreamContext();
  return nppiGammaFwd_8u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, ctx);
}

NppStatus nppiGammaInv_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx = createDefaultStreamContext();
  return nppiGammaInv_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiGammaInv_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx = createDefaultStreamContext();
  return nppiGammaInv_8u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, ctx);
}
