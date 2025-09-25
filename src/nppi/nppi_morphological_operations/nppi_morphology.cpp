#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

//=============================================================================
// Erosion Functions
//=============================================================================

// 8-bit unsigned single channel 3x3 erosion
NppStatus nppiErode3x3_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiErode3x3_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI) {
  printf("not implement\n");
  return NPP_ERROR;
}

// 32-bit float single channel 3x3 erosion
NppStatus nppiErode3x3_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiErode3x3_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI) {
  printf("not implement\n");
  return NPP_ERROR;
}

//=============================================================================
// Dilation Functions
//=============================================================================

// 8-bit unsigned single channel 3x3 dilation
NppStatus nppiDilate3x3_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiDilate3x3_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI) {
  printf("not implement\n");
  return NPP_ERROR;
}

// 32-bit float single channel 3x3 dilation
NppStatus nppiDilate3x3_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiDilate3x3_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI) {
  printf("not implement\n");
  return NPP_ERROR;
}

//=============================================================================
// General Erosion Functions
//=============================================================================

// 8-bit unsigned single channel general erosion
NppStatus nppiErode_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                               NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiErode_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                           const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  printf("not implement\n");
  return NPP_ERROR;
}

// 8-bit unsigned 4-channel general erosion
NppStatus nppiErode_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                               NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiErode_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                           const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  printf("not implement\n");
  return NPP_ERROR;
}

// 32-bit float single channel general erosion
NppStatus nppiErode_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiErode_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  printf("not implement\n");
  return NPP_ERROR;
}

// 32-bit float 4-channel general erosion
NppStatus nppiErode_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiErode_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  printf("not implement\n");
  return NPP_ERROR;
}

//=============================================================================
// General Dilation Functions
//=============================================================================

// 8-bit unsigned single channel general dilation
NppStatus nppiDilate_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiDilate_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  printf("not implement\n");
  return NPP_ERROR;
}

// 8-bit unsigned 4-channel general dilation
NppStatus nppiDilate_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiDilate_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  printf("not implement\n");
  return NPP_ERROR;
}

// 32-bit float single channel general dilation
NppStatus nppiDilate_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiDilate_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  printf("not implement\n");
  return NPP_ERROR;
}

// 32-bit float 4-channel general dilation
NppStatus nppiDilate_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiDilate_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor) {
  printf("not implement\n");
  return NPP_ERROR;
}
