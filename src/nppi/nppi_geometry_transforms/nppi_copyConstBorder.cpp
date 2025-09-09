#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

/**
 * NPP Image Copy with Constant Border Functions Implementation
 * Implements nppiCopyConstBorder functions for copying images with constant border extension
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiCopyConstBorder_8u_C1R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                              Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth,
                                              Npp8u nValue, NppStreamContext nppStreamCtx);
NppStatus nppiCopyConstBorder_8u_C3R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                              Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                              int nTopBorderHeight, int nLeftBorderWidth,
                                              const Npp8u aValue[3], NppStreamContext nppStreamCtx);
NppStatus nppiCopyConstBorder_16s_C1R_Ctx_cuda(const Npp16s* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp16s* pDst, int nDstStep, NppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth,
                                               Npp16s nValue, NppStreamContext nppStreamCtx);
NppStatus nppiCopyConstBorder_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                               Npp32f* pDst, int nDstStep, NppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth,
                                               Npp32f nValue, NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateCopyConstBorderInputs(const void* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                     void* pDst, int nDstStep, NppiSize oDstSizeROI,
                                                     int nTopBorderHeight, int nLeftBorderWidth) {
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSrcSizeROI.width <= 0 || oSrcSizeROI.height <= 0 ||
        oDstSizeROI.width <= 0 || oDstSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrcStep <= 0 || nDstStep <= 0) {
        return NPP_STEP_ERROR;
    }
    
    if (nTopBorderHeight < 0 || nLeftBorderWidth < 0) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    // Check if destination is large enough to contain source + borders
    int nRightBorderWidth = oDstSizeROI.width - oSrcSizeROI.width - nLeftBorderWidth;
    int nBottomBorderHeight = oDstSizeROI.height - oSrcSizeROI.height - nTopBorderHeight;
    
    if (nRightBorderWidth < 0 || nBottomBorderHeight < 0) {
        return NPP_SIZE_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 8-bit unsigned single channel copy with constant border
NppStatus nppiCopyConstBorder_8u_C1R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                         Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth,
                                         Npp8u nValue, NppStreamContext nppStreamCtx) {
    NppStatus status = validateCopyConstBorderInputs(pSrc, nSrcStep, oSrcSizeROI, 
                                                    pDst, nDstStep, oDstSizeROI,
                                                    nTopBorderHeight, nLeftBorderWidth);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCopyConstBorder_8u_C1R_Ctx_cuda(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI,
                                              nTopBorderHeight, nLeftBorderWidth, nValue, nppStreamCtx);
}

NppStatus nppiCopyConstBorder_8u_C1R(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                    Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                    int nTopBorderHeight, int nLeftBorderWidth, Npp8u nValue) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCopyConstBorder_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI,
                                         nTopBorderHeight, nLeftBorderWidth, nValue, nppStreamCtx);
}

// 8-bit unsigned three channel copy with constant border
NppStatus nppiCopyConstBorder_8u_C3R_Ctx(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                         Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight, int nLeftBorderWidth,
                                         const Npp8u aValue[3], NppStreamContext nppStreamCtx) {
    NppStatus status = validateCopyConstBorderInputs(pSrc, nSrcStep, oSrcSizeROI, 
                                                    pDst, nDstStep, oDstSizeROI,
                                                    nTopBorderHeight, nLeftBorderWidth);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    if (!aValue) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    return nppiCopyConstBorder_8u_C3R_Ctx_cuda(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI,
                                              nTopBorderHeight, nLeftBorderWidth, aValue, nppStreamCtx);
}

NppStatus nppiCopyConstBorder_8u_C3R(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                    Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                    int nTopBorderHeight, int nLeftBorderWidth, const Npp8u aValue[3]) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCopyConstBorder_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI,
                                         nTopBorderHeight, nLeftBorderWidth, aValue, nppStreamCtx);
}

// 16-bit signed single channel copy with constant border
NppStatus nppiCopyConstBorder_16s_C1R_Ctx(const Npp16s* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                          Npp16s* pDst, int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Npp16s nValue, NppStreamContext nppStreamCtx) {
    NppStatus status = validateCopyConstBorderInputs(pSrc, nSrcStep, oSrcSizeROI, 
                                                    pDst, nDstStep, oDstSizeROI,
                                                    nTopBorderHeight, nLeftBorderWidth);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCopyConstBorder_16s_C1R_Ctx_cuda(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI,
                                               nTopBorderHeight, nLeftBorderWidth, nValue, nppStreamCtx);
}

NppStatus nppiCopyConstBorder_16s_C1R(const Npp16s* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                     Npp16s* pDst, int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth, Npp16s nValue) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCopyConstBorder_16s_C1R_Ctx(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI,
                                          nTopBorderHeight, nLeftBorderWidth, nValue, nppStreamCtx);
}

// 32-bit float single channel copy with constant border
NppStatus nppiCopyConstBorder_32f_C1R_Ctx(const Npp32f* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                          Npp32f* pDst, int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Npp32f nValue, NppStreamContext nppStreamCtx) {
    NppStatus status = validateCopyConstBorderInputs(pSrc, nSrcStep, oSrcSizeROI, 
                                                    pDst, nDstStep, oDstSizeROI,
                                                    nTopBorderHeight, nLeftBorderWidth);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCopyConstBorder_32f_C1R_Ctx_cuda(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI,
                                               nTopBorderHeight, nLeftBorderWidth, nValue, nppStreamCtx);
}

NppStatus nppiCopyConstBorder_32f_C1R(const Npp32f* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                     Npp32f* pDst, int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth, Npp32f nValue) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCopyConstBorder_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSizeROI, pDst, nDstStep, oDstSizeROI,
                                          nTopBorderHeight, nLeftBorderWidth, nValue, nppStreamCtx);
}