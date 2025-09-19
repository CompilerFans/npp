#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

/**
 * NPP Image Copy Functions Implementation
 * Implements basic nppiCopy functions for image data copying
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiCopy_8u_C1R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, 
                                   Npp8u* pDst, int nDstStep, 
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_8u_C3R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, 
                                   Npp8u* pDst, int nDstStep, 
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_8u_C4R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, 
                                   Npp8u* pDst, int nDstStep, 
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, 
                                    Npp32f* pDst, int nDstStep, 
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32f_C3R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, 
                                    Npp32f* pDst, int nDstStep, 
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiCopy_32f_C3P3R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, 
                                      Npp32f* const pDst[3], int nDstStep, 
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateCopyInputs(const void* pSrc, int nSrcStep, 
                                           void* pDst, int nDstStep, 
                                           NppiSize oSizeROI) {
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrcStep <= 0 || nDstStep <= 0) {
        return NPP_STEP_ERROR;
    }
    
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 8-bit unsigned single channel copy
NppStatus nppiCopy_8u_C1R_Ctx(const Npp8u* pSrc, Npp32s nSrcStep,
                              Npp8u* pDst, Npp32s nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateCopyInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCopy_8u_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_8u_C1R(const Npp8u* pSrc, Npp32s nSrcStep,
                         Npp8u* pDst, Npp32s nDstStep,
                         NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCopy_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 8-bit unsigned three channel copy
NppStatus nppiCopy_8u_C3R_Ctx(const Npp8u* pSrc, Npp32s nSrcStep,
                              Npp8u* pDst, Npp32s nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateCopyInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCopy_8u_C3R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_8u_C3R(const Npp8u* pSrc, Npp32s nSrcStep,
                         Npp8u* pDst, Npp32s nDstStep,
                         NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCopy_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 8-bit unsigned four channel copy
NppStatus nppiCopy_8u_C4R_Ctx(const Npp8u* pSrc, Npp32s nSrcStep,
                              Npp8u* pDst, Npp32s nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateCopyInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCopy_8u_C4R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_8u_C4R(const Npp8u* pSrc, Npp32s nSrcStep,
                         Npp8u* pDst, Npp32s nDstStep,
                         NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCopy_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 32-bit float single channel copy
NppStatus nppiCopy_32f_C1R_Ctx(const Npp32f* pSrc, Npp32s nSrcStep,
                               Npp32f* pDst, Npp32s nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateCopyInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCopy_32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C1R(const Npp32f* pSrc, Npp32s nSrcStep,
                          Npp32f* pDst, Npp32s nDstStep,
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCopy_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 32-bit float three channel copy
NppStatus nppiCopy_32f_C3R_Ctx(const Npp32f* pSrc, Npp32s nSrcStep,
                               Npp32f* pDst, Npp32s nDstStep,
                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    NppStatus status = validateCopyInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    if (status != NPP_SUCCESS) {
        return status;
    }
    
    return nppiCopy_32f_C3R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C3R(const Npp32f* pSrc, Npp32s nSrcStep,
                          Npp32f* pDst, Npp32s nDstStep,
                          NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCopy_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 32-bit float three channel packed to planar copy (C3P3R = packed to planar)
NppStatus nppiCopy_32f_C3P3R_Ctx(const Npp32f* pSrc, Npp32s nSrcStep,
                                 Npp32f* const pDst[3], Npp32s nDstStep,
                                 NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    // Validate input
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    // Validate all channel pointers
    for (int c = 0; c < 3; c++) {
        if (!pDst[c]) {
            return NPP_NULL_POINTER_ERROR;
        }
    }
    
    if (nSrcStep <= 0 || nDstStep <= 0) {
        return NPP_STEP_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    return nppiCopy_32f_C3P3R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C3P3R(const Npp32f* pSrc, Npp32s nSrcStep,
                            Npp32f* const pDst[3], Npp32s nDstStep,
                            NppiSize oSizeROI) {
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    return nppiCopy_32f_C3P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}