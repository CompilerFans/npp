#include "npp.h"
#include <cuda_runtime.h>

/**
 * NV12 to RGB Color Conversion Implementation
 * 
 * NV12 format: Planar YUV 4:2:0 format with interleaved U and V components
 * - Y plane: width x height luminance values
 * - UV plane: (width/2) x (height/2) interleaved chroma values
 */

// Forward declarations for CUDA kernels
extern "C" {
    cudaError_t nppiNV12ToRGB_8u_P2C3R_kernel(
        const Npp8u* pSrcY, int nSrcYStep,
        const Npp8u* pSrcUV, int nSrcUVStep,
        Npp8u* pDst, int nDstStep,
        NppiSize oSizeROI,
        cudaStream_t stream
    );
    
    cudaError_t nppiNV12ToRGB_709CSC_8u_P2C3R_kernel(
        const Npp8u* pSrcY, int nSrcYStep,
        const Npp8u* pSrcUV, int nSrcUVStep,
        Npp8u* pDst, int nDstStep,
        NppiSize oSizeROI,
        cudaStream_t stream
    );
}

/**
 * Convert NV12 format image to RGB.
 * Standard YUV to RGB conversion using ITU-R BT.601 coefficients.
 */
NppStatus nppiNV12ToRGB_8u_P2C3R_Ctx(
    const Npp8u * const pSrc[2], int rSrcStep,
    Npp8u * pDst, int nDstStep,
    NppiSize oSizeROI,
    NppStreamContext nppStreamCtx)
{
    // 参数验证
    if (!pSrc || !pSrc[0] || !pSrc[1] || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_WRONG_INTERSECTION_ROI_ERROR;
    }
    
    if (rSrcStep <= 0 || nDstStep <= 0) {
        return NPP_STEP_ERROR;
    }
    
    // NV12 format requires even dimensions for proper chroma handling
    if (oSizeROI.width % 2 != 0 || oSizeROI.height % 2 != 0) {
        return NPP_WRONG_INTERSECTION_ROI_ERROR;
    }
    
    // 调用CUDA内核
    cudaError_t cudaStatus = nppiNV12ToRGB_8u_P2C3R_kernel(
        pSrc[0], rSrcStep,           // Y plane
        pSrc[1], rSrcStep,           // UV plane (interleaved)
        pDst, nDstStep,              // RGB output
        oSizeROI,
        nppStreamCtx.hStream
    );
    
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_NO_ERROR;
}

/**
 * Convert NV12 format image to RGB (default stream context).
 */
NppStatus nppiNV12ToRGB_8u_P2C3R(
    const Npp8u * const pSrc[2], int rSrcStep,
    Npp8u * pDst, int nDstStep,
    NppiSize oSizeROI)
{
    NppStreamContext ctx;
    ctx.hStream = 0;  // 默认流
    return nppiNV12ToRGB_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

/**
 * Convert NV12 format image to RGB using ITU-R BT.709 color space conversion.
 * This function is specifically used by torchcodec for HDTV color space.
 */
NppStatus nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(
    const Npp8u * const pSrc[2], int rSrcStep,
    Npp8u * pDst, int nDstStep,
    NppiSize oSizeROI,
    NppStreamContext nppStreamCtx)
{
    // 参数验证
    if (!pSrc || !pSrc[0] || !pSrc[1] || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_WRONG_INTERSECTION_ROI_ERROR;
    }
    
    if (rSrcStep <= 0 || nDstStep <= 0) {
        return NPP_STEP_ERROR;
    }
    
    // NV12 format requires even dimensions
    if (oSizeROI.width % 2 != 0 || oSizeROI.height % 2 != 0) {
        return NPP_WRONG_INTERSECTION_ROI_ERROR;
    }
    
    // 调用BT.709 CUDA内核
    cudaError_t cudaStatus = nppiNV12ToRGB_709CSC_8u_P2C3R_kernel(
        pSrc[0], rSrcStep,           // Y plane
        pSrc[1], rSrcStep,           // UV plane
        pDst, nDstStep,              // RGB output
        oSizeROI,
        nppStreamCtx.hStream
    );
    
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_NO_ERROR;
}

/**
 * Convert NV12 format image to RGB using BT.709 (default stream context).
 */
NppStatus nppiNV12ToRGB_709CSC_8u_P2C3R(
    const Npp8u * const pSrc[2], int rSrcStep,
    Npp8u * pDst, int nDstStep,
    NppiSize oSizeROI)
{
    NppStreamContext ctx;
    ctx.hStream = 0;  // 默认流
    return nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

/**
 * Convert NV12 format image to RGB using ITU-R BT.709 HDTV coefficients.
 * This is an alias for the 709CSC function for compatibility.
 */
NppStatus nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
    const Npp8u * const pSrc[2], int rSrcStep,
    Npp8u * pDst, int nDstStep,
    NppiSize oSizeROI,
    NppStreamContext nppStreamCtx)
{
    return nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, 
                                             oSizeROI, nppStreamCtx);
}

/**
 * Convert NV12 format image to RGB using BT.709 HDTV (default stream context).
 */
NppStatus nppiNV12ToRGB_709HDTV_8u_P2C3R(
    const Npp8u * const pSrc[2], int rSrcStep,
    Npp8u * pDst, int nDstStep,
    NppiSize oSizeROI)
{
    NppStreamContext ctx;
    ctx.hStream = 0;  // 默认流
    return nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}