#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * CUDA Kernels for NPP Image Div Functions
 */

// ============================================================================
// Device kernels
// ============================================================================

// 8-bit unsigned div with scaling, single channel
__global__ void nppiDiv_8u_C1RSfs_kernel(
    const Npp8u* pSrc1, int nSrc1Step,
    const Npp8u* pSrc2, int nSrc2Step,
    Npp8u* pDst, int nDstStep,
    int width, int height, int nScaleFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp8u* src1Row = (const Npp8u*)((const char*)pSrc1 + y * nSrc1Step);
        const Npp8u* src2Row = (const Npp8u*)((const char*)pSrc2 + y * nSrc2Step);
        Npp8u* dstRow = (Npp8u*)((char*)pDst + y * nDstStep);
        
        // Division with scaling (pSrc2 / pSrc1) - following NPP convention
        int src1Val = (int)src1Row[x];  // divisor
        int src2Val = (int)src2Row[x];  // dividend
        int result;
        
        if (src1Val == 0) {
            // Division by zero - saturate to maximum value
            result = 255;
        } else {
            // Scale up numerator before division to maintain precision
            if (nScaleFactor > 0) {
                src2Val <<= nScaleFactor;
            }
            result = src2Val / src1Val;
        }
        
        // Saturate to 8-bit range (clamp to 0-255)
        dstRow[x] = (Npp8u)(result < 0 ? 0 : (result > 255 ? 255 : result));
    }
}

// 8-bit unsigned div with scaling, 3 channels
__global__ void nppiDiv_8u_C3RSfs_kernel(
    const Npp8u* pSrc1, int nSrc1Step,
    const Npp8u* pSrc2, int nSrc2Step,
    Npp8u* pDst, int nDstStep,
    int width, int height, int nScaleFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp8u* src1Row = (const Npp8u*)((const char*)pSrc1 + y * nSrc1Step);
        const Npp8u* src2Row = (const Npp8u*)((const char*)pSrc2 + y * nSrc2Step);
        Npp8u* dstRow = (Npp8u*)((char*)pDst + y * nDstStep);
        
        int idx = x * 3;
        
        // Process 3 channels
        for (int c = 0; c < 3; c++) {
            int src1Val = (int)src1Row[idx + c];
            int src2Val = (int)src2Row[idx + c];
            int result;
            
            if (src1Val == 0) {
                // Division by zero - saturate to maximum value
                result = 255;
            } else {
                // Scale up numerator before division to maintain precision
                if (nScaleFactor > 0) {
                    src2Val <<= nScaleFactor;
                }
                result = src2Val / src1Val;
            }
            
            // Saturate to 8-bit range
            dstRow[idx + c] = (Npp8u)(result < 0 ? 0 : (result > 255 ? 255 : result));
        }
    }
}

// 16-bit unsigned div with scaling, single channel
__global__ void nppiDiv_16u_C1RSfs_kernel(
    const Npp16u* pSrc1, int nSrc1Step,
    const Npp16u* pSrc2, int nSrc2Step,
    Npp16u* pDst, int nDstStep,
    int width, int height, int nScaleFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp16u* src1Row = (const Npp16u*)((const char*)pSrc1 + y * nSrc1Step);
        const Npp16u* src2Row = (const Npp16u*)((const char*)pSrc2 + y * nSrc2Step);
        Npp16u* dstRow = (Npp16u*)((char*)pDst + y * nDstStep);
        
        // Division with scaling - use larger intermediate type
        long long src1Val = (long long)src1Row[x];
        long long src2Val = (long long)src2Row[x];
        long long result;
        
        if (src1Val == 0) {
            // Division by zero - saturate to maximum value
            result = 65535;
        } else {
            // Scale up numerator before division to maintain precision
            if (nScaleFactor > 0) {
                src2Val <<= nScaleFactor;
            }
            result = src2Val / src1Val;
        }
        
        // Saturate to 16-bit range
        dstRow[x] = (Npp16u)(result < 0 ? 0 : (result > 65535 ? 65535 : result));
    }
}

// 16-bit signed div with scaling, single channel
__global__ void nppiDiv_16s_C1RSfs_kernel(
    const Npp16s* pSrc1, int nSrc1Step,
    const Npp16s* pSrc2, int nSrc2Step,
    Npp16s* pDst, int nDstStep,
    int width, int height, int nScaleFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp16s* src1Row = (const Npp16s*)((const char*)pSrc1 + y * nSrc1Step);
        const Npp16s* src2Row = (const Npp16s*)((const char*)pSrc2 + y * nSrc2Step);
        Npp16s* dstRow = (Npp16s*)((char*)pDst + y * nDstStep);
        
        // Division with scaling - use larger intermediate type
        long long src1Val = (long long)src1Row[x];
        long long src2Val = (long long)src2Row[x];
        long long result;
        
        if (src1Val == 0) {
            // Division by zero - saturate based on sign of numerator  
            result = (src2Val >= 0) ? 32767 : -32768;
        } else {
            // Scale up numerator before division to maintain precision
            if (nScaleFactor > 0) {
                src2Val <<= nScaleFactor;
            }
            result = src2Val / src1Val;
        }
        
        // Saturate to 16-bit signed range
        dstRow[x] = (Npp16s)(result < -32768 ? -32768 : (result > 32767 ? 32767 : result));
    }
}

// 32-bit float div, single channel (no scaling)
__global__ void nppiDiv_32f_C1R_kernel(
    const Npp32f* pSrc1, int nSrc1Step,
    const Npp32f* pSrc2, int nSrc2Step,
    Npp32f* pDst, int nDstStep,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp32f* src1Row = (const Npp32f*)((const char*)pSrc1 + y * nSrc1Step);
        const Npp32f* src2Row = (const Npp32f*)((const char*)pSrc2 + y * nSrc2Step);
        Npp32f* dstRow = (Npp32f*)((char*)pDst + y * nDstStep);
        
        // Float division (pSrc2 / pSrc1) - handles IEEE 754 division by zero automatically
        dstRow[x] = src2Row[x] / src1Row[x];
    }
}

// 32-bit float div, 3 channels (no scaling)
__global__ void nppiDiv_32f_C3R_kernel(
    const Npp32f* pSrc1, int nSrc1Step,
    const Npp32f* pSrc2, int nSrc2Step,
    Npp32f* pDst, int nDstStep,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp32f* src1Row = (const Npp32f*)((const char*)pSrc1 + y * nSrc1Step);
        const Npp32f* src2Row = (const Npp32f*)((const char*)pSrc2 + y * nSrc2Step);
        Npp32f* dstRow = (Npp32f*)((char*)pDst + y * nDstStep);
        
        int idx = x * 3;
        
        // Process 3 channels (pSrc2 / pSrc1) - IEEE 754 handles division by zero
        dstRow[idx] = src2Row[idx] / src1Row[idx];       // R
        dstRow[idx + 1] = src2Row[idx + 1] / src1Row[idx + 1]; // G
        dstRow[idx + 2] = src2Row[idx + 2] / src1Row[idx + 2]; // B
    }
}

// ============================================================================
// Host functions
// ============================================================================

extern "C" {

// 8-bit unsigned
NppStatus nppiDiv_8u_C1RSfs_Ctx_cuda(
    const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
    Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
    NppStreamContext nppStreamCtx)
{
    dim3 block(16, 16);
    dim3 grid((oSizeROI.width + block.x - 1) / block.x,
              (oSizeROI.height + block.y - 1) / block.y);
              
    nppiDiv_8u_C1RSfs_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, nScaleFactor);
        
    return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDiv_8u_C3RSfs_Ctx_cuda(
    const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
    Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
    NppStreamContext nppStreamCtx)
{
    dim3 block(16, 16);
    dim3 grid((oSizeROI.width + block.x - 1) / block.x,
              (oSizeROI.height + block.y - 1) / block.y);
              
    nppiDiv_8u_C3RSfs_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, nScaleFactor);
        
    return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// 16-bit unsigned
NppStatus nppiDiv_16u_C1RSfs_Ctx_cuda(
    const Npp16u* pSrc1, int nSrc1Step, const Npp16u* pSrc2, int nSrc2Step,
    Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
    NppStreamContext nppStreamCtx)
{
    dim3 block(16, 16);
    dim3 grid((oSizeROI.width + block.x - 1) / block.x,
              (oSizeROI.height + block.y - 1) / block.y);
              
    nppiDiv_16u_C1RSfs_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, nScaleFactor);
        
    return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// 16-bit signed
NppStatus nppiDiv_16s_C1RSfs_Ctx_cuda(
    const Npp16s* pSrc1, int nSrc1Step, const Npp16s* pSrc2, int nSrc2Step,
    Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
    NppStreamContext nppStreamCtx)
{
    dim3 block(16, 16);
    dim3 grid((oSizeROI.width + block.x - 1) / block.x,
              (oSizeROI.height + block.y - 1) / block.y);
              
    nppiDiv_16s_C1RSfs_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, nScaleFactor);
        
    return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// 32-bit float
NppStatus nppiDiv_32f_C1R_Ctx_cuda(
    const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step,
    Npp32f* pDst, int nDstStep, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx)
{
    dim3 block(16, 16);
    dim3 grid((oSizeROI.width + block.x - 1) / block.x,
              (oSizeROI.height + block.y - 1) / block.y);
              
    nppiDiv_32f_C1R_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height);
        
    return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDiv_32f_C3R_Ctx_cuda(
    const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step,
    Npp32f* pDst, int nDstStep, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx)
{
    dim3 block(16, 16);
    dim3 grid((oSizeROI.width + block.x - 1) / block.x,
              (oSizeROI.height + block.y - 1) / block.y);
              
    nppiDiv_32f_C3R_kernel<<<grid, block, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height);
        
    return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}