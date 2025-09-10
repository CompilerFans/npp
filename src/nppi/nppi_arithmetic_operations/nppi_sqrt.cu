#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * CUDA kernels for NPP Square Root Operations
 */

/**
 * CUDA kernel for 8-bit unsigned square root with scaling
 */
__global__ void nppiSqrt_8u_C1RSfs_kernel(const Npp8u* pSrc, int nSrcStep,
                                           Npp8u* pDst, int nDstStep,
                                           int width, int height, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp8u* src_pixel = pSrc + y * nSrcStep + x;
        Npp8u* dst_pixel = pDst + y * nDstStep + x;
        
        int src_val = *src_pixel;
        // Compute square root and apply scaling
        float sqrt_val = sqrtf((float)src_val);
        int result = (int)(sqrt_val * (1 << nScaleFactor) + 0.5f);
        
        // Saturate to 8-bit range
        *dst_pixel = (Npp8u)min(result, 255);
    }
}

/**
 * CUDA kernel for 16-bit unsigned square root with scaling
 */
__global__ void nppiSqrt_16u_C1RSfs_kernel(const Npp16u* pSrc, int nSrcStep,
                                            Npp16u* pDst, int nDstStep,
                                            int width, int height, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp16u* src_pixel = (const Npp16u*)((const char*)pSrc + y * nSrcStep) + x;
        Npp16u* dst_pixel = (Npp16u*)((char*)pDst + y * nDstStep) + x;
        
        int src_val = *src_pixel;
        // Compute square root and apply scaling
        float sqrt_val = sqrtf((float)src_val);
        int result = (int)(sqrt_val * (1 << nScaleFactor) + 0.5f);
        
        // Saturate to 16-bit unsigned range
        *dst_pixel = (Npp16u)min(result, 65535);
    }
}

/**
 * CUDA kernel for 16-bit signed square root with scaling
 */
__global__ void nppiSqrt_16s_C1RSfs_kernel(const Npp16s* pSrc, int nSrcStep,
                                            Npp16s* pDst, int nDstStep,
                                            int width, int height, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp16s* src_pixel = (const Npp16s*)((const char*)pSrc + y * nSrcStep) + x;
        Npp16s* dst_pixel = (Npp16s*)((char*)pDst + y * nDstStep) + x;
        
        int src_val = *src_pixel;
        
        // Handle negative values - square root of negative is undefined
        if (src_val < 0) {
            *dst_pixel = 0; // Set to 0 for negative inputs
        } else {
            // Compute square root and apply scaling
            float sqrt_val = sqrtf((float)src_val);
            int result = (int)(sqrt_val * (1 << nScaleFactor) + 0.5f);
            
            // Saturate to 16-bit signed range
            *dst_pixel = (Npp16s)max(min(result, 32767), -32768);
        }
    }
}

/**
 * CUDA kernel for 32-bit float square root (no scaling)
 */
__global__ void nppiSqrt_32f_C1R_kernel(const Npp32f* pSrc, int nSrcStep,
                                         Npp32f* pDst, int nDstStep,
                                         int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp32f* src_pixel = (const Npp32f*)((const char*)pSrc + y * nSrcStep) + x;
        Npp32f* dst_pixel = (Npp32f*)((char*)pDst + y * nDstStep) + x;
        
        float src_val = *src_pixel;
        
        // Handle negative values - return NaN or 0 based on typical NPP behavior
        if (src_val < 0.0f) {
            *dst_pixel = 0.0f; // Set to 0 for negative inputs
        } else {
            *dst_pixel = sqrtf(src_val);
        }
    }
}

extern "C" {

/**
 * 8-bit unsigned square root with scaling
 */
NppStatus nppiSqrt_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiSqrt_8u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    // Synchronize to ensure kernel completion
    if (nppStreamCtx.hStream == 0) {
        cudaStatus = cudaDeviceSynchronize();
    } else {
        cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
    }
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

/**
 * 16-bit unsigned square root with scaling
 */
NppStatus nppiSqrt_16u_C1RSfs_Ctx_cuda(const Npp16u* pSrc, int nSrcStep, Npp16u* pDst, int nDstStep,
                                        NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiSqrt_16u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    // Synchronize to ensure kernel completion
    if (nppStreamCtx.hStream == 0) {
        cudaStatus = cudaDeviceSynchronize();
    } else {
        cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
    }
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

/**
 * 16-bit signed square root with scaling
 */
NppStatus nppiSqrt_16s_C1RSfs_Ctx_cuda(const Npp16s* pSrc, int nSrcStep, Npp16s* pDst, int nDstStep,
                                        NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiSqrt_16s_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    // Synchronize to ensure kernel completion
    if (nppStreamCtx.hStream == 0) {
        cudaStatus = cudaDeviceSynchronize();
    } else {
        cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
    }
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

/**
 * 32-bit float square root (no scaling)
 */
NppStatus nppiSqrt_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiSqrt_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    // Synchronize to ensure kernel completion
    if (nppStreamCtx.hStream == 0) {
        cudaStatus = cudaDeviceSynchronize();
    } else {
        cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
    }
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

} // extern "C"