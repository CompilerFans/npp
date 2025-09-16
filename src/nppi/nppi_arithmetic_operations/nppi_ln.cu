#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * CUDA kernels for NPP Natural Logarithm Operations
 */

/**
 * CUDA kernel for 8-bit unsigned natural logarithm with scaling
 */
__global__ void nppiLn_8u_C1RSfs_kernel(const Npp8u* pSrc, int nSrcStep,
                                         Npp8u* pDst, int nDstStep,
                                         int width, int height, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp8u* src_pixel = pSrc + y * nSrcStep + x;
        Npp8u* dst_pixel = pDst + y * nDstStep + x;
        
        int src_val = *src_pixel;
        
        // Handle zero or negative values - ln(0) is undefined
        if (src_val <= 0) {
            *dst_pixel = 0; // Set to 0 for non-positive inputs
        } else {
            // Compute natural logarithm and apply NVIDIA NPP scaling: 2^nScaleFactor
            float ln_val = logf((float)src_val);
            int result = (int)(ln_val * (1 << nScaleFactor) + 0.5f);
            
            // Saturate to 8-bit range
            *dst_pixel = (Npp8u)max(min(result, 255), 0);
        }
    }
}

/**
 * CUDA kernel for 16-bit unsigned natural logarithm with scaling
 */
__global__ void nppiLn_16u_C1RSfs_kernel(const Npp16u* pSrc, int nSrcStep,
                                          Npp16u* pDst, int nDstStep,
                                          int width, int height, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp16u* src_pixel = (const Npp16u*)((const char*)pSrc + y * nSrcStep) + x;
        Npp16u* dst_pixel = (Npp16u*)((char*)pDst + y * nDstStep) + x;
        
        int src_val = *src_pixel;
        
        // Handle zero value - ln(0) is undefined
        if (src_val <= 0) {
            *dst_pixel = 0; // Set to 0 for zero input
        } else {
            // Compute natural logarithm and apply scaling
            float ln_val = logf((float)src_val);
            int result = (int)(ln_val * (1 << nScaleFactor) + 0.5f);
            
            // Saturate to 16-bit unsigned range
            *dst_pixel = (Npp16u)max(min(result, 65535), 0);
        }
    }
}

/**
 * CUDA kernel for 16-bit signed natural logarithm with scaling
 */
__global__ void nppiLn_16s_C1RSfs_kernel(const Npp16s* pSrc, int nSrcStep,
                                          Npp16s* pDst, int nDstStep,
                                          int width, int height, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp16s* src_pixel = (const Npp16s*)((const char*)pSrc + y * nSrcStep) + x;
        Npp16s* dst_pixel = (Npp16s*)((char*)pDst + y * nDstStep) + x;
        
        int src_val = *src_pixel;
        
        // Handle non-positive values - ln(x) is undefined for x <= 0
        if (src_val <= 0) {
            *dst_pixel = 0; // Set to 0 for non-positive inputs
        } else {
            // Compute natural logarithm and apply scaling
            float ln_val = logf((float)src_val);
            int result = (int)(ln_val * (1 << nScaleFactor) + 0.5f);
            
            // Saturate to 16-bit signed range
            *dst_pixel = (Npp16s)max(min(result, 32767), -32768);
        }
    }
}

/**
 * CUDA kernel for 32-bit float natural logarithm (no scaling)
 */
__global__ void nppiLn_32f_C1R_kernel(const Npp32f* pSrc, int nSrcStep,
                                       Npp32f* pDst, int nDstStep,
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Correct pointer arithmetic for step sizes
        const Npp32f* src_row = (const Npp32f*)((const char*)pSrc + y * nSrcStep);
        Npp32f* dst_row = (Npp32f*)((char*)pDst + y * nDstStep);
        
        float src_val = src_row[x];
        
        // Handle non-positive values - ln(x) is undefined for x <= 0
        if (src_val <= 0.0f) {
            // For compatibility with NVIDIA NPP, return NaN for non-positive values
            dst_row[x] = __fdividef(0.0f, 0.0f); // Generate NaN
        } else {
            dst_row[x] = logf(src_val);
        }
    }
}

extern "C" {

/**
 * 8-bit unsigned natural logarithm with scaling
 */
NppStatus nppiLn_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep,
                                     NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiLn_8u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
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
 * 16-bit unsigned natural logarithm with scaling
 */
NppStatus nppiLn_16u_C1RSfs_Ctx_cuda(const Npp16u* pSrc, int nSrcStep, Npp16u* pDst, int nDstStep,
                                      NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiLn_16u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
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
 * 16-bit signed natural logarithm with scaling
 */
NppStatus nppiLn_16s_C1RSfs_Ctx_cuda(const Npp16s* pSrc, int nSrcStep, Npp16s* pDst, int nDstStep,
                                      NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiLn_16s_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
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
 * 32-bit float natural logarithm (no scaling)
 */
NppStatus nppiLn_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiLn_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
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