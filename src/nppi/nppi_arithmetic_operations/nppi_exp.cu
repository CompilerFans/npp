#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * CUDA kernels for NPP Exponential Operations
 */

/**
 * CUDA kernel for 8-bit unsigned exponential with scaling
 */
__global__ void nppiExp_8u_C1RSfs_kernel(const Npp8u* pSrc, int nSrcStep,
                                          Npp8u* pDst, int nDstStep,
                                          int width, int height, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp8u* src_pixel = pSrc + y * nSrcStep + x;
        Npp8u* dst_pixel = pDst + y * nDstStep + x;
        
        int src_val = *src_pixel;
        
        // Since input is 8-bit, we need to scale down to avoid overflow
        // Typical approach: divide by a factor before exp, then scale result
        float scaled_input = (float)src_val / 255.0f * 5.0f; // Scale to [0, 5] range
        
        // Compute exponential and apply scaling
        float exp_val = expf(scaled_input);
        int result = (int)(exp_val * (1 << nScaleFactor) + 0.5f);
        
        // Saturate to 8-bit range
        *dst_pixel = (Npp8u)min(result, 255);
    }
}

/**
 * CUDA kernel for 16-bit unsigned exponential with scaling
 */
__global__ void nppiExp_16u_C1RSfs_kernel(const Npp16u* pSrc, int nSrcStep,
                                           Npp16u* pDst, int nDstStep,
                                           int width, int height, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp16u* src_pixel = (const Npp16u*)((const char*)pSrc + y * nSrcStep) + x;
        Npp16u* dst_pixel = (Npp16u*)((char*)pDst + y * nDstStep) + x;
        
        int src_val = *src_pixel;
        
        // Scale input to reasonable range for exp
        float scaled_input = (float)src_val / 65535.0f * 10.0f; // Scale to [0, 10] range
        
        // Compute exponential and apply scaling
        float exp_val = expf(scaled_input);
        int result = (int)(exp_val * (1 << nScaleFactor) + 0.5f);
        
        // Saturate to 16-bit unsigned range
        *dst_pixel = (Npp16u)min(result, 65535);
    }
}

/**
 * CUDA kernel for 16-bit signed exponential with scaling
 */
__global__ void nppiExp_16s_C1RSfs_kernel(const Npp16s* pSrc, int nSrcStep,
                                           Npp16s* pDst, int nDstStep,
                                           int width, int height, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp16s* src_pixel = (const Npp16s*)((const char*)pSrc + y * nSrcStep) + x;
        Npp16s* dst_pixel = (Npp16s*)((char*)pDst + y * nDstStep) + x;
        
        int src_val = *src_pixel;
        
        // Scale input to reasonable range for exp (can be negative)
        float scaled_input = (float)src_val / 32768.0f * 5.0f; // Scale to [-5, 5] range
        
        // Compute exponential and apply scaling
        float exp_val = expf(scaled_input);
        int result = (int)(exp_val * (1 << nScaleFactor) + 0.5f);
        
        // Saturate to 16-bit signed range
        *dst_pixel = (Npp16s)max(min(result, 32767), -32768);
    }
}

/**
 * CUDA kernel for 32-bit float exponential (no scaling)
 */
__global__ void nppiExp_32f_C1R_kernel(const Npp32f* pSrc, int nSrcStep,
                                        Npp32f* pDst, int nDstStep,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp32f* src_pixel = (const Npp32f*)((const char*)pSrc + y * nSrcStep) + x;
        Npp32f* dst_pixel = (Npp32f*)((char*)pDst + y * nDstStep) + x;
        
        float src_val = *src_pixel;
        
        // Compute exponential directly
        *dst_pixel = expf(src_val);
    }
}

extern "C" {

/**
 * 8-bit unsigned exponential with scaling
 */
NppStatus nppiExp_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep,
                                      NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiExp_8u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
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
 * 16-bit unsigned exponential with scaling
 */
NppStatus nppiExp_16u_C1RSfs_Ctx_cuda(const Npp16u* pSrc, int nSrcStep, Npp16u* pDst, int nDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiExp_16u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
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
 * 16-bit signed exponential with scaling
 */
NppStatus nppiExp_16s_C1RSfs_Ctx_cuda(const Npp16s* pSrc, int nSrcStep, Npp16s* pDst, int nDstStep,
                                       NppiSize oSizeROI, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiExp_16s_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
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
 * 32-bit float exponential (no scaling)
 */
NppStatus nppiExp_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiExp_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
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