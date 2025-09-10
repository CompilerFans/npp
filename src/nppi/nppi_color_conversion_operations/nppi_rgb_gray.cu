#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * CUDA kernels for NPP RGB to Grayscale Conversion Functions
 */

// ITU-R BT.709 standard luminance weights
#define WEIGHT_R 0.299f
#define WEIGHT_G 0.587f
#define WEIGHT_B 0.114f

/**
 * CUDA kernel for converting RGB (3-channel) to grayscale (8-bit)
 */
__global__ void nppiRGBToGray_8u_C3C1R_kernel(const Npp8u* pSrc, int nSrcStep, 
                                               Npp8u* pDst, int nDstStep,
                                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Calculate byte offsets correctly
        const Npp8u* src_pixel = pSrc + y * nSrcStep + x * 3;
        Npp8u* dst_pixel = pDst + y * nDstStep + x;
        
        // Get RGB values
        Npp8u r = src_pixel[0];
        Npp8u g = src_pixel[1];
        Npp8u b = src_pixel[2];
        
        // Calculate grayscale using ITU-R BT.709 weights
        float gray = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
        *dst_pixel = (Npp8u)(gray + 0.5f); // Round to nearest
    }
}

/**
 * CUDA kernel for converting RGBA (4-channel) to grayscale (8-bit), ignoring alpha
 */
__global__ void nppiRGBToGray_8u_AC4C1R_kernel(const Npp8u* pSrc, int nSrcStep, 
                                                Npp8u* pDst, int nDstStep,
                                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Calculate byte offsets correctly
        const Npp8u* src_pixel = pSrc + y * nSrcStep + x * 4;
        Npp8u* dst_pixel = pDst + y * nDstStep + x;
        
        // Get RGB values (ignore alpha channel)
        Npp8u r = src_pixel[0];
        Npp8u g = src_pixel[1];
        Npp8u b = src_pixel[2];
        // Alpha channel at index 3 is ignored
        
        // Calculate grayscale using ITU-R BT.709 weights
        float gray = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
        *dst_pixel = (Npp8u)(gray + 0.5f); // Round to nearest
    }
}

/**
 * CUDA kernel for converting RGB (3-channel) to grayscale (32-bit float)
 */
__global__ void nppiRGBToGray_32f_C3C1R_kernel(const Npp32f* pSrc, int nSrcStep, 
                                                Npp32f* pDst, int nDstStep,
                                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Calculate byte offsets correctly for 32-bit float
        const Npp32f* src_pixel = (const Npp32f*)((const char*)pSrc + y * nSrcStep) + x * 3;
        Npp32f* dst_pixel = (Npp32f*)((char*)pDst + y * nDstStep) + x;
        
        // Get RGB values
        Npp32f r = src_pixel[0];
        Npp32f g = src_pixel[1];
        Npp32f b = src_pixel[2];
        
        // Calculate grayscale using ITU-R BT.709 weights
        *dst_pixel = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
    }
}

/**
 * CUDA kernel for converting RGBA (4-channel) to grayscale (32-bit float), ignoring alpha
 */
__global__ void nppiRGBToGray_32f_AC4C1R_kernel(const Npp32f* pSrc, int nSrcStep, 
                                                 Npp32f* pDst, int nDstStep,
                                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Calculate byte offsets correctly for 32-bit float
        const Npp32f* src_pixel = (const Npp32f*)((const char*)pSrc + y * nSrcStep) + x * 4;
        Npp32f* dst_pixel = (Npp32f*)((char*)pDst + y * nDstStep) + x;
        
        // Get RGB values (ignore alpha channel)
        Npp32f r = src_pixel[0];
        Npp32f g = src_pixel[1];
        Npp32f b = src_pixel[2];
        // Alpha channel at index 3 is ignored
        
        // Calculate grayscale using ITU-R BT.709 weights
        *dst_pixel = WEIGHT_R * r + WEIGHT_G * g + WEIGHT_B * b;
    }
}

extern "C" {

/**
 * Convert 3-channel RGB to single-channel grayscale (8-bit)
 */
NppStatus nppiRGBToGray_8u_C3C1R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep,
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiRGBToGray_8u_C3C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    // Synchronize to ensure kernel completion
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

/**
 * Convert 4-channel RGBA to single-channel grayscale (8-bit), ignoring alpha channel
 */
NppStatus nppiRGBToGray_8u_AC4C1R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep,
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiRGBToGray_8u_AC4C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    // Synchronize to ensure kernel completion
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

/**
 * Convert 3-channel RGB to single-channel grayscale (32-bit float)
 */
NppStatus nppiRGBToGray_32f_C3C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiRGBToGray_32f_C3C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    // Synchronize to ensure kernel completion
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

/**
 * Convert 4-channel RGBA to single-channel grayscale (32-bit float), ignoring alpha channel
 */
NppStatus nppiRGBToGray_32f_AC4C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiRGBToGray_32f_AC4C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    // Synchronize to ensure kernel completion
    cudaStatus = cudaStreamSynchronize(nppStreamCtx.hStream);
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

} // extern "C"