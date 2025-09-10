#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * CUDA kernels for NPP MulScale Operations
 * Multiplies two images and scales by maximum value for pixel bit width
 */

/**
 * CUDA kernel for 8-bit unsigned multiplication with scaling
 * Formula: dst = (src1 * src2) / 255
 */
__global__ void nppiMulScale_8u_C1R_kernel(const Npp8u* pSrc1, int nSrc1Step,
                                            const Npp8u* pSrc2, int nSrc2Step,
                                            Npp8u* pDst, int nDstStep,
                                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp8u* src1_pixel = (const Npp8u*)((const char*)pSrc1 + y * nSrc1Step) + x;
        const Npp8u* src2_pixel = (const Npp8u*)((const char*)pSrc2 + y * nSrc2Step) + x;
        Npp8u* dst_pixel = (Npp8u*)((char*)pDst + y * nDstStep) + x;
        
        int src1_val = *src1_pixel;
        int src2_val = *src2_pixel;
        
        // Multiply and scale by maximum value (255 for 8-bit)
        int result = (src1_val * src2_val) / 255;
        
        // Saturate to 8-bit range
        *dst_pixel = (Npp8u)min(result, 255);
    }
}

/**
 * CUDA kernel for 16-bit unsigned multiplication with scaling
 * Formula: dst = (src1 * src2) / 65535
 */
__global__ void nppiMulScale_16u_C1R_kernel(const Npp16u* pSrc1, int nSrc1Step,
                                             const Npp16u* pSrc2, int nSrc2Step,
                                             Npp16u* pDst, int nDstStep,
                                             int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp16u* src1_pixel = (const Npp16u*)((const char*)pSrc1 + y * nSrc1Step) + x;
        const Npp16u* src2_pixel = (const Npp16u*)((const char*)pSrc2 + y * nSrc2Step) + x;
        Npp16u* dst_pixel = (Npp16u*)((char*)pDst + y * nDstStep) + x;
        
        int src1_val = *src1_pixel;
        int src2_val = *src2_pixel;
        
        // Multiply and scale by maximum value (65535 for 16-bit)
        // Use 64-bit intermediate to avoid overflow
        long long result = ((long long)src1_val * src2_val) / 65535;
        
        // Saturate to 16-bit unsigned range
        *dst_pixel = (Npp16u)min(result, 65535LL);
    }
}

extern "C" {

/**
 * 8-bit unsigned multiplication with scaling
 */
NppStatus nppiMulScale_8u_C1R_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
                                        Npp8u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiMulScale_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    
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
 * 16-bit unsigned multiplication with scaling
 */
NppStatus nppiMulScale_16u_C1R_Ctx_cuda(const Npp16u* pSrc1, int nSrc1Step, const Npp16u* pSrc2, int nSrc2Step,
                                         Npp16u* pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiMulScale_16u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    
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