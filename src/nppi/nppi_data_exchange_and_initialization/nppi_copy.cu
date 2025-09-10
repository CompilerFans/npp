#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

/**
 * CUDA kernels for NPP Image Copy Functions
 * Implements efficient GPU-based image copying operations
 */

// Kernel for 8-bit unsigned single channel copy
__global__ void nppiCopy_8u_C1R_kernel(const Npp8u* pSrc, int nSrcStep,
                                       Npp8u* pDst, int nDstStep,
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const Npp8u* src_row = (const Npp8u*)((const char*)pSrc + y * nSrcStep);
    Npp8u* dst_row = (Npp8u*)((char*)pDst + y * nDstStep);
    
    dst_row[x] = src_row[x];
}

// Kernel for 8-bit unsigned three channel copy
__global__ void nppiCopy_8u_C3R_kernel(const Npp8u* pSrc, int nSrcStep,
                                       Npp8u* pDst, int nDstStep,
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const Npp8u* src_row = (const Npp8u*)((const char*)pSrc + y * nSrcStep);
    Npp8u* dst_row = (Npp8u*)((char*)pDst + y * nDstStep);
    
    // Copy all three channels
    dst_row[x * 3 + 0] = src_row[x * 3 + 0]; // R or B
    dst_row[x * 3 + 1] = src_row[x * 3 + 1]; // G
    dst_row[x * 3 + 2] = src_row[x * 3 + 2]; // B or R
}

// Kernel for 8-bit unsigned four channel copy
__global__ void nppiCopy_8u_C4R_kernel(const Npp8u* pSrc, int nSrcStep,
                                       Npp8u* pDst, int nDstStep,
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const Npp8u* src_row = (const Npp8u*)((const char*)pSrc + y * nSrcStep);
    Npp8u* dst_row = (Npp8u*)((char*)pDst + y * nDstStep);
    
    // Copy all four channels (RGBA/BGRA)
    dst_row[x * 4 + 0] = src_row[x * 4 + 0];
    dst_row[x * 4 + 1] = src_row[x * 4 + 1];
    dst_row[x * 4 + 2] = src_row[x * 4 + 2];
    dst_row[x * 4 + 3] = src_row[x * 4 + 3];
}

// Kernel for 32-bit float single channel copy
__global__ void nppiCopy_32f_C1R_kernel(const Npp32f* pSrc, int nSrcStep,
                                        Npp32f* pDst, int nDstStep,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const Npp32f* src_row = (const Npp32f*)((const char*)pSrc + y * nSrcStep);
    Npp32f* dst_row = (Npp32f*)((char*)pDst + y * nDstStep);
    
    dst_row[x] = src_row[x];
}

// Optimized kernel using vectorized loads for better memory throughput
__global__ void nppiCopy_8u_C4R_vectorized_kernel(const Npp8u* pSrc, int nSrcStep,
                                                  Npp8u* pDst, int nDstStep,
                                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const uint32_t* src_row = (const uint32_t*)((const char*)pSrc + y * nSrcStep);
    uint32_t* dst_row = (uint32_t*)((char*)pDst + y * nDstStep);
    
    // Copy 4 bytes (1 pixel) at once
    dst_row[x] = src_row[x];
}

extern "C" {

// 8-bit unsigned single channel copy implementation
NppStatus nppiCopy_8u_C1R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep,
                                   Npp8u* pDst, int nDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiCopy_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 8-bit unsigned three channel copy implementation
NppStatus nppiCopy_8u_C3R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep,
                                   Npp8u* pDst, int nDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiCopy_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 8-bit unsigned four channel copy implementation
NppStatus nppiCopy_8u_C4R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep,
                                   Npp8u* pDst, int nDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    // Use vectorized version if width is aligned to 4-byte boundaries
    if (oSizeROI.width % 4 == 0 && nSrcStep % 4 == 0 && nDstStep % 4 == 0) {
        dim3 blockSize(16, 16);
        dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                      (oSizeROI.height + blockSize.y - 1) / blockSize.y);
        
        nppiCopy_8u_C4R_vectorized_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
            pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    } else {
        // Fall back to regular implementation
        dim3 blockSize(16, 16);
        dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                      (oSizeROI.height + blockSize.y - 1) / blockSize.y);
        
        nppiCopy_8u_C4R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
            pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    }
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 32-bit float single channel copy implementation
NppStatus nppiCopy_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep,
                                    Npp32f* pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiCopy_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

}