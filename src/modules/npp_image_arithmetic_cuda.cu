#include "npp_image_arithmetic.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * CUDA kernel for adding constant to 8-bit unsigned image
 * 
 * Each thread processes one pixel
 */
__global__ void addC_8u_C1RSfs_kernel(const Npp8u* __restrict__ pSrc, int nSrcStep, 
                                      Npp8u nConstant, Npp8u* __restrict__ pDst, 
                                      int nDstStep, int width, int height, int nScaleFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate memory addresses
    const Npp8u* srcPixel = pSrc + y * nSrcStep + x;
    Npp8u* dstPixel = pDst + y * nDstStep + x;
    
    // Load source pixel
    Npp8u srcValue = *srcPixel;
    
    // Add constant and scale
    int result = static_cast<int>(srcValue) + static_cast<int>(nConstant);
    result = result >> nScaleFactor;
    
    // Clamp to 8-bit range
    result = max(0, min(255, result));
    
    // Store result
    *dstPixel = static_cast<Npp8u>(result);
}

/**
 * CUDA implementation of nppiAddC_8u_C1RSfs_Ctx
 */
NppStatus 
nppiAddC_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                             Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                             NppStreamContext nppStreamCtx)
{
    // Parameter validation
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }
    
    if (nSrc1Step < oSizeROI.width || nDstStep < oSizeROI.width) {
        return NPP_STRIDE_ERROR;
    }
    
    if (nScaleFactor < 0 || nScaleFactor > 16) {
        return NPP_BAD_ARGUMENT_ERROR;
    }
    
    // Set up CUDA grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel with the specified CUDA stream
    addC_8u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
        oSizeROI.width, oSizeROI.height, nScaleFactor
    );
    
    // Check for kernel launch errors
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_NO_ERROR;
}

/**
 * Optimized CUDA kernel using shared memory for better performance
 * (when processing larger images)
 */
__global__ void addC_8u_C1RSfs_shared_kernel(const Npp8u* __restrict__ pSrc, int nSrcStep, 
                                              Npp8u nConstant, Npp8u* __restrict__ pDst, 
                                              int nDstStep, int width, int height, int nScaleFactor)
{
    __shared__ Npp8u sharedMem[16][16];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int localX = threadIdx.x;
    int localY = threadIdx.y;
    
    if (x < width && y < height) {
        // Load to shared memory
        sharedMem[localY][localX] = pSrc[y * nSrcStep + x];
    }
    __syncthreads();
    
    if (x < width && y < height) {
        // Process data
        Npp8u srcValue = sharedMem[localY][localX];
        int result = static_cast<int>(srcValue) + static_cast<int>(nConstant);
        result = result >> nScaleFactor;
        result = max(0, min(255, result));
        pDst[y * nDstStep + x] = static_cast<Npp8u>(result);
    }
}