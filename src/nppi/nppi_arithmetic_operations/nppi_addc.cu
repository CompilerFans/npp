#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * CUDA kernels for NPP Image Add Constant operations
 */

/**
 * CUDA kernel for adding constant to 8-bit unsigned 1-channel image
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
    
    // Add constant and scale - NVIDIA NPP uses round-to-nearest-even for 8u
    int result = static_cast<int>(srcValue) + static_cast<int>(nConstant);
    if (nScaleFactor > 0) {
        // Standard rounding: add 0.5 equivalent before right shift
        result += (1 << (nScaleFactor - 1));
        result = result >> nScaleFactor;
    }
    
    // Clamp to 8-bit range
    result = max(0, min(255, result));
    
    // Store result
    *dstPixel = static_cast<Npp8u>(result);
}

/**
 * CUDA kernel for adding constants to 8-bit unsigned 3-channel image
 */
__global__ void addC_8u_C3RSfs_kernel(const Npp8u* __restrict__ pSrc, int nSrcStep,
                                      Npp8u nConstant0, Npp8u nConstant1, Npp8u nConstant2,
                                      Npp8u* __restrict__ pDst, int nDstStep,
                                      int width, int height, int nScaleFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate memory addresses for 3-channel data
    const Npp8u* srcPixel = pSrc + y * nSrcStep + x * 3;
    Npp8u* dstPixel = pDst + y * nDstStep + x * 3;
    
    // Process each channel
    int result;
    
    // Channel 0
    result = static_cast<int>(srcPixel[0]) + static_cast<int>(nConstant0);
    if (nScaleFactor > 0) {
        result += (1 << (nScaleFactor - 1));
        result = result >> nScaleFactor;
    }
    dstPixel[0] = static_cast<Npp8u>(max(0, min(255, result)));
    
    // Channel 1
    result = static_cast<int>(srcPixel[1]) + static_cast<int>(nConstant1);
    if (nScaleFactor > 0) {
        result += (1 << (nScaleFactor - 1));
        result = result >> nScaleFactor;
    }
    dstPixel[1] = static_cast<Npp8u>(max(0, min(255, result)));
    
    // Channel 2
    result = static_cast<int>(srcPixel[2]) + static_cast<int>(nConstant2);
    if (nScaleFactor > 0) {
        result += (1 << (nScaleFactor - 1));
        result = result >> nScaleFactor;
    }
    dstPixel[2] = static_cast<Npp8u>(max(0, min(255, result)));
}

/**
 * CUDA kernel for adding constants to 8-bit unsigned 4-channel image with alpha
 */
__global__ void addC_8u_C4RSfs_kernel(const Npp8u* __restrict__ pSrc, int nSrcStep,
                                      Npp8u nConstant0, Npp8u nConstant1, 
                                      Npp8u nConstant2, Npp8u nConstant3,
                                      Npp8u* __restrict__ pDst, int nDstStep,
                                      int width, int height, int nScaleFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate memory addresses for 4-channel data
    const Npp8u* srcPixel = pSrc + y * nSrcStep + x * 4;
    Npp8u* dstPixel = pDst + y * nDstStep + x * 4;
    
    // Process each channel
    int result;
    
    // Channels 0-3
    result = static_cast<int>(srcPixel[0]) + static_cast<int>(nConstant0);
    if (nScaleFactor > 0) {
        result += (1 << (nScaleFactor - 1));
        result = result >> nScaleFactor;
    }
    dstPixel[0] = static_cast<Npp8u>(max(0, min(255, result)));
    
    result = static_cast<int>(srcPixel[1]) + static_cast<int>(nConstant1);
    if (nScaleFactor > 0) {
        result += (1 << (nScaleFactor - 1));
        result = result >> nScaleFactor;
    }
    dstPixel[1] = static_cast<Npp8u>(max(0, min(255, result)));
    
    result = static_cast<int>(srcPixel[2]) + static_cast<int>(nConstant2);
    if (nScaleFactor > 0) {
        result += (1 << (nScaleFactor - 1));
        result = result >> nScaleFactor;
    }
    dstPixel[2] = static_cast<Npp8u>(max(0, min(255, result)));
    
    result = static_cast<int>(srcPixel[3]) + static_cast<int>(nConstant3);
    if (nScaleFactor > 0) {
        result += (1 << (nScaleFactor - 1));
        result = result >> nScaleFactor;
    }
    dstPixel[3] = static_cast<Npp8u>(max(0, min(255, result)));
}

/**
 * CUDA kernel for adding constants to 8-bit unsigned 4-channel image (AC4 - alpha unchanged)
 */
__global__ void addC_8u_AC4RSfs_kernel(const Npp8u* __restrict__ pSrc, int nSrcStep,
                                       Npp8u nConstant0, Npp8u nConstant1, Npp8u nConstant2,
                                       Npp8u* __restrict__ pDst, int nDstStep,
                                       int width, int height, int nScaleFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate memory addresses for 4-channel data
    const Npp8u* srcPixel = pSrc + y * nSrcStep + x * 4;
    Npp8u* dstPixel = pDst + y * nDstStep + x * 4;
    
    // Process RGB channels only
    int result;
    
    // Channel 0 (R)
    result = static_cast<int>(srcPixel[0]) + static_cast<int>(nConstant0);
    if (nScaleFactor > 0) {
        result += (1 << (nScaleFactor - 1));
        result = result >> nScaleFactor;
    }
    dstPixel[0] = static_cast<Npp8u>(max(0, min(255, result)));
    
    // Channel 1 (G)
    result = static_cast<int>(srcPixel[1]) + static_cast<int>(nConstant1);
    if (nScaleFactor > 0) {
        result += (1 << (nScaleFactor - 1));
        result = result >> nScaleFactor;
    }
    dstPixel[1] = static_cast<Npp8u>(max(0, min(255, result)));
    
    // Channel 2 (B)
    result = static_cast<int>(srcPixel[2]) + static_cast<int>(nConstant2);
    if (nScaleFactor > 0) {
        result += (1 << (nScaleFactor - 1));
        result = result >> nScaleFactor;
    }
    dstPixel[2] = static_cast<Npp8u>(max(0, min(255, result)));
    
    // Channel 3 (Alpha) - copy unchanged
    dstPixel[3] = srcPixel[3];
}

/**
 * CUDA kernel for adding constant to 16-bit unsigned 1-channel image
 */
__global__ void addC_16u_C1RSfs_kernel(const Npp16u* __restrict__ pSrc, int nSrcStep, 
                                       Npp16u nConstant, Npp16u* __restrict__ pDst, 
                                       int nDstStep, int width, int height, int nScaleFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate memory addresses (16-bit addressing)
    const Npp16u* srcPixel = (const Npp16u*)((const char*)pSrc + y * nSrcStep) + x;
    Npp16u* dstPixel = (Npp16u*)((char*)pDst + y * nDstStep) + x;
    
    // Load source pixel
    Npp16u srcValue = *srcPixel;
    
    // Add constant and scale with rounding
    int result = static_cast<int>(srcValue) + static_cast<int>(nConstant);
    if (nScaleFactor > 0) {
        // Add rounding bias before right shift to match NVIDIA NPP behavior
        result = (result + (1 << (nScaleFactor - 1))) >> nScaleFactor;
    }
    
    // Clamp to 16-bit unsigned range
    result = max(0, min(65535, result));
    
    // Store result
    *dstPixel = static_cast<Npp16u>(result);
}

/**
 * CUDA kernel for adding constant to 16-bit signed 1-channel image
 */
__global__ void addC_16s_C1RSfs_kernel(const Npp16s* __restrict__ pSrc, int nSrcStep, 
                                       Npp16s nConstant, Npp16s* __restrict__ pDst, 
                                       int nDstStep, int width, int height, int nScaleFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate memory addresses (16-bit addressing)
    const Npp16s* srcPixel = (const Npp16s*)((const char*)pSrc + y * nSrcStep) + x;
    Npp16s* dstPixel = (Npp16s*)((char*)pDst + y * nDstStep) + x;
    
    // Load source pixel
    Npp16s srcValue = *srcPixel;
    
    // Add constant and scale with rounding
    int result = static_cast<int>(srcValue) + static_cast<int>(nConstant);
    if (nScaleFactor > 0) {
        // Add rounding bias before right shift to match NVIDIA NPP behavior
        result = (result + (1 << (nScaleFactor - 1))) >> nScaleFactor;
    }
    
    // Clamp to 16-bit signed range
    result = max(-32768, min(32767, result));
    
    // Store result
    *dstPixel = static_cast<Npp16s>(result);
}

/**
 * CUDA kernel for adding constant to 32-bit float 1-channel image
 */
__global__ void addC_32f_C1R_kernel(const Npp32f* __restrict__ pSrc, int nSrcStep, 
                                    Npp32f nConstant, Npp32f* __restrict__ pDst, 
                                    int nDstStep, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate memory addresses (32-bit addressing)
    const Npp32f* srcPixel = (const Npp32f*)((const char*)pSrc + y * nSrcStep) + x;
    Npp32f* dstPixel = (Npp32f*)((char*)pDst + y * nDstStep) + x;
    
    // Load source pixel
    Npp32f srcValue = *srcPixel;
    
    // Add constant (no scaling for float)
    Npp32f result = srcValue + nConstant;
    
    // Store result
    *dstPixel = result;
}

extern "C" {

/**
 * CUDA implementation of nppiAddC_8u_C1RSfs_Ctx
 */
NppStatus nppiAddC_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx)
{
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
 * CUDA implementation of nppiAddC_8u_C3RSfs_Ctx
 */
NppStatus nppiAddC_8u_C3RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u aConstants[3],
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx)
{
    // Set up CUDA grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel with the specified CUDA stream
    addC_8u_C3RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, aConstants[0], aConstants[1], aConstants[2],
        pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor
    );
    
    // Check for kernel launch errors
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_NO_ERROR;
}

/**
 * CUDA implementation of nppiAddC_8u_C4RSfs_Ctx
 */
NppStatus nppiAddC_8u_C4RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u aConstants[4],
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx)
{
    // Set up CUDA grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel with the specified CUDA stream
    addC_8u_C4RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, aConstants[0], aConstants[1], aConstants[2], aConstants[3],
        pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor
    );
    
    // Check for kernel launch errors
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_NO_ERROR;
}

/**
 * CUDA implementation of nppiAddC_8u_AC4RSfs_Ctx
 */
NppStatus nppiAddC_8u_AC4RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u aConstants[3],
                                       Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx)
{
    // Set up CUDA grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel with the specified CUDA stream
    addC_8u_AC4RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, aConstants[0], aConstants[1], aConstants[2],
        pDst, nDstStep, oSizeROI.width, oSizeROI.height, nScaleFactor
    );
    
    // Check for kernel launch errors
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_NO_ERROR;
}

/**
 * CUDA implementation of nppiAddC_16u_C1RSfs_Ctx
 */
NppStatus nppiAddC_16u_C1RSfs_Ctx_cuda(const Npp16u* pSrc1, int nSrc1Step, const Npp16u nConstant,
                                       Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx)
{
    // Set up CUDA grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel with the specified CUDA stream
    addC_16u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
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
 * CUDA implementation of nppiAddC_16s_C1RSfs_Ctx
 */
NppStatus nppiAddC_16s_C1RSfs_Ctx_cuda(const Npp16s* pSrc1, int nSrc1Step, const Npp16s nConstant,
                                       Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx)
{
    // Set up CUDA grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel with the specified CUDA stream
    addC_16s_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
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
 * CUDA implementation of nppiAddC_32f_C1R_Ctx
 */
NppStatus nppiAddC_32f_C1R_Ctx_cuda(const Npp32f* pSrc1, int nSrc1Step, const Npp32f nConstant,
                                    Npp32f* pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx)
{
    // Set up CUDA grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel with the specified CUDA stream
    addC_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, nConstant, pDst, nDstStep, 
        oSizeROI.width, oSizeROI.height
    );
    
    // Check for kernel launch errors
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_NO_ERROR;
}

} // extern "C"