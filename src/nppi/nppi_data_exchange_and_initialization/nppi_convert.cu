#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * CUDA kernels for NPP Image Convert operations
 */

/**
 * CUDA kernel for converting 8-bit unsigned to 32-bit float, single channel
 */
__global__ void convert_8u32f_C1R_kernel(const Npp8u* __restrict__ pSrc, int nSrcStep,
                                         Npp32f* __restrict__ pDst, int nDstStep,
                                         int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const Npp8u* srcRow = (const Npp8u*)(((const char*)pSrc) + y * nSrcStep);
    Npp32f* dstRow = (Npp32f*)(((char*)pDst) + y * nDstStep);
    
    // Convert 8-bit unsigned to 32-bit float (0-255 -> 0.0-255.0)
    dstRow[x] = (Npp32f)srcRow[x];
}

extern "C" {

/**
 * CUDA implementation for nppiConvert_8u32f_C1R_Ctx
 */
NppStatus nppiConvert_8u32f_C1R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, 
                                         Npp32f* pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
    
    // Setup kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (oSizeROI.width + blockSize.x - 1) / blockSize.x,
        (oSizeROI.height + blockSize.y - 1) / blockSize.y
    );
    
    // Launch kernel with the specified CUDA stream
    convert_8u32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, 
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