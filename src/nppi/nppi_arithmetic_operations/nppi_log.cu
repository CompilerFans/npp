#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * CUDA kernels for NPP Common Logarithm Operations
 * Computes base-10 logarithm of input image values
 */

/**
 * CUDA kernel for 32-bit float common logarithm (base 10)
 */
__global__ void nppiLog_32f_C1R_kernel(const Npp32f* pSrc, int nSrcStep,
                                        Npp32f* pDst, int nDstStep,
                                        int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp32f* src_pixel = (const Npp32f*)((const char*)pSrc + y * nSrcStep) + x;
        Npp32f* dst_pixel = (Npp32f*)((char*)pDst + y * nDstStep) + x;
        
        float src_val = *src_pixel;
        
        // 计算以10为底的对数
        float result;
        if (src_val <= 0.0f) {
            // 负数或0的对数未定义，设为负无穷
            result = -HUGE_VALF;
        } else if (src_val == 1.0f) {
            // log10(1) = 0，特殊优化
            result = 0.0f;
        } else if (src_val == 10.0f) {
            // log10(10) = 1，特殊优化
            result = 1.0f;
        } else {
            // 一般情况：log10(x) = ln(x) / ln(10)
            result = log10f(src_val);
        }
        
        *dst_pixel = result;
    }
}

extern "C" {

/**
 * 32-bit float common logarithm (base 10)
 */
NppStatus nppiLog_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiLog_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    // 同步等待内核完成
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