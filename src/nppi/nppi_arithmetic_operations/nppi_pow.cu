#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * CUDA kernels for NPP Power Operations
 * Computes power of input image values
 */

/**
 * CUDA kernel for 8-bit unsigned power with scaling
 */
__global__ void nppiPow_8u_C1RSfs_kernel(const Npp8u* pSrc, int nSrcStep,
                                          Npp8u* pDst, int nDstStep,
                                          int width, int height, int nPower, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp8u* src_pixel = (const Npp8u*)((const char*)pSrc + y * nSrcStep) + x;
        Npp8u* dst_pixel = (Npp8u*)((char*)pDst + y * nDstStep) + x;
        
        float src_val = (float)(*src_pixel);
        
        // 计算幂
        float result;
        if (nPower == 0) {
            result = 1.0f;
        } else if (nPower == 1) {
            result = src_val;
        } else if (nPower == 2) {
            result = src_val * src_val;
        } else if (nPower > 0) {
            result = powf(src_val, (float)nPower);
        } else {
            // 负幂
            if (src_val == 0.0f) {
                result = 0.0f;  // 避免除零
            } else {
                result = powf(src_val, (float)nPower);
            }
        }
        
        // 应用缩放因子
        result *= (1 << nScaleFactor);
        
        // 饱和到8位范围
        *dst_pixel = (Npp8u)fmaxf(0.0f, fminf(255.0f, result));
    }
}

/**
 * CUDA kernel for 16-bit unsigned power with scaling
 */
__global__ void nppiPow_16u_C1RSfs_kernel(const Npp16u* pSrc, int nSrcStep,
                                           Npp16u* pDst, int nDstStep,
                                           int width, int height, int nPower, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp16u* src_pixel = (const Npp16u*)((const char*)pSrc + y * nSrcStep) + x;
        Npp16u* dst_pixel = (Npp16u*)((char*)pDst + y * nDstStep) + x;
        
        float src_val = (float)(*src_pixel);
        
        // 计算幂
        float result;
        if (nPower == 0) {
            result = 1.0f;
        } else if (nPower == 1) {
            result = src_val;
        } else if (nPower == 2) {
            result = src_val * src_val;
        } else if (nPower > 0) {
            result = powf(src_val, (float)nPower);
        } else {
            // 负幂
            if (src_val == 0.0f) {
                result = 0.0f;  // 避免除零
            } else {
                result = powf(src_val, (float)nPower);
            }
        }
        
        // 应用缩放因子
        result *= (1 << nScaleFactor);
        
        // 饱和到16位无符号范围
        *dst_pixel = (Npp16u)fmaxf(0.0f, fminf(65535.0f, result));
    }
}

/**
 * CUDA kernel for 16-bit signed power with scaling
 */
__global__ void nppiPow_16s_C1RSfs_kernel(const Npp16s* pSrc, int nSrcStep,
                                           Npp16s* pDst, int nDstStep,
                                           int width, int height, int nPower, int nScaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp16s* src_pixel = (const Npp16s*)((const char*)pSrc + y * nSrcStep) + x;
        Npp16s* dst_pixel = (Npp16s*)((char*)pDst + y * nDstStep) + x;
        
        float src_val = (float)(*src_pixel);
        
        // 计算幂
        float result;
        if (nPower == 0) {
            result = 1.0f;
        } else if (nPower == 1) {
            result = src_val;
        } else if (nPower == 2) {
            result = src_val * src_val;
        } else if (nPower > 0) {
            result = powf(src_val, (float)nPower);
        } else {
            // 负幂
            if (src_val == 0.0f) {
                result = 0.0f;  // 避免除零
            } else {
                result = powf(src_val, (float)nPower);
            }
        }
        
        // 应用缩放因子
        result *= (1 << nScaleFactor);
        
        // 饱和到16位有符号范围
        *dst_pixel = (Npp16s)fmaxf(-32768.0f, fminf(32767.0f, result));
    }
}

/**
 * CUDA kernel for 32-bit float power (no scaling)
 */
__global__ void nppiPow_32f_C1R_kernel(const Npp32f* pSrc, int nSrcStep,
                                        Npp32f* pDst, int nDstStep,
                                        int width, int height, float nPower) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const Npp32f* src_pixel = (const Npp32f*)((const char*)pSrc + y * nSrcStep) + x;
        Npp32f* dst_pixel = (Npp32f*)((char*)pDst + y * nDstStep) + x;
        
        float src_val = *src_pixel;
        
        // 计算幂
        float result;
        if (nPower == 0.0f) {
            result = 1.0f;
        } else if (nPower == 1.0f) {
            result = src_val;
        } else if (nPower == 2.0f) {
            result = src_val * src_val;
        } else if (nPower == 0.5f) {
            result = sqrtf(fabsf(src_val));  // 平方根，取绝对值避免负数
        } else {
            // 一般情况
            if (src_val < 0.0f && floorf(nPower) != nPower) {
                // 负数的非整数幂，设为NaN
                result = nanf("");
            } else if (src_val == 0.0f && nPower < 0.0f) {
                // 0的负幂，设为无穷大
                result = INFINITY;
            } else {
                result = powf(src_val, nPower);
            }
        }
        
        *dst_pixel = result;
    }
}

extern "C" {

/**
 * 8-bit unsigned power with scaling
 */
NppStatus nppiPow_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep,
                                      NppiSize oSizeROI, int nPower, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiPow_8u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nPower, nScaleFactor);
    
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

/**
 * 16-bit unsigned power with scaling
 */
NppStatus nppiPow_16u_C1RSfs_Ctx_cuda(const Npp16u* pSrc, int nSrcStep, Npp16u* pDst, int nDstStep,
                                       NppiSize oSizeROI, int nPower, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiPow_16u_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nPower, nScaleFactor);
    
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

/**
 * 16-bit signed power with scaling
 */
NppStatus nppiPow_16s_C1RSfs_Ctx_cuda(const Npp16s* pSrc, int nSrcStep, Npp16s* pDst, int nDstStep,
                                       NppiSize oSizeROI, int nPower, int nScaleFactor, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiPow_16s_C1RSfs_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nPower, nScaleFactor);
    
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

/**
 * 32-bit float power (no scaling)
 */
NppStatus nppiPow_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep,
                                    NppiSize oSizeROI, Npp32f nPower, NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiPow_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nPower);
    
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