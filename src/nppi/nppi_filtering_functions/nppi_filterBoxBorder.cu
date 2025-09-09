#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * CUDA kernels for NPP Box Filter with Border Functions
 */

// Device function to handle border pixel access
template<typename T>
__device__ T getBorderPixel(const T* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                           int x, int y, NppiBorderType eBorderType, T borderValue = 0) {
    switch (eBorderType) {
        case NPP_BORDER_REPLICATE:
            x = max(0, min(x, oSrcSizeROI.width - 1));
            y = max(0, min(y, oSrcSizeROI.height - 1));
            break;
        case NPP_BORDER_WRAP:
            x = (x + oSrcSizeROI.width) % oSrcSizeROI.width;
            y = (y + oSrcSizeROI.height) % oSrcSizeROI.height;
            break;
        case NPP_BORDER_MIRROR:
            if (x < 0) x = -x - 1;
            if (x >= oSrcSizeROI.width) x = 2 * oSrcSizeROI.width - x - 1;
            if (y < 0) y = -y - 1;
            if (y >= oSrcSizeROI.height) y = 2 * oSrcSizeROI.height - y - 1;
            break;
        case NPP_BORDER_CONSTANT:
            if (x < 0 || x >= oSrcSizeROI.width || y < 0 || y >= oSrcSizeROI.height) {
                return borderValue;
            }
            break;
        default:
            return borderValue;
    }
    
    const T* src_row = (const T*)((const char*)pSrc + y * nSrcStep);
    return src_row[x];
}

// Kernel for 8-bit unsigned single channel box filter with border
__global__ void nppiFilterBoxBorder_8u_C1R_kernel(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                  NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                                  NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < oDstSizeROI.width && dst_y < oDstSizeROI.height) {
        Npp8u* dst_row = (Npp8u*)((char*)pDst + dst_y * nDstStep);
        
        int sum = 0;
        int count = 0;
        
        // Apply box filter
        for (int ky = 0; ky < oMaskSize.height; ky++) {
            for (int kx = 0; kx < oMaskSize.width; kx++) {
                int src_x = dst_x + oSrcOffset.x + kx - oAnchor.x;
                int src_y = dst_y + oSrcOffset.y + ky - oAnchor.y;
                
                Npp8u pixel = getBorderPixel<Npp8u>(pSrc, nSrcStep, oSrcSizeROI, 
                                                    src_x, src_y, eBorderType, 0);
                sum += pixel;
                count++;
            }
        }
        
        dst_row[dst_x] = (count > 0) ? (Npp8u)(sum / count) : 0;
    }
}

// Kernel for 8-bit unsigned three channel box filter with border
__global__ void nppiFilterBoxBorder_8u_C3R_kernel(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                  NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                                  NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < oDstSizeROI.width && dst_y < oDstSizeROI.height) {
        Npp8u* dst_row = (Npp8u*)((char*)pDst + dst_y * nDstStep);
        
        int sum[3] = {0, 0, 0};
        int count = 0;
        
        // Apply box filter to each channel
        for (int ky = 0; ky < oMaskSize.height; ky++) {
            for (int kx = 0; kx < oMaskSize.width; kx++) {
                int src_x = dst_x + oSrcOffset.x + kx - oAnchor.x;
                int src_y = dst_y + oSrcOffset.y + ky - oAnchor.y;
                
                // Handle border for each channel
                for (int c = 0; c < 3; c++) {
                    if (src_x >= 0 && src_x < oSrcSizeROI.width && 
                        src_y >= 0 && src_y < oSrcSizeROI.height) {
                        const Npp8u* src_row = (const Npp8u*)((const char*)pSrc + src_y * nSrcStep);
                        sum[c] += src_row[src_x * 3 + c];
                    } else {
                        // Apply border handling per channel
                        switch (eBorderType) {
                            case NPP_BORDER_REPLICATE: {
                                int border_x = max(0, min(src_x, oSrcSizeROI.width - 1));
                                int border_y = max(0, min(src_y, oSrcSizeROI.height - 1));
                                const Npp8u* border_row = (const Npp8u*)((const char*)pSrc + border_y * nSrcStep);
                                sum[c] += border_row[border_x * 3 + c];
                                break;
                            }
                            case NPP_BORDER_CONSTANT:
                            default:
                                sum[c] += 0;  // Constant border value
                                break;
                        }
                    }
                }
                count++;
            }
        }
        
        // Store averaged results
        int dst_idx = dst_x * 3;
        for (int c = 0; c < 3; c++) {
            dst_row[dst_idx + c] = (count > 0) ? (Npp8u)(sum[c] / count) : 0;
        }
    }
}

// Kernel for 16-bit signed single channel box filter with border
__global__ void nppiFilterBoxBorder_16s_C1R_kernel(const Npp16s* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                   NppiPoint oSrcOffset, Npp16s* pDst, int nDstStep, NppiSize oDstSizeROI,
                                                   NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < oDstSizeROI.width && dst_y < oDstSizeROI.height) {
        Npp16s* dst_row = (Npp16s*)((char*)pDst + dst_y * nDstStep);
        
        int sum = 0;
        int count = 0;
        
        // Apply box filter
        for (int ky = 0; ky < oMaskSize.height; ky++) {
            for (int kx = 0; kx < oMaskSize.width; kx++) {
                int src_x = dst_x + oSrcOffset.x + kx - oAnchor.x;
                int src_y = dst_y + oSrcOffset.y + ky - oAnchor.y;
                
                Npp16s pixel = getBorderPixel<Npp16s>(pSrc, nSrcStep, oSrcSizeROI, 
                                                     src_x, src_y, eBorderType, 0);
                sum += pixel;
                count++;
            }
        }
        
        dst_row[dst_x] = (count > 0) ? (Npp16s)(sum / count) : 0;
    }
}

// Kernel for 32-bit float single channel box filter with border
__global__ void nppiFilterBoxBorder_32f_C1R_kernel(const Npp32f* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                                   NppiPoint oSrcOffset, Npp32f* pDst, int nDstStep, NppiSize oDstSizeROI,
                                                   NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < oDstSizeROI.width && dst_y < oDstSizeROI.height) {
        Npp32f* dst_row = (Npp32f*)((char*)pDst + dst_y * nDstStep);
        
        float sum = 0.0f;
        int count = 0;
        
        // Apply box filter
        for (int ky = 0; ky < oMaskSize.height; ky++) {
            for (int kx = 0; kx < oMaskSize.width; kx++) {
                int src_x = dst_x + oSrcOffset.x + kx - oAnchor.x;
                int src_y = dst_y + oSrcOffset.y + ky - oAnchor.y;
                
                Npp32f pixel = getBorderPixel<Npp32f>(pSrc, nSrcStep, oSrcSizeROI, 
                                                      src_x, src_y, eBorderType, 0.0f);
                sum += pixel;
                count++;
            }
        }
        
        dst_row[dst_x] = (count > 0) ? (sum / count) : 0.0f;
    }
}

extern "C" {

// 8-bit unsigned single channel implementation
NppStatus nppiFilterBoxBorder_8u_C1R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                              NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                              NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType,
                                              NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiFilterBoxBorder_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI,
        oMaskSize, oAnchor, eBorderType);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 8-bit unsigned three channel implementation
NppStatus nppiFilterBoxBorder_8u_C3R_Ctx_cuda(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                              NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oDstSizeROI,
                                              NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType,
                                              NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiFilterBoxBorder_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI,
        oMaskSize, oAnchor, eBorderType);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 16-bit signed single channel implementation
NppStatus nppiFilterBoxBorder_16s_C1R_Ctx_cuda(const Npp16s* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                               NppiPoint oSrcOffset, Npp16s* pDst, int nDstStep, NppiSize oDstSizeROI,
                                               NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType,
                                               NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiFilterBoxBorder_16s_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI,
        oMaskSize, oAnchor, eBorderType);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 32-bit float single channel implementation
NppStatus nppiFilterBoxBorder_32f_C1R_Ctx_cuda(const Npp32f* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
                                               NppiPoint oSrcOffset, Npp32f* pDst, int nDstStep, NppiSize oDstSizeROI,
                                               NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType,
                                               NppStreamContext nppStreamCtx) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiFilterBoxBorder_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSizeROI, oSrcOffset, pDst, nDstStep, oDstSizeROI,
        oMaskSize, oAnchor, eBorderType);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

} // extern "C"