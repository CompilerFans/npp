#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

/**
 * CUDA kernels for NPP Gradient Vector Functions
 */

// Prewitt operator kernels (3x3)
__constant__ float prewitt_x_3x3[9] = {
    -1.0f, 0.0f, 1.0f,
    -1.0f, 0.0f, 1.0f,
    -1.0f, 0.0f, 1.0f
};

__constant__ float prewitt_y_3x3[9] = {
    -1.0f, -1.0f, -1.0f,
     0.0f,  0.0f,  0.0f,
     1.0f,  1.0f,  1.0f
};

// Device function to handle border pixel access
template<typename T>
__device__ T getBorderPixelGradient(const T* pSrc, int nSrcStep, NppiSize oSrcSizeROI,
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

// Kernel for 8u to 16s gradient vector with Prewitt operator
__global__ void nppiGradientVectorPrewittBorder_8u16s_C1R_kernel(
    const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
    Npp16s* pDstMag, int nDstMagStep, Npp16s* pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
    NppiMaskSize eMaskSize, NppiBorderType eBorderType) {
    
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < oDstSizeROI.width && dst_y < oDstSizeROI.height) {
        Npp16s* mag_row = (Npp16s*)((char*)pDstMag + dst_y * nDstMagStep);
        Npp16s* dir_row = (Npp16s*)((char*)pDstDir + dst_y * nDstDirStep);
        
        float gx = 0.0f, gy = 0.0f;
        
        // Apply Prewitt 3x3 operator
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int src_x = dst_x + oSrcOffset.x + kx;
                int src_y = dst_y + oSrcOffset.y + ky;
                
                Npp8u pixel = getBorderPixelGradient<Npp8u>(pSrc, nSrcStep, oSrcSizeROI, 
                                                           src_x, src_y, eBorderType, 0);
                
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                gx += pixel * prewitt_x_3x3[kernel_idx];
                gy += pixel * prewitt_y_3x3[kernel_idx];
            }
        }
        
        // Calculate magnitude and direction
        float magnitude = sqrtf(gx * gx + gy * gy);
        float direction = atan2f(gy, gx) * 180.0f / M_PI;  // Convert to degrees
        
        // Clamp magnitude to 16-bit signed range
        magnitude = fmaxf(-32768.0f, fminf(32767.0f, magnitude));
        
        // Normalize direction to [0, 360) range
        if (direction < 0.0f) direction += 360.0f;
        
        mag_row[dst_x] = (Npp16s)magnitude;
        dir_row[dst_x] = (Npp16s)direction;
    }
}

// Kernel for 8u to 32f gradient vector with Prewitt operator
__global__ void nppiGradientVectorPrewittBorder_8u32f_C1R_kernel(
    const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
    Npp32f* pDstMag, int nDstMagStep, Npp32f* pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
    NppiMaskSize eMaskSize, NppiBorderType eBorderType) {
    
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < oDstSizeROI.width && dst_y < oDstSizeROI.height) {
        Npp32f* mag_row = (Npp32f*)((char*)pDstMag + dst_y * nDstMagStep);
        Npp32f* dir_row = (Npp32f*)((char*)pDstDir + dst_y * nDstDirStep);
        
        float gx = 0.0f, gy = 0.0f;
        
        // Apply Prewitt 3x3 operator
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int src_x = dst_x + oSrcOffset.x + kx;
                int src_y = dst_y + oSrcOffset.y + ky;
                
                Npp8u pixel = getBorderPixelGradient<Npp8u>(pSrc, nSrcStep, oSrcSizeROI, 
                                                           src_x, src_y, eBorderType, 0);
                
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                gx += pixel * prewitt_x_3x3[kernel_idx];
                gy += pixel * prewitt_y_3x3[kernel_idx];
            }
        }
        
        // Calculate magnitude and direction
        float magnitude = sqrtf(gx * gx + gy * gy);
        float direction = atan2f(gy, gx) * 180.0f / M_PI;  // Convert to degrees
        
        // Normalize direction to [0, 360) range
        if (direction < 0.0f) direction += 360.0f;
        
        mag_row[dst_x] = magnitude;
        dir_row[dst_x] = direction;
    }
}

// Kernel for 16s gradient vector with Prewitt operator
__global__ void nppiGradientVectorPrewittBorder_16s_C1R_kernel(
    const Npp16s* pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
    Npp16s* pDstMag, int nDstMagStep, Npp16s* pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
    NppiMaskSize eMaskSize, NppiBorderType eBorderType) {
    
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < oDstSizeROI.width && dst_y < oDstSizeROI.height) {
        Npp16s* mag_row = (Npp16s*)((char*)pDstMag + dst_y * nDstMagStep);
        Npp16s* dir_row = (Npp16s*)((char*)pDstDir + dst_y * nDstDirStep);
        
        float gx = 0.0f, gy = 0.0f;
        
        // Apply Prewitt 3x3 operator
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int src_x = dst_x + oSrcOffset.x + kx;
                int src_y = dst_y + oSrcOffset.y + ky;
                
                Npp16s pixel = getBorderPixelGradient<Npp16s>(pSrc, nSrcStep, oSrcSizeROI, 
                                                             src_x, src_y, eBorderType, 0);
                
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                gx += pixel * prewitt_x_3x3[kernel_idx];
                gy += pixel * prewitt_y_3x3[kernel_idx];
            }
        }
        
        // Calculate magnitude and direction
        float magnitude = sqrtf(gx * gx + gy * gy);
        float direction = atan2f(gy, gx) * 180.0f / M_PI;  // Convert to degrees
        
        // Clamp magnitude to 16-bit signed range
        magnitude = fmaxf(-32768.0f, fminf(32767.0f, magnitude));
        
        // Normalize direction to [0, 360) range
        if (direction < 0.0f) direction += 360.0f;
        
        mag_row[dst_x] = (Npp16s)magnitude;
        dir_row[dst_x] = (Npp16s)direction;
    }
}

extern "C" {

// 8u to 16s implementation
NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx_cuda(
    const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
    Npp16s* pDstMag, int nDstMagStep, Npp16s* pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
    NppiMaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiGradientVectorPrewittBorder_8u16s_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSizeROI, oSrcOffset,
        pDstMag, nDstMagStep, pDstDir, nDstDirStep, oDstSizeROI,
        eMaskSize, eBorderType);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 8u to 32f implementation
NppStatus nppiGradientVectorPrewittBorder_8u32f_C1R_Ctx_cuda(
    const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
    Npp32f* pDstMag, int nDstMagStep, Npp32f* pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
    NppiMaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiGradientVectorPrewittBorder_8u32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSizeROI, oSrcOffset,
        pDstMag, nDstMagStep, pDstDir, nDstDirStep, oDstSizeROI,
        eMaskSize, eBorderType);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

// 16s implementation
NppStatus nppiGradientVectorPrewittBorder_16s_C1R_Ctx_cuda(
    const Npp16s* pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
    Npp16s* pDstMag, int nDstMagStep, Npp16s* pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
    NppiMaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiGradientVectorPrewittBorder_16s_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSizeROI, oSrcOffset,
        pDstMag, nDstMagStep, pDstDir, nDstDirStep, oDstSizeROI,
        eMaskSize, eBorderType);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}


// 原始版本重命名为带_magdir后缀（保持原始实现）
NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx_cuda_magdir(
    const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSizeROI, NppiPoint oSrcOffset,
    Npp16s* pDstMag, int nDstMagStep, Npp16s* pDstDir, int nDstDirStep, NppiSize oDstSizeROI,
    NppiMaskSize eMaskSize, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((oDstSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oDstSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    nppiGradientVectorPrewittBorder_8u16s_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, oSrcSizeROI, oSrcOffset,
        pDstMag, nDstMagStep, pDstDir, nDstDirStep, oDstSizeROI,
        eMaskSize, eBorderType);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

} // extern "C"

// 实现新的X/Y分量输出版本
__global__ void prewittGradientXY_8u16s_kernel(
    const Npp8u* pSrc, int nSrcStep, int srcWidth, int srcHeight,
    NppiPoint oSrcOffset, Npp16s* pDstX, int nDstXStep, 
    Npp16s* pDstY, int nDstYStep, Npp16s* pDstMag, int nDstMagStep,
    Npp32f* pDstAngle, int nDstAngleStep, int dstWidth, int dstHeight,
    NppiNorm eNorm, NppiBorderType eBorderType) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dstWidth || y >= dstHeight) return;
    
    // 应用源偏移
    int srcX = x + oSrcOffset.x;
    int srcY = y + oSrcOffset.y;
    
    // 计算Prewitt梯度
    float gradX = 0.0f;
    float gradY = 0.0f;
    
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = srcX + kx;
            int py = srcY + ky;
            
            // 处理边界
            if (eBorderType == NPP_BORDER_REPLICATE) {
                px = max(0, min(px, srcWidth - 1));
                py = max(0, min(py, srcHeight - 1));
            } else if (eBorderType == NPP_BORDER_CONSTANT) {
                if (px < 0 || px >= srcWidth || py < 0 || py >= srcHeight) {
                    continue; // 边界外使用0
                }
            }
            
            const Npp8u* src_row = (const Npp8u*)((const char*)pSrc + py * nSrcStep);
            float pixel = (float)src_row[px];
            
            int idx = (ky + 1) * 3 + (kx + 1);
            gradX += pixel * prewitt_x_3x3[idx];
            gradY += pixel * prewitt_y_3x3[idx];
        }
    }
    
    // 保存X和Y梯度
    if (pDstX) {
        Npp16s* dstXRow = (Npp16s*)((char*)pDstX + y * nDstXStep);
        dstXRow[x] = (Npp16s)gradX;
    }
    
    if (pDstY) {
        Npp16s* dstYRow = (Npp16s*)((char*)pDstY + y * nDstYStep);
        dstYRow[x] = (Npp16s)gradY;
    }
    
    // 计算幅度
    if (pDstMag) {
        float mag = 0.0f;
        if (eNorm == nppiNormL1) {
            mag = fabsf(gradX) + fabsf(gradY);
        } else if (eNorm == nppiNormL2) {
            mag = sqrtf(gradX * gradX + gradY * gradY);
        } else { // nppiNormInf
            mag = fmaxf(fabsf(gradX), fabsf(gradY));
        }
        Npp16s* dstMagRow = (Npp16s*)((char*)pDstMag + y * nDstMagStep);
        dstMagRow[x] = (Npp16s)mag;
    }
    
    // 计算角度（弧度）
    if (pDstAngle) {
        float angle = atan2f(gradY, gradX);
        Npp32f* dstAngleRow = (Npp32f*)((char*)pDstAngle + y * nDstAngleStep);
        dstAngleRow[x] = angle;
    }
}

extern "C" NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx_cuda_xy(
    const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16s* pDstX, int nDstXStep, Npp16s* pDstY, int nDstYStep,
    Npp16s* pDstMag, int nDstMagStep, Npp32f* pDstAngle, int nDstAngleStep,
    NppiSize oSizeROI, NppiMaskSize eMaskSize, NppiNorm eNorm,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
    
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    
    // 目前只支持3x3
    if (eMaskSize == NPP_MASK_SIZE_3_X_3) {
        prewittGradientXY_8u16s_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
            pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset,
            pDstX, nDstXStep, pDstY, nDstYStep, pDstMag, nDstMagStep,
            pDstAngle, nDstAngleStep, oSizeROI.width, oSizeROI.height,
            eNorm, eBorderType);
    } else {
        // TODO: 实现5x5版本
        return NPP_MASK_SIZE_ERROR;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}