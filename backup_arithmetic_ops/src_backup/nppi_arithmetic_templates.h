#ifndef NPPI_ARITHMETIC_TEMPLATES_H
#define NPPI_ARITHMETIC_TEMPLATES_H

#include "nppi_arithmetic_functors.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Forward declarations of kernel functions
template<typename T, int Channels, typename BinaryOp>
__global__ void binaryOpKernel(const T *pSrc1, int nSrc1Step, 
                               const T *pSrc2, int nSrc2Step,
                               T *pDst, int nDstStep, 
                               int width, int height, 
                               BinaryOp op, int scaleFactor);

template<typename T, int Channels, typename UnaryOp>
__global__ void unaryOpKernel(const T *pSrc, int nSrcStep,
                              T *pDst, int nDstStep,
                              int width, int height,
                              UnaryOp op, int scaleFactor);

template<typename T, int Channels, typename ConstOp>
__global__ void constOpKernel(const T *pSrc, int nSrcStep,
                              T *pDst, int nDstStep,
                              int width, int height,
                              ConstOp op, int scaleFactor);

namespace nppi {
namespace templates {

// Generic binary operation kernel
template<typename T, int Channels, typename BinaryOp>
__global__ void binaryOpKernel(const T *pSrc1, int nSrc1Step, 
                               const T *pSrc2, int nSrc2Step,
                               T *pDst, int nDstStep, 
                               int width, int height, 
                               BinaryOp op, int scaleFactor = 0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const T *src1Row = (const T *)((const char *)pSrc1 + y * nSrc1Step);
        const T *src2Row = (const T *)((const char *)pSrc2 + y * nSrc2Step);
        T *dstRow = (T *)((char *)pDst + y * nDstStep);

        if constexpr (Channels == 1) {
            dstRow[x] = op(src1Row[x], src2Row[x], scaleFactor);
        } else {
            int idx = x * Channels;
            for (int c = 0; c < Channels; c++) {
                dstRow[idx + c] = op(src1Row[idx + c], src2Row[idx + c], scaleFactor);
            }
        }
    }
}

// Generic unary operation kernel  
template<typename T, int Channels, typename UnaryOp>
__global__ void unaryOpKernel(const T *pSrc, int nSrcStep,
                              T *pDst, int nDstStep,
                              int width, int height,
                              UnaryOp op, int scaleFactor = 0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
        T *dstRow = (T *)((char *)pDst + y * nDstStep);

        if constexpr (Channels == 1) {
            dstRow[x] = op(srcRow[x], scaleFactor);
        } else {
            int idx = x * Channels;
            for (int c = 0; c < Channels; c++) {
                dstRow[idx + c] = op(srcRow[idx + c], scaleFactor);
            }
        }
    }
}

// Generic constant operation kernel
template<typename T, int Channels, typename ConstOp>
__global__ void constOpKernel(const T *pSrc, int nSrcStep,
                              T *pDst, int nDstStep,
                              int width, int height,
                              ConstOp op, int scaleFactor = 0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
        T *dstRow = (T *)((char *)pDst + y * nDstStep);

        if constexpr (Channels == 1) {
            dstRow[x] = op(srcRow[x], scaleFactor);
        } else {
            int idx = x * Channels;
            for (int c = 0; c < Channels; c++) {
                dstRow[idx + c] = op(srcRow[idx + c], scaleFactor);
            }
        }
    }
}

// Host parameter validation
inline NppStatus validateBinaryOpParameters(const void *pSrc1, int nSrc1Step, 
                                           const void *pSrc2, int nSrc2Step,
                                           const void *pDst, int nDstStep, 
                                           NppiSize oSizeROI, int elementSize) {
    if (!pSrc1 || !pSrc2 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }

    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
        return NPP_SIZE_ERROR;
    }

    if (oSizeROI.width == 0 || oSizeROI.height == 0) {
        return NPP_NO_ERROR; // Early return for zero-size ROI
    }

    int minStep = oSizeROI.width * elementSize;
    if (nSrc1Step < minStep || nSrc2Step < minStep || nDstStep < minStep) {
        return NPP_STRIDE_ERROR;
    }

    return NPP_NO_ERROR;
}

inline NppStatus validateUnaryOpParameters(const void *pSrc, int nSrcStep,
                                          const void *pDst, int nDstStep,
                                          NppiSize oSizeROI, int elementSize) {
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }

    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
        return NPP_SIZE_ERROR;
    }

    if (oSizeROI.width == 0 || oSizeROI.height == 0) {
        return NPP_NO_ERROR;
    }

    int minStep = oSizeROI.width * elementSize;
    if (nSrcStep < minStep || nDstStep < minStep) {
        return NPP_STRIDE_ERROR;
    }

    return NPP_NO_ERROR;
}

// Generic binary operation launcher
template<typename T, int Channels, typename BinaryOp>
NppStatus launchBinaryOp(const T *pSrc1, int nSrc1Step, 
                         const T *pSrc2, int nSrc2Step,
                         T *pDst, int nDstStep, 
                         NppiSize oSizeROI, 
                         BinaryOp op, int scaleFactor = 0,
                         cudaStream_t stream = 0) {
    
    // Parameter validation
    NppStatus status = validateBinaryOpParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                                  pDst, nDstStep, oSizeROI, 
                                                  sizeof(T) * Channels);
    if (status != NPP_NO_ERROR) {
        return status;
    }

    // Early return for zero-size ROI
    if (oSizeROI.width == 0 || oSizeROI.height == 0) {
        return NPP_NO_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    binaryOpKernel<T, Channels, BinaryOp><<<gridSize, blockSize, 0, stream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, 
        oSizeROI.width, oSizeROI.height, op, scaleFactor);

    cudaError_t cudaStatus = cudaGetLastError();
    return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// Generic unary operation launcher
template<typename T, int Channels, typename UnaryOp>
NppStatus launchUnaryOp(const T *pSrc, int nSrcStep,
                        T *pDst, int nDstStep,
                        NppiSize oSizeROI,
                        UnaryOp op, int scaleFactor = 0,
                        cudaStream_t stream = 0) {
    
    // Parameter validation
    NppStatus status = validateUnaryOpParameters(pSrc, nSrcStep, pDst, nDstStep, 
                                                 oSizeROI, sizeof(T) * Channels);
    if (status != NPP_NO_ERROR) {
        return status;
    }

    if (oSizeROI.width == 0 || oSizeROI.height == 0) {
        return NPP_NO_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    unaryOpKernel<T, Channels, UnaryOp><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, 
        oSizeROI.width, oSizeROI.height, op, scaleFactor);

    cudaError_t cudaStatus = cudaGetLastError();
    return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// Generic constant operation launcher
template<typename T, int Channels, typename ConstOp>
NppStatus launchConstOp(const T *pSrc, int nSrcStep,
                        T *pDst, int nDstStep,
                        NppiSize oSizeROI,
                        ConstOp op, int scaleFactor = 0,
                        cudaStream_t stream = 0) {
    
    // Parameter validation
    NppStatus status = validateUnaryOpParameters(pSrc, nSrcStep, pDst, nDstStep, 
                                                 oSizeROI, sizeof(T) * Channels);
    if (status != NPP_NO_ERROR) {
        return status;
    }

    if (oSizeROI.width == 0 || oSizeROI.height == 0) {
        return NPP_NO_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    constOpKernel<T, Channels, ConstOp><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, 
        oSizeROI.width, oSizeROI.height, op, scaleFactor);

    cudaError_t cudaStatus = cudaGetLastError();
    return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

} // namespace templates
} // namespace nppi

#endif // NPPI_ARITHMETIC_TEMPLATES_H