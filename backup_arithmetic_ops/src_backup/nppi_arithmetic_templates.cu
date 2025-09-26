#include "nppi_arithmetic_templates.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Generic binary operation kernel implementation
template<typename T, int Channels, typename BinaryOp>
__global__ void binaryOpKernel(const T *pSrc1, int nSrc1Step, 
                               const T *pSrc2, int nSrc2Step,
                               T *pDst, int nDstStep, 
                               int width, int height, 
                               BinaryOp op, int scaleFactor) {
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

// Generic unary operation kernel implementation
template<typename T, int Channels, typename UnaryOp>
__global__ void unaryOpKernel(const T *pSrc, int nSrcStep,
                              T *pDst, int nDstStep,
                              int width, int height,
                              UnaryOp op, int scaleFactor) {
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

// Generic constant operation kernel implementation
template<typename T, int Channels, typename ConstOp>
__global__ void constOpKernel(const T *pSrc, int nSrcStep,
                              T *pDst, int nDstStep,
                              int width, int height,
                              ConstOp op, int scaleFactor) {
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

// Explicit instantiations for common types and operations
// Binary operations for Add
template __global__ void binaryOpKernel<Npp8u, 1, nppi::functors::AddFunctor<Npp8u>>(
    const Npp8u*, int, const Npp8u*, int, Npp8u*, int, int, int, 
    nppi::functors::AddFunctor<Npp8u>, int);

template __global__ void binaryOpKernel<Npp8u, 3, nppi::functors::AddFunctor<Npp8u>>(
    const Npp8u*, int, const Npp8u*, int, Npp8u*, int, int, int, 
    nppi::functors::AddFunctor<Npp8u>, int);

template __global__ void binaryOpKernel<Npp32f, 1, nppi::functors::AddFunctor<Npp32f>>(
    const Npp32f*, int, const Npp32f*, int, Npp32f*, int, int, int, 
    nppi::functors::AddFunctor<Npp32f>, int);

template __global__ void binaryOpKernel<Npp32f, 3, nppi::functors::AddFunctor<Npp32f>>(
    const Npp32f*, int, const Npp32f*, int, Npp32f*, int, int, int, 
    nppi::functors::AddFunctor<Npp32f>, int);