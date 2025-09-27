#pragma once

#include "npp.h"
#include <cuda_runtime.h>
#include <type_traits>

namespace nppi {
namespace arithmetic {

// Type-specific saturating cast
template<typename T>
__device__ __host__ inline T saturate_cast(double value) {
    if constexpr (std::is_same_v<T, Npp8u>) {
        return static_cast<T>(value < 0.0 ? 0.0 : (value > 255.0 ? 255.0 : value));
    } else if constexpr (std::is_same_v<T, Npp16u>) {
        return static_cast<T>(value < 0.0 ? 0.0 : (value > 65535.0 ? 65535.0 : value));
    } else if constexpr (std::is_same_v<T, Npp16s>) {
        return static_cast<T>(value < -32768.0 ? -32768.0 : (value > 32767.0 ? 32767.0 : value));
    } else if constexpr (std::is_same_v<T, Npp32s>) {
        return static_cast<T>(value < -2147483648.0 ? -2147483648.0 : (value > 2147483647.0 ? 2147483647.0 : value));
    } else if constexpr (std::is_same_v<T, Npp32f>) {
        return static_cast<T>(value);
    }
    return static_cast<T>(value);
}

// Parameter validation utility
template<typename T>
inline NppStatus validateBinaryParameters(const T *pSrc1, int nSrc1Step, 
                                           const T *pSrc2, int nSrc2Step,
                                           T *pDst, int nDstStep, 
                                           NppiSize oSizeROI, int channels) {
    if (!pSrc1 || !pSrc2 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
        return NPP_SIZE_ERROR;
    }
    if (oSizeROI.width == 0 || oSizeROI.height == 0) {
        return NPP_NO_ERROR;
    }
    int minStep = oSizeROI.width * channels * sizeof(T);
    if (nSrc1Step < minStep || nSrc2Step < minStep || nDstStep < minStep) {
        return NPP_STRIDE_ERROR;
    }
    return NPP_NO_ERROR;
}

template<typename T>
inline NppStatus validateUnaryParameters(const T *pSrc, int nSrcStep,
                                          T *pDst, int nDstStep, 
                                          NppiSize oSizeROI, int channels) {
    if (!pSrc || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
        return NPP_SIZE_ERROR;
    }
    if (oSizeROI.width == 0 || oSizeROI.height == 0) {
        return NPP_NO_ERROR;
    }
    int minStep = oSizeROI.width * channels * sizeof(T);
    if (nSrcStep < minStep || nDstStep < minStep) {
        return NPP_STRIDE_ERROR;
    }
    return NPP_NO_ERROR;
}

// Operation functors
template<typename T>
struct AddOp {
    __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
        double result = static_cast<double>(a) + static_cast<double>(b);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

template<typename T>
struct SubOp {
    __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
        // NPP Sub convention: pSrc2 - pSrc1, so b - a
        double result = static_cast<double>(b) - static_cast<double>(a);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

template<typename T>
struct MulOp {
    __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
        double result = static_cast<double>(a) * static_cast<double>(b);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

template<typename T>
struct DivOp {
    __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
        if (a == T(0)) {
            // Division by zero - saturate to maximum value for integers
            if constexpr (std::is_same_v<T, Npp8u>) return T(255);
            else if constexpr (std::is_same_v<T, Npp16u>) return T(65535);
            else if constexpr (std::is_same_v<T, Npp16s>) return T(32767);
            else return T(0);
        }
        // NPP Div convention: pSrc2 / pSrc1, so b / a
        double result = static_cast<double>(b) / static_cast<double>(a);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

template<typename T>
struct AndOp {
    __device__ __host__ T operator()(T a, T b, int = 0) const {
        if constexpr (std::is_integral_v<T>) {
            return a & b;
        }
        return T(0);
    }
};

template<typename T>
struct OrOp {
    __device__ __host__ T operator()(T a, T b, int = 0) const {
        if constexpr (std::is_integral_v<T>) {
            return a | b;
        }
        return T(0);
    }
};

template<typename T>
struct AbsDiffOp {
    __device__ __host__ T operator()(T a, T b, int scaleFactor = 0) const {
        double result = std::abs(static_cast<double>(a) - static_cast<double>(b));
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

// Unary operations
template<typename T>
struct AbsOp {
    __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
        return std::abs(a);
    }
};

template<typename T>
struct SqrOp {
    __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
        double result = static_cast<double>(a) * static_cast<double>(a);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

template<typename T>
struct SqrtOp {
    __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
        double result = std::sqrt(static_cast<double>(a));
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

// Constant operations
template<typename T>
class AddConstOp {
private:
    T constant;
public:
    __device__ __host__ AddConstOp(T c) : constant(c) {}
    
    __device__ __host__ T operator()(T a, T = T(0), int scaleFactor = 0) const {
        double result = static_cast<double>(a) + static_cast<double>(constant);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

template<typename T>
class AndConstOp {
private:
    T constant;
public:
    __device__ __host__ AndConstOp(T c) : constant(c) {}
    
    __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
        if constexpr (std::is_integral_v<T>) {
            return a & constant;
        }
        return T(0);
    }
};

template<typename T>
class OrConstOp {
private:
    T constant;
public:
    __device__ __host__ OrConstOp(T c) : constant(c) {}
    
    __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
        if constexpr (std::is_integral_v<T>) {
            return a | constant;
        }
        return T(0);
    }
};

// Unified kernel templates
template<typename T, int Channels, typename BinaryOp>
__global__ void binaryKernel(const T *pSrc1, int nSrc1Step, 
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

template<typename T, int Channels, typename UnaryOp>
__global__ void unaryKernel(const T *pSrc, int nSrcStep,
                            T *pDst, int nDstStep,
                            int width, int height,
                            UnaryOp op, int scaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
        T *dstRow = (T *)((char *)pDst + y * nDstStep);

        if constexpr (Channels == 1) {
            dstRow[x] = op(srcRow[x], T(0), scaleFactor);
        } else {
            int idx = x * Channels;
            for (int c = 0; c < Channels; c++) {
                dstRow[idx + c] = op(srcRow[idx + c], T(0), scaleFactor);
            }
        }
    }
}

// Unified executor class for binary operations
template<typename T, int Channels, typename OpType>
class BinaryOperationExecutor {
public:
    static NppStatus execute(const T *pSrc1, int nSrc1Step, 
                             const T *pSrc2, int nSrc2Step,
                             T *pDst, int nDstStep, 
                             NppiSize oSizeROI, int scaleFactor, 
                             cudaStream_t stream) {
        // Validate parameters
        NppStatus status = validateBinaryParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, 
                                                    pDst, nDstStep, oSizeROI, Channels);
        if (status != NPP_NO_ERROR) return status;
        if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;

        // Launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                      (oSizeROI.height + blockSize.y - 1) / blockSize.y);

        OpType op;
        binaryKernel<T, Channels, OpType><<<gridSize, blockSize, 0, stream>>>(
            pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
            oSizeROI.width, oSizeROI.height, op, scaleFactor);

        return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
};

// Unified executor class for unary operations
template<typename T, int Channels, typename OpType>
class UnaryOperationExecutor {
public:
    static NppStatus execute(const T *pSrc, int nSrcStep,
                             T *pDst, int nDstStep,
                             NppiSize oSizeROI, int scaleFactor,
                             cudaStream_t stream) {
        // Validate parameters
        NppStatus status = validateUnaryParameters(pSrc, nSrcStep, pDst, nDstStep, 
                                                   oSizeROI, Channels);
        if (status != NPP_NO_ERROR) return status;
        if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;

        // Launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                      (oSizeROI.height + blockSize.y - 1) / blockSize.y);

        OpType op;
        unaryKernel<T, Channels, OpType><<<gridSize, blockSize, 0, stream>>>(
            pSrc, nSrcStep, pDst, nDstStep,
            oSizeROI.width, oSizeROI.height, op, scaleFactor);

        return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
};

// Unified executor for constant operations
template<typename T, int Channels, typename ConstOpType>
class ConstOperationExecutor {
public:
    static NppStatus execute(const T *pSrc, int nSrcStep,
                             T *pDst, int nDstStep,
                             NppiSize oSizeROI, int scaleFactor,
                             cudaStream_t stream, ConstOpType op) {
        // Validate parameters
        NppStatus status = validateUnaryParameters(pSrc, nSrcStep, pDst, nDstStep, 
                                                   oSizeROI, Channels);
        if (status != NPP_NO_ERROR) return status;
        if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;

        // Launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                      (oSizeROI.height + blockSize.y - 1) / blockSize.y);

        unaryKernel<T, Channels, ConstOpType><<<gridSize, blockSize, 0, stream>>>(
            pSrc, nSrcStep, pDst, nDstStep,
            oSizeROI.width, oSizeROI.height, op, scaleFactor);

        return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
};

} // namespace arithmetic
} // namespace nppi