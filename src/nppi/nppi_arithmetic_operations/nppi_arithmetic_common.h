#pragma once

#include "npp.h"
#include <cuda_runtime.h>
#include <functional>
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

// Standard library based binary operators
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
        return T(0); // Bitwise operations not defined for floating point
    }
};

template<typename T>
struct OrOp {
    __device__ __host__ T operator()(T a, T b, int = 0) const {
        if constexpr (std::is_integral_v<T>) {
            return a | b;
        }
        return T(0); // Bitwise operations not defined for floating point
    }
};

template<typename T>
struct XorOp {
    __device__ __host__ T operator()(T a, T b, int = 0) const {
        if constexpr (std::is_integral_v<T>) {
            return a ^ b;
        }
        return T(0); // Bitwise operations not defined for floating point
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

template<typename T>
struct MaxOp {
    __device__ __host__ T operator()(T a, T b, int = 0) const {
        return a > b ? a : b;
    }
};

template<typename T>
struct MinOp {
    __device__ __host__ T operator()(T a, T b, int = 0) const {
        return a < b ? a : b;
    }
};

// Constant operations (binary op with constant)
template<typename T>
struct AddConstOp {
    T constant;
    __device__ __host__ AddConstOp(T c) : constant(c) {}
    
    __device__ __host__ T operator()(T a, T, int scaleFactor = 0) const {
        double result = static_cast<double>(a) + static_cast<double>(constant);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

template<typename T>
struct SubConstOp {
    T constant;
    __device__ __host__ SubConstOp(T c) : constant(c) {}
    
    __device__ __host__ T operator()(T a, T, int scaleFactor = 0) const {
        double result = static_cast<double>(a) - static_cast<double>(constant);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

template<typename T>
struct MulConstOp {
    T constant;
    __device__ __host__ MulConstOp(T c) : constant(c) {}
    
    __device__ __host__ T operator()(T a, T, int scaleFactor = 0) const {
        double result = static_cast<double>(a) * static_cast<double>(constant);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

template<typename T>
struct DivConstOp {
    T constant;
    __device__ __host__ DivConstOp(T c) : constant(c) {}
    
    __device__ __host__ T operator()(T a, T, int scaleFactor = 0) const {
        if (constant == T(0)) return T(0); // Zero protection
        double result = static_cast<double>(a) / static_cast<double>(constant);
        if (scaleFactor > 0 && std::is_integral_v<T>) {
            result = (result + (1 << (scaleFactor - 1))) / (1 << scaleFactor);
        }
        return saturate_cast<T>(result);
    }
};

// Unary operations
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

template<typename T>
struct AbsOp {
    __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
        return std::abs(a);
    }
};

template<typename T>
struct ExpOp {
    __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
        return saturate_cast<T>(std::exp(static_cast<double>(a)));
    }
};

template<typename T>
struct LnOp {
    __device__ __host__ T operator()(T a, T = T(0), int = 0) const {
        if (a <= T(0)) return T(0); // Log of non-positive numbers
        return saturate_cast<T>(std::log(static_cast<double>(a)));
    }
};

// Unified binary operation CUDA kernels
template<typename T, int Channels, typename BinaryOp>
__global__ void binaryOpKernel_C(const T *pSrc1, int nSrc1Step, 
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

// Parameter validation helper
static inline NppStatus validateParameters(const void *pSrc1, int nSrc1Step, 
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
        return NPP_NO_ERROR;
    }
    int minStep = oSizeROI.width * elementSize;
    if (nSrc1Step < minStep || nSrc2Step < minStep || nDstStep < minStep) {
        return NPP_STRIDE_ERROR;
    }
    return NPP_NO_ERROR;
}

// Unified kernel launcher template
template<typename T, int Channels, typename BinaryOp>
static NppStatus launchBinaryOpKernel(const T *pSrc1, int nSrc1Step, 
                                       const T *pSrc2, int nSrc2Step,
                                       T *pDst, int nDstStep, 
                                       NppiSize oSizeROI, int scaleFactor, 
                                       cudaStream_t stream, BinaryOp op) {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    binaryOpKernel_C<T, Channels, BinaryOp><<<gridSize, blockSize, 0, stream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, op, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// Template API generator macros
#define IMPLEMENT_BINARY_OP_IMPL(opName, OpType, dataType, channels, scaleSuffix) \
extern "C" { \
NppStatus nppi##opName##_##dataType##_C##channels##R##scaleSuffix##_Ctx_impl( \
    const Npp##dataType *pSrc1, int nSrc1Step, const Npp##dataType *pSrc2, int nSrc2Step, \
    Npp##dataType *pDst, int nDstStep, NppiSize oSizeROI, \
    int nScaleFactor, NppStreamContext nppStreamCtx) { \
    OpType<Npp##dataType> op; \
    return launchBinaryOpKernel<Npp##dataType, channels>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, \
        pDst, nDstStep, oSizeROI, nScaleFactor, nppStreamCtx.hStream, op); \
} \
}

#define IMPLEMENT_BINARY_OP_IMPL_NO_SCALE(opName, OpType, dataType, channels) \
extern "C" { \
NppStatus nppi##opName##_##dataType##_C##channels##R_Ctx_impl( \
    const Npp##dataType *pSrc1, int nSrc1Step, const Npp##dataType *pSrc2, int nSrc2Step, \
    Npp##dataType *pDst, int nDstStep, NppiSize oSizeROI, \
    NppStreamContext nppStreamCtx) { \
    OpType<Npp##dataType> op; \
    return launchBinaryOpKernel<Npp##dataType, channels>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, \
        pDst, nDstStep, oSizeROI, 0, nppStreamCtx.hStream, op); \
} \
}

#define IMPLEMENT_BINARY_OP_API(opName, dataType, channels, scaleSuffix, hasScale) \
NppStatus nppi##opName##_##dataType##_C##channels##R##scaleSuffix##_Ctx( \
    const Npp##dataType *pSrc1, int nSrc1Step, const Npp##dataType *pSrc2, int nSrc2Step, \
    Npp##dataType *pDst, int nDstStep, NppiSize oSizeROI, \
    int nScaleFactor, NppStreamContext nppStreamCtx) { \
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, \
                                          oSizeROI, channels * sizeof(Npp##dataType)); \
    if (status != NPP_NO_ERROR) return status; \
    if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR; \
    if (hasScale && (nScaleFactor < 0 || nScaleFactor > 31)) return NPP_BAD_ARGUMENT_ERROR; \
    return nppi##opName##_##dataType##_C##channels##R##scaleSuffix##_Ctx_impl( \
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, \
        nScaleFactor, nppStreamCtx); \
}

#define IMPLEMENT_BINARY_OP_API_NO_SCALE(opName, dataType, channels) \
NppStatus nppi##opName##_##dataType##_C##channels##R_Ctx( \
    const Npp##dataType *pSrc1, int nSrc1Step, const Npp##dataType *pSrc2, int nSrc2Step, \
    Npp##dataType *pDst, int nDstStep, NppiSize oSizeROI, \
    NppStreamContext nppStreamCtx) { \
    NppStatus status = validateParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, \
                                          oSizeROI, channels * sizeof(Npp##dataType)); \
    if (status != NPP_NO_ERROR) return status; \
    if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR; \
    return nppi##opName##_##dataType##_C##channels##R_Ctx_impl( \
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx); \
}

} // namespace arithmetic  
} // namespace nppi