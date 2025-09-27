#include "nppi_arithmetic_executor.h"
#include "nppi_arithmetic_ops.h"

using namespace nppi::arithmetic;

// ============================================================================
// AlphaPremul Implementation
// nppiAlphaPremulC: Alpha premultiplication with constant alpha value
// nppiAlphaPremul: Alpha premultiplication with pixel alpha value
// ============================================================================

template <typename T>
class AlphaPremulConstOp {
private:
    T alpha;

public:
    __device__ __host__ AlphaPremulConstOp(T a) : alpha(a) {}

    __device__ __host__ T operator()(T src, int = 0) const {
        // Simple alpha premultiplication: src * alpha / max_value
        if constexpr (std::is_same_v<T, Npp8u>) {
            return static_cast<T>((static_cast<int>(src) * static_cast<int>(alpha)) / 255);
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            return static_cast<T>((static_cast<int64_t>(src) * static_cast<int64_t>(alpha)) / 65535);
        } else {
            return static_cast<T>(src * alpha);
        }
    }
};

template <typename T>
class AlphaPremulPixelOp {
public:
    __device__ __host__ T operator()(T src, T alpha, int = 0) const {
        // Simple alpha premultiplication: src * alpha / max_value
        if constexpr (std::is_same_v<T, Npp8u>) {
            return static_cast<T>((static_cast<int>(src) * static_cast<int>(alpha)) / 255);
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            return static_cast<T>((static_cast<int64_t>(src) * static_cast<int64_t>(alpha)) / 65535);
        } else {
            return static_cast<T>(src * alpha);
        }
    }
};

// Custom kernels for AlphaPremul operations
template <typename T, int Channels>
__global__ void alphaPremulConstKernel(const T *pSrc, int nSrcStep, T alpha,
                                       T *pDst, int nDstStep, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
        T *dstRow = (T *)((char *)pDst + y * nDstStep);

::AlphaPremulConstOp<T> op(alpha);

        if constexpr (Channels == 1) {
            dstRow[x] = op(srcRow[x]);
        } else {
            int idx = x * Channels;
            for (int c = 0; c < Channels; c++) {
                dstRow[idx + c] = op(srcRow[idx + c]);
            }
        }
    }
}

template <typename T, int Channels>
__global__ void alphaPremulConstInPlaceKernel(T alpha, T *pSrcDst, int nSrcDstStep, 
                                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        T *srcDstRow = (T *)((char *)pSrcDst + y * nSrcDstStep);

::AlphaPremulConstOp<T> op(alpha);

        if constexpr (Channels == 1) {
            srcDstRow[x] = op(srcDstRow[x]);
        } else {
            int idx = x * Channels;
            for (int c = 0; c < Channels; c++) {
                srcDstRow[idx + c] = op(srcDstRow[idx + c]);
            }
        }
    }
}

// Special kernel for AC4 format (alpha channel is in the 4th position)
template <typename T>
__global__ void alphaPremulAC4Kernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, 
                                      int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
        T *dstRow = (T *)((char *)pDst + y * nDstStep);

        int idx = x * 4;  // AC4 format has 4 channels
        T alpha = srcRow[idx + 3];  // Alpha is in 4th channel
        
::AlphaPremulPixelOp<T> op;
        
        // Premultiply RGB channels with alpha (don't modify alpha channel)
        dstRow[idx + 0] = op(srcRow[idx + 0], alpha);  // R
        dstRow[idx + 1] = op(srcRow[idx + 1], alpha);  // G
        dstRow[idx + 2] = op(srcRow[idx + 2], alpha);  // B
        dstRow[idx + 3] = srcRow[idx + 3];             // A (unchanged)
    }
}

template <typename T>
__global__ void alphaPremulAC4InPlaceKernel(T *pSrcDst, int nSrcDstStep, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        T *srcDstRow = (T *)((char *)pSrcDst + y * nSrcDstStep);

        int idx = x * 4;  // AC4 format has 4 channels
        T alpha = srcDstRow[idx + 3];  // Alpha is in 4th channel
        
::AlphaPremulPixelOp<T> op;
        
        // Premultiply RGB channels with alpha (don't modify alpha channel)
        srcDstRow[idx + 0] = op(srcDstRow[idx + 0], alpha);  // R
        srcDstRow[idx + 1] = op(srcDstRow[idx + 1], alpha);  // G
        srcDstRow[idx + 2] = op(srcDstRow[idx + 2], alpha);  // B
        // Alpha channel remains unchanged
    }
}

extern "C" {

// ============================================================================
// AlphaPremulC - Constant Alpha Premultiplication
// ============================================================================

// 8u versions
NppStatus nppiAlphaPremulC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstKernel<Npp8u, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremulC_8u_C1R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremulC_8u_C1IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrcDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstInPlaceKernel<Npp8u, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        nAlpha1, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_8u_C1IR(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremulC_8u_C1IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 8u C3 versions
NppStatus nppiAlphaPremulC_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstKernel<Npp8u, 3><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremulC_8u_C3R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremulC_8u_C3IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrcDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstInPlaceKernel<Npp8u, 3><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        nAlpha1, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_8u_C3IR(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremulC_8u_C3IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 8u C4 versions
NppStatus nppiAlphaPremulC_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstKernel<Npp8u, 4><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_8u_C4R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremulC_8u_C4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremulC_8u_C4IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, 
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrcDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstInPlaceKernel<Npp8u, 4><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        nAlpha1, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_8u_C4IR(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremulC_8u_C4IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 8u AC4 versions
NppStatus nppiAlphaPremulC_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1,
                                       Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel for AC4 format (3 channels plus alpha, alpha channel not affected)
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstKernel<Npp8u, 4><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremulC_8u_AC4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremulC_8u_AC4IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrcDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstInPlaceKernel<Npp8u, 4><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        nAlpha1, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_8u_AC4IR(Npp8u nAlpha1, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremulC_8u_AC4IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 16u versions
NppStatus nppiAlphaPremulC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1,
                                       Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstKernel<Npp16u, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1,
                                   Npp16u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremulC_16u_C1R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremulC_16u_C1IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrcDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstInPlaceKernel<Npp16u, 1><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        nAlpha1, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_16u_C1IR(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremulC_16u_C1IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// 16u C3, C4, AC4 versions (similar pattern as 8u)
NppStatus nppiAlphaPremulC_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1,
                                       Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
    if (!pSrc1 || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstKernel<Npp16u, 3><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_16u_C3R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1,
                                   Npp16u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0;
    return nppiAlphaPremulC_16u_C3R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremulC_16u_C3IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    if (!pSrcDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstInPlaceKernel<Npp16u, 3><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        nAlpha1, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_16u_C3IR(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0;
    return nppiAlphaPremulC_16u_C3IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremulC_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1,
                                       Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
    if (!pSrc1 || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstKernel<Npp16u, 4><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_16u_C4R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1,
                                   Npp16u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0;
    return nppiAlphaPremulC_16u_C4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremulC_16u_C4IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, 
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    if (!pSrcDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstInPlaceKernel<Npp16u, 4><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        nAlpha1, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_16u_C4IR(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0;
    return nppiAlphaPremulC_16u_C4IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremulC_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1,
                                        Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
    if (!pSrc1 || !pDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstKernel<Npp16u, 4><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nAlpha1,
                                    Npp16u *pDst, int nDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0;
    return nppiAlphaPremulC_16u_AC4R_Ctx(pSrc1, nSrc1Step, nAlpha1, pDst, nDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremulC_16u_AC4IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, 
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    if (!pSrcDst) return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulConstInPlaceKernel<Npp16u, 4><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        nAlpha1, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremulC_16u_AC4IR(Npp16u nAlpha1, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0;
    return nppiAlphaPremulC_16u_AC4IR_Ctx(nAlpha1, pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

// ============================================================================
// AlphaPremul - Pixel Alpha Premultiplication (AC4 format only)
// ============================================================================

NppStatus nppiAlphaPremul_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u *pDst, int nDstStep, 
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulAC4Kernel<Npp8u><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremul_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step, Npp8u *pDst, int nDstStep, 
                                  NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremul_8u_AC4R_Ctx(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremul_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, 
                                       NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrcDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulAC4InPlaceKernel<Npp8u><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremul_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremul_8u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremul_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step, Npp16u *pDst, int nDstStep, 
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrc1 || !pDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulAC4Kernel<Npp16u><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremul_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step, Npp16u *pDst, int nDstStep, 
                                   NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremul_16u_AC4R_Ctx(pSrc1, nSrc1Step, pDst, nDstStep, oSizeROI, defaultCtx);
}

NppStatus nppiAlphaPremul_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, 
                                        NppStreamContext nppStreamCtx) {
    // Validate parameters
    if (!pSrcDst) {
        return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
        return NPP_SIZE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, 
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    alphaPremulAC4InPlaceKernel<Npp16u><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiAlphaPremul_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
    NppStreamContext defaultCtx = {};
    defaultCtx.hStream = 0; // Default CUDA stream
    return nppiAlphaPremul_16u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, defaultCtx);
}

} // extern "C"