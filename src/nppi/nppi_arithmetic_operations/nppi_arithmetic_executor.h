#pragma once

#include "nppi_arithmetic_ops.h"
#include <cuda_runtime.h>

namespace nppi {
namespace arithmetic {

// ============================================================================
// Parameter Validation Utilities
// ============================================================================

template <typename T>
inline NppStatus validateBinaryParameters(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst,
                                          int nDstStep, NppiSize oSizeROI, int channels) {
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

template <typename T>
inline NppStatus validateUnaryParameters(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                                         int channels) {
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

// ============================================================================
// CUDA Kernel Templates
// ============================================================================

template <typename T, int Channels, typename BinaryOp>
__global__ void binaryKernel(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                             int width, int height, BinaryOp op, int scaleFactor) {
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

template <typename T, int Channels, typename UnaryOp>
__global__ void unaryKernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height, UnaryOp op,
                            int scaleFactor) {
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

template <typename T, int Channels, typename CompareOp>
__global__ void compareKernel(const T *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width, int height,
                              CompareOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);

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

template <typename T, int Channels, typename TernaryOp>
__global__ void ternaryKernel(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, const T *pSrc3,
                              int nSrc3Step, T *pDst, int nDstStep, int width, int height, TernaryOp op,
                              int scaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *src1Row = (const T *)((const char *)pSrc1 + y * nSrc1Step);
    const T *src2Row = (const T *)((const char *)pSrc2 + y * nSrc2Step);
    const T *src3Row = (const T *)((const char *)pSrc3 + y * nSrc3Step);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    if constexpr (Channels == 1) {
      dstRow[x] = op(src1Row[x], src2Row[x], src3Row[x], scaleFactor);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        dstRow[idx + c] = op(src1Row[idx + c], src2Row[idx + c], src3Row[idx + c], scaleFactor);
      }
    }
  }
}

// ============================================================================
// Executor Classes
// ============================================================================

template <typename T, int Channels, typename OpType> class BinaryOperationExecutor {
public:
  static NppStatus execute(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                           NppiSize oSizeROI, int scaleFactor, cudaStream_t stream) {
    // Validate parameters
    NppStatus status = validateBinaryParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, Channels);
    if (status != NPP_NO_ERROR)
      return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    binaryKernel<T, Channels, OpType><<<gridSize, blockSize, 0, stream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, op, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

template <typename T, int Channels, typename OpType> class UnaryOperationExecutor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, int scaleFactor,
                           cudaStream_t stream) {
    // Validate parameters
    NppStatus status = validateUnaryParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, Channels);
    if (status != NPP_NO_ERROR)
      return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    unaryKernel<T, Channels, OpType><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                         oSizeROI.height, op, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

template <typename T, int Channels, typename ConstOpType> class ConstOperationExecutor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, int scaleFactor,
                           cudaStream_t stream, ConstOpType op) {
    // Validate parameters
    NppStatus status = validateUnaryParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, Channels);
    if (status != NPP_NO_ERROR)
      return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    unaryKernel<T, Channels, ConstOpType><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, op, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// Multi-channel constant kernel
template <typename T, int Channels, typename ConstOpType>
__global__ void multiConstKernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height,
                                 const T *constants, int scaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    if constexpr (Channels == 1) {
      ConstOpType op(constants[0]);
      dstRow[x] = op(srcRow[x], T(0), scaleFactor);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        ConstOpType op(constants[c]);
        dstRow[idx + c] = op(srcRow[idx + c], T(0), scaleFactor);
      }
    }
  }
}

// Multi-channel constant operation executor
template <typename T, int Channels, typename ConstOpType> class MultiConstOperationExecutor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, int scaleFactor,
                           cudaStream_t stream, const T *constants) {
    // Validate parameters
    NppStatus status = validateUnaryParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, Channels);
    if (status != NPP_NO_ERROR)
      return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    // Copy constants to device
    T *d_constants;
    cudaError_t err = cudaMalloc(&d_constants, Channels * sizeof(T));
    if (err != cudaSuccess)
      return NPP_MEMORY_ALLOCATION_ERR;

    err = cudaMemcpy(d_constants, constants, Channels * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cudaFree(d_constants);
      return NPP_MEMORY_ALLOCATION_ERR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    multiConstKernel<T, Channels, ConstOpType><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_constants, scaleFactor);

    cudaError_t kernelResult = cudaGetLastError();
    cudaFree(d_constants);

    return (kernelResult == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

template <typename T, int Channels, typename CompareOpType> class CompareOperationExecutor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                           cudaStream_t stream, CompareOpType op) {
    // Validate input parameters
    if (!pSrc || !pDst) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
      return NPP_SIZE_ERROR;
    }
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    int minSrcStep = oSizeROI.width * Channels * sizeof(T);
    int minDstStep = oSizeROI.width * Channels * sizeof(Npp8u);
    if (nSrcStep < minSrcStep || nDstStep < minDstStep) {
      return NPP_STRIDE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    compareKernel<T, Channels, CompareOpType>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, op);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

template <typename T, int Channels, typename OpType> class TernaryOperationExecutor {
public:
  static NppStatus execute(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, const T *pSrc3, int nSrc3Step,
                           T *pDst, int nDstStep, NppiSize oSizeROI, int scaleFactor, cudaStream_t stream) {
    // Validate parameters
    if (!pSrc1 || !pSrc2 || !pSrc3 || !pDst) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
      return NPP_SIZE_ERROR;
    }
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    int minStep = oSizeROI.width * Channels * sizeof(T);
    if (nSrc1Step < minStep || nSrc2Step < minStep || nSrc3Step < minStep || nDstStep < minStep) {
      return NPP_STRIDE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    ternaryKernel<T, Channels, OpType><<<gridSize, blockSize, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrc3,
                                                                           nSrc3Step, pDst, nDstStep, oSizeROI.width,
                                                                           oSizeROI.height, op, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// ============================================================================
// Three Operand Operation Helper Functions
// ============================================================================

template <typename SrcType1, typename SrcType2, typename DstType, typename OpType, bool InPlace>
NppStatus executeThreeOperandOperation(const SrcType1 *pSrc1, int nSrc1Step, const SrcType2 *pSrc2, int nSrc2Step,
                                       const DstType *pSrc3, int nSrc3Step, DstType *pDst, int nDstStep,
                                       NppiSize oSizeROI, int scaleFactor, NppStreamContext nppStreamCtx) {
  // For AddProduct, the third operand is the destination that gets accumulated
  // This handles the IR (in-place result) operations where result = dst + (src1 * src2)
  if constexpr (InPlace) {
    // For in-place operations, pSrc3 and pDst should be the same
    if (pSrc3 != pDst) {
      return NPP_BAD_ARGUMENT_ERROR;
    }
  }

  // Validate parameters
  if (!pSrc1 || !pSrc2 || !pSrc3 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  // For three operand operations, we need special handling
  // AddProduct: result = src3 + (src1 * src2)
  return TernaryOperationExecutor<DstType, 1, OpType>::execute(
      (const DstType *)pSrc3, nSrc3Step, (const DstType *)pSrc1, nSrc1Step, (const DstType *)pSrc2, nSrc2Step, pDst,
      nDstStep, oSizeROI, scaleFactor, nppStreamCtx.hStream);
}

// ============================================================================
// Mixed Type Ternary Operation Kernels and Executors
// ============================================================================

template <typename SrcType1, typename SrcType2, typename DstType, int Channels, typename TernaryOp>
__global__ void mixedTernaryKernel(const DstType *pSrc1, int nSrc1Step, const SrcType1 *pSrc2, int nSrc2Step,
                                   const SrcType2 *pSrc3, int nSrc3Step, DstType *pDst, int nDstStep, int width,
                                   int height, TernaryOp op, int scaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const DstType *src1Row = (const DstType *)((const char *)pSrc1 + y * nSrc1Step);
    const SrcType1 *src2Row = (const SrcType1 *)((const char *)pSrc2 + y * nSrc2Step);
    const SrcType2 *src3Row = (const SrcType2 *)((const char *)pSrc3 + y * nSrc3Step);
    DstType *dstRow = (DstType *)((char *)pDst + y * nDstStep);

    if constexpr (Channels == 1) {
      // Convert types and apply operation: result = src1 + (src2 * src3)
      DstType val1 = src1Row[x];
      DstType val2 = static_cast<DstType>(src2Row[x]);
      DstType val3 = static_cast<DstType>(src3Row[x]);
      dstRow[x] = op(val1, val2, val3, scaleFactor);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        DstType val1 = src1Row[idx + c];
        DstType val2 = static_cast<DstType>(src2Row[idx + c]);
        DstType val3 = static_cast<DstType>(src3Row[idx + c]);
        dstRow[idx + c] = op(val1, val2, val3, scaleFactor);
      }
    }
  }
}

template <typename SrcType1, typename SrcType2, typename DstType, int Channels, typename OpType>
class MixedTernaryOperationExecutor {
public:
  static NppStatus execute(const DstType *pSrc1, int nSrc1Step, const SrcType1 *pSrc2, int nSrc2Step,
                           const SrcType2 *pSrc3, int nSrc3Step, DstType *pDst, int nDstStep, NppiSize oSizeROI,
                           int scaleFactor, cudaStream_t stream) {
    // Validate parameters
    if (!pSrc1 || !pSrc2 || !pSrc3 || !pDst) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
      return NPP_SIZE_ERROR;
    }
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    int minSrc1Step = oSizeROI.width * Channels * sizeof(DstType);
    int minSrc2Step = oSizeROI.width * Channels * sizeof(SrcType1);
    int minSrc3Step = oSizeROI.width * Channels * sizeof(SrcType2);
    int minDstStep = oSizeROI.width * Channels * sizeof(DstType);

    if (nSrc1Step < minSrc1Step || nSrc2Step < minSrc2Step || nSrc3Step < minSrc3Step || nDstStep < minDstStep) {
      return NPP_STRIDE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    mixedTernaryKernel<SrcType1, SrcType2, DstType, Channels, OpType>
        <<<gridSize, blockSize, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrc3, nSrc3Step, pDst, nDstStep,
                                             oSizeROI.width, oSizeROI.height, op, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// ============================================================================
// Quaternary Operation Kernels and Executors (for AlphaComp)
// ============================================================================

template <typename T, int Channels, typename QuaternaryOp>
__global__ void quaternaryKernel(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, const T *pSrc3,
                                 int nSrc3Step, const T *pSrc4, int nSrc4Step, T *pDst, int nDstStep, int width,
                                 int height, QuaternaryOp op, int scaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *src1Row = (const T *)((const char *)pSrc1 + y * nSrc1Step);
    const T *src2Row = (const T *)((const char *)pSrc2 + y * nSrc2Step);
    const T *src3Row = (const T *)((const char *)pSrc3 + y * nSrc3Step);
    const T *src4Row = (const T *)((const char *)pSrc4 + y * nSrc4Step);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    if constexpr (Channels == 1) {
      dstRow[x] = op(src1Row[x], src2Row[x], src3Row[x], src4Row[x], scaleFactor);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        dstRow[idx + c] = op(src1Row[idx + c], src2Row[idx + c], src3Row[idx + c], src4Row[idx + c], scaleFactor);
      }
    }
  }
}

template <typename T, int Channels, typename OpType> class QuaternaryOperationExecutor {
public:
  static NppStatus execute(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, const T *pSrc3, int nSrc3Step,
                           const T *pSrc4, int nSrc4Step, T *pDst, int nDstStep, NppiSize oSizeROI, int scaleFactor,
                           cudaStream_t stream) {
    // Validate parameters
    if (!pSrc1 || !pSrc2 || !pSrc3 || !pSrc4 || !pDst) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
      return NPP_SIZE_ERROR;
    }
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    int minStep = oSizeROI.width * Channels * sizeof(T);
    if (nSrc1Step < minStep || nSrc2Step < minStep || nSrc3Step < minStep || nSrc4Step < minStep ||
        nDstStep < minStep) {
      return NPP_STRIDE_ERROR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    quaternaryKernel<T, Channels, OpType>
        <<<gridSize, blockSize, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pSrc3, nSrc3Step, pSrc4, nSrc4Step,
                                             pDst, nDstStep, oSizeROI.width, oSizeROI.height, op, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// ============================================================================
// Multi-Channel Shift Operation Kernels and Executors
// ============================================================================

template <typename T, int Channels, typename ShiftOp>
__global__ void shiftMultiKernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height,
                                 ShiftOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    int idx = x * Channels;
    for (int c = 0; c < Channels; c++) {
      T srcVal = srcRow[idx + c];
      dstRow[idx + c] = op.applyShift(srcVal, c);
    }
  }
}

template <typename T, int Channels, typename ShiftOpType> class ShiftMultiOperationExecutor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, const Npp32u aConstants[Channels], T *pDst, int nDstStep,
                           NppiSize oSizeROI, cudaStream_t stream) {

    NppStatus status = validateUnaryParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, Channels);
    if (status != NPP_NO_ERROR)
      return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    NppStatus shiftStatus = validateShiftConstants<T>(aConstants, Channels);
    if (shiftStatus != NPP_NO_ERROR)
      return shiftStatus;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    ShiftOpType op(aConstants);
    shiftMultiKernel<T, Channels, ShiftOpType>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, op);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

private:
  template <typename U> static NppStatus validateShiftConstants(const Npp32u aConstants[Channels], int channels) {
    for (int i = 0; i < channels; i++) {
      if constexpr (std::is_same_v<U, Npp8u>) {
        if (aConstants[i] > 7)
          return NPP_BAD_ARGUMENT_ERROR;
      } else if constexpr (std::is_same_v<U, Npp16u> || std::is_same_v<U, Npp16s>) {
        if (aConstants[i] > 15)
          return NPP_BAD_ARGUMENT_ERROR;
      } else if constexpr (std::is_same_v<U, Npp32u> || std::is_same_v<U, Npp32s>) {
        if (aConstants[i] > 31)
          return NPP_BAD_ARGUMENT_ERROR;
      }
    }
    return NPP_NO_ERROR;
  }
};

template <typename T, int Channels>
using LShiftMultiOperationExecutor = ShiftMultiOperationExecutor<T, Channels, LShiftConstMultiOp<T, Channels>>;

template <typename T, int Channels>
using RShiftMultiOperationExecutor = ShiftMultiOperationExecutor<T, Channels, RShiftConstMultiOp<T, Channels>>;

// ============================================================================
// DivRound Operation Kernel and Executor (with rounding mode)
// ============================================================================

template <typename T, int Channels>
__global__ void divRoundKernel(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                               int width, int height, NppRoundMode rndMode, int scaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *src1Row = (const T *)((const char *)pSrc1 + y * nSrc1Step);
    const T *src2Row = (const T *)((const char *)pSrc2 + y * nSrc2Step);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    DivRoundOp<T> op;
    if constexpr (Channels == 1) {
      dstRow[x] = op(src1Row[x], src2Row[x], scaleFactor, rndMode);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        dstRow[idx + c] = op(src1Row[idx + c], src2Row[idx + c], scaleFactor, rndMode);
      }
    }
  }
}

template <typename T, int Channels> class DivRoundOperationExecutor {
public:
  static NppStatus execute(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                           NppiSize oSizeROI, NppRoundMode rndMode, int scaleFactor, cudaStream_t stream) {
    // Validate parameters
    NppStatus status = validateBinaryParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, Channels);
    if (status != NPP_NO_ERROR)
      return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    divRoundKernel<T, Channels><<<gridSize, blockSize, 0, stream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, rndMode, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

} // namespace arithmetic
} // namespace nppi