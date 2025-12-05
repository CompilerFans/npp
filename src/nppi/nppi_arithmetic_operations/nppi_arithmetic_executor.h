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
      dstRow[x] = op(srcRow[x], T{}, scaleFactor);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        dstRow[idx + c] = op(srcRow[idx + c], T{}, scaleFactor);
      }
    }
  }
}

// AC4 Unary kernel: processes only first 3 channels, preserves alpha (4th channel)
// Alpha in dst is left unchanged (matches NVIDIA NPP behavior)
template <typename T, typename UnaryOp>
__global__ void unaryAC4Kernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height, UnaryOp op,
                               int scaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    // Process first 3 channels only, leave alpha (4th channel) unchanged
    dstRow[idx + 0] = op(srcRow[idx + 0], T{}, scaleFactor);
    dstRow[idx + 1] = op(srcRow[idx + 1], T{}, scaleFactor);
    dstRow[idx + 2] = op(srcRow[idx + 2], T{}, scaleFactor);
    // Alpha channel (idx + 3) is NOT modified
  }
}

// AC4 Binary kernel: processes only first 3 channels, preserves alpha (4th channel)
// Alpha in dst is left unchanged (matches NVIDIA NPP behavior)
template <typename T, typename BinaryOp>
__global__ void binaryAC4Kernel(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                                int width, int height, BinaryOp op, int scaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *src1Row = (const T *)((const char *)pSrc1 + y * nSrc1Step);
    const T *src2Row = (const T *)((const char *)pSrc2 + y * nSrc2Step);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    // Process first 3 channels only, leave alpha (4th channel) unchanged
    dstRow[idx + 0] = op(src1Row[idx + 0], src2Row[idx + 0], scaleFactor);
    dstRow[idx + 1] = op(src1Row[idx + 1], src2Row[idx + 1], scaleFactor);
    dstRow[idx + 2] = op(src1Row[idx + 2], src2Row[idx + 2], scaleFactor);
    // Alpha channel (idx + 3) is NOT modified
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

// AC4 Unary Operation Executor: processes only first 3 channels, preserves alpha
template <typename T, typename OpType> class UnaryAC4OperationExecutor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, int scaleFactor,
                           cudaStream_t stream) {
    // Validate parameters (use 4 channels for stride validation)
    NppStatus status = validateUnaryParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
    if (status != NPP_NO_ERROR)
      return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    unaryAC4Kernel<T, OpType><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                  oSizeROI.height, op, scaleFactor);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// AC4 Binary Operation Executor: processes only first 3 channels, preserves alpha
template <typename T, typename OpType> class BinaryAC4OperationExecutor {
public:
  static NppStatus execute(const T *pSrc1, int nSrc1Step, const T *pSrc2, int nSrc2Step, T *pDst, int nDstStep,
                           NppiSize oSizeROI, int scaleFactor, cudaStream_t stream) {
    // Validate parameters (use 4 channels for stride validation)
    NppStatus status = validateBinaryParameters(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, 4);
    if (status != NPP_NO_ERROR)
      return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    binaryAC4Kernel<T, OpType><<<gridSize, blockSize, 0, stream>>>(
        pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI.width, oSizeROI.height, op, scaleFactor);

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
      dstRow[x] = op(srcRow[x], T{}, scaleFactor);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        ConstOpType op(constants[c]);
        dstRow[idx + c] = op(srcRow[idx + c], T{}, scaleFactor);
      }
    }
  }
}

// AC4 Multi-channel constant kernel: processes only first 3 channels, preserves alpha
// Alpha in dst is left unchanged (matches NVIDIA NPP behavior)
template <typename T, typename ConstOpType>
__global__ void multiConstAC4Kernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height,
                                    const T *constants, int scaleFactor) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    // Process only first 3 channels, leave alpha (4th channel) unchanged
    for (int c = 0; c < 3; c++) {
      ConstOpType op(constants[c]);
      dstRow[idx + c] = op(srcRow[idx + c], T{}, scaleFactor);
    }
    // Alpha channel (idx + 3) is NOT modified
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

// AC4 Multi-channel constant operation executor: processes only first 3 channels, preserves alpha
template <typename T, typename ConstOpType> class MultiConstAC4OperationExecutor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, int scaleFactor,
                           cudaStream_t stream, const T *constants) {
    // Validate parameters (use 4 channels for stride validation)
    NppStatus status = validateUnaryParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
    if (status != NPP_NO_ERROR)
      return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    // Copy 3 constants to device
    T *d_constants;
    cudaError_t err = cudaMalloc(&d_constants, 3 * sizeof(T));
    if (err != cudaSuccess)
      return NPP_MEMORY_ALLOCATION_ERR;

    err = cudaMemcpy(d_constants, constants, 3 * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cudaFree(d_constants);
      return NPP_MEMORY_ALLOCATION_ERR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    multiConstAC4Kernel<T, ConstOpType><<<gridSize, blockSize, 0, stream>>>(
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
      if constexpr (std::is_same_v<U, Npp8u> || std::is_same_v<U, Npp8s>) {
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

// AC4 Multi-Channel Shift kernel: processes only first 3 channels, preserves alpha (4th channel)
// Alpha in dst is left unchanged (matches NVIDIA NPP behavior)
template <typename T, typename ShiftOp>
__global__ void shiftMultiAC4Kernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height,
                                    ShiftOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    // Process only first 3 channels, leave alpha (4th channel) unchanged
    dstRow[idx + 0] = op.applyShift(srcRow[idx + 0], 0);
    dstRow[idx + 1] = op.applyShift(srcRow[idx + 1], 1);
    dstRow[idx + 2] = op.applyShift(srcRow[idx + 2], 2);
    // Alpha channel (idx + 3) is NOT modified
  }
}

template <typename T, typename ShiftOpType> class ShiftMultiAC4OperationExecutor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, const Npp32u aConstants[3], T *pDst, int nDstStep,
                           NppiSize oSizeROI, cudaStream_t stream) {

    NppStatus status = validateUnaryParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
    if (status != NPP_NO_ERROR)
      return status;
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    NppStatus shiftStatus = validateShiftConstants<T>(aConstants, 3);
    if (shiftStatus != NPP_NO_ERROR)
      return shiftStatus;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    // Create a 4-channel op but only use first 3 constants
    Npp32u extConstants[4] = {aConstants[0], aConstants[1], aConstants[2], 0};
    ShiftOpType op(extConstants);
    shiftMultiAC4Kernel<T, ShiftOpType>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, op);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

private:
  template <typename U> static NppStatus validateShiftConstants(const Npp32u aConstants[3], int channels) {
    for (int i = 0; i < channels; i++) {
      if constexpr (std::is_same_v<U, Npp8u> || std::is_same_v<U, Npp8s>) {
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

template <typename T>
using LShiftMultiAC4OperationExecutor = ShiftMultiAC4OperationExecutor<T, LShiftConstMultiOp<T, 4>>;

template <typename T>
using RShiftMultiAC4OperationExecutor = ShiftMultiAC4OperationExecutor<T, RShiftConstMultiOp<T, 4>>;

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

// ============================================================================
// 16f Multi-Channel Constant Operation Kernels and Executors
// Note: For 16f operations, constants are Npp32f type
// ============================================================================

template <int Channels, typename Op16fFactory>
__global__ void multiConst16fKernel(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, int width, int height,
                                    const Npp32f *constants) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16f *srcRow = (const Npp16f *)((const char *)pSrc + y * nSrcStep);
    Npp16f *dstRow = (Npp16f *)((char *)pDst + y * nDstStep);

    int idx = x * Channels;
    for (int c = 0; c < Channels; c++) {
      Op16fFactory op(constants[c]);
      dstRow[idx + c] = op(srcRow[idx + c]);
    }
  }
}

template <int Channels, typename Op16fFactory> class MultiConst16fOperationExecutor {
public:
  static NppStatus execute(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                           cudaStream_t stream, const Npp32f *constants) {
    // Validate parameters
    if (!pSrc || !pDst || !constants) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
      return NPP_SIZE_ERROR;
    }
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    // Copy constants to device
    Npp32f *d_constants;
    cudaError_t err = cudaMalloc(&d_constants, Channels * sizeof(Npp32f));
    if (err != cudaSuccess)
      return NPP_MEMORY_ALLOCATION_ERR;

    err = cudaMemcpyAsync(d_constants, constants, Channels * sizeof(Npp32f), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      cudaFree(d_constants);
      return NPP_MEMORY_ALLOCATION_ERR;
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    multiConst16fKernel<Channels, Op16fFactory><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, d_constants);

    cudaStreamSynchronize(stream);
    cudaError_t kernelResult = cudaGetLastError();
    cudaFree(d_constants);

    return (kernelResult == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// ============================================================================
// 16f DeviceC Multi-Channel Operation Kernels and Executors
// ============================================================================

template <int Channels, typename DeviceOp16fFactory>
__global__ void multiDeviceConst16fKernel(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, int width,
                                          int height, const Npp32f *pConstants) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp16f *srcRow = (const Npp16f *)((const char *)pSrc + y * nSrcStep);
    Npp16f *dstRow = (Npp16f *)((char *)pDst + y * nDstStep);

    int idx = x * Channels;
    for (int c = 0; c < Channels; c++) {
      DeviceOp16fFactory op(&pConstants[Channels == 1 ? 0 : c]);
      dstRow[idx + c] = op(srcRow[idx + c]);
    }
  }
}

template <int Channels, typename DeviceOp16fFactory> class MultiDeviceConst16fOperationExecutor {
public:
  static NppStatus execute(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                           cudaStream_t stream, const Npp32f *pConstants) {
    // Validate parameters
    if (!pSrc || !pDst || !pConstants) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width < 0 || oSizeROI.height < 0) {
      return NPP_SIZE_ERROR;
    }
    if (oSizeROI.width == 0 || oSizeROI.height == 0)
      return NPP_NO_ERROR;

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    multiDeviceConst16fKernel<Channels, DeviceOp16fFactory><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pConstants);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// ============================================================================
// Masked Binary Operation Kernels and Executors
// Updates destination only when mask is non-zero
// ============================================================================

template <typename SrcType, typename DstType, int Channels, typename BinaryOp>
__global__ void maskedBinaryKernel(const SrcType *pSrc1, int nSrc1Step, const SrcType *pSrc2, int nSrc2Step,
                                   const Npp8u *pMask, int nMaskStep, DstType *pSrcDst, int nSrcDstStep, int width,
                                   int height, BinaryOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *maskRow = (const Npp8u *)((const char *)pMask + y * nMaskStep);

    if (maskRow[x] != 0) {
      const SrcType *src1Row = (const SrcType *)((const char *)pSrc1 + y * nSrc1Step);
      const SrcType *src2Row = (const SrcType *)((const char *)pSrc2 + y * nSrc2Step);
      DstType *dstRow = (DstType *)((char *)pSrcDst + y * nSrcDstStep);

      if constexpr (Channels == 1) {
        dstRow[x] = op(dstRow[x], static_cast<DstType>(src1Row[x]), static_cast<DstType>(src2Row[x]), 0);
      } else {
        int idx = x * Channels;
        for (int c = 0; c < Channels; c++) {
          dstRow[idx + c] =
              op(dstRow[idx + c], static_cast<DstType>(src1Row[idx + c]), static_cast<DstType>(src2Row[idx + c]), 0);
        }
      }
    }
  }
}

template <typename SrcType, typename DstType, int Channels, typename OpType> class MaskedBinaryOperationExecutor {
public:
  static NppStatus execute(const SrcType *pSrc1, int nSrc1Step, const SrcType *pSrc2, int nSrc2Step, const Npp8u *pMask,
                           int nMaskStep, DstType *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, cudaStream_t stream) {
    if (!pSrc1 || !pSrc2 || !pMask || !pSrcDst)
      return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0)
      return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    maskedBinaryKernel<SrcType, DstType, Channels, OpType>
        <<<gridSize, blockSize, 0, stream>>>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pMask, nMaskStep, pSrcDst, nSrcDstStep,
                                             oSizeROI.width, oSizeROI.height, op);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// ============================================================================
// Masked Unary Operation Kernels and Executors
// ============================================================================

template <typename SrcType, typename DstType, int Channels, typename UnaryOp>
__global__ void maskedUnaryKernel(const SrcType *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                  DstType *pSrcDst, int nSrcDstStep, int width, int height, UnaryOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *maskRow = (const Npp8u *)((const char *)pMask + y * nMaskStep);

    if (maskRow[x] != 0) {
      const SrcType *srcRow = (const SrcType *)((const char *)pSrc + y * nSrcStep);
      DstType *dstRow = (DstType *)((char *)pSrcDst + y * nSrcDstStep);

      if constexpr (Channels == 1) {
        dstRow[x] = op(dstRow[x], static_cast<DstType>(srcRow[x]), 0);
      } else {
        int idx = x * Channels;
        for (int c = 0; c < Channels; c++) {
          dstRow[idx + c] = op(dstRow[idx + c], static_cast<DstType>(srcRow[idx + c]), 0);
        }
      }
    }
  }
}

template <typename SrcType, typename DstType, int Channels, typename OpType> class MaskedUnaryOperationExecutor {
public:
  static NppStatus execute(const SrcType *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, DstType *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI, cudaStream_t stream) {
    if (!pSrc || !pMask || !pSrcDst)
      return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0)
      return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    maskedUnaryKernel<SrcType, DstType, Channels, OpType><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, op);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// ============================================================================
// Mixed Type Unary Operation Kernels and Executors
// For operations like AddSquare: dst(32f) = dst + src(8u)^2
// ============================================================================

template <typename SrcType, typename DstType, int Channels, typename UnaryOp>
__global__ void mixedUnaryKernel(const SrcType *pSrc, int nSrcStep, DstType *pSrcDst, int nSrcDstStep, int width,
                                 int height, UnaryOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const SrcType *srcRow = (const SrcType *)((const char *)pSrc + y * nSrcStep);
    DstType *dstRow = (DstType *)((char *)pSrcDst + y * nSrcDstStep);

    if constexpr (Channels == 1) {
      DstType srcVal = static_cast<DstType>(srcRow[x]);
      dstRow[x] = op(dstRow[x], srcVal, 0);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        DstType srcVal = static_cast<DstType>(srcRow[idx + c]);
        dstRow[idx + c] = op(dstRow[idx + c], srcVal, 0);
      }
    }
  }
}

template <typename SrcType, typename DstType, int Channels, typename OpType> class MixedUnaryOperationExecutor {
public:
  static NppStatus execute(const SrcType *pSrc, int nSrcStep, DstType *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           cudaStream_t stream) {
    if (!pSrc || !pSrcDst)
      return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0)
      return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    mixedUnaryKernel<SrcType, DstType, Channels, OpType>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, op);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// ============================================================================
// Weighted Operation Kernels and Executors
// For AddWeighted: dst = src * alpha + dst * (1 - alpha)
// ============================================================================

template <typename SrcType, typename DstType, int Channels, typename WeightedOp>
__global__ void weightedKernel(const SrcType *pSrc, int nSrcStep, DstType *pSrcDst, int nSrcDstStep, int width,
                               int height, DstType alpha, WeightedOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const SrcType *srcRow = (const SrcType *)((const char *)pSrc + y * nSrcStep);
    DstType *dstRow = (DstType *)((char *)pSrcDst + y * nSrcDstStep);

    if constexpr (Channels == 1) {
      dstRow[x] = op(static_cast<DstType>(srcRow[x]), dstRow[x], alpha);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        dstRow[idx + c] = op(static_cast<DstType>(srcRow[idx + c]), dstRow[idx + c], alpha);
      }
    }
  }
}

template <typename SrcType, typename DstType, int Channels, typename OpType> class WeightedOperationExecutor {
public:
  static NppStatus execute(const SrcType *pSrc, int nSrcStep, DstType *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           DstType alpha, cudaStream_t stream) {
    if (!pSrc || !pSrcDst)
      return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0)
      return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    weightedKernel<SrcType, DstType, Channels, OpType><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, alpha, op);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// ============================================================================
// Masked Weighted Operation Executors
// ============================================================================

template <typename SrcType, typename DstType, int Channels, typename WeightedOp>
__global__ void maskedWeightedKernel(const SrcType *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep,
                                     DstType *pSrcDst, int nSrcDstStep, int width, int height, DstType alpha,
                                     WeightedOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *maskRow = (const Npp8u *)((const char *)pMask + y * nMaskStep);

    if (maskRow[x] != 0) {
      const SrcType *srcRow = (const SrcType *)((const char *)pSrc + y * nSrcStep);
      DstType *dstRow = (DstType *)((char *)pSrcDst + y * nSrcDstStep);

      if constexpr (Channels == 1) {
        dstRow[x] = op(static_cast<DstType>(srcRow[x]), dstRow[x], alpha);
      } else {
        int idx = x * Channels;
        for (int c = 0; c < Channels; c++) {
          dstRow[idx + c] = op(static_cast<DstType>(srcRow[idx + c]), dstRow[idx + c], alpha);
        }
      }
    }
  }
}

template <typename SrcType, typename DstType, int Channels, typename OpType> class MaskedWeightedOperationExecutor {
public:
  static NppStatus execute(const SrcType *pSrc, int nSrcStep, const Npp8u *pMask, int nMaskStep, DstType *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI, DstType alpha, cudaStream_t stream) {
    if (!pSrc || !pMask || !pSrcDst)
      return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0)
      return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    maskedWeightedKernel<SrcType, DstType, Channels, OpType><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pMask, nMaskStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, alpha, op);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// ============================================================================
// Alpha Premul Operation Kernels and Executors
// ============================================================================

template <typename T, int Channels, typename PremulOp>
__global__ void alphaPremulConstKernelT(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height,
                                        T alpha, PremulOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    if constexpr (Channels == 1) {
      dstRow[x] = op(srcRow[x], alpha);
    } else {
      int idx = x * Channels;
      for (int c = 0; c < Channels; c++) {
        dstRow[idx + c] = op(srcRow[idx + c], alpha);
      }
    }
  }
}

template <typename T, int Channels, typename OpType> class AlphaPremulConstExecutor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, T alpha,
                           cudaStream_t stream) {
    if (!pSrc || !pDst)
      return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0)
      return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    alphaPremulConstKernelT<T, Channels, OpType><<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, alpha, op);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  static NppStatus executeInplace(T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, T alpha, cudaStream_t stream) {
    return execute(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, alpha, stream);
  }
};

// AC4 Alpha Premul - alpha from 4th channel
template <typename T, typename PremulOp>
__global__ void alphaPremulAC4KernelT(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height,
                                      PremulOp op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
    T *dstRow = (T *)((char *)pDst + y * nDstStep);

    int idx = x * 4;
    T alpha = srcRow[idx + 3];

    dstRow[idx + 0] = op(srcRow[idx + 0], alpha);
    dstRow[idx + 1] = op(srcRow[idx + 1], alpha);
    dstRow[idx + 2] = op(srcRow[idx + 2], alpha);
    dstRow[idx + 3] = srcRow[idx + 3]; // Alpha unchanged
  }
}

template <typename T, typename OpType> class AlphaPremulAC4Executor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
    if (!pSrc || !pDst)
      return NPP_NULL_POINTER_ERROR;
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0)
      return NPP_SIZE_ERROR;

    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    OpType op;
    alphaPremulAC4KernelT<T, OpType>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, op);

    return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  static NppStatus executeInplace(T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, cudaStream_t stream) {
    return execute(pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI, stream);
  }
};

} // namespace arithmetic
} // namespace nppi