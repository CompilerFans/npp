#pragma once

#include "npp.h"
#include <cuda_runtime.h>
#include <type_traits>

namespace nppi {
namespace statistics {

// ============================================================================
// MinEvery/MaxEvery Operation Functors
// ============================================================================

template <typename T> struct MaxEveryOp {
  __device__ __host__ T operator()(T a, T b) const { return a > b ? a : b; }
};

template <typename T> struct MinEveryOp {
  __device__ __host__ T operator()(T a, T b) const { return a < b ? a : b; }
};

// ============================================================================
// Binary Inplace Operation Kernel
// ============================================================================

template <typename T, int Channels, typename Op>
__global__ void binaryInplaceKernel(const T *pSrc, int nSrcStep, T *pSrcDst, int nSrcDstStep, int width, int height,
                                    Op op) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const T *srcRow = reinterpret_cast<const T *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
    T *dstRow = reinterpret_cast<T *>(reinterpret_cast<char *>(pSrcDst) + y * nSrcDstStep);

    for (int c = 0; c < Channels; c++) {
      int idx = x * Channels + c;
      dstRow[idx] = op(srcRow[idx], dstRow[idx]);
    }
  }
}

// ============================================================================
// Binary Inplace Operation Executor
// ============================================================================

template <typename T, int Channels, typename Op> class BinaryInplaceExecutor {
public:
  static NppStatus execute(const T *pSrc, int nSrcStep, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           cudaStream_t stream) {
    if (!pSrc || !pSrcDst) {
      return NPP_NULL_POINTER_ERROR;
    }
    if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
      return NPP_SIZE_ERROR;
    }

    dim3 block(16, 16);
    dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);

    Op op;
    binaryInplaceKernel<T, Channels, Op>
        <<<grid, block, 0, stream>>>(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height, op);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
};

// ============================================================================
// Binary Inplace Operation API
// ============================================================================

template <typename T, int Channels, template <typename> class OpTemplate> class BinaryInplaceOpAPI {
public:
  using Op = OpTemplate<T>;

  static NppStatus executeInplace(const T *pSrc, int nSrcStep, T *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
    return BinaryInplaceExecutor<T, Channels, Op>::execute(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI,
                                                           nppStreamCtx.hStream);
  }
};

// ============================================================================
// Helper: Get default stream context
// ============================================================================

inline NppStreamContext getDefaultStreamContext() {
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  return ctx;
}

} // namespace statistics
} // namespace nppi
