#pragma once

// Common kernel-launch utilities shared by the nppi / npps executors
// (introduced by package p0.1 as a pure refactoring; see
// docs/agent_packages/p0.1.md).
//
// This header is host-visible: it is included from plain .cpp translation
// units (e.g. the generated npps wrappers). It therefore never declares
// __global__ kernels. Device helper functions are marked
// __device__ __forceinline__ with plain-C++ bodies (tolerated by the host
// compilers used in this project, like the existing npps executor headers).
// getThreadCoords references the CUDA block/thread index builtins and is
// only declared when compiling with nvcc (__CUDACC__).
//
// Semantics are pinned by the differential tests against NVIDIA NPP 13.4;
// every definition here is moved verbatim from
// nppi/nppi_arithmetic_operations/nppi_arithmetic_executor.h.

#include "npp.h"

#include <cuda_runtime.h>

namespace npp_kernel {

// ============================================================================
// Block Size Configuration
// ============================================================================

// Block configuration for 2D image processing kernels. The block size is a
// template parameter so callers can deviate from the 16x16 default without
// duplicating the grid computation.
template <int BlockX = 16, int BlockY = 16> struct BlockConfig {
  static constexpr int kBlockX = BlockX;
  static constexpr int kBlockY = BlockY;

  static dim3 blockSize() { return dim3(kBlockX, kBlockY); }

  static dim3 gridSize(int width, int height) {
    return dim3((width + kBlockX - 1) / kBlockX, (height + kBlockY - 1) / kBlockY);
  }
};

// ============================================================================
// Kernel Helper Utilities
// ============================================================================

// Get row pointer with step offset
template <typename T> __device__ __forceinline__ const T *getSrcRow(const T *ptr, int y, int step) {
  return (const T *)((const char *)ptr + y * step);
}

template <typename T> __device__ __forceinline__ T *getDstRow(T *ptr, int y, int step) {
  return (T *)((char *)ptr + y * step);
}

#if defined(__CUDACC__)
// Calculate thread coordinates and check bounds (device-only: references the
// CUDA block/thread index builtins)
__device__ __forceinline__ bool getThreadCoords(int &x, int &y, int width, int height) {
  x = blockIdx.x * blockDim.x + threadIdx.x;
  y = blockIdx.y * blockDim.y + threadIdx.y;
  return (x < width && y < height);
}
#endif // defined(__CUDACC__)

// Check CUDA kernel execution result
inline NppStatus checkCudaKernelError() {
  return (cudaGetLastError() == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// ============================================================================
// Parameter Validation Utilities
// ============================================================================

// Validate shift constants based on type bit width
template <typename T> inline NppStatus validateShiftConstants(const Npp32u *aConstants, int channels) {
  // constexpr Npp32u maxShift = sizeof(T) * 8 - 1;
  constexpr Npp32u typeBits = sizeof(T) * 8;
  constexpr Npp32u maxShift = (typeBits >= 32) ? 31 : typeBits;
  for (int i = 0; i < channels; i++) {
    if (aConstants[i] > maxShift)
      return NPP_BAD_ARGUMENT_ERROR;
  }
  return NPP_NO_ERROR;
}

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

// Masked-style validation used by the in-place masked / weighted /
// alpha-premultiplied executors: every pointer must be non-null and the ROI
// strictly positive. Unlike validateBinaryParameters /
// validateUnaryParameters there is no minimum-step check and a zero-sized
// ROI is reported as NPP_SIZE_ERROR.
template <typename... PtrTypes>
inline NppStatus validateMaskedParameters(NppiSize oSizeROI, const PtrTypes *...pointers) {
  if (((pointers == nullptr) || ...)) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  return NPP_NO_ERROR;
}

} // namespace npp_kernel
