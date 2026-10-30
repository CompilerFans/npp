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
#include <cstdint>

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

// ============================================================================
// 128-bit Vectorized Access Support (package opt_elem)
// ============================================================================
// Purely element-wise kernels (binary / unary / constant / compare) move data
// in 16-byte packets when every base pointer is 16B-aligned and every row
// step is a multiple of 16 bytes. Element types whose size does not divide
// the 16-byte packet, and any unaligned pointer / step, fall back to the
// original scalar kernels; a partial trailing packet is handled element-wise
// inside the vector kernels. Op functors are invoked per element exactly as
// in the scalar kernels, so numerics are unchanged.

// Threads per block for the flat grid-stride vector kernels.
constexpr int kVecBlock = 256;

// Elements of type T moved per 16-byte packet. Every Npp* element type has a
// size of 1/2/4/8/16 bytes, so this is >= 1 for all of them.
template <typename T> constexpr int vecElemsPerPacket() { return 16 / int(sizeof(T)); }

// Total number of 16-byte packets covering an image of width*channels
// elements of T over height rows. The vectorized kernels index these packets
// with 32-bit arithmetic, so executors only take the vectorized path when
// this count fits in 31 bits; 2^31 packets is 32 GiB of traffic, far beyond
// any realistic image, so the guard never fires in practice.
template <typename T> inline size_t vecPacketCount(int width, int channels, int height) {
  constexpr int E = 16 / int(sizeof(T));
  return ((size_t)width * channels + E - 1) / E * (size_t)height;
}

// True when two base pointers and their row steps permit 128-bit row access.
// Steps are validated (>= minimum row size > 0) before this is called.
inline bool vecAligned2(const void *p1, int step1, const void *p2, int step2) {
  return ((reinterpret_cast<uintptr_t>(p1) | reinterpret_cast<uintptr_t>(p2)) & 15u) == 0 &&
         ((static_cast<unsigned>(step1) | static_cast<unsigned>(step2)) & 15u) == 0;
}

// Three-pointer variant (two sources + destination).
inline bool vecAligned3(const void *p1, int step1, const void *p2, int step2, const void *p3, int step3) {
  return ((reinterpret_cast<uintptr_t>(p1) | reinterpret_cast<uintptr_t>(p2) | reinterpret_cast<uintptr_t>(p3)) &
          15u) == 0 &&
         ((static_cast<unsigned>(step1) | static_cast<unsigned>(step2) | static_cast<unsigned>(step3)) & 15u) == 0;
}

// One 16-byte packet reinterpreted as E elements of T. Lives here so every
// vectorized executor (nppi arithmetic / copy, npps element-wise) shares one
// definition; the union is only ever type-punned between .raw loads/stores
// and .elem[] functor application, never both at once.
namespace vec {
template <typename T, int E> union Packet {
  uint4 raw;
  T elem[E];
};
} // namespace vec

// Packets covering a flat 1D signal of n elements (npps element-wise ops).
template <typename T> inline size_t vecPacketCount1D(size_t n) {
  constexpr int E = 16 / int(sizeof(T));
  return (n + E - 1) / E;
}

// True when the base pointers (no row steps) permit 128-bit packet access.
inline bool vecAlignedPtr2(const void *p1, const void *p2) {
  return ((reinterpret_cast<uintptr_t>(p1) | reinterpret_cast<uintptr_t>(p2)) & 15u) == 0;
}

// Three-pointer variant (two sources + destination).
inline bool vecAlignedPtr3(const void *p1, const void *p2, const void *p3) {
  return ((reinterpret_cast<uintptr_t>(p1) | reinterpret_cast<uintptr_t>(p2) |
           reinterpret_cast<uintptr_t>(p3)) & 15u) == 0;
}

// True when a flat 1D signal of n elements of T can be walked with 32-bit
// packet index math: the end of the last (possibly partial) packet must stay
// below 2^32 elements. Larger signals keep the 64-bit-index scalar kernels.
// For every element type this bound is at or above 4 GiB of signal data.
template <typename T> inline bool vec1dIndexSafe(size_t n) {
  return n <= ((size_t)0xFFFFFFF0u) / sizeof(T);
}

// Grid size covering totalWork packets with one packet per thread
// (grid-stride loops absorb anything beyond the cap).
inline unsigned vecGridSize(size_t totalWork) {
  const size_t grid = (totalWork + kVecBlock - 1) / kVecBlock;
  return static_cast<unsigned>(grid < 65535u ? grid : 65535u);
}

} // namespace npp_kernel
