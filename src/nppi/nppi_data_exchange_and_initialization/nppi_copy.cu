#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>
#include "npp_kernel_utils.h"

// Generic multi-channel copy kernel template
template <typename T, int CHANNELS>
__global__ void nppiCopy_kernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);
  T *dst_row = (T *)((char *)pDst + y * nDstStep);

#pragma unroll
  for (int c = 0; c < CHANNELS; c++) {
    dst_row[x * CHANNELS + c] = src_row[x * CHANNELS + c];
  }
}

// 128-bit flat copy kernel (package opt_elem): each thread moves one uint4
// packet of vecElemsPerPacket<T>() elements; a trailing partial packet is
// finished element-wise by its owning thread, so ROIs whose row is not a
// multiple of the packet size stay correct. A copy is channel-agnostic, so
// the same kernel serves every packed-pixel C1/C3/C4 variant. Index math is
// 32-bit; nppiCopy_impl only launches this kernel when the packet count
// fits in 31 bits (2^31 packets is 32 GiB of traffic, so the guard never
// fires in practice).
template <typename T, int CHANNELS>
__global__ void nppiCopy_kernel_vec(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height) {
  constexpr int E = npp_kernel::vecElemsPerPacket<T>();
  static_assert(E > 0, "element size must divide the 16-byte packet size");
  const int rowElems = width * CHANNELS;
  const unsigned packets = (unsigned)((rowElems + E - 1) / E); // includes a possible partial packet
  const unsigned total = packets * (unsigned)height;

  for (unsigned i = (unsigned)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
    const unsigned y = i / packets;
    const unsigned base = (i - y * packets) * (unsigned)E;
    const T *src_row = (const T *)((const char *)pSrc + (size_t)y * nSrcStep);
    T *dst_row = (T *)((char *)pDst + (size_t)y * nDstStep);

    if (base + (unsigned)E <= (unsigned)rowElems) {
      *reinterpret_cast<uint4 *>(dst_row + base) = *reinterpret_cast<const uint4 *>(src_row + base);
    } else {
      for (unsigned g = base; g < (unsigned)rowElems; ++g)
        dst_row[g] = src_row[g];
    }
  }
}

// Generic packed to planar copy kernel (fallback)
template <typename T, int CHANNELS>
__global__ void nppiCopy_CxPxR_kernel_optimized(const T *pSrc, int nSrcStep, T *pDst0, T *pDst1, T *pDst2, int nDstStep,
                                                int width, int height) {
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y >= height)
    return;

  const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);
  T *dst_row0 = (T *)((char *)pDst0 + y * nDstStep);
  T *dst_row1 = (T *)((char *)pDst1 + y * nDstStep);
  T *dst_row2 = (T *)((char *)pDst2 + y * nDstStep);

  if (x + 3 < width) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int idx = x + i;
      dst_row0[idx] = src_row[idx * CHANNELS];
      dst_row1[idx] = src_row[idx * CHANNELS + 1];
      if (CHANNELS >= 3)
        dst_row2[idx] = src_row[idx * CHANNELS + 2];
    }
  } else {
    for (int i = 0; i < 4 && x + i < width; i++) {
      int idx = x + i;
      dst_row0[idx] = src_row[idx * CHANNELS];
      dst_row1[idx] = src_row[idx * CHANNELS + 1];
      if (CHANNELS >= 3)
        dst_row2[idx] = src_row[idx * CHANNELS + 2];
    }
  }
}

// Pointers passed to the planar <-> packed fallback kernels by value, so the
// Ctx entry points stay asynchronous and do not need a temporary device buffer.
template <typename T, int CHANNELS>
struct NppiCopyPointers {
  T *p[CHANNELS];
};

// Legacy kernel for C4P4R
template <typename T, int CHANNELS>
__global__ void nppiCopy_CxPxR_kernel(const T *pSrc, int nSrcStep, NppiCopyPointers<T, CHANNELS> pDst, int nDstStep,
                                      int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);

#pragma unroll
  for (int c = 0; c < CHANNELS; c++) {
    T *dst_row = (T *)((char *)pDst.p[c] + y * nDstStep);
    dst_row[x] = src_row[x * CHANNELS + c];
  }
}

// Generic planar to packed copy kernel (fallback)
template <typename T, int CHANNELS>
__global__ void nppiCopy_PxCxR_kernel_optimized(const T *pSrc0, const T *pSrc1, const T *pSrc2, int nSrcStep, T *pDst,
                                                int nDstStep, int width, int height) {
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y >= height)
    return;

  const T *src_row0 = (const T *)((const char *)pSrc0 + y * nSrcStep);
  const T *src_row1 = (const T *)((const char *)pSrc1 + y * nSrcStep);
  const T *src_row2 = (const T *)((const char *)pSrc2 + y * nSrcStep);
  T *dst_row = (T *)((char *)pDst + y * nDstStep);

  if (x + 3 < width) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int idx = x + i;
      dst_row[idx * CHANNELS] = src_row0[idx];
      dst_row[idx * CHANNELS + 1] = src_row1[idx];
      if (CHANNELS >= 3)
        dst_row[idx * CHANNELS + 2] = src_row2[idx];
    }
  } else {
    for (int i = 0; i < 4 && x + i < width; i++) {
      int idx = x + i;
      dst_row[idx * CHANNELS] = src_row0[idx];
      dst_row[idx * CHANNELS + 1] = src_row1[idx];
      if (CHANNELS >= 3)
        dst_row[idx * CHANNELS + 2] = src_row2[idx];
    }
  }
}

// Legacy kernel for P4C4R
template <typename T, int CHANNELS>
__global__ void nppiCopy_PxCxR_kernel(NppiCopyPointers<T, CHANNELS> pSrc, int nSrcStep, T *pDst, int nDstStep,
                                      int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  T *dst_row = (T *)((char *)pDst + y * nDstStep);

#pragma unroll
  for (int c = 0; c < CHANNELS; c++) {
    const T *src_row = (const T *)((const char *)pSrc.p[c] + y * nSrcStep);
    dst_row[x * CHANNELS + c] = src_row[x];
  }
}

// ============================================================================
// Planar <-> packed copy kernels (package opt_planar)
// ============================================================================
// A packed <-> planar copy is a CHANNELS-way transpose.  The original
// implementation staged each 16-pixel unit through per-plane local arrays
// and moved it with 128-bit loads/stores; ptxas miscompiles that register
// transpose for sm_120 (measured: the middle plane's last four bytes alias
// the third source packet, corrupting channel 1 at fixed unit offsets, at
// -O0 as well).  The kernels below keep the grid-stride launch shape and
// move elements with plain offsets instead, which compiles correctly on
// both CUDA and MACA.
//
// The executors launch these kernels only when every base pointer is 16B
// aligned and both steps are multiples of 16; anything else (unaligned plane
// pointers, odd steps, oversized images) keeps the scalar fallback kernels
// above, so results and status codes are unchanged for those inputs.
template <typename T, int CHANNELS>
__global__ void nppiCopy_CxPxR_kernel_vec(const T *pSrc, int nSrcStep, NppiCopyPointers<T, CHANNELS> pDst, int nDstStep,
                                          int width, int height) {
  const unsigned total = (unsigned)width * (unsigned)height;

  for (unsigned i = (unsigned)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
    const unsigned y = i / (unsigned)width;
    const int x = (int)(i - y * (unsigned)width);
    const T *src_row = (const T *)((const char *)pSrc + (size_t)y * nSrcStep);
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
      T *dst_row = (T *)((char *)pDst.p[c] + (size_t)y * nDstStep);
      dst_row[x] = src_row[x * CHANNELS + c];
    }
  }
}

// Planar -> packed (P3C3R / P4C4R) mirror of nppiCopy_CxPxR_kernel_vec.
template <typename T, int CHANNELS>
__global__ void nppiCopy_PxCxR_kernel_vec(NppiCopyPointers<T, CHANNELS> pSrc, int nSrcStep, T *pDst, int nDstStep,
                                          int width, int height) {
  const unsigned total = (unsigned)width * (unsigned)height;

  for (unsigned i = (unsigned)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
    const unsigned y = i / (unsigned)width;
    const int x = (int)(i - y * (unsigned)width);
    T *dst_row = (T *)((char *)pDst + (size_t)y * nDstStep);
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
      const T *src_row = (const T *)((const char *)pSrc.p[c] + (size_t)y * nSrcStep);
      dst_row[x * CHANNELS + c] = src_row[x];
    }
  }
}

// Generic copy implementation template (outside extern "C")
template <typename T, int CHANNELS>
NppStatus nppiCopy_impl(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                        NppStreamContext nppStreamCtx) {
  // 128-bit vectorized flat copy when the layout permits 16-byte accesses
  // (base pointers 16B-aligned, row steps multiples of 16 bytes, at least one
  // full packet per row, packet count fits in 31 bits); scalar
  // one-pixel-per-thread kernel otherwise (package opt_elem). Both kernels
  // copy exactly width*CHANNELS elements per row, so results are identical.
  constexpr int E = npp_kernel::vecElemsPerPacket<T>();
  const size_t vecPackets = npp_kernel::vecPacketCount<T>(oSizeROI.width, CHANNELS, oSizeROI.height);
  if (npp_kernel::vecAligned2(pSrc, nSrcStep, pDst, nDstStep) &&
      (size_t)oSizeROI.width * CHANNELS >= (size_t)E && vecPackets < ((size_t)1 << 31)) {
    const dim3 grid(npp_kernel::vecGridSize(vecPackets));
    nppiCopy_kernel_vec<T, CHANNELS><<<grid, npp_kernel::kVecBlock, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  } else {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    nppiCopy_kernel<T, CHANNELS><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                   oSizeROI.width, oSizeROI.height);
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

extern "C" {

// Explicit instantiations for 8u
NppStatus nppiCopy_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp8u, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp8u, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  // The old special case used 32-bit accesses (one pixel per thread) and did
  // not check base-pointer alignment; the generic impl now dispatches to the
  // 128-bit flat copy kernel with a checked alignment condition and a scalar
  // fallback (package opt_elem), which subsumes it.
  return nppiCopy_impl<Npp8u, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Explicit instantiations for 16u
NppStatus nppiCopy_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16u, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16u, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16u, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Explicit instantiations for 16s
NppStatus nppiCopy_16s_C1R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16s, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16s_C3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16s, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16s_C4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp16s, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Explicit instantiations for 32s
NppStatus nppiCopy_32s_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32s, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32s_C3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32s, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32s_C4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32s, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// Explicit instantiations for 32f
NppStatus nppiCopy_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32f, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32f, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiCopy_impl<Npp32f, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

} // extern "C"

// True when the packed base pointer, every plane base pointer and both row
// steps permit 16-byte packet access (package opt_planar). Steps are already
// validated to be positive by the wrappers.
template <typename T, int CHANNELS>
static bool planarCopyVecReady(const void *pPacked, int nPackedStep, const NppiCopyPointers<T, CHANNELS> &planes,
                               int nPlaneStep) {
  uintptr_t ptrs = reinterpret_cast<uintptr_t>(pPacked);
  for (int c = 0; c < CHANNELS; ++c) {
    ptrs |= reinterpret_cast<uintptr_t>(planes.p[c]);
  }
  return (ptrs & 15u) == 0u && (((unsigned)nPackedStep | (unsigned)nPlaneStep) & 15u) == 0u;
}

// Packed -> planar copy (C3P3R / C4P4R) for every data type.
//
// Empty ROIs have always reached the kernel launch with a zero grid
// dimension, which the runtime rejects, so NPP_CUDA_KERNEL_EXECUTION_ERROR
// is the pinned status for that case (package opt_planar keeps it instead of
// turning empty ROIs into silent no-ops). Aligned layouts take the 128-bit
// transpose kernel; everything else keeps the historical scalar fallback
// kernels with their original launch configurations.
template <typename T, int CHANNELS>
NppStatus nppiCopy_CxPxR_impl(const T *pSrc, int nSrcStep, T *const pDst[], int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  constexpr int P = npp_kernel::vecElemsPerPacket<T>();
  NppiCopyPointers<T, CHANNELS> dst{};
  for (int c = 0; c < CHANNELS; ++c) {
    dst.p[c] = pDst[c];
  }

  // Units of P pixels per row; the kernel indexes them with 32-bit math, so
  // only take the vectorized path when the count fits in 31 bits (at or
  // above 32 GiB of traffic for every element type).
  const size_t units = ((size_t)oSizeROI.width + P - 1) / P * (size_t)oSizeROI.height;
  if (units < ((size_t)1 << 31) && planarCopyVecReady<T, CHANNELS>(pSrc, nSrcStep, dst, nDstStep)) {
    nppiCopy_CxPxR_kernel_vec<T, CHANNELS>
        <<<npp_kernel::vecGridSize(units), npp_kernel::kVecBlock, 0, nppStreamCtx.hStream>>>(
            pSrc, nSrcStep, dst, nDstStep, oSizeROI.width, oSizeROI.height);
  } else if constexpr (CHANNELS == 3) {
    dim3 blockSize(32, 8);
    dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    nppiCopy_CxPxR_kernel_optimized<T, 3><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, dst.p[0], dst.p[1], dst.p[2], nDstStep, oSizeROI.width, oSizeROI.height);
  } else {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    nppiCopy_CxPxR_kernel<T, CHANNELS><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, dst, nDstStep, oSizeROI.width, oSizeROI.height);
  }

  return npp_kernel::checkCudaKernelError();
}

// Planar -> packed copy (P3C3R / P4C4R) for every data type; see
// nppiCopy_CxPxR_impl for the empty-ROI / fallback policy.
template <typename T, int CHANNELS>
NppStatus nppiCopy_PxCxR_impl(T *const pSrc[], int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  constexpr int P = npp_kernel::vecElemsPerPacket<T>();
  NppiCopyPointers<T, CHANNELS> src{};
  for (int c = 0; c < CHANNELS; ++c) {
    src.p[c] = pSrc[c];
  }

  const size_t units = ((size_t)oSizeROI.width + P - 1) / P * (size_t)oSizeROI.height;
  if (units < ((size_t)1 << 31) && planarCopyVecReady<T, CHANNELS>(pDst, nDstStep, src, nSrcStep)) {
    nppiCopy_PxCxR_kernel_vec<T, CHANNELS>
        <<<npp_kernel::vecGridSize(units), npp_kernel::kVecBlock, 0, nppStreamCtx.hStream>>>(
            src, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  } else if constexpr (CHANNELS == 3) {
    dim3 blockSize(32, 8);
    dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    nppiCopy_PxCxR_kernel_optimized<T, 3><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        src.p[0], src.p[1], src.p[2], nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  } else {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    nppiCopy_PxCxR_kernel<T, CHANNELS><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        src, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  }

  return npp_kernel::checkCudaKernelError();
}

extern "C" {

// C3P3R implementations for all data types
NppStatus nppiCopy_8u_C3P3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *const pDst[3], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp8u, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_C3P3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp16u, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16s_C3P3R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp16s, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32s_C3P3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp32s, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C3P3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *const pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp32f, 3>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// C4P4R implementations for all data types
NppStatus nppiCopy_8u_C4P4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *const pDst[4], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp8u, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_C4P4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *const pDst[4], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp16u, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16s_C4P4R_Ctx_impl(const Npp16s *pSrc, int nSrcStep, Npp16s *const pDst[4], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp16s, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32s_C4P4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *const pDst[4], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp32s, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_C4P4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *const pDst[4], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_CxPxR_impl<Npp32f, 4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// P3C3R implementations for all data types
NppStatus nppiCopy_8u_P3C3R_Ctx_impl(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp8u, 3>(const_cast<Npp8u **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_P3C3R_Ctx_impl(const Npp16u *const pSrc[3], int nSrcStep, Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp16u, 3>(const_cast<Npp16u **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI,
                                        nppStreamCtx);
}

NppStatus nppiCopy_16s_P3C3R_Ctx_impl(const Npp16s *const pSrc[3], int nSrcStep, Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp16s, 3>(const_cast<Npp16s **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI,
                                        nppStreamCtx);
}

NppStatus nppiCopy_32s_P3C3R_Ctx_impl(const Npp32s *const pSrc[3], int nSrcStep, Npp32s *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp32s, 3>(const_cast<Npp32s **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI,
                                        nppStreamCtx);
}

NppStatus nppiCopy_32f_P3C3R_Ctx_impl(const Npp32f *const pSrc[3], int nSrcStep, Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp32f, 3>(const_cast<Npp32f **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI,
                                        nppStreamCtx);
}

// P4C4R implementations for all data types
NppStatus nppiCopy_8u_P4C4R_Ctx_impl(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp8u, 4>(const_cast<Npp8u **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16u_P4C4R_Ctx_impl(const Npp16u *const pSrc[4], int nSrcStep, Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp16u, 4>(const_cast<Npp16u **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_16s_P4C4R_Ctx_impl(const Npp16s *const pSrc[4], int nSrcStep, Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp16s, 4>(const_cast<Npp16s **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32s_P4C4R_Ctx_impl(const Npp32s *const pSrc[4], int nSrcStep, Npp32s *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp32s, 4>(const_cast<Npp32s **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiCopy_32f_P4C4R_Ctx_impl(const Npp32f *const pSrc[4], int nSrcStep, Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiCopy_PxCxR_impl<Npp32f, 4>(const_cast<Npp32f **>(pSrc), nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
}