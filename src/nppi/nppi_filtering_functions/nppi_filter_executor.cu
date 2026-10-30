// nppiFilter32f / nppiFilterBorder(32f) / nppiFilterRow(32f) / nppiFilterColumn(32f)
// kernels and executor entry points.  See nppi_filter_executor.h for the family
// contract (semantics measured against NVIDIA NPP 13.4 on an RTX 5090).
#include "npp.h"
#include "nppi_filter_executor.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace nppi {
namespace filter {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

inline NppStatus finish_launch() {
  if (cudaGetLastError() == cudaSuccess) return NPP_NO_ERROR;
  return NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Source sample as float (all supported source types are arithmetic).
template <typename T>
__host__ __device__ __forceinline__ float to_float(const T &s) {
  return static_cast<float>(s);
}

// Truncate a float accumulation toward zero and saturate into the destination
// integer type (NPP behaviour: -1.5 -> -1, 0.75 -> 0, 30000*2 -> 32767 for 16s).
template <typename Dst>
__host__ __device__ __forceinline__ Dst emit_int(float acc) {
  const float t = truncf(acc);
  if constexpr (std::is_same_v<Dst, Npp32s>) {
    if (t >= 2147483648.0f) return 2147483647;
    if (t <= -2147483648.0f) return -2147483647 - 1;
  } else if constexpr (std::is_same_v<Dst, Npp16s>) {
    if (t < -32768.0f) return -32768;
    if (t > 32767.0f) return 32767;
  } else if constexpr (std::is_same_v<Dst, Npp16u>) {
    if (t < 0.0f) return 0;
    if (t > 65535.0f) return 65535;
  } else if constexpr (std::is_same_v<Dst, Npp8s>) {
    if (t < -128.0f) return -128;
    if (t > 127.0f) return 127;
  } else {  // Npp8u
    if (t < 0.0f) return 0;
    if (t > 255.0f) return 255;
  }
  return static_cast<Dst>(t);
}

// Emit a float accumulation into the destination: integers truncate+clamp,
// Npp32f passes the value through unchanged.
template <typename Dst>
__host__ __device__ __forceinline__ Dst emit_dst(float acc) {
  if constexpr (std::is_same_v<Dst, Npp32f>) {
    return acc;
  } else {
    return emit_int<Dst>(acc);
  }
}

// Saturate an already-integral (post-division) result into the destination.
template <typename Dst>
__host__ __device__ __forceinline__ Dst sat_i32(int v) {
  if constexpr (std::is_same_v<Dst, Npp16s>) {
    if (v < -32768) return -32768;
    if (v > 32767) return 32767;
  } else if constexpr (std::is_same_v<Dst, Npp16u>) {
    if (v < 0) return 0;
    if (v > 65535) return 65535;
  } else if constexpr (std::is_same_v<Dst, Npp8s>) {
    if (v < -128) return -128;
    if (v > 127) return 127;
  } else if constexpr (std::is_same_v<Dst, Npp8u>) {
    if (v < 0) return 0;
    if (v > 255) return 255;
  }
  return static_cast<Dst>(v);
}

// ---------------------------------------------------------------------------
// 2D generic filter, no border.  Reads are raw: taps may leave the ROI.
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter2d_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst,
                                int nDstStep, int W, int H, const Npp32f *__restrict__ pKernel,
                                int kw, int kh, int ax, int ay) {
  constexpr int kNch = AC4 ? 3 : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  for (int c = 0; c < kNch; ++c) {
    float acc = 0.0f;
    for (int j = 0; j < kh; ++j) {
      const char *srow = (const char *)pSrc + (long long)(y + ay - j) * nSrcStep;
      const Npp32f *krow = pKernel + (long long)j * kw;
      for (int i = 0; i < kw; ++i) {
        const Src *sp = (const Src *)(srow + ((long long)(x + ax - i) * CH + c) * (long long)sizeof(Src));
        acc += to_float(*sp) * krow[i];
      }
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = emit_dst<Dst>(acc);
  }
}

// ---------------------------------------------------------------------------
// 2D filter with replicate border (32f kernel).  Image origin is
// pSrc - oSrcOffset; every tap coordinate is clamped into [0, srcSize-1].
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter2d_border32f_kernel(const Src *__restrict__ pSrc, int nSrcStep, int sw,
                                          int sh, int offx, int offy, Dst *__restrict__ pDst,
                                          int nDstStep, int W, int H,
                                          const Npp32f *__restrict__ pKernel, int kw, int kh,
                                          int ax, int ay) {
  constexpr int kNch = AC4 ? 3 : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *origin =
      (const char *)pSrc - (long long)offy * nSrcStep - (long long)offx * CH * (long long)sizeof(Src);
  for (int c = 0; c < kNch; ++c) {
    float acc = 0.0f;
    for (int j = 0; j < kh; ++j) {
      const int iy = clampi(y + offy + ay - j, 0, sh - 1);
      const char *srow = origin + (long long)iy * nSrcStep;
      const Npp32f *krow = pKernel + (long long)j * kw;
      for (int i = 0; i < kw; ++i) {
        const int ix = clampi(x + offx + ax - i, 0, sw - 1);
        const Src *sp = (const Src *)(srow + ((long long)ix * CH + c) * (long long)sizeof(Src));
        acc += to_float(*sp) * krow[i];
      }
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = emit_dst<Dst>(acc);
  }
}

// ---------------------------------------------------------------------------
// 2D filter with replicate border (32s kernel + divisor).  Accumulation wraps
// at 32 bits (ring arithmetic: int64 accumulation + one wrap is equivalent),
// division truncates toward zero, the quotient saturates.
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter2d_border_div_kernel(const Src *__restrict__ pSrc, int nSrcStep, int sw,
                                           int sh, int offx, int offy, Dst *__restrict__ pDst,
                                           int nDstStep, int W, int H,
                                           const Npp32s *__restrict__ pKernel, int kw, int kh,
                                           int ax, int ay, int nDivisor) {
  constexpr int kNch = AC4 ? 3 : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *origin =
      (const char *)pSrc - (long long)offy * nSrcStep - (long long)offx * CH * (long long)sizeof(Src);
  for (int c = 0; c < kNch; ++c) {
    long long acc = 0;
    for (int j = 0; j < kh; ++j) {
      const int iy = clampi(y + offy + ay - j, 0, sh - 1);
      const char *srow = origin + (long long)iy * nSrcStep;
      const Npp32s *krow = pKernel + (long long)j * kw;
      for (int i = 0; i < kw; ++i) {
        const int ix = clampi(x + offx + ax - i, 0, sw - 1);
        const Src *sp = (const Src *)(srow + ((long long)ix * CH + c) * (long long)sizeof(Src));
        acc += (long long)krow[i] * (long long)(int)*sp;
      }
    }
    const int wrapped = (int)(unsigned int)((unsigned long long)acc & 0xffffffffull);
    int q;
    if (wrapped == INT32_MIN && nDivisor == -1) {
      q = INT32_MIN;
    } else {
      q = wrapped / nDivisor;
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = sat_i32<Dst>(q);
  }
}

// ---------------------------------------------------------------------------
// Separable row/column filters (no border; raw reads).
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter_row_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst,
                                  int nDstStep, int W, int H, const Npp32f *__restrict__ pKernel,
                                  int mask, int ax) {
  constexpr int kNch = AC4 ? 3 : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *srow = (const char *)pSrc + (long long)y * nSrcStep;
  for (int c = 0; c < kNch; ++c) {
    float acc = 0.0f;
    for (int i = 0; i < mask; ++i) {
      const Src *sp = (const Src *)(srow + ((long long)(x + ax - i) * CH + c) * (long long)sizeof(Src));
      acc += to_float(*sp) * pKernel[i];
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = emit_dst<Dst>(acc);
  }
}

template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter_col_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst,
                                  int nDstStep, int W, int H, const Npp32f *__restrict__ pKernel,
                                  int mask, int ay) {
  constexpr int kNch = AC4 ? 3 : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  for (int c = 0; c < kNch; ++c) {
    float acc = 0.0f;
    for (int i = 0; i < mask; ++i) {
      const char *srow = (const char *)pSrc + (long long)(y + ay - i) * nSrcStep;
      const Src *sp = (const Src *)(srow + ((long long)x * CH + c) * (long long)sizeof(Src));
      acc += to_float(*sp) * pKernel[i];
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = emit_dst<Dst>(acc);
  }
}

// ---------------------------------------------------------------------------
// Separable row/column filters with replicate border.  The non-filtered axis
// is replicate-clamped as well (matches NVIDIA NPP 13.4 measured behaviour).
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter_row_border32f_kernel(const Src *__restrict__ pSrc, int nSrcStep, int sw,
                                            int sh, int offx, int offy, Dst *__restrict__ pDst,
                                            int nDstStep, int W, int H,
                                            const Npp32f *__restrict__ pKernel, int mask, int ax) {
  constexpr int kNch = AC4 ? 3 : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *origin =
      (const char *)pSrc - (long long)offy * nSrcStep - (long long)offx * CH * (long long)sizeof(Src);
  const int iy = clampi(y + offy, 0, sh - 1);
  const char *srow = origin + (long long)iy * nSrcStep;
  for (int c = 0; c < kNch; ++c) {
    float acc = 0.0f;
    for (int i = 0; i < mask; ++i) {
      const int ix = clampi(x + offx + ax - i, 0, sw - 1);
      const Src *sp = (const Src *)(srow + ((long long)ix * CH + c) * (long long)sizeof(Src));
      acc += to_float(*sp) * pKernel[i];
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = emit_dst<Dst>(acc);
  }
}

template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter_col_border32f_kernel(const Src *__restrict__ pSrc, int nSrcStep, int sw,
                                            int sh, int offx, int offy, Dst *__restrict__ pDst,
                                            int nDstStep, int W, int H,
                                            const Npp32f *__restrict__ pKernel, int mask, int ay) {
  constexpr int kNch = AC4 ? 3 : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *origin =
      (const char *)pSrc - (long long)offy * nSrcStep - (long long)offx * CH * (long long)sizeof(Src);
  const int ix = clampi(x + offx, 0, sw - 1);
  for (int c = 0; c < kNch; ++c) {
    float acc = 0.0f;
    for (int i = 0; i < mask; ++i) {
      const int iy = clampi(y + offy + ay - i, 0, sh - 1);
      const char *srow = origin + (long long)iy * nSrcStep;
      const Src *sp = (const Src *)(srow + ((long long)ix * CH + c) * (long long)sizeof(Src));
      acc += to_float(*sp) * pKernel[i];
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = emit_dst<Dst>(acc);
  }
}

inline bool step_too_small(int nDstStep, int rowBytes, int H) {
  return H > 0 && (Npp32u)nDstStep < (Npp32u)rowBytes;
}

}  // namespace

// ---------------------------------------------------------------------------
// nppiFilter32f
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filter32f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp32f *pKernel, NppiSize oKernelSize, NppiPoint oAnchor,
                    cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int kw = oKernelSize.width;
  const int kh = oKernelSize.height;
  if (oAnchor.x < 1 - oSizeROI.width || oAnchor.x > kw || oAnchor.y < 1 - oSizeROI.height ||
      oAnchor.y > kh) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  filter2d_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pKernel, kw, kh,
      oAnchor.x, oAnchor.y);
  return finish_launch();
}

// ---------------------------------------------------------------------------
// nppiFilterBorder32f / nppiFilterBorder_32f
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterBorder32f(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                          Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f *pKernel,
                          NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType,
                          cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  filter2d_border32f_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
      oSizeROI.width, oSizeROI.height, pKernel, oKernelSize.width, oKernelSize.height, oAnchor.x,
      oAnchor.y);
  return finish_launch();
}

// ---------------------------------------------------------------------------
// nppiFilterBorder (integer kernels with divisor)
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterBorderDiv(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                          Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32s *pKernel,
                          NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                          NppiBorderType eBorderType, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (nDivisor == 0) return NPP_DIVISOR_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  filter2d_border_div_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
      oSizeROI.width, oSizeROI.height, pKernel, oKernelSize.width, oKernelSize.height, oAnchor.x,
      oAnchor.y, nDivisor);
  return finish_launch();
}

// ---------------------------------------------------------------------------
// nppiFilterRow32f / nppiFilterColumn32f
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterRow32f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                       const Npp32f *pKernel, int nMaskSize, int nAnchor, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oROI.width < 0 || oROI.height < 0) return NPP_SIZE_ERROR;
  if (nAnchor < 1 - oROI.width || nAnchor > nMaskSize) return NPP_OUT_OFF_RANGE_ERROR;
  const int rowBytes = oROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oROI.height)) return NPP_STEP_ERROR;
  if (oROI.width == 0 || oROI.height == 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  filter_row_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pKernel, nMaskSize, nAnchor);
  return finish_launch();
}

template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterColumn32f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                          const Npp32f *pKernel, int nMaskSize, int nAnchor, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oROI.width < 0 || oROI.height < 0) return NPP_SIZE_ERROR;
  if (nAnchor < 1 - oROI.height || nAnchor > nMaskSize) return NPP_OUT_OFF_RANGE_ERROR;
  const int rowBytes = oROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oROI.height)) return NPP_STEP_ERROR;
  if (oROI.width == 0 || oROI.height == 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  filter_col_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pKernel, nMaskSize, nAnchor);
  return finish_launch();
}

// ---------------------------------------------------------------------------
// nppiFilterRowBorder32f / nppiFilterColumnBorder32f
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterRowBorder32f(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                             Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f *pKernel,
                             int nMaskSize, int nAnchor, NppiBorderType eBorderType,
                             cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  filter_row_border32f_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
      oSizeROI.width, oSizeROI.height, pKernel, nMaskSize, nAnchor);
  return finish_launch();
}

template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterColumnBorder32f(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                                Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32f *pKernel,
                                int nMaskSize, int nAnchor, NppiBorderType eBorderType,
                                cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  filter_col_border32f_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
      oSizeROI.width, oSizeROI.height, pKernel, nMaskSize, nAnchor);
  return finish_launch();
}

// ---------------------------------------------------------------------------
// Explicit instantiations.
// Generated lists; keep in sync with the family tables in the header.

// filter32f
template NppStatus filter32f<Npp8u, Npp8u, 1, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8u, Npp8u, 2, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8u, Npp8u, 3, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8u, Npp8u, 4, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8u, Npp8u, 4, true>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8s, Npp8s, 1, false>(const Npp8s *, int, Npp8s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8s, Npp8s, 2, false>(const Npp8s *, int, Npp8s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8s, Npp8s, 3, false>(const Npp8s *, int, Npp8s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8s, Npp8s, 4, false>(const Npp8s *, int, Npp8s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8s, Npp8s, 4, true>(const Npp8s *, int, Npp8s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp16u, Npp16u, 1, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp16u, Npp16u, 3, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp16u, Npp16u, 4, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp16u, Npp16u, 4, true>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp16s, Npp16s, 1, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp16s, Npp16s, 3, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp16s, Npp16s, 4, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp16s, Npp16s, 4, true>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp32s, Npp32s, 1, false>(const Npp32s *, int, Npp32s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp32s, Npp32s, 3, false>(const Npp32s *, int, Npp32s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp32s, Npp32s, 4, false>(const Npp32s *, int, Npp32s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp32s, Npp32s, 4, true>(const Npp32s *, int, Npp32s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8u, Npp16s, 1, false>(const Npp8u *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8u, Npp16s, 3, false>(const Npp8u *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8u, Npp16s, 4, false>(const Npp8u *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8u, Npp16s, 4, true>(const Npp8u *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8s, Npp16s, 1, false>(const Npp8s *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8s, Npp16s, 3, false>(const Npp8s *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8s, Npp16s, 4, false>(const Npp8s *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp8s, Npp16s, 4, true>(const Npp8s *, int, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);

// filterBorder32f
template NppStatus filterBorder32f<Npp8u, Npp8u, 1, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8u, Npp8u, 2, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8u, Npp8u, 3, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8u, Npp8u, 4, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8u, Npp8u, 4, true>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8s, Npp8s, 1, false>(const Npp8s *, int, NppiSize, NppiPoint, Npp8s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8s, Npp8s, 2, false>(const Npp8s *, int, NppiSize, NppiPoint, Npp8s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8s, Npp8s, 3, false>(const Npp8s *, int, NppiSize, NppiPoint, Npp8s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8s, Npp8s, 4, false>(const Npp8s *, int, NppiSize, NppiPoint, Npp8s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8s, Npp8s, 4, true>(const Npp8s *, int, NppiSize, NppiPoint, Npp8s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp16u, Npp16u, 1, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp16u, Npp16u, 3, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp16u, Npp16u, 4, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp16u, Npp16u, 4, true>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp16s, Npp16s, 1, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp16s, Npp16s, 3, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp16s, Npp16s, 4, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp16s, Npp16s, 4, true>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp32s, Npp32s, 1, false>(const Npp32s *, int, NppiSize, NppiPoint, Npp32s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp32s, Npp32s, 3, false>(const Npp32s *, int, NppiSize, NppiPoint, Npp32s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp32s, Npp32s, 4, false>(const Npp32s *, int, NppiSize, NppiPoint, Npp32s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp32s, Npp32s, 4, true>(const Npp32s *, int, NppiSize, NppiPoint, Npp32s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8u, Npp16s, 1, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8u, Npp16s, 3, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8u, Npp16s, 4, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8u, Npp16s, 4, true>(const Npp8u *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8s, Npp16s, 1, false>(const Npp8s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8s, Npp16s, 3, false>(const Npp8s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8s, Npp16s, 4, false>(const Npp8s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp8s, Npp16s, 4, true>(const Npp8s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp32f, Npp32f, 1, false>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp32f, Npp32f, 2, false>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp32f, Npp32f, 3, false>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp32f, Npp32f, 4, false>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp32f, Npp32f, 4, true>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);

// filterBorderDiv
template NppStatus filterBorderDiv<Npp8u, Npp8u, 1, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterBorderDiv<Npp8u, Npp8u, 3, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterBorderDiv<Npp8u, Npp8u, 4, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterBorderDiv<Npp8u, Npp8u, 4, true>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterBorderDiv<Npp16u, Npp16u, 1, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterBorderDiv<Npp16u, Npp16u, 3, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterBorderDiv<Npp16u, Npp16u, 4, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterBorderDiv<Npp16u, Npp16u, 4, true>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterBorderDiv<Npp16s, Npp16s, 1, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterBorderDiv<Npp16s, Npp16s, 3, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterBorderDiv<Npp16s, Npp16s, 4, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterBorderDiv<Npp16s, Npp16s, 4, true>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, NppiBorderType, cudaStream_t);

// filterRow32f
template NppStatus filterRow32f<Npp8u, Npp8u, 1, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp8u, Npp8u, 3, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp8u, Npp8u, 4, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp8u, Npp8u, 4, true>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp16u, Npp16u, 1, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp16u, Npp16u, 3, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp16u, Npp16u, 4, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp16u, Npp16u, 4, true>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp16s, Npp16s, 1, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp16s, Npp16s, 3, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp16s, Npp16s, 4, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp16s, Npp16s, 4, true>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);

// filterColumn32f
template NppStatus filterColumn32f<Npp8u, Npp8u, 1, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp8u, Npp8u, 3, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp8u, Npp8u, 4, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp8u, Npp8u, 4, true>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp16u, Npp16u, 1, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp16u, Npp16u, 3, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp16u, Npp16u, 4, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp16u, Npp16u, 4, true>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp16s, Npp16s, 1, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp16s, Npp16s, 3, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp16s, Npp16s, 4, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp16s, Npp16s, 4, true>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);

// filterRowBorder32f
template NppStatus filterRowBorder32f<Npp8u, Npp8u, 1, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp8u, Npp8u, 3, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp8u, Npp8u, 4, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp8u, Npp8u, 4, true>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp16u, Npp16u, 1, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp16u, Npp16u, 3, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp16u, Npp16u, 4, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp16u, Npp16u, 4, true>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp16s, Npp16s, 1, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp16s, Npp16s, 3, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp16s, Npp16s, 4, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp16s, Npp16s, 4, true>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);

// filterColumnBorder32f
template NppStatus filterColumnBorder32f<Npp8u, Npp8u, 1, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp8u, Npp8u, 3, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp8u, Npp8u, 4, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp8u, Npp8u, 4, true>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp16u, Npp16u, 1, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp16u, Npp16u, 3, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp16u, Npp16u, 4, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp16u, Npp16u, 4, true>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp16s, Npp16s, 1, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp16s, Npp16s, 3, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp16s, Npp16s, 4, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp16s, Npp16s, 4, true>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);

}  // namespace filter
}  // namespace nppi
