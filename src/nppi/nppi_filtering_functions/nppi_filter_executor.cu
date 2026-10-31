// nppiFilter32f / nppiFilterBorder(32f) / nppiFilterRow(32f) / nppiFilterColumn(32f)
// kernels and executor entry points.  See nppi_filter_executor.h for the family
// contract (semantics measured against NVIDIA NPP 13.4 on an RTX 5090).
#include "npp.h"
#include "nppi_filter_executor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// NPP's empty/negative-kernel and zero-mask behaviour (5090): the destination
// is zero-filled over [0, limitX) x [0, limitY) with the per-axis limit
//   limit(dim, k) = clamp(dim - 1 + k, 0, dim)   (k = kernel dim / mask size)
// i.e. a kernel of size 0 leaves the last line untouched, a size of 1 or more
// fills the whole axis, and a negative size shrinks the filled rectangle.
// AC4 filters zero only the RGB channels and preserve the alpha channel.
template <typename Dst, int CH, bool AC4 = false>
__global__ void zeroFillKernel(Dst *pDst, int nDstStep, int limitX, int limitY) {
  constexpr int kNch = AC4 ? (CH < 3 ? CH : 3) : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= limitX || y >= limitY) return;
  char *row = (char *)pDst + (long long)y * nDstStep;
  Dst *pix = reinterpret_cast<Dst *>(row);
  const Dst zero{};
  for (int c = 0; c < kNch; ++c) pix[x * CH + c] = zero;
}

template <typename Dst, int CH, bool AC4 = false>
inline NppStatus launchZeroFill(Dst *pDst, int nDstStep, int limitX, int limitY, cudaStream_t stream) {
  if (limitX <= 0 || limitY <= 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((limitX + kBlockX - 1) / kBlockX, (limitY + kBlockY - 1) / kBlockY);
  zeroFillKernel<Dst, CH, AC4><<<grid, block, 0, stream>>>(pDst, nDstStep, limitX, limitY);
  return finish_launch();
}

// Filled extent along one axis for a degenerate kernel/mask of size k.
inline int fill_limit(int dim, int k) {
  const int lim = dim - 1 + k;
  if (lim <= 0) return 0;
  return lim > dim ? dim : lim;
}

// NPP takes kernel/mask arrays as HOST pointers and stages them internally.
// cudaMemcpyDefault lets the same path also accept a device pointer (the
// differential harness passes device-side kernel buffers).
template <typename K>
inline NppStatus stage_kernel(const K *pKernel, int count, cudaStream_t stream, K **out) {
  K *d = nullptr;
  if (cudaMallocAsync(reinterpret_cast<void **>(&d), (size_t)count * sizeof(K), stream) != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  if (cudaMemcpyAsync(d, pKernel, (size_t)count * sizeof(K), cudaMemcpyDefault, stream) != cudaSuccess) {
    cudaFreeAsync(d, stream);
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  *out = d;
  return NPP_NO_ERROR;
}

// Launch helper: run the kernel with the staged copy, then free it on the same
// stream (ordered after the kernel).
template <typename K, typename F>
inline NppStatus with_staged_kernel(const K *pKernel, int count, cudaStream_t stream, F &&launch) {
  K *d = nullptr;
  NppStatus st = stage_kernel(pKernel, count, stream, &d);
  if (st != NPP_NO_ERROR) return st;
  launch(d);
  cudaError_t e = cudaGetLastError();
  cudaFreeAsync(d, stream);
  return e == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Source sample as float (all supported source types are arithmetic).
template <typename T>
__host__ __device__ __forceinline__ float to_float(const T &s) {
  return static_cast<float>(s);
}

// 16f samples are converted through float (NPP accepts 16f filters on sm_120).
__host__ __device__ __forceinline__ float to_float(const Npp16f &s) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(s.fp16)));
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

template <>
__host__ __device__ __forceinline__ Npp16f emit_dst<Npp16f>(float acc) {
  Npp16f out;
  out.fp16 = static_cast<short>(__half_as_ushort(__float2half(acc)));
  return out;
}

// NPP's 8u C4/AC4 1D filter path (row/column, border and non-border) does its
// accumulation in unsigned 32-bit ring arithmetic and divides unsigned; a
// negative signed sum therefore wraps high and saturates to 255, not 0.
// Measured on the 5090 (1 * -1 / (2^31-1) -> 2; 8 * 2^29 -> 0, i.e. wrap).
__host__ __device__ __forceinline__ Npp8u emit_u8_unsigned(unsigned acc, int nDivisor) {
  const unsigned q = acc / (unsigned)nDivisor;
  return static_cast<Npp8u>(q > 255u ? 255u : q);
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
// 2D filter, integer kernel + divisor, no border.  Reads are raw (taps may
// leave the ROI); the arithmetic is the same ring as the border variant.
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter2d_div_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst,
                                    int nDstStep, int W, int H, const Npp32s *__restrict__ pKernel,
                                    int kw, int kh, int ax, int ay, int nDivisor) {
  constexpr int kNch = AC4 ? 3 : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  for (int c = 0; c < kNch; ++c) {
    long long acc = 0;
    for (int j = 0; j < kh; ++j) {
      const char *srow = (const char *)pSrc + (long long)(y + ay - j) * nSrcStep;
      const Npp32s *krow = pKernel + (long long)j * kw;
      for (int i = 0; i < kw; ++i) {
        const Src *sp = (const Src *)(srow + ((long long)(x + ax - i) * CH + c) * (long long)sizeof(Src));
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
// 2D filter, 64f kernel, no border.  Accumulates in double; only Npp64f
// destinations are instantiated.
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter2d_64f_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst,
                                    int nDstStep, int W, int H, const Npp64f *__restrict__ pKernel,
                                    int kw, int kh, int ax, int ay) {
  constexpr int kNch = AC4 ? 3 : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  for (int c = 0; c < kNch; ++c) {
    double acc = 0.0;
    for (int j = 0; j < kh; ++j) {
      const char *srow = (const char *)pSrc + (long long)(y + ay - j) * nSrcStep;
      const Npp64f *krow = pKernel + (long long)j * kw;
      for (int i = 0; i < kw; ++i) {
        const Src *sp = (const Src *)(srow + ((long long)(x + ax - i) * CH + c) * (long long)sizeof(Src));
        acc += (double)to_float(*sp) * (double)krow[i];
      }
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = static_cast<Dst>(acc);
  }
}

// ---------------------------------------------------------------------------
// 32f C1R with an odd square kernel: NPP uses a specialised path that clamps
// every tap into the ROI (replicate inside the ROI), unlike every other kernel
// shape/layout, which reads raw.  Measured on the 5090 with shape probes.
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH>
__global__ void filter2d_clamp32f_kernel(const Src *__restrict__ pSrc, int nSrcStep,
                                         Dst *__restrict__ pDst, int nDstStep, int W, int H,
                                         const Npp32f *__restrict__ pKernel, int kw, int kh,
                                         int ax, int ay) {
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  for (int c = 0; c < CH; ++c) {
    float acc = 0.0f;
    for (int j = 0; j < kh; ++j) {
      const int sy = clampi(y + ay - j, 0, H - 1);
      const Npp32f *krow = pKernel + (long long)j * kw;
      for (int i = 0; i < kw; ++i) {
        const int sx = clampi(x + ax - i, 0, W - 1);
        const Src *sp = (const Src *)((const char *)pSrc + (long long)sy * nSrcStep +
                                      ((long long)sx * CH + c) * (long long)sizeof(Src));
        acc += to_float(*sp) * krow[i];
      }
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = emit_dst<Dst>(acc);
  }
}

// ---------------------------------------------------------------------------
// 32f C1R row/column with an odd mask >= 3: like the 2D C1R square kernels,
// NPP clamps the taps into the ROI along the filter axis.  Measured on the
// 5090 (the C3/C4/AC4 and even-mask families read raw).
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH>
__global__ void filter_row_clamp32f_kernel(const Src *__restrict__ pSrc, int nSrcStep,
                                           Dst *__restrict__ pDst, int nDstStep, int W, int H,
                                           const Npp32f *__restrict__ pKernel, int mask, int ax) {
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *srow = (const char *)pSrc + (long long)y * nSrcStep;
  for (int c = 0; c < CH; ++c) {
    float acc = 0.0f;
    for (int i = 0; i < mask; ++i) {
      const int ix = clampi(x + ax - i, 0, W - 1);
      const Src *sp = (const Src *)(srow + ((long long)ix * CH + c) * (long long)sizeof(Src));
      acc += to_float(*sp) * pKernel[i];
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = emit_dst<Dst>(acc);
  }
}

template <typename Src, typename Dst, int CH>
__global__ void filter_col_clamp32f_kernel(const Src *__restrict__ pSrc, int nSrcStep,
                                           Dst *__restrict__ pDst, int nDstStep, int W, int H,
                                           const Npp32f *__restrict__ pKernel, int mask, int ay) {
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  for (int c = 0; c < CH; ++c) {
    float acc = 0.0f;
    for (int i = 0; i < mask; ++i) {
      const int iy = clampi(y + ay - i, 0, H - 1);
      const char *srow = (const char *)pSrc + (long long)iy * nSrcStep;
      const Src *sp = (const Src *)(srow + ((long long)x * CH + c) * (long long)sizeof(Src));
      acc += to_float(*sp) * pKernel[i];
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = emit_dst<Dst>(acc);
  }
}

// ---------------------------------------------------------------------------
// 64f row/column with nMaskSize <= 0: NPP performs a single tap at the anchor
// with k[0] (the signed anchor is used as-is).  Measured on the 5090.
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH>
__global__ void filter_row_1tap_64f_kernel(const Src *__restrict__ pSrc, int nSrcStep,
                                           Dst *__restrict__ pDst, int nDstStep, int W, int H,
                                           const Npp64f *__restrict__ pKernel, int ax) {
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *srow = (const char *)pSrc + (long long)y * nSrcStep;
  for (int c = 0; c < CH; ++c) {
    const Src *sp = (const Src *)(srow + ((long long)(x + ax) * CH + c) * (long long)sizeof(Src));
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = static_cast<Dst>((double)to_float(*sp) * pKernel[0]);
  }
}

template <typename Src, typename Dst, int CH>
__global__ void filter_col_1tap_64f_kernel(const Src *__restrict__ pSrc, int nSrcStep,
                                           Dst *__restrict__ pDst, int nDstStep, int W, int H,
                                           const Npp64f *__restrict__ pKernel, int ay) {
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *srow = (const char *)pSrc + (long long)(y + ay) * nSrcStep;
  for (int c = 0; c < CH; ++c) {
    const Src *sp = (const Src *)(srow + ((long long)x * CH + c) * (long long)sizeof(Src));
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = static_cast<Dst>((double)to_float(*sp) * pKernel[0]);
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
// Separable row/column filters, integer kernel + divisor (no border; raw
// reads).  Same ring arithmetic as the 2D div kernel.
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter_row_div_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst,
                                      int nDstStep, int W, int H, const Npp32s *__restrict__ pKernel,
                                      int mask, int ax, int nDivisor) {
  constexpr int kNch = AC4 ? 3 : CH;
  // NPP's 8u C4/AC4 1D path uses unsigned ring arithmetic (see emit_u8_unsigned).
  constexpr bool kUnsignedU8x4 = std::is_same_v<Src, Npp8u> && std::is_same_v<Dst, Npp8u> && CH == 4;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *srow = (const char *)pSrc + (long long)y * nSrcStep;
  for (int c = 0; c < kNch; ++c) {
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    if constexpr (kUnsignedU8x4) {
      unsigned uacc = 0;
      for (int i = 0; i < mask; ++i) {
        const Src *sp = (const Src *)(srow + ((long long)(x + ax - i) * CH + c) * (long long)sizeof(Src));
        uacc += (unsigned)((unsigned long long)((long long)pKernel[i] * (long long)(int)*sp) & 0xffffffffull);
      }
      *reinterpret_cast<Dst *>(dp) = emit_u8_unsigned(uacc, nDivisor);
      continue;
    }
    long long acc = 0;
    for (int i = 0; i < mask; ++i) {
      const Src *sp = (const Src *)(srow + ((long long)(x + ax - i) * CH + c) * (long long)sizeof(Src));
      acc += (long long)pKernel[i] * (long long)(int)*sp;
    }
    const int wrapped = (int)(unsigned int)((unsigned long long)acc & 0xffffffffull);
    int q;
    if (wrapped == INT32_MIN && nDivisor == -1) {
      q = INT32_MIN;
    } else {
      q = wrapped / nDivisor;
    }
    *reinterpret_cast<Dst *>(dp) = sat_i32<Dst>(q);
  }
}

template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter_col_div_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst,
                                      int nDstStep, int W, int H, const Npp32s *__restrict__ pKernel,
                                      int mask, int ay, int nDivisor) {
  constexpr int kNch = AC4 ? 3 : CH;
  constexpr bool kUnsignedU8x4 = std::is_same_v<Src, Npp8u> && std::is_same_v<Dst, Npp8u> && CH == 4;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  for (int c = 0; c < kNch; ++c) {
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    if constexpr (kUnsignedU8x4) {
      unsigned uacc = 0;
      for (int i = 0; i < mask; ++i) {
        const char *srow = (const char *)pSrc + (long long)(y + ay - i) * nSrcStep;
        const Src *sp = (const Src *)(srow + ((long long)x * CH + c) * (long long)sizeof(Src));
        uacc += (unsigned)((unsigned long long)((long long)pKernel[i] * (long long)(int)*sp) & 0xffffffffull);
      }
      *reinterpret_cast<Dst *>(dp) = emit_u8_unsigned(uacc, nDivisor);
      continue;
    }
    long long acc = 0;
    for (int i = 0; i < mask; ++i) {
      const char *srow = (const char *)pSrc + (long long)(y + ay - i) * nSrcStep;
      const Src *sp = (const Src *)(srow + ((long long)x * CH + c) * (long long)sizeof(Src));
      acc += (long long)pKernel[i] * (long long)(int)*sp;
    }
    const int wrapped = (int)(unsigned int)((unsigned long long)acc & 0xffffffffull);
    int q;
    if (wrapped == INT32_MIN && nDivisor == -1) {
      q = INT32_MIN;
    } else {
      q = wrapped / nDivisor;
    }
    *reinterpret_cast<Dst *>(dp) = sat_i32<Dst>(q);
  }
}

// ---------------------------------------------------------------------------
// Separable row/column filters, 64f kernel (no border; raw reads).  The
// accumulation is done in double; only Npp64f destinations are instantiated.
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter_row_64f_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst,
                                      int nDstStep, int W, int H, const Npp64f *__restrict__ pKernel,
                                      int mask, int ax) {
  constexpr int kNch = AC4 ? 3 : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *srow = (const char *)pSrc + (long long)y * nSrcStep;
  for (int c = 0; c < kNch; ++c) {
    double acc = 0.0;
    for (int i = 0; i < mask; ++i) {
      const Src *sp = (const Src *)(srow + ((long long)(x + ax - i) * CH + c) * (long long)sizeof(Src));
      acc += (double)to_float(*sp) * (double)pKernel[i];
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = static_cast<Dst>(acc);
  }
}

template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter_col_64f_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst,
                                      int nDstStep, int W, int H, const Npp64f *__restrict__ pKernel,
                                      int mask, int ay) {
  constexpr int kNch = AC4 ? 3 : CH;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  for (int c = 0; c < kNch; ++c) {
    double acc = 0.0;
    for (int i = 0; i < mask; ++i) {
      const char *srow = (const char *)pSrc + (long long)(y + ay - i) * nSrcStep;
      const Src *sp = (const Src *)(srow + ((long long)x * CH + c) * (long long)sizeof(Src));
      acc += (double)to_float(*sp) * (double)pKernel[i];
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    *reinterpret_cast<Dst *>(dp) = static_cast<Dst>(acc);
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

// ---------------------------------------------------------------------------
// Separable row/column filters with replicate border, integer kernel +
// divisor.  Same ring arithmetic as the 2D border div kernel.
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter_row_border_div_kernel(const Src *__restrict__ pSrc, int nSrcStep, int sw,
                                             int sh, int offx, int offy, Dst *__restrict__ pDst,
                                             int nDstStep, int W, int H,
                                             const Npp32s *__restrict__ pKernel, int mask, int ax,
                                             int nDivisor) {
  constexpr int kNch = AC4 ? 3 : CH;
  constexpr bool kUnsignedU8x4 = std::is_same_v<Src, Npp8u> && std::is_same_v<Dst, Npp8u> && CH == 4;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *origin =
      (const char *)pSrc - (long long)offy * nSrcStep - (long long)offx * CH * (long long)sizeof(Src);
  const int iy = clampi(y + offy, 0, sh - 1);
  const char *srow = origin + (long long)iy * nSrcStep;
  for (int c = 0; c < kNch; ++c) {
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    if constexpr (kUnsignedU8x4) {
      unsigned uacc = 0;
      for (int i = 0; i < mask; ++i) {
        const int ix = clampi(x + offx + ax - i, 0, sw - 1);
        const Src *sp = (const Src *)(srow + ((long long)ix * CH + c) * (long long)sizeof(Src));
        uacc += (unsigned)((unsigned long long)((long long)pKernel[i] * (long long)(int)*sp) & 0xffffffffull);
      }
      *reinterpret_cast<Dst *>(dp) = emit_u8_unsigned(uacc, nDivisor);
      continue;
    }
    long long acc = 0;
    for (int i = 0; i < mask; ++i) {
      const int ix = clampi(x + offx + ax - i, 0, sw - 1);
      const Src *sp = (const Src *)(srow + ((long long)ix * CH + c) * (long long)sizeof(Src));
      acc += (long long)pKernel[i] * (long long)(int)*sp;
    }
    const int wrapped = (int)(unsigned int)((unsigned long long)acc & 0xffffffffull);
    int q;
    if (wrapped == INT32_MIN && nDivisor == -1) {
      q = INT32_MIN;
    } else {
      q = wrapped / nDivisor;
    }
    *reinterpret_cast<Dst *>(dp) = sat_i32<Dst>(q);
  }
}

template <typename Src, typename Dst, int CH, bool AC4>
__global__ void filter_col_border_div_kernel(const Src *__restrict__ pSrc, int nSrcStep, int sw,
                                             int sh, int offx, int offy, Dst *__restrict__ pDst,
                                             int nDstStep, int W, int H,
                                             const Npp32s *__restrict__ pKernel, int mask, int ay,
                                             int nDivisor) {
  constexpr int kNch = AC4 ? 3 : CH;
  constexpr bool kUnsignedU8x4 = std::is_same_v<Src, Npp8u> && std::is_same_v<Dst, Npp8u> && CH == 4;
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *origin =
      (const char *)pSrc - (long long)offy * nSrcStep - (long long)offx * CH * (long long)sizeof(Src);
  const int ix = clampi(x + offx, 0, sw - 1);
  for (int c = 0; c < kNch; ++c) {
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Dst);
    if constexpr (kUnsignedU8x4) {
      unsigned uacc = 0;
      for (int i = 0; i < mask; ++i) {
        const int iy = clampi(y + offy + ay - i, 0, sh - 1);
        const char *srow = origin + (long long)iy * nSrcStep;
        const Src *sp = (const Src *)(srow + ((long long)ix * CH + c) * (long long)sizeof(Src));
        uacc += (unsigned)((unsigned long long)((long long)pKernel[i] * (long long)(int)*sp) & 0xffffffffull);
      }
      *reinterpret_cast<Dst *>(dp) = emit_u8_unsigned(uacc, nDivisor);
      continue;
    }
    long long acc = 0;
    for (int i = 0; i < mask; ++i) {
      const int iy = clampi(y + offy + ay - i, 0, sh - 1);
      const char *srow = origin + (long long)iy * nSrcStep;
      const Src *sp = (const Src *)(srow + ((long long)ix * CH + c) * (long long)sizeof(Src));
      acc += (long long)pKernel[i] * (long long)(int)*sp;
    }
    const int wrapped = (int)(unsigned int)((unsigned long long)acc & 0xffffffffull);
    int q;
    if (wrapped == INT32_MIN && nDivisor == -1) {
      q = INT32_MIN;
    } else {
      q = wrapped / nDivisor;
    }
    *reinterpret_cast<Dst *>(dp) = sat_i32<Dst>(q);
  }
}

// ---------------------------------------------------------------------------
// Running-window sums (unit weights) into Npp32f.  The window is the same tap
// range as the row/column filters ([x+ax-mask+1, x+ax]); the non-border
// variants skip taps outside the ROI (clipped window sum).
// ---------------------------------------------------------------------------
template <typename Src, int CH>
__global__ void sum_window_row_kernel(const Src *__restrict__ pSrc, int nSrcStep,
                                      Npp32f *__restrict__ pDst, int nDstStep, int W, int H,
                                      int mask, int ax) {
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *srow = (const char *)pSrc + (long long)y * nSrcStep;
  for (int c = 0; c < CH; ++c) {
    float acc = 0.0f;
    // Non-border sum-window reads raw source taps (no clipping/replication);
    // the differential buffers carry margins for the window overhang.  The
    // rightmost tap coordinate is clamped to W + mask - 1 (measured on the
    // 5090 with impulse probes; the column kernel has no such clamp).
    const int kmax = W + mask - 1;
    for (int i = 0; i < mask; ++i) {
      int k = x - ax + i;  // NPP window: [x-ax, x-ax+mask-1]
      if (k > kmax) k = kmax;
      const Src *sp = (const Src *)(srow + ((long long)k * CH + c) * (long long)sizeof(Src));
      acc += to_float(*sp);
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Npp32f);
    *reinterpret_cast<Npp32f *>(dp) = acc;
  }
}

template <typename Src, int CH>
__global__ void sum_window_col_kernel(const Src *__restrict__ pSrc, int nSrcStep,
                                      Npp32f *__restrict__ pDst, int nDstStep, int W, int H,
                                      int mask, int ay) {
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  for (int c = 0; c < CH; ++c) {
    float acc = 0.0f;
    for (int i = 0; i < mask; ++i) {
      const int k = y - ay + i;  // NPP window: [y-ay, y-ay+mask-1]
      const char *srow = (const char *)pSrc + (long long)k * nSrcStep;
      const Src *sp = (const Src *)(srow + ((long long)x * CH + c) * (long long)sizeof(Src));
      acc += to_float(*sp);
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Npp32f);
    *reinterpret_cast<Npp32f *>(dp) = acc;
  }
}

// ---------------------------------------------------------------------------
// Running-window sums with replicate border: every tap coordinate is clamped
// into [0, srcSize-1]; the non-filtered axis is replicate-clamped as well.
// ---------------------------------------------------------------------------
template <typename Src, int CH>
__global__ void sum_window_row_border_kernel(const Src *__restrict__ pSrc, int nSrcStep, int sw,
                                             int sh, int offx, int offy, Npp32f *__restrict__ pDst,
                                             int nDstStep, int W, int H, int mask, int ax) {
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *origin =
      (const char *)pSrc - (long long)offy * nSrcStep - (long long)offx * CH * (long long)sizeof(Src);
  const int iy = clampi(y + offy, 0, sh - 1);
  const char *srow = origin + (long long)iy * nSrcStep;
  for (int c = 0; c < CH; ++c) {
    float acc = 0.0f;
    for (int i = 0; i < mask; ++i) {
      const int ix = clampi(x + offx - ax + i, 0, sw - 1);
      const Src *sp = (const Src *)(srow + ((long long)ix * CH + c) * (long long)sizeof(Src));
      acc += to_float(*sp);
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Npp32f);
    *reinterpret_cast<Npp32f *>(dp) = acc;
  }
}

template <typename Src, int CH>
__global__ void sum_window_col_border_kernel(const Src *__restrict__ pSrc, int nSrcStep, int sw,
                                             int sh, int offx, int offy, Npp32f *__restrict__ pDst,
                                             int nDstStep, int W, int H, int mask, int ay) {
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= W || y >= H) return;
  const char *origin =
      (const char *)pSrc - (long long)offy * nSrcStep - (long long)offx * CH * (long long)sizeof(Src);
  const int ix = clampi(x + offx, 0, sw - 1);
  for (int c = 0; c < CH; ++c) {
    float acc = 0.0f;
    for (int i = 0; i < mask; ++i) {
      const int iy = clampi(y + offy - ay + i, 0, sh - 1);
      const char *srow = origin + (long long)iy * nSrcStep;
      const Src *sp = (const Src *)(srow + ((long long)ix * CH + c) * (long long)sizeof(Src));
      acc += to_float(*sp);
    }
    char *dp = (char *)pDst + (long long)y * nDstStep + ((long long)x * CH + c) * (long long)sizeof(Npp32f);
    *reinterpret_cast<Npp32f *>(dp) = acc;
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
  if (kw <= 0 || kh <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, fill_limit(oSizeROI.width, kw),
                                        fill_limit(oSizeROI.height, kh), stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  // NPP 13.4: the 32f C1R family clamps taps into the ROI for odd square
  // kernels of size >= 3 (all other shapes/layouts read raw).
  if constexpr (std::is_same_v<Src, Npp32f> && std::is_same_v<Dst, Npp32f> && CH == 1 && !AC4) {
    if (kw == kh && kw >= 3 && (kw & 1) != 0) {
      return with_staged_kernel(pKernel, kw * kh, stream, [&](const Npp32f *pk) {
        filter2d_clamp32f_kernel<Src, Dst, CH><<<grid, block, 0, stream>>>(
            pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pk, kw, kh,
            oAnchor.x, oAnchor.y);
      });
    }
  }
  return with_staged_kernel(pKernel, kw * kh, stream, [&](const Npp32f *pk) {
    filter2d_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pk, kw, kh,
        oAnchor.x, oAnchor.y);
  });
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
  if (oKernelSize.width <= 0 || oKernelSize.height <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, fill_limit(oSizeROI.width, oKernelSize.width),
                                        fill_limit(oSizeROI.height, oKernelSize.height), stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  return with_staged_kernel(pKernel, oKernelSize.width * oKernelSize.height, stream,
                            [&](const Npp32f *pk) {
    filter2d_border32f_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, pk, oKernelSize.width, oKernelSize.height, oAnchor.x,
        oAnchor.y);
  });
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
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (nDivisor == 0) return NPP_DIVISOR_ERROR;
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (oKernelSize.width <= 0 || oKernelSize.height <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, fill_limit(oSizeROI.width, oKernelSize.width),
                                        fill_limit(oSizeROI.height, oKernelSize.height), stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  return with_staged_kernel(pKernel, oKernelSize.width * oKernelSize.height, stream,
                            [&](const Npp32s *pk) {
    filter2d_border_div_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, pk, oKernelSize.width, oKernelSize.height, oAnchor.x,
        oAnchor.y, nDivisor);
  });
}

// ---------------------------------------------------------------------------
// nppiFilter_<dt>_<lay>R (integer kernel + divisor)
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterDiv(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp32s *pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor,
                    cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int kw = oKernelSize.width;
  const int kh = oKernelSize.height;
  if (oAnchor.x < 1 - oSizeROI.width || oAnchor.x > kw || oAnchor.y < 1 - oSizeROI.height ||
      oAnchor.y > kh) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  // 5090: step size and the empty-ROI no-op are checked before the divisor.
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (nDivisor == 0) return NPP_DIVISOR_ERROR;
  if (kw <= 0 || kh <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, fill_limit(oSizeROI.width, kw),
                                        fill_limit(oSizeROI.height, kh), stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  return with_staged_kernel(pKernel, kw * kh, stream, [&](const Npp32s *pk) {
    filter2d_div_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pk, kw, kh,
        oAnchor.x, oAnchor.y, nDivisor);
  });
}

// ---------------------------------------------------------------------------
// nppiFilter_64f_C1R
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filter64f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                    const Npp64f *pKernel, NppiSize oKernelSize, NppiPoint oAnchor,
                    cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  const int kw = oKernelSize.width;
  const int kh = oKernelSize.height;
  // 5090: nppiFilter_64f_C1R does NOT validate the anchor (anchor {4,1} with a
  // 3x3 kernel is NO_ERROR); the integer/32f variants keep the check.
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (kw <= 0 || kh <= 0) {
    // 64f zeroes the whole ROI (unlike the integer/32f families, which leave
    // the last line untouched for a zero kernel dimension).
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, oSizeROI.width, oSizeROI.height, stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  return with_staged_kernel(pKernel, kw * kh, stream, [&](const Npp64f *pk) {
    filter2d_64f_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pk, kw, kh,
        oAnchor.x, oAnchor.y);
  });
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
  if (nMaskSize <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, fill_limit(oROI.width, nMaskSize),
                                        oROI.height, stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  if constexpr (std::is_same_v<Src, Npp32f> && std::is_same_v<Dst, Npp32f> && CH == 1 && !AC4) {
    if (nMaskSize >= 3 && (nMaskSize & 1) != 0) {
      return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp32f *pk) {
        filter_row_clamp32f_kernel<Src, Dst, CH><<<grid, block, 0, stream>>>(
            pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pk, nMaskSize, nAnchor);
      });
    }
  }
  return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp32f *pk) {
    filter_row_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pk, nMaskSize, nAnchor);
  });
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
  if (nMaskSize <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, oROI.width,
                                        fill_limit(oROI.height, nMaskSize), stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  if constexpr (std::is_same_v<Src, Npp32f> && std::is_same_v<Dst, Npp32f> && CH == 1 && !AC4) {
    if (nMaskSize >= 3 && (nMaskSize & 1) != 0) {
      return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp32f *pk) {
        filter_col_clamp32f_kernel<Src, Dst, CH><<<grid, block, 0, stream>>>(
            pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pk, nMaskSize, nAnchor);
      });
    }
  }
  return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp32f *pk) {
    filter_col_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pk, nMaskSize, nAnchor);
  });
}

// ---------------------------------------------------------------------------
// nppiFilterRow_<dt>_<lay>R / nppiFilterColumn_<dt>_<lay>R (integer kernel +
// divisor)
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterRowDiv(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                       const Npp32s *pKernel, int nMaskSize, int nAnchor, Npp32s nDivisor,
                       cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oROI.width < 0 || oROI.height < 0) return NPP_SIZE_ERROR;
  // 5090: the filter-axis empty ROI is the no-op and is checked before the
  // anchor; a zero cross-axis is a SIZE error.  The anchor is validated against
  // [1 - roiDim, mask] (inclusive), before the step and divisor checks.
  if (oROI.width == 0) return NPP_NO_ERROR;
  if (oROI.height == 0) return NPP_SIZE_ERROR;  // cross axis: SIZE, not a no-op
  if (nAnchor < 1 - oROI.width || nAnchor > nMaskSize) return NPP_OUT_OFF_RANGE_ERROR;
  const int rowBytes = oROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oROI.height)) return NPP_STEP_ERROR;
  if (nDivisor == 0) return NPP_DIVISOR_ERROR;
  if (nMaskSize <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, fill_limit(oROI.width, nMaskSize),
                                        oROI.height, stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp32s *pk) {
    filter_row_div_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pk, nMaskSize, nAnchor,
        nDivisor);
  });
}

template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterColumnDiv(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                          const Npp32s *pKernel, int nMaskSize, int nAnchor, Npp32s nDivisor,
                          cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oROI.width < 0 || oROI.height < 0) return NPP_SIZE_ERROR;
  if (oROI.height == 0) return NPP_NO_ERROR;
  if (oROI.width == 0) return NPP_SIZE_ERROR;  // cross axis: SIZE
  if (nAnchor < 1 - oROI.height || nAnchor > nMaskSize) return NPP_OUT_OFF_RANGE_ERROR;
  const int rowBytes = oROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oROI.height)) return NPP_STEP_ERROR;
  if (nDivisor == 0) return NPP_DIVISOR_ERROR;
  if (nMaskSize <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, oROI.width,
                                        fill_limit(oROI.height, nMaskSize), stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp32s *pk) {
    filter_col_div_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pk, nMaskSize, nAnchor,
        nDivisor);
  });
}

// ---------------------------------------------------------------------------
// nppiFilterRow_64f_C1R / nppiFilterColumn_64f_C1R
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterRow64f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                       const Npp64f *pKernel, int nMaskSize, int nAnchor, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oROI.width < 0 || oROI.height < 0) return NPP_SIZE_ERROR;
  // 5090: nppiFilterRow/Column_64f_C1R never validate the anchor.  A mask of
  // zero or below is a single tap at the (signed) anchor with k[0]; a positive
  // mask is the usual window, with the anchor clamped into [0, mask-1] to keep
  // raw taps in bounds (NPP accepts out-of-range anchors here).
  const int rowBytes = oROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oROI.height)) return NPP_STEP_ERROR;
  if (oROI.width == 0 || oROI.height == 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  if (nMaskSize <= 0) {
    return with_staged_kernel(pKernel, 1, stream, [&](const Npp64f *pk) {
      filter_row_1tap_64f_kernel<Src, Dst, CH><<<grid, block, 0, stream>>>(
          pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pk, nAnchor);
    });
  }
  const int ax = nAnchor < 0 ? 0 : (nAnchor > nMaskSize - 1 ? nMaskSize - 1 : nAnchor);
  return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp64f *pk) {
    filter_row_64f_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pk, nMaskSize, ax);
  });
}

template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterColumn64f(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oROI,
                          const Npp64f *pKernel, int nMaskSize, int nAnchor, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oROI.width < 0 || oROI.height < 0) return NPP_SIZE_ERROR;
  const int rowBytes = oROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oROI.height)) return NPP_STEP_ERROR;
  if (oROI.width == 0 || oROI.height == 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  if (nMaskSize <= 0) {
    return with_staged_kernel(pKernel, 1, stream, [&](const Npp64f *pk) {
      filter_col_1tap_64f_kernel<Src, Dst, CH><<<grid, block, 0, stream>>>(
          pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pk, nAnchor);
    });
  }
  const int ay = nAnchor < 0 ? 0 : (nAnchor > nMaskSize - 1 ? nMaskSize - 1 : nAnchor);
  return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp64f *pk) {
    filter_col_64f_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, pk, nMaskSize, ay);
  });
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
  if (nMaskSize <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, fill_limit(oSizeROI.width, nMaskSize),
                                        oSizeROI.height, stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp32f *pk) {
    filter_row_border32f_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, pk, nMaskSize, nAnchor);
  });
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
  if (nMaskSize <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, oSizeROI.width,
                                        fill_limit(oSizeROI.height, nMaskSize), stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp32f *pk) {
    filter_col_border32f_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, pk, nMaskSize, nAnchor);
  });
}

// ---------------------------------------------------------------------------
// nppiFilterRowBorder_<dt>_<lay>R / nppiFilterColumnBorder_<dt>_<lay>R
// (integer kernel + divisor)
// ---------------------------------------------------------------------------
template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterRowBorderDiv(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                             Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32s *pKernel,
                             int nMaskSize, int nAnchor, Npp32s nDivisor,
                             NppiBorderType eBorderType, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  // 5090: the source-offset check runs before the border-mode check.
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (nDivisor == 0) return NPP_DIVISOR_ERROR;
  if (nMaskSize <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, fill_limit(oSizeROI.width, nMaskSize),
                                        oSizeROI.height, stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp32s *pk) {
    filter_row_border_div_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, pk, nMaskSize, nAnchor, nDivisor);
  });
}

template <typename Src, typename Dst, int CH, bool AC4>
NppStatus filterColumnBorderDiv(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                                Dst *pDst, int nDstStep, NppiSize oSizeROI, const Npp32s *pKernel,
                                int nMaskSize, int nAnchor, Npp32s nDivisor,
                                NppiBorderType eBorderType, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  const int rowBytes = oSizeROI.width * CH * (int)sizeof(Dst);
  if (step_too_small(nDstStep, rowBytes, oSizeROI.height)) return NPP_STEP_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (nDivisor == 0) return NPP_DIVISOR_ERROR;
  if (nMaskSize <= 0) {
    return launchZeroFill<Dst, CH, AC4>(pDst, nDstStep, oSizeROI.width,
                                        fill_limit(oSizeROI.height, nMaskSize), stream);
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX,
            (oSizeROI.height + kBlockY - 1) / kBlockY);
  return with_staged_kernel(pKernel, nMaskSize, stream, [&](const Npp32s *pk) {
    filter_col_border_div_kernel<Src, Dst, CH, AC4><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
        oSizeROI.width, oSizeROI.height, pk, nMaskSize, nAnchor, nDivisor);
  });
}

// ---------------------------------------------------------------------------
// nppiSumWindowRow_<dt>32f_<lay>R / nppiSumWindowColumn_<dt>32f_<lay>R
// ---------------------------------------------------------------------------
template <typename Src, int CH>
NppStatus sumWindowRow(const Src *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                       int nMaskSize, int nAnchor, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oROI.width < 0 || oROI.height < 0) return NPP_SIZE_ERROR;
  // 5090 sum-window contract: mask <= 0 -> MASK_SIZE; anchor >= mask ->
  // ANCHOR; anchor < -mask -> OUT_OFF_RANGE; a zero cross-axis is a SIZE
  // error while a zero filter-axis is the no-op.
  if (oROI.width == 0) return NPP_NO_ERROR;   // filter axis: empty no-op
  if (oROI.height == 0) return NPP_SIZE_ERROR;  // cross axis: SIZE
  if (nMaskSize <= 0) return NPP_MASK_SIZE_ERROR;
  if (nAnchor >= nMaskSize) return NPP_ANCHOR_ERROR;
  if (nAnchor < -nMaskSize) return NPP_OUT_OFF_RANGE_ERROR;
  const int rowBytes = oROI.width * CH * (int)sizeof(Npp32f);
  if (step_too_small(nDstStep, rowBytes, oROI.height)) return NPP_STEP_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  sum_window_row_kernel<Src, CH><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, nMaskSize, nAnchor);
  return finish_launch();
}

template <typename Src, int CH>
NppStatus sumWindowColumn(const Src *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                          int nMaskSize, int nAnchor, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oROI.width < 0 || oROI.height < 0) return NPP_SIZE_ERROR;
  if (oROI.height == 0) return NPP_NO_ERROR;   // filter axis: empty no-op
  if (oROI.width == 0) return NPP_SIZE_ERROR;   // cross axis: SIZE
  if (nMaskSize <= 0) return NPP_MASK_SIZE_ERROR;
  if (nAnchor >= nMaskSize) return NPP_ANCHOR_ERROR;
  if (nAnchor < -nMaskSize) return NPP_OUT_OFF_RANGE_ERROR;
  const int rowBytes = oROI.width * CH * (int)sizeof(Npp32f);
  if (step_too_small(nDstStep, rowBytes, oROI.height)) return NPP_STEP_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  sum_window_col_kernel<Src, CH><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, nMaskSize, nAnchor);
  return finish_launch();
}

// ---------------------------------------------------------------------------
// nppiSumWindowRowBorder_<dt>32f_<lay>R / nppiSumWindowColumnBorder_<dt>32f_<lay>R
// ---------------------------------------------------------------------------
template <typename Src, int CH>
NppStatus sumWindowRowBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                             Npp32f *pDst, int nDstStep, NppiSize oROI, int nMaskSize, int nAnchor,
                             NppiBorderType eBorderType, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oROI.width < 0 || oROI.height < 0) return NPP_SIZE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (nMaskSize <= 0) return NPP_MASK_SIZE_ERROR;
  if (nAnchor >= nMaskSize) return NPP_ANCHOR_ERROR;
  const int rowBytes = oROI.width * CH * (int)sizeof(Npp32f);
  if (step_too_small(nDstStep, rowBytes, oROI.height)) return NPP_STEP_ERROR;
  if (oROI.width == 0 || oROI.height == 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  sum_window_row_border_kernel<Src, CH><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
      oROI.width, oROI.height, nMaskSize, nAnchor);
  return finish_launch();
}

template <typename Src, int CH>
NppStatus sumWindowColumnBorder(const Src *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                                Npp32f *pDst, int nDstStep, NppiSize oROI, int nMaskSize, int nAnchor,
                                NppiBorderType eBorderType, cudaStream_t stream) {
  if (pSrc == nullptr || pDst == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oROI.width < 0 || oROI.height < 0) return NPP_SIZE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height) {
    return NPP_OUT_OFF_RANGE_ERROR;
  }
  if (nMaskSize <= 0) return NPP_MASK_SIZE_ERROR;
  if (nAnchor >= nMaskSize) return NPP_ANCHOR_ERROR;
  const int rowBytes = oROI.width * CH * (int)sizeof(Npp32f);
  if (step_too_small(nDstStep, rowBytes, oROI.height)) return NPP_STEP_ERROR;
  if (oROI.width == 0 || oROI.height == 0) return NPP_NO_ERROR;
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oROI.width + kBlockX - 1) / kBlockX, (oROI.height + kBlockY - 1) / kBlockY);
  sum_window_col_border_kernel<Src, CH><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, pDst, nDstStep,
      oROI.width, oROI.height, nMaskSize, nAnchor);
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


// f12 additions: filterDiv / filter64f / 32f-data filter32f
template NppStatus filterDiv<Npp8u, Npp8u, 1, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filterDiv<Npp8u, Npp8u, 3, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filterDiv<Npp8u, Npp8u, 4, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filterDiv<Npp8u, Npp8u, 4, true>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filterDiv<Npp16u, Npp16u, 1, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filterDiv<Npp16u, Npp16u, 3, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filterDiv<Npp16u, Npp16u, 4, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filterDiv<Npp16u, Npp16u, 4, true>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filterDiv<Npp16s, Npp16s, 1, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filterDiv<Npp16s, Npp16s, 3, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filterDiv<Npp16s, Npp16s, 4, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filterDiv<Npp16s, Npp16s, 4, true>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, NppiSize, NppiPoint, Npp32s, cudaStream_t);
template NppStatus filter64f<Npp64f, Npp64f, 1, false>(const Npp64f *, int, Npp64f *, int, NppiSize, const Npp64f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp32f, Npp32f, 1, false>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp32f, Npp32f, 2, false>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp32f, Npp32f, 3, false>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp32f, Npp32f, 4, false>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp32f, Npp32f, 4, true>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);

// f12 additions: filterRowDiv / filterColumnDiv / 32f-data row/column filters
template NppStatus filterRowDiv<Npp8u, Npp8u, 1, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRowDiv<Npp8u, Npp8u, 3, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRowDiv<Npp8u, Npp8u, 4, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRowDiv<Npp8u, Npp8u, 4, true>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRowDiv<Npp16u, Npp16u, 1, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRowDiv<Npp16u, Npp16u, 3, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRowDiv<Npp16u, Npp16u, 4, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRowDiv<Npp16u, Npp16u, 4, true>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRowDiv<Npp16s, Npp16s, 1, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRowDiv<Npp16s, Npp16s, 3, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRowDiv<Npp16s, Npp16s, 4, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRowDiv<Npp16s, Npp16s, 4, true>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp8u, Npp8u, 1, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp8u, Npp8u, 3, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp8u, Npp8u, 4, false>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp8u, Npp8u, 4, true>(const Npp8u *, int, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp16u, Npp16u, 1, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp16u, Npp16u, 3, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp16u, Npp16u, 4, false>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp16u, Npp16u, 4, true>(const Npp16u *, int, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp16s, Npp16s, 1, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp16s, Npp16s, 3, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp16s, Npp16s, 4, false>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterColumnDiv<Npp16s, Npp16s, 4, true>(const Npp16s *, int, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, cudaStream_t);
template NppStatus filterRow64f<Npp64f, Npp64f, 1, false>(const Npp64f *, int, Npp64f *, int, NppiSize, const Npp64f *, int, int, cudaStream_t);
template NppStatus filterColumn64f<Npp64f, Npp64f, 1, false>(const Npp64f *, int, Npp64f *, int, NppiSize, const Npp64f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp32f, Npp32f, 1, false>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp32f, Npp32f, 3, false>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp32f, Npp32f, 4, false>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterRow32f<Npp32f, Npp32f, 4, true>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp32f, Npp32f, 1, false>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp32f, Npp32f, 3, false>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp32f, Npp32f, 4, false>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);
template NppStatus filterColumn32f<Npp32f, Npp32f, 4, true>(const Npp32f *, int, Npp32f *, int, NppiSize, const Npp32f *, int, int, cudaStream_t);

// f12 additions: filterRowBorderDiv / filterColumnBorderDiv / 32f-data border
// row/column filters
template NppStatus filterRowBorderDiv<Npp8u, Npp8u, 1, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorderDiv<Npp8u, Npp8u, 3, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorderDiv<Npp8u, Npp8u, 4, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorderDiv<Npp8u, Npp8u, 4, true>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorderDiv<Npp16u, Npp16u, 1, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorderDiv<Npp16u, Npp16u, 3, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorderDiv<Npp16u, Npp16u, 4, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorderDiv<Npp16u, Npp16u, 4, true>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorderDiv<Npp16s, Npp16s, 1, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorderDiv<Npp16s, Npp16s, 3, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorderDiv<Npp16s, Npp16s, 4, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorderDiv<Npp16s, Npp16s, 4, true>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp8u, Npp8u, 1, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp8u, Npp8u, 3, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp8u, Npp8u, 4, false>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp8u, Npp8u, 4, true>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp16u, Npp16u, 1, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp16u, Npp16u, 3, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp16u, Npp16u, 4, false>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp16u, Npp16u, 4, true>(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp16s, Npp16s, 1, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp16s, Npp16s, 3, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp16s, Npp16s, 4, false>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorderDiv<Npp16s, Npp16s, 4, true>(const Npp16s *, int, NppiSize, NppiPoint, Npp16s *, int, NppiSize, const Npp32s *, int, int, Npp32s, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp32f, Npp32f, 1, false>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp32f, Npp32f, 3, false>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp32f, Npp32f, 4, false>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterRowBorder32f<Npp32f, Npp32f, 4, true>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp32f, Npp32f, 1, false>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp32f, Npp32f, 3, false>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp32f, Npp32f, 4, false>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);
template NppStatus filterColumnBorder32f<Npp32f, Npp32f, 4, true>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, const Npp32f *, int, int, NppiBorderType, cudaStream_t);

// f12 additions: sum-window row/column (border and non-border)
template NppStatus sumWindowRow<Npp8u, 1>(const Npp8u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowRow<Npp8u, 3>(const Npp8u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowRow<Npp8u, 4>(const Npp8u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowRow<Npp16u, 1>(const Npp16u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowRow<Npp16u, 3>(const Npp16u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowRow<Npp16u, 4>(const Npp16u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowRow<Npp16s, 1>(const Npp16s *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowRow<Npp16s, 3>(const Npp16s *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowRow<Npp16s, 4>(const Npp16s *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowColumn<Npp8u, 1>(const Npp8u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowColumn<Npp8u, 3>(const Npp8u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowColumn<Npp8u, 4>(const Npp8u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowColumn<Npp16u, 1>(const Npp16u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowColumn<Npp16u, 3>(const Npp16u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowColumn<Npp16u, 4>(const Npp16u *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowColumn<Npp16s, 1>(const Npp16s *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowColumn<Npp16s, 3>(const Npp16s *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowColumn<Npp16s, 4>(const Npp16s *, int, Npp32f *, int, NppiSize, int, int, cudaStream_t);
template NppStatus sumWindowRowBorder<Npp8u, 1>(const Npp8u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowRowBorder<Npp8u, 3>(const Npp8u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowRowBorder<Npp8u, 4>(const Npp8u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowRowBorder<Npp16u, 1>(const Npp16u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowRowBorder<Npp16u, 3>(const Npp16u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowRowBorder<Npp16u, 4>(const Npp16u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowRowBorder<Npp16s, 1>(const Npp16s *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowRowBorder<Npp16s, 3>(const Npp16s *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowRowBorder<Npp16s, 4>(const Npp16s *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowColumnBorder<Npp8u, 1>(const Npp8u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowColumnBorder<Npp8u, 3>(const Npp8u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowColumnBorder<Npp8u, 4>(const Npp8u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowColumnBorder<Npp16u, 1>(const Npp16u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowColumnBorder<Npp16u, 3>(const Npp16u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowColumnBorder<Npp16u, 4>(const Npp16u *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowColumnBorder<Npp16s, 1>(const Npp16s *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowColumnBorder<Npp16s, 3>(const Npp16s *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);
template NppStatus sumWindowColumnBorder<Npp16s, 4>(const Npp16s *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize, int, int, NppiBorderType, cudaStream_t);

// 16f variants (NPP accepts these on sm_120; work in float internally)
template NppStatus filter32f<Npp16f, Npp16f, 1, false>(const Npp16f *, int, Npp16f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp16f, Npp16f, 3, false>(const Npp16f *, int, Npp16f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filter32f<Npp16f, Npp16f, 4, false>(const Npp16f *, int, Npp16f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, cudaStream_t);
template NppStatus filterBorder32f<Npp16f, Npp16f, 1, false>(const Npp16f *, int, NppiSize, NppiPoint, Npp16f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp16f, Npp16f, 3, false>(const Npp16f *, int, NppiSize, NppiPoint, Npp16f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);
template NppStatus filterBorder32f<Npp16f, Npp16f, 4, false>(const Npp16f *, int, NppiSize, NppiPoint, Npp16f *, int, NppiSize, const Npp32f *, NppiSize, NppiPoint, NppiBorderType, cudaStream_t);

}  // namespace filter
}  // namespace nppi
