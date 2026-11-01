// Window (rank-statistics) filter kernels and executor entry points for
// nppiFilterBox / BoxBorder / Min / Max (+Border) / Median (+Border).
// See nppi_filter_window_executor.h for the measured family contract.
#include "npp.h"
#include "nppi_filter_window_executor.h"
#include "nppi_median_bufsize_table.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace nppi {
namespace filter {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

inline NppStatus finish_launch() {
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// Step checks shared by the standard families (measured): the unsigned
// comparison makes a negative dst step fail the even-step test instead of the
// size test; alignment uses the element size, except for 4-channel layouts
// where the whole 4-element pixel must be aligned.
template <typename T, int Ch>
inline int step_alignment() {
  return (int)sizeof(T) * (Ch == 4 ? 4 : 1);
}

template <typename T, int Ch>
inline NppStatus check_dst_step(int nDstStep, int width) {
  const Npp32u rowBytes = (Npp32u)width * (Npp32u)(sizeof(T) * Ch);
  if ((Npp32u)nDstStep < rowBytes) return NPP_STEP_ERROR;
  if (nDstStep % step_alignment<T, Ch>() != 0) return NPP_NOT_EVEN_STEP_ERROR;
  return NPP_NO_ERROR;
}

template <typename T>
__device__ __forceinline__ T clamp_int(long long v) {
  if constexpr (std::is_signed_v<T>) {
    const long long lo = -(1ll << (8 * sizeof(T) - 1));
    const long long hi = (1ll << (8 * sizeof(T) - 1)) - 1;
    if (v < lo) return (T)lo;
    if (v > hi) return (T)hi;
  } else {
    const unsigned long long hi = (1ull << (8 * sizeof(T))) - 1ull;
    if (v < 0) return (T)0;
    if ((unsigned long long)v > hi) return (T)hi;
  }
  return (T)v;
}

enum class WOp { Box, Min, Max, Median };

__device__ __forceinline__ int map_border(int v, int n, int mode) {
  if (n <= 0) return 0;
  if (mode == NPP_BORDER_REPLICATE) return v < 0 ? 0 : (v >= n ? n - 1 : v);
  if (n == 1) return 0;
  if (mode == NPP_BORDER_MIRROR) {  // 4: reflect without repeating the edge
    const int p = 2 * (n - 1);
    v %= p;
    if (v < 0) v += p;
    return v < n ? v : p - v;
  }
  // value 5: reflect repeating the edge (BoxBorder accepts it as well)
  const int p = 2 * n;
  v %= p;
  if (v < 0) v += p;
  return v < n ? v : p - 1 - v;
}

template <typename T>
__device__ __forceinline__ unsigned long long order_key(T v) {
  if constexpr (std::is_floating_point_v<T>) {
    const unsigned u = __float_as_uint((float)v);
    return (u & 0x80000000u) ? (unsigned long long)(~u)
                             : ((unsigned long long)u | 0x80000000ull);
  } else if constexpr (std::is_signed_v<T>) {
    using U = std::make_unsigned_t<T>;
    const U u = (U)v;
    return (unsigned long long)(u ^ (U)((U)1 << (8 * sizeof(T) - 1)));
  } else {
    return (unsigned long long)(std::make_unsigned_t<T>)v;
  }
}

template <typename T>
__device__ __forceinline__ T from_order_key(unsigned long long k) {
  if constexpr (std::is_floating_point_v<T>) {
    unsigned u = (unsigned)k;
    u = (u & 0x80000000u) ? (u ^ 0x80000000u) : ~u;
    return (T)__uint_as_float(u);
  } else if constexpr (std::is_signed_v<T>) {
    using U = std::make_unsigned_t<T>;
    return (T)((U)k ^ (U)((U)1 << (8 * sizeof(T) - 1)));
  } else {
    return (T)k;
  }
}

template <typename T>
__device__ __forceinline__ unsigned long long key_max() {
  if constexpr (std::is_floating_point_v<T>) {
    return 0xffffffffull;
  } else {
    return (1ull << (8 * sizeof(T))) - 1ull;
  }
}

// One window tap.  x/y are destination coordinates, i/j the mask offsets.
template <typename T, int Ch, bool Border>
__device__ __forceinline__ T window_tap(const T *pSrc, int nSrcStep, int x, int y, int i, int j,
                                        int ax, int ay, int offx, int offy, NppiSize srcSize,
                                        int borderMode, int c) {
  int sx = x + i - ax;
  int sy = y + j - ay;
  if constexpr (Border) {
    int ix = sx + offx;
    int iy = sy + offy;
    ix = map_border(ix, srcSize.width, borderMode);
    iy = map_border(iy, srcSize.height, borderMode);
    sx = ix - offx;
    sy = iy - offy;
  }
  const T *row = reinterpret_cast<const T *>(reinterpret_cast<const char *>(pSrc) +
                                             (long long)sy * nSrcStep);
  return row[(long long)sx * Ch + c];
}

template <typename T, int Ch, bool Border>
__device__ __forceinline__ T kth_sample(const T *pSrc, int nSrcStep, int x, int y, int maskW,
                                        int maskH, int ax, int ay, int offx, int offy,
                                        NppiSize srcSize, int borderMode, int c, int k) {
  unsigned long long lo = 0;
  unsigned long long hi = key_max<T>();
  while (lo < hi) {
    const unsigned long long mid = lo + (hi - lo) / 2;
    int count = 0;
    for (int j = 0; j < maskH; ++j)
      for (int i = 0; i < maskW; ++i) {
        const T v = window_tap<T, Ch, Border>(pSrc, nSrcStep, x, y, i, j, ax, ay, offx, offy,
                                              srcSize, borderMode, c);
        if (order_key<T>(v) <= mid) ++count;
      }
    if (count > k)
      hi = mid;
    else
      lo = mid + 1;
  }
  return from_order_key<T>(lo);
}

template <typename T, int Ch, bool AlphaPreserve, WOp Op, bool Border>
__global__ void window_kernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize roi,
                              NppiSize mask, NppiPoint anchor, int offx, int offy,
                              NppiSize srcSize, int borderMode) {
  const int x = blockIdx.x * kBlockX + threadIdx.x;
  const int y = blockIdx.y * kBlockY + threadIdx.y;
  if (x >= roi.width || y >= roi.height) return;

  constexpr int kNch = AlphaPreserve ? 3 : Ch;
  const int area = mask.width * mask.height;
  const int nsamp = area;

  T *drow = reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + (long long)y * nDstStep);
  for (int c = 0; c < kNch; ++c) {
    T out;
    if constexpr (Op == WOp::Box) {
      if constexpr (std::is_floating_point_v<T> && sizeof(T) == 8) {
        double sum = 0.0;
        for (int j = 0; j < mask.height; ++j)
          for (int i = 0; i < mask.width; ++i)
            sum += (double)window_tap<T, Ch, Border>(pSrc, nSrcStep, x, y, i, j, anchor.x,
                                                     anchor.y, offx, offy, srcSize, borderMode, c);
        out = (T)(sum / (double)area);
      } else if constexpr (std::is_floating_point_v<T>) {
        float sum = 0.0f;
        for (int j = 0; j < mask.height; ++j)
          for (int i = 0; i < mask.width; ++i)
            sum += (float)window_tap<T, Ch, Border>(pSrc, nSrcStep, x, y, i, j, anchor.x,
                                                    anchor.y, offx, offy, srcSize, borderMode, c);
        out = (T)(sum / (float)area);
      } else {
        long long sum = 0;
        for (int j = 0; j < mask.height; ++j)
          for (int i = 0; i < mask.width; ++i)
            sum += (long long)window_tap<T, Ch, Border>(pSrc, nSrcStep, x, y, i, j, anchor.x,
                                                         anchor.y, offx, offy, srcSize,
                                                         borderMode, c);
        out = clamp_int<T>(sum / (long long)area);
      }
    } else if constexpr (Op == WOp::Min || Op == WOp::Max) {
      T best = window_tap<T, Ch, Border>(pSrc, nSrcStep, x, y, 0, 0, anchor.x, anchor.y, offx,
                                         offy, srcSize, borderMode, c);
      for (int j = 0; j < mask.height; ++j)
        for (int i = 0; i < mask.width; ++i) {
          if (i == 0 && j == 0) continue;
          const T v = window_tap<T, Ch, Border>(pSrc, nSrcStep, x, y, i, j, anchor.x, anchor.y,
                                                offx, offy, srcSize, borderMode, c);
          if constexpr (Op == WOp::Min) {
            if (v < best) best = v;
          } else {
            if (v > best) best = v;
          }
        }
      out = best;
    } else {
      const int k1 = (nsamp - 1) / 2;
      const int k2 = nsamp / 2;
      const T v1 = kth_sample<T, Ch, Border>(pSrc, nSrcStep, x, y, mask.width, mask.height,
                                             anchor.x, anchor.y, offx, offy, srcSize, borderMode,
                                             c, k1);
      if (k1 == k2) {
        out = v1;
      } else if constexpr (std::is_floating_point_v<T>) {
        const T v2 = kth_sample<T, Ch, Border>(pSrc, nSrcStep, x, y, mask.width, mask.height,
                                               anchor.x, anchor.y, offx, offy, srcSize,
                                               borderMode, c, k2);
        out = (T)(((float)v1 + (float)v2) * 0.5f);
      } else {
        const T v2 = kth_sample<T, Ch, Border>(pSrc, nSrcStep, x, y, mask.width, mask.height,
                                               anchor.x, anchor.y, offx, offy, srcSize,
                                               borderMode, c, k2);
        out = clamp_int<T>(((long long)v1 + (long long)v2) / 2);
      }
    }
    drow[(long long)x * Ch + c] = out;
  }
}

template <typename T, int Ch, bool AlphaPreserve, WOp Op, bool Border>
inline NppStatus launch_window(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize roi,
                               NppiSize mask, NppiPoint anchor, int offx, int offy,
                               NppiSize srcSize, int borderMode, cudaStream_t stream) {
  dim3 block(kBlockX, kBlockY);
  dim3 grid((roi.width + kBlockX - 1) / kBlockX, (roi.height + kBlockY - 1) / kBlockY);
  window_kernel<T, Ch, AlphaPreserve, Op, Border><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, roi, mask, anchor, offx, offy, srcSize, borderMode);
  return finish_launch();
}

}  // namespace

// nppiFilterBox: raw reads, mask must fit in the ROI; zero ROI is reported as
// MASK_SIZE (the mask > ROI test runs first), negative masks are
// OUT_OFF_RANGE, zero masks succeed without writing, the anchor is not checked.
template <typename T, int Ch, bool AlphaPreserve>
NppStatus filterBox(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                    NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width < 0 || oMaskSize.height < 0)
    return (oSizeROI.width == 0 || oSizeROI.height == 0) ? NPP_SIZE_ERROR : NPP_OUT_OFF_RANGE_ERROR;
  if (oMaskSize.width > oSizeROI.width || oMaskSize.height > oSizeROI.height)
    return NPP_MASK_SIZE_ERROR;
  NppStatus st = check_dst_step<T, Ch>(nDstStep, oSizeROI.width);
  if (st != NPP_NO_ERROR) return st;
  if (oMaskSize.width == 0 || oMaskSize.height == 0) return NPP_NO_ERROR;
  return launch_window<T, Ch, AlphaPreserve, WOp::Box, false>(pSrc, nSrcStep, pDst, nDstStep,
                                                             oSizeROI, oMaskSize, oAnchor, 0, 0,
                                                             NppiSize{0, 0}, 0, stream);
}

// nppiFilterBoxBorder: border modes 2, 4 and 5 only; source image origin is
// pSrc - oSrcOffset.
template <typename T, int Ch, bool AlphaPreserve>
NppStatus filterBoxBorder(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                          T *pDst, int nDstStep, NppiSize oSizeROI, NppiSize oMaskSize,
                          NppiPoint oAnchor, NppiBorderType eBorderType, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  const int bt = (int)eBorderType;
  if (bt != NPP_BORDER_REPLICATE && bt != NPP_BORDER_MIRROR && bt != 5)
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height)
    return NPP_OUT_OFF_RANGE_ERROR;
  if (oMaskSize.width < 0 || oMaskSize.height < 0) return NPP_OUT_OFF_RANGE_ERROR;
  if (oMaskSize.width > oSizeROI.width || oMaskSize.height > oSizeROI.height)
    return NPP_MASK_SIZE_ERROR;
  NppStatus st = check_dst_step<T, Ch>(nDstStep, oSizeROI.width);
  if (st != NPP_NO_ERROR) return st;
  if (oMaskSize.width == 0 || oMaskSize.height == 0) return NPP_NO_ERROR;
  return launch_window<T, Ch, AlphaPreserve, WOp::Box, true>(pSrc, nSrcStep, pDst, nDstStep,
                                                            oSizeROI, oMaskSize, oAnchor,
                                                            oSrcOffset.x, oSrcOffset.y, oSrcSize,
                                                            bt, stream);
}

// nppiFilterMin/Max: zero or negative ROI and mask are SIZE, no mask > ROI
// test, anchor is not checked.
template <typename T, int Ch, bool AlphaPreserve, bool IsMax>
NppStatus filterMinMax(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                       NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_SIZE_ERROR;
  NppStatus st = check_dst_step<T, Ch>(nDstStep, oSizeROI.width);
  if (st != NPP_NO_ERROR) return st;
  return launch_window<T, Ch, AlphaPreserve, IsMax ? WOp::Max : WOp::Min, false>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, 0, 0, NppiSize{0, 0}, 0,
      stream);
}

// nppiFilterMinBorder/MaxBorder: only NPP_BORDER_REPLICATE.
template <typename T, int Ch, bool AlphaPreserve, bool IsMax>
NppStatus filterMinMaxBorder(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
                             T *pDst, int nDstStep, NppiSize oSizeROI, NppiSize oMaskSize,
                             NppiPoint oAnchor, NppiBorderType eBorderType, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) return NPP_SIZE_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height)
    return NPP_OUT_OFF_RANGE_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_SIZE_ERROR;
  NppStatus st = check_dst_step<T, Ch>(nDstStep, oSizeROI.width);
  if (st != NPP_NO_ERROR) return st;
  return launch_window<T, Ch, AlphaPreserve, IsMax ? WOp::Max : WOp::Min, true>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, oSrcOffset.x, oSrcOffset.y,
      oSrcSize, (int)eBorderType, stream);
}

// nppiFilterMedian: zero ROI succeeds (no work) before the step checks; the
// destination step only has the size test (no alignment); zero mask succeeds;
// pBuffer is neither validated nor read.
template <typename T, int Ch, bool AlphaPreserve>
NppStatus filterMedian(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI,
                       NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer, cudaStream_t stream) {
  (void)pBuffer;
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_NO_ERROR;
  const Npp32u rowBytes = (Npp32u)oSizeROI.width * (Npp32u)(sizeof(T) * Ch);
  if ((Npp32u)nDstStep < rowBytes) return NPP_STEP_ERROR;
  return launch_window<T, Ch, AlphaPreserve, WOp::Median, false>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, 0, 0, NppiSize{0, 0}, 0,
      stream);
}

// nppiFilterMedianBorder: zero ROI and zero mask short-circuit before every
// other check; source size, offset and dst step come next, border last.
template <typename T, int Ch, bool AlphaPreserve>
NppStatus filterMedianBorder(const T *pSrc, int nSrcStep, NppiSize oSrcSize,
                             NppiPoint oSrcOffset, T *pDst, int nDstStep, NppiSize oSizeROI,
                             NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
                             NppiBorderType eBorderType, cudaStream_t stream) {
  (void)pBuffer;
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  if (oMaskSize.width <= 0 || oMaskSize.height <= 0) return NPP_NO_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) return NPP_SIZE_ERROR;
  if (oSrcOffset.x < 0 || oSrcOffset.x >= oSrcSize.width || oSrcOffset.y < 0 ||
      oSrcOffset.y >= oSrcSize.height)
    return NPP_OUT_OFF_RANGE_ERROR;
  NppStatus st = check_dst_step<T, Ch>(nDstStep, oSizeROI.width);
  if (st != NPP_NO_ERROR) return st;
  if (eBorderType != NPP_BORDER_REPLICATE && eBorderType != NPP_BORDER_MIRROR)
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  return launch_window<T, Ch, AlphaPreserve, WOp::Median, true>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, oSrcOffset.x, oSrcOffset.y,
      oSrcSize, (int)eBorderType, stream);
}

// GetBufferSize is ROI-independent (measured); the plain dispatch is a table
// measured on the 5090 for mask dimensions up to 40 in each direction.
template <int Ch>
inline int layout_index() {
  return Ch == 1 ? 0 : (Ch == 3 ? 1 : 2);
}

template <typename T, int Ch>
NppStatus medianGetBufferSize(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u *pBufferSize) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width < 0 || oMaskSize.height < 0) return NPP_SIZE_ERROR;
  const int layout = layout_index<Ch>();
  const int dtypeIdx = std::is_same_v<T, Npp8u> ? 0
                      : std::is_same_v<T, Npp16u> ? 1
                      : std::is_same_v<T, Npp16s> ? 2
                                                  : 3;
  const int idx = dtypeIdx * 4 + layout;
  const int mw = oMaskSize.width;
  const int mh = oMaskSize.height;
  bool buffered;
  if (mw == 0 || mh == 0) {
    buffered = false;
  } else if (mw <= kMedianBufMaxDim && mh <= kMedianBufMaxDim) {
    buffered = ((kMedianPlainBufMask[idx][mh - 1] >> (mw - 1)) & 1ull) != 0;
  } else {
    // fallback outside the measured 40x40 range (see the header notes)
    buffered = (long long)mw * mh * (long long)sizeof(T) > 192;
  }
  const long long value = buffered ? 49152ll * (long long)sizeof(T) * mw * mh : 0;
  *pBufferSize = (Npp32u)value;
  return NPP_NO_ERROR;
}

template <typename T, int Ch>
NppStatus medianBorderGetBufferSize(NppiSize oSizeROI, NppiSize oMaskSize, Npp32u *pBufferSize,
                                    NppiBorderType eBorderType) {
  (void)eBorderType;  // any border type is accepted by the size query (measured)
  *pBufferSize = 0;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oMaskSize.width < 0 || oMaskSize.height < 0) return NPP_SIZE_ERROR;
  const int mw = oMaskSize.width;
  const int mh = oMaskSize.height;
  const bool buffered = (long long)mw * mh * (long long)sizeof(T) > 192;
  *pBufferSize = (Npp32u)(buffered ? 49152ll * (long long)sizeof(T) * mw * mh : 0);
  return NPP_NO_ERROR;
}

// ---------------------------------------------------------------------------
// explicit instantiations (host-visible templates)
// ---------------------------------------------------------------------------
#define NPPI_WINDOW_INSTANTIATE(T)                                                             \
  template NppStatus filterBox<T, 1, false>(const T *, int, T *, int, NppiSize, NppiSize,      \
                                            NppiPoint, cudaStream_t);                          \
  template NppStatus filterBox<T, 3, false>(const T *, int, T *, int, NppiSize, NppiSize,      \
                                            NppiPoint, cudaStream_t);                          \
  template NppStatus filterBox<T, 4, false>(const T *, int, T *, int, NppiSize, NppiSize,      \
                                            NppiPoint, cudaStream_t);                          \
  template NppStatus filterBox<T, 4, true>(const T *, int, T *, int, NppiSize, NppiSize,       \
                                           NppiPoint, cudaStream_t);                           \
  template NppStatus filterBoxBorder<T, 1, false>(const T *, int, NppiSize, NppiPoint, T *,    \
                                                  int, NppiSize, NppiSize, NppiPoint,          \
                                                  NppiBorderType, cudaStream_t);               \
  template NppStatus filterBoxBorder<T, 3, false>(const T *, int, NppiSize, NppiPoint, T *,    \
                                                  int, NppiSize, NppiSize, NppiPoint,          \
                                                  NppiBorderType, cudaStream_t);               \
  template NppStatus filterBoxBorder<T, 4, false>(const T *, int, NppiSize, NppiPoint, T *,    \
                                                  int, NppiSize, NppiSize, NppiPoint,          \
                                                  NppiBorderType, cudaStream_t);               \
  template NppStatus filterBoxBorder<T, 4, true>(const T *, int, NppiSize, NppiPoint, T *,     \
                                                 int, NppiSize, NppiSize, NppiPoint,           \
                                                 NppiBorderType, cudaStream_t);                \
  template NppStatus filterMinMax<T, 1, false, false>(const T *, int, T *, int, NppiSize,      \
                                                      NppiSize, NppiPoint, cudaStream_t);      \
  template NppStatus filterMinMax<T, 3, false, false>(const T *, int, T *, int, NppiSize,      \
                                                      NppiSize, NppiPoint, cudaStream_t);      \
  template NppStatus filterMinMax<T, 4, false, false>(const T *, int, T *, int, NppiSize,      \
                                                      NppiSize, NppiPoint, cudaStream_t);      \
  template NppStatus filterMinMax<T, 4, true, false>(const T *, int, T *, int, NppiSize,       \
                                                     NppiSize, NppiPoint, cudaStream_t);       \
  template NppStatus filterMinMax<T, 1, false, true>(const T *, int, T *, int, NppiSize,       \
                                                     NppiSize, NppiPoint, cudaStream_t);       \
  template NppStatus filterMinMax<T, 3, false, true>(const T *, int, T *, int, NppiSize,       \
                                                     NppiSize, NppiPoint, cudaStream_t);       \
  template NppStatus filterMinMax<T, 4, false, true>(const T *, int, T *, int, NppiSize,       \
                                                     NppiSize, NppiPoint, cudaStream_t);       \
  template NppStatus filterMinMax<T, 4, true, true>(const T *, int, T *, int, NppiSize,        \
                                                    NppiSize, NppiPoint, cudaStream_t);        \
  template NppStatus filterMinMaxBorder<T, 1, false, false>(const T *, int, NppiSize,          \
                                                            NppiPoint, T *, int, NppiSize,     \
                                                            NppiSize, NppiPoint,               \
                                                            NppiBorderType, cudaStream_t);     \
  template NppStatus filterMinMaxBorder<T, 3, false, false>(const T *, int, NppiSize,          \
                                                            NppiPoint, T *, int, NppiSize,     \
                                                            NppiSize, NppiPoint,               \
                                                            NppiBorderType, cudaStream_t);     \
  template NppStatus filterMinMaxBorder<T, 4, false, false>(const T *, int, NppiSize,          \
                                                            NppiPoint, T *, int, NppiSize,     \
                                                            NppiSize, NppiPoint,               \
                                                            NppiBorderType, cudaStream_t);     \
  template NppStatus filterMinMaxBorder<T, 4, true, false>(const T *, int, NppiSize,           \
                                                           NppiPoint, T *, int, NppiSize,      \
                                                           NppiSize, NppiPoint,                \
                                                           NppiBorderType, cudaStream_t);      \
  template NppStatus filterMinMaxBorder<T, 1, false, true>(const T *, int, NppiSize,           \
                                                           NppiPoint, T *, int, NppiSize,      \
                                                           NppiSize, NppiPoint,                \
                                                           NppiBorderType, cudaStream_t);      \
  template NppStatus filterMinMaxBorder<T, 3, false, true>(const T *, int, NppiSize,           \
                                                           NppiPoint, T *, int, NppiSize,      \
                                                           NppiSize, NppiPoint,                \
                                                           NppiBorderType, cudaStream_t);      \
  template NppStatus filterMinMaxBorder<T, 4, false, true>(const T *, int, NppiSize,           \
                                                           NppiPoint, T *, int, NppiSize,      \
                                                           NppiSize, NppiPoint,                \
                                                           NppiBorderType, cudaStream_t);      \
  template NppStatus filterMinMaxBorder<T, 4, true, true>(const T *, int, NppiSize,            \
                                                          NppiPoint, T *, int, NppiSize,       \
                                                          NppiSize, NppiPoint,                 \
                                                          NppiBorderType, cudaStream_t);       \
  template NppStatus filterMedian<T, 1, false>(const T *, int, T *, int, NppiSize, NppiSize,   \
                                               NppiPoint, Npp8u *, cudaStream_t);              \
  template NppStatus filterMedian<T, 3, false>(const T *, int, T *, int, NppiSize, NppiSize,   \
                                               NppiPoint, Npp8u *, cudaStream_t);              \
  template NppStatus filterMedian<T, 4, false>(const T *, int, T *, int, NppiSize, NppiSize,   \
                                               NppiPoint, Npp8u *, cudaStream_t);              \
  template NppStatus filterMedian<T, 4, true>(const T *, int, T *, int, NppiSize, NppiSize,    \
                                              NppiPoint, Npp8u *, cudaStream_t);               \
  template NppStatus filterMedianBorder<T, 1, false>(const T *, int, NppiSize, NppiPoint, T *, \
                                                     int, NppiSize, NppiSize, NppiPoint,       \
                                                     Npp8u *, NppiBorderType, cudaStream_t);    \
  template NppStatus filterMedianBorder<T, 3, false>(const T *, int, NppiSize, NppiPoint, T *, \
                                                     int, NppiSize, NppiSize, NppiPoint,       \
                                                     Npp8u *, NppiBorderType, cudaStream_t);    \
  template NppStatus filterMedianBorder<T, 4, false>(const T *, int, NppiSize, NppiPoint, T *, \
                                                     int, NppiSize, NppiSize, NppiPoint,       \
                                                     Npp8u *, NppiBorderType, cudaStream_t);    \
  template NppStatus filterMedianBorder<T, 4, true>(const T *, int, NppiSize, NppiPoint, T *,  \
                                                    int, NppiSize, NppiSize, NppiPoint,        \
                                                    Npp8u *, NppiBorderType, cudaStream_t);     \
  template NppStatus medianGetBufferSize<T, 1>(NppiSize, NppiSize, Npp32u *);                  \
  template NppStatus medianGetBufferSize<T, 3>(NppiSize, NppiSize, Npp32u *);                  \
  template NppStatus medianGetBufferSize<T, 4>(NppiSize, NppiSize, Npp32u *);                  \
  template NppStatus medianBorderGetBufferSize<T, 1>(NppiSize, NppiSize, Npp32u *,             \
                                                     NppiBorderType);                           \
  template NppStatus medianBorderGetBufferSize<T, 3>(NppiSize, NppiSize, Npp32u *,             \
                                                     NppiBorderType);                           \
  template NppStatus medianBorderGetBufferSize<T, 4>(NppiSize, NppiSize, Npp32u *,             \
                                                     NppiBorderType);

NPPI_WINDOW_INSTANTIATE(Npp8u)
NPPI_WINDOW_INSTANTIATE(Npp16u)
NPPI_WINDOW_INSTANTIATE(Npp16s)
NPPI_WINDOW_INSTANTIATE(Npp32f)

// nppiFilterBox_64f_C1R only (no border / min / max / median 64f variants).
template NppStatus filterBox<Npp64f, 1, false>(const Npp64f *, int, Npp64f *, int, NppiSize,
                                               NppiSize, NppiPoint, cudaStream_t);

}  // namespace filter
}  // namespace nppi
