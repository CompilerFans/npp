// nppiConvert_<src><dst>_<layout>R kernels and executor entry points.
// See nppi_convert_executor.h for the family contract.
#include "npp.h"
#include "nppi_convert_executor.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace nppi {
namespace convert {
namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

template <typename D>
struct IsFloatDst : std::integral_constant<bool,
                       std::is_floating_point_v<D> || std::is_same_v<D, Npp16f>> {};

inline NppStatus finish_launch() {
  if (cudaGetLastError() == cudaSuccess) return NPP_NO_ERROR;
  return NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

__host__ __device__ __forceinline__ double inv_pow2(int sf) {
  return ldexp(1.0, -sf);
}

// Real value of a source sample as a double.
template <typename T>
__host__ __device__ __forceinline__ double to_double(const T &s) {
  if constexpr (std::is_same_v<T, Npp16f>) {
    const unsigned short bits = static_cast<unsigned short>(s.fp16);
    const __half h = *reinterpret_cast<const __half *>(&bits);
    return static_cast<double>(__half2float(h));
  } else {
    return static_cast<double>(s);
  }
}

// Emit a real value (already rounded if required) into the destination type.
template <typename Dst>
__host__ __device__ __forceinline__ Dst to_real(double v) {
  if constexpr (std::is_same_v<Dst, Npp16f>) {
    const __half h = __float2half(static_cast<float>(v));
    Npp16f out;
    out.fp16 = static_cast<short>(*reinterpret_cast<const unsigned short *>(&h));
    return out;
  } else {
    return static_cast<Dst>(v);
  }
}

// Saturating integer emission (no wrap).
template <typename Dst>
__host__ __device__ __forceinline__ Dst saturate_int(double v) {
  if (std::isnan(v)) v = -std::numeric_limits<double>::infinity();
  if constexpr (std::is_same_v<Dst, Npp8s>) {
    return v <= -128.0 ? static_cast<Dst>(-128) : (v >= 127.0 ? static_cast<Dst>(127) : static_cast<Dst>(v));
  } else if constexpr (std::is_unsigned_v<Dst>) {
    return static_cast<Dst>(v <= 0.0 ? 0.0 : (v >= (double)std::numeric_limits<Dst>::max()
                                                       ? (double)std::numeric_limits<Dst>::max()
                                                       : v));
  } else {
    return static_cast<Dst>(v <= (double)std::numeric_limits<Dst>::min()
                                  ? (double)std::numeric_limits<Dst>::min()
                                  : (v >= (double)std::numeric_limits<Dst>::max()
                                         ? (double)std::numeric_limits<Dst>::max()
                                         : v));
  }
}

// Round a real value per NppRoundMode (NEAR = ties-to-even, FINANCIAL = ties
// away from zero, ZERO = toward zero).
__host__ __device__ __forceinline__ double round_mode(double v, NppRoundMode mode) {
  if (std::isnan(v)) return v;
  if (mode == NPP_RND_FINANCIAL) {
    return v >= 0.0 ? std::floor(v + 0.5) : std::ceil(v - 0.5);
  }
  if (mode == NPP_RND_ZERO) {
    return v >= 0.0 ? std::floor(v) : std::ceil(v);
  }
  // NPP_RND_NEAR (default): nearest, ties to even.
  const double av = std::fabs(v);
  const double lo = std::floor(av);
  const double fr = av - lo;
  long long m = static_cast<long long>(lo);
  const bool up = (fr > 0.5) || (fr == 0.5 && (m & 1LL));
  const double mag = m + (up ? 1.0 : 0.0);
  return v < 0.0 ? -mag : mag;
}

// Core per-pixel conversion.
//   Round: apply eRoundMode before emission.
//   Scale: divide by 2^sf first.
template <typename Src, typename Dst, bool Round, bool Scale>
__host__ __device__ __forceinline__ Dst cvt(const Src &s, NppRoundMode mode, int sf) {
  double v = to_double(s);
  if constexpr (Scale) v *= inv_pow2(sf);
  if constexpr (IsFloatDst<Dst>::value) {
    // Converting to a floating destination keeps the fractional value; the
    // format conversion (fp32 -> fp16) applies nearest-even rounding on the
    // mantissa, which __half reported round_mode's integer rounding would destroy.
    return to_real<Dst>(v);
  } else {
    const long long i = (Round || Scale) ? static_cast<long long>(round_mode(v, mode))
                                         : static_cast<long long>(v);
    return saturate_int<Dst>(static_cast<double>(i));
  }
}

// 2D ROI convert kernel.  Channels is the buffer channel count; AlphaPr=true
// converts only the first three channels and leaves the 4th (alpha) untouched.
template <typename Src, typename Dst, int Channels, bool AlphaPr, bool Round, bool Scale>
__global__ void convert_kernel(const Src *__restrict__ pSrc, int nSrcStep, Dst *__restrict__ pDst, int nDstStep,
                               int width, int height, NppRoundMode mode, int sf) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const Src *srow = reinterpret_cast<const Src *>(reinterpret_cast<const char *>(pSrc) + y * nSrcStep);
  Dst *drow = reinterpret_cast<Dst *>(reinterpret_cast<char *>(pDst) + y * nDstStep);
  const int base = x * Channels;
  const int n = AlphaPr ? 3 : Channels;
  for (int c = 0; c < n; ++c) {
    drow[base + c] = cvt<Src, Dst, Round, Scale>(srow[base + c], mode, sf);
  }
}

inline NppStatus validate_common(const void *pSrc, void *pDst, NppiSize oSizeROI, int nSrcStep, int nDstStep,
                                 int elemSrcBytes, int elemDstBytes, int channels) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0) return NPP_SIZE_ERROR;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  const int minSrc = oSizeROI.width * channels * elemSrcBytes;
  const int minDst = oSizeROI.width * channels * elemDstBytes;
  if (nSrcStep < minSrc || nDstStep < minDst) return NPP_STEP_ERROR;
  return NPP_NO_ERROR;
}

template <typename Src, typename Dst, int Channels, bool AlphaPr, bool Round, bool Scale>
NppStatus run(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode mode, int sf,
              cudaStream_t stream) {
  NppStatus st = validate_common(pSrc, pDst, oSizeROI, nSrcStep, nDstStep, (int)sizeof(Src), (int)sizeof(Dst),
                                 Channels);
  if (st != NPP_NO_ERROR) return st;
  if (oSizeROI.width == 0 || oSizeROI.height == 0) return NPP_NO_ERROR;
  (void)cudaGetLastError();
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + kBlockX - 1) / kBlockX, (oSizeROI.height + kBlockY - 1) / kBlockY);
  convert_kernel<Src, Dst, Channels, AlphaPr, Round, Scale><<<grid, block, 0, stream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, mode, sf);
  const cudaError_t lerr = cudaGetLastError();
  if (lerr != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  cudaStreamSynchronize(stream);
  return finish_launch();
}

}  // namespace

template <typename Src, typename Dst, int Channels, bool AlphaPr>
NppStatus convert(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  return run<Src, Dst, Channels, AlphaPr, false, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, NPP_RND_NEAR, 0,
                                                        stream);
}

template <typename Src, typename Dst, int Channels, bool AlphaPr>
NppStatus convertRnd(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI, NppRoundMode eRoundMode,
                     cudaStream_t stream) {
  return run<Src, Dst, Channels, AlphaPr, true, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eRoundMode, 0,
                                                       stream);
}

template <typename Src, typename Dst, int Channels, bool AlphaPr>
NppStatus convertRndSfs(const Src *pSrc, int nSrcStep, Dst *pDst, int nDstStep, NppiSize oSizeROI,
                        NppRoundMode eRoundMode, int nScaleFactor, cudaStream_t stream) {
  return run<Src, Dst, Channels, AlphaPr, true, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eRoundMode,
                                                      nScaleFactor, stream);
}

}  // namespace convert
}  // namespace nppi
template NppStatus nppi::convert::convert<Npp16f, Npp32f, 1, false>(const Npp16f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16f, Npp32f, 3, false>(const Npp16f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16f, Npp32f, 4, false>(const Npp16f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16f, Npp32f, 4, true>(const Npp16f*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp16u, 1, false>(const Npp16s*, int, Npp16u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp32f, 1, false>(const Npp16s*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp32f, 3, false>(const Npp16s*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp32f, 4, false>(const Npp16s*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp32f, 4, true>(const Npp16s*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp32s, 1, false>(const Npp16s*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp32s, 3, false>(const Npp16s*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp32s, 4, false>(const Npp16s*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp32s, 4, true>(const Npp16s*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp32u, 1, false>(const Npp16s*, int, Npp32u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp8u, 1, false>(const Npp16s*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp8u, 3, false>(const Npp16s*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp8u, 4, false>(const Npp16s*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16s, Npp8u, 4, true>(const Npp16s*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp32f, 3, false>(const Npp16u*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp32f, 4, false>(const Npp16u*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp32f, 4, true>(const Npp16u*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp32s, 1, false>(const Npp16u*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp32s, 3, false>(const Npp16u*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp32s, 4, false>(const Npp16u*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp32s, 4, true>(const Npp16u*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp32u, 1, false>(const Npp16u*, int, Npp32u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp8u, 1, false>(const Npp16u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp8u, 3, false>(const Npp16u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp8u, 4, false>(const Npp16u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp16u, Npp8u, 4, true>(const Npp16u*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp32s, Npp32f, 1, false>(const Npp32s*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp32s, Npp32u, 1, false>(const Npp32s*, int, Npp32u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp32s, Npp8s, 1, false>(const Npp32s*, int, Npp8s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp32s, Npp8s, 3, false>(const Npp32s*, int, Npp8s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp32s, Npp8s, 4, false>(const Npp32s*, int, Npp8s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp32s, Npp8s, 4, true>(const Npp32s*, int, Npp8s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp32s, Npp8u, 1, false>(const Npp32s*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp32s, Npp8u, 3, false>(const Npp32s*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp32s, Npp8u, 4, false>(const Npp32s*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp32s, Npp8u, 4, true>(const Npp32s*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp32u, Npp32f, 1, false>(const Npp32u*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp16s, 1, false>(const Npp8s*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp16u, 1, false>(const Npp8s*, int, Npp16u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp32f, 1, false>(const Npp8s*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp32f, 3, false>(const Npp8s*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp32f, 4, false>(const Npp8s*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp32f, 4, true>(const Npp8s*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp32s, 1, false>(const Npp8s*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp32s, 3, false>(const Npp8s*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp32s, 4, false>(const Npp8s*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp32s, 4, true>(const Npp8s*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp32u, 1, false>(const Npp8s*, int, Npp32u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8s, Npp8u, 1, false>(const Npp8s*, int, Npp8u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp16s, 1, false>(const Npp8u*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp16s, 3, false>(const Npp8u*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp16s, 4, false>(const Npp8u*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp16s, 4, true>(const Npp8u*, int, Npp16s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp16u, 3, false>(const Npp8u*, int, Npp16u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp16u, 4, false>(const Npp8u*, int, Npp16u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp16u, 4, true>(const Npp8u*, int, Npp16u*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp32f, 4, true>(const Npp8u*, int, Npp32f*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp32s, 1, false>(const Npp8u*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp32s, 3, false>(const Npp8u*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp32s, 4, false>(const Npp8u*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convert<Npp8u, Npp32s, 4, true>(const Npp8u*, int, Npp32s*, int, NppiSize, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16f, 1, false>(const Npp32f*, int, Npp16f*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16f, 3, false>(const Npp32f*, int, Npp16f*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16f, 4, false>(const Npp32f*, int, Npp16f*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16f, 4, true>(const Npp32f*, int, Npp16f*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16s, 1, false>(const Npp32f*, int, Npp16s*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16s, 3, false>(const Npp32f*, int, Npp16s*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16s, 4, false>(const Npp32f*, int, Npp16s*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16s, 4, true>(const Npp32f*, int, Npp16s*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16u, 1, false>(const Npp32f*, int, Npp16u*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16u, 3, false>(const Npp32f*, int, Npp16u*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16u, 4, false>(const Npp32f*, int, Npp16u*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp16u, 4, true>(const Npp32f*, int, Npp16u*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp8s, 1, false>(const Npp32f*, int, Npp8s*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp8s, 3, false>(const Npp32f*, int, Npp8s*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp8s, 4, false>(const Npp32f*, int, Npp8s*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp8s, 4, true>(const Npp32f*, int, Npp8s*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp8u, 4, false>(const Npp32f*, int, Npp8u*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRnd<Npp32f, Npp8u, 4, true>(const Npp32f*, int, Npp8u*, int, NppiSize, NppRoundMode, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp16s, Npp8s, 1, false>(const Npp16s*, int, Npp8s*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp16u, Npp16s, 1, false>(const Npp16u*, int, Npp16s*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp16u, Npp8s, 1, false>(const Npp16u*, int, Npp8s*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32f, Npp16s, 1, false>(const Npp32f*, int, Npp16s*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32f, Npp16u, 1, false>(const Npp32f*, int, Npp16u*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32f, Npp32s, 1, false>(const Npp32f*, int, Npp32s*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32f, Npp32u, 1, false>(const Npp32f*, int, Npp32u*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32f, Npp8s, 1, false>(const Npp32f*, int, Npp8s*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32f, Npp8u, 1, false>(const Npp32f*, int, Npp8u*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32s, Npp16s, 1, false>(const Npp32s*, int, Npp16s*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32s, Npp16u, 1, false>(const Npp32s*, int, Npp16u*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32u, Npp16s, 1, false>(const Npp32u*, int, Npp16s*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32u, Npp16u, 1, false>(const Npp32u*, int, Npp16u*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32u, Npp32s, 1, false>(const Npp32u*, int, Npp32s*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32u, Npp8s, 1, false>(const Npp32u*, int, Npp8s*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp32u, Npp8u, 1, false>(const Npp32u*, int, Npp8u*, int, NppiSize, NppRoundMode, int, cudaStream_t);
template NppStatus nppi::convert::convertRndSfs<Npp8u, Npp8s, 1, false>(const Npp8u*, int, Npp8s*, int, NppiSize, NppRoundMode, int, cudaStream_t);
