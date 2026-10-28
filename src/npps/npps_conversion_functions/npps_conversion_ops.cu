// nppsConvert_* kernels + executors.
// Mirrors npps_binary_ops.cu / npps_unary_ops.cu structure.
#include "npp.h"
#include "npp_version_compat.h"
#include "npps_conversion_executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace npps {
namespace convert {

namespace ed = npps::elementwise;
namespace ek = npps::elementwise::detail;

namespace {

constexpr int kBlock = 256;

inline NppStatus finish_launch() {
  return (cudaGetLastError() == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

__device__ __host__ __forceinline__ double pow2d(int sf) {
  double r = 1.0;
  for (int i = 0; i < sf && i < 63; ++i) r *= 2.0;
  return r;
}

// Saturation for converting a (already rounded) real value into an integer
// destination type.
template <typename D>
__device__ __host__ __forceinline__ D sat(double v) {
  if constexpr (std::is_same_v<D, Npp8s>)  { return v <= -128.0 ? (D)-128 : (v >= 127.0 ? (D)127 : (D)v); }
  else if constexpr (std::is_same_v<D, Npp8u>)   { return v <= 0.0 ? (D)0 : (v >= 255.0 ? (D)255 : (D)v); }
  else if constexpr (std::is_same_v<D, Npp16s>)  { return v <= -32768.0 ? (D)-32768 : (v >= 32767.0 ? (D)32767 : (D)v); }
  else if constexpr (std::is_same_v<D, Npp16u>)  { return v <= 0.0 ? (D)0 : (v >= 65535.0 ? (D)65535 : (D)v); }
  else if constexpr (std::is_same_v<D, Npp32s>)  { return v <= -2147483648.0 ? (D)-2147483647 - 1 : (v >= 2147483647.0 ? (D)2147483647 : (D)v); }
  else if constexpr (std::is_same_v<D, Npp32u>)  { return v <= 0.0 ? (D)0 : (v >= 4294967295.0 ? (D)4294967295u : (D)v); }
  else if constexpr (std::is_same_v<D, Npp64s>)  { return (D)v; }
  else if constexpr (std::is_same_v<D, Npp64u>)  { return v <= 0.0 ? (D)0 : (D)v; }
  else { return (D)v; }
}

template <typename D>
__device__ __host__ __forceinline__ D rnd_sat(double v, NppRoundMode mode) {
  long long r;
  if (mode == NPP_RND_FINANCIAL) {
    r = (long long)(v >= 0.0 ? v + 0.5 : v - 0.5);
  } else if (mode == NPP_RND_ZERO) {
    r = (long long)v;
  } else {  // NPP_RND_NEAR : nearest, ties to even
    const double av = std::fabs(v);
    const double lo = std::floor(av);
    const double f = av - lo;
    long long m = (long long)lo;
    const bool up = (f > 0.5) || (f == 0.5 && (m & 1LL));
    r = (v < 0.0) ? -(m + (up ? 1 : 0)) : (m + (up ? 1 : 0));
  }
  return sat<D>((double)r);
}

template <typename A, typename D>
__global__ void convert_kernel(const A* src, D* dst, NppSignalLength n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) {
    if constexpr (std::is_floating_point_v<D>) {
      dst[i] = (D)((double)ek::re_l(src[i]));
    } else {
      dst[i] = ek::sat_ll<D>(ek::re_ll(src[i]));
    }
  }
}

template <typename A, typename D>
__global__ void convert_scale_kernel(const A* src, D* dst, NppSignalLength n, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) {
    const double val = (double)ek::re_l(src[i]) / pow2d(sf);
    if constexpr (std::is_floating_point_v<D>) {
      dst[i] = (D)val;
    } else {
      // Integer-destination Sfs (no explicit round mode) uses round-half-away-from-zero.
      dst[i] = rnd_sat<D>(val, NPP_RND_FINANCIAL);
    }
  }
}

template <typename A, typename D>
__global__ void convert_round_kernel(const A* src, D* dst, NppSignalLength n, NppRoundMode mode,
                                     int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) {
    const double val = (double)ek::re_l(src[i]) / pow2d(sf);
    dst[i] = rnd_sat<D>(val, mode);
  }
}

}  // namespace

template <typename A, typename D>
NppStatus convert(const A* src, D* dst, NppSignalLength nLength, cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  convert_kernel<A, D><<<(nLength + kBlock - 1) / kBlock, kBlock, 0, stream>>>(src, dst, nLength);
  return finish_launch();
}

template <typename A, typename D>
NppStatus convertScale(const A* src, D* dst, NppSignalLength nLength, int nScaleFactor,
                       cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  convert_scale_kernel<A, D><<<(nLength + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      src, dst, nLength, nScaleFactor);
  return finish_launch();
}

template <typename A, typename D>
NppStatus convertRound(const A* src, D* dst, NppSignalLength nLength, NppRoundMode mode,
                       int nScaleFactor, cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  convert_round_kernel<A, D><<<(nLength + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      src, dst, nLength, mode, nScaleFactor);
  return finish_launch();
}

// -- explicit instantiations -------------------------------------------------
// plain: int<->float and int->int
template NppStatus convert<Npp8s, Npp16s>(const Npp8s*, Npp16s*, NppSignalLength, cudaStream_t);
template NppStatus convert<Npp8s, Npp32f>(const Npp8s*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus convert<Npp8u, Npp32f>(const Npp8u*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus convert<Npp16s, Npp32f>(const Npp16s*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus convert<Npp16s, Npp32s>(const Npp16s*, Npp32s*, NppSignalLength, cudaStream_t);
template NppStatus convert<Npp16u, Npp32f>(const Npp16u*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus convert<Npp32s, Npp16s>(const Npp32s*, Npp16s*, NppSignalLength, cudaStream_t);
template NppStatus convert<Npp32s, Npp32f>(const Npp32s*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus convert<Npp32s, Npp64f>(const Npp32s*, Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus convert<Npp64s, Npp64f>(const Npp64s*, Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus convert<Npp32f, Npp64f>(const Npp32f*, Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus convert<Npp64f, Npp32f>(const Npp64f*, Npp32f*, NppSignalLength, cudaStream_t);

// convertScale (int nScaleFactor only, no round mode)
template NppStatus convertScale<Npp16s, Npp32f>(const Npp16s*, Npp32f*, NppSignalLength, int, cudaStream_t);
template NppStatus convertScale<Npp16s, Npp64f>(const Npp16s*, Npp64f*, NppSignalLength, int, cudaStream_t);
template NppStatus convertScale<Npp32s, Npp16s>(const Npp32s*, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus convertScale<Npp32s, Npp32f>(const Npp32s*, Npp32f*, NppSignalLength, int, cudaStream_t);
template NppStatus convertScale<Npp32s, Npp64f>(const Npp32s*, Npp64f*, NppSignalLength, int, cudaStream_t);

// convertRound (eRoundMode + nScaleFactor), dest integer
template NppStatus convertRound<Npp16s, Npp8s>(const Npp16s*, Npp8s*, NppSignalLength, NppRoundMode, int, cudaStream_t);
template NppStatus convertRound<Npp32f, Npp16s>(const Npp32f*, Npp16s*, NppSignalLength, NppRoundMode, int, cudaStream_t);
template NppStatus convertRound<Npp32f, Npp16u>(const Npp32f*, Npp16u*, NppSignalLength, NppRoundMode, int, cudaStream_t);
template NppStatus convertRound<Npp32f, Npp8s>(const Npp32f*, Npp8s*, NppSignalLength, NppRoundMode, int, cudaStream_t);
template NppStatus convertRound<Npp32f, Npp8u>(const Npp32f*, Npp8u*, NppSignalLength, NppRoundMode, int, cudaStream_t);
template NppStatus convertRound<Npp32f, Npp32s>(const Npp32f*, Npp32s*, NppSignalLength, NppRoundMode, int, cudaStream_t);
template NppStatus convertRound<Npp64f, Npp16s>(const Npp64f*, Npp16s*, NppSignalLength, NppRoundMode, int, cudaStream_t);
template NppStatus convertRound<Npp64f, Npp32s>(const Npp64f*, Npp32s*, NppSignalLength, NppRoundMode, int, cudaStream_t);
template NppStatus convertRound<Npp64f, Npp64s>(const Npp64f*, Npp64s*, NppSignalLength, NppRoundMode, int, cudaStream_t);
template NppStatus convertRound<Npp64s, Npp32s>(const Npp64s*, Npp32s*, NppSignalLength, NppRoundMode, int, cudaStream_t);

}  // namespace convert
}  // namespace npps
