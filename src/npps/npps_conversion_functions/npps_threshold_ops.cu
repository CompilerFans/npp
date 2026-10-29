// nppsThreshold_* kernels + executors (plain / GT / LT / GTVal / LTVal).
#include "npp.h"
#include "npp_version_compat.h"
#include "npps_conversion_executor.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace npps {
namespace threshold {

namespace ed = npps::elementwise;

namespace {

constexpr int kBlock = 256;

// Kernel-launch error check moved to the shared header npp_kernel_utils.h
// (package p0.1); re-exported here so the executors below keep using the
// unqualified name.
using npp_kernel::checkCudaKernelError;

// Compare the scalar threshold level against the real part of a sample.
template <typename T, typename L>
__device__ __host__ __forceinline__ bool cmp(const T& s, L level, NppCmpOp op) {
  if constexpr (ed::is_complex_v<T>) {
    using CF = std::conditional_t<std::is_same_v<T, Npp64fc>, double, float>;
    const CF mag = sqrt((CF)s.re * (CF)s.re + (CF)s.im * (CF)s.im);
    const CF lv = (CF)level;
    switch (op) {
      case NPP_CMP_LESS:       return mag <  lv;
      case NPP_CMP_LESS_EQ:    return mag <= lv;
      case NPP_CMP_EQ:         return mag == lv;
      case NPP_CMP_GREATER_EQ: return mag >= lv;
      case NPP_CMP_GREATER:    return mag >  lv;
      default:                 return false;
    }
  } else {
    switch (op) {
      case NPP_CMP_LESS:       return s <  level;
      case NPP_CMP_LESS_EQ:    return s <= level;
      case NPP_CMP_EQ:       return s == level;
      case NPP_CMP_GREATER_EQ: return s >= level;
      case NPP_CMP_GREATER:    return s >  level;
      default:                 return false;
    }
  }
}

template <typename T, typename L>
__device__ __host__ __forceinline__ T set_value(const T& s, L level, bool pass) {
  if (!pass) return s;
  if constexpr (ed::is_complex_v<T>) {
    // Passing complex samples are rescaled to have magnitude == level
    // (dst = src * level / |src|), with component precision matching the type.
    using F = std::conditional_t<std::is_same_v<T, Npp64fc>, double, float>;
    const F re = (F)s.re;
    const F im = (F)s.im;
    const F mag = sqrt(re * re + im * im);
    const F f = (F)level / mag;
    T r{};
    r.re = (decltype(r.re))(re * f);
    r.im = (decltype(r.im))(im * f);
    return r;
  } else {
    return (T)level;
  }
}

template <typename T, typename L>
__global__ void threshold_kernel(const T* src, T* dst, NppSignalLength n, L level, NppCmpOp op) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) dst[i] = set_value(src[i], level, cmp(src[i], level, op));
}

template <typename T, typename L>
__global__ void threshold_val_kernel(const T* src, T* dst, NppSignalLength n, L level, T value,
                                     NppCmpOp op) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) dst[i] = cmp(src[i], level, op) ? value : src[i];
}

template <typename T, typename L>
NppStatus th_run(const T* src, T* dst, NppSignalLength n, L level, NppCmpOp op, cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (n == 0) return NPP_SIZE_ERROR;
  threshold_kernel<T, L><<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(src, dst, n, level, op);
  return checkCudaKernelError();
}

template <typename T, typename L>
NppStatus th_run_val(const T* src, T* dst, NppSignalLength n, L level, T value, NppCmpOp op,
                     cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (n == 0) return NPP_SIZE_ERROR;
  threshold_val_kernel<T, L><<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      src, dst, n, level, value, op);
  return checkCudaKernelError();
}

}  // namespace

template <typename T, typename L>
NppStatus thresholdOp(const T* src, T* dst, NppSignalLength nLength, L nLevel, NppCmpOp relOp,
                      cudaStream_t stream) {
  return th_run(src, dst, nLength, nLevel, relOp, stream);
}

template <typename T, typename L>
NppStatus thresholdGT(const T* src, T* dst, NppSignalLength nLength, L nLevel, cudaStream_t stream) {
  return th_run(src, dst, nLength, nLevel, NPP_CMP_GREATER, stream);
}

template <typename T, typename L>
NppStatus thresholdLT(const T* src, T* dst, NppSignalLength nLength, L nLevel, cudaStream_t stream) {
  return th_run(src, dst, nLength, nLevel, NPP_CMP_LESS, stream);
}

template <typename T, typename L>
NppStatus thresholdGTVal(const T* src, T* dst, NppSignalLength nLength, L nLevel, T nValue,
                         cudaStream_t stream) {
  return th_run_val(src, dst, nLength, nLevel, nValue, NPP_CMP_GREATER, stream);
}

template <typename T, typename L>
NppStatus thresholdLTVal(const T* src, T* dst, NppSignalLength nLength, L nLevel, T nValue,
                         cudaStream_t stream) {
  return th_run_val(src, dst, nLength, nLevel, nValue, NPP_CMP_LESS, stream);
}

// -- explicit instantiations -------------------------------------------------
// nppsThreshold (relOp family): 16s,16sc,32f,32fc,64f,64fc
template NppStatus thresholdOp<Npp16s, Npp16s>(const Npp16s*, Npp16s*, NppSignalLength, Npp16s, NppCmpOp, cudaStream_t);
template NppStatus thresholdOp<Npp16sc, Npp16s>(const Npp16sc*, Npp16sc*, NppSignalLength, Npp16s, NppCmpOp, cudaStream_t);
template NppStatus thresholdOp<Npp32f, Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, Npp32f, NppCmpOp, cudaStream_t);
template NppStatus thresholdOp<Npp32fc, Npp32f>(const Npp32fc*, Npp32fc*, NppSignalLength, Npp32f, NppCmpOp, cudaStream_t);
template NppStatus thresholdOp<Npp64f, Npp64f>(const Npp64f*, Npp64f*, NppSignalLength, Npp64f, NppCmpOp, cudaStream_t);
template NppStatus thresholdOp<Npp64fc, Npp64f>(const Npp64fc*, Npp64fc*, NppSignalLength, Npp64f, NppCmpOp, cudaStream_t);

// thresholdGT / LT (same type set).
template NppStatus thresholdGT<Npp16s, Npp16s>(const Npp16s*, Npp16s*, NppSignalLength, Npp16s, cudaStream_t);
template NppStatus thresholdGT<Npp16sc, Npp16s>(const Npp16sc*, Npp16sc*, NppSignalLength, Npp16s, cudaStream_t);
template NppStatus thresholdGT<Npp32f, Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, Npp32f, cudaStream_t);
template NppStatus thresholdGT<Npp32fc, Npp32f>(const Npp32fc*, Npp32fc*, NppSignalLength, Npp32f, cudaStream_t);
template NppStatus thresholdGT<Npp64f, Npp64f>(const Npp64f*, Npp64f*, NppSignalLength, Npp64f, cudaStream_t);
template NppStatus thresholdGT<Npp64fc, Npp64f>(const Npp64fc*, Npp64fc*, NppSignalLength, Npp64f, cudaStream_t);

template NppStatus thresholdLT<Npp16s, Npp16s>(const Npp16s*, Npp16s*, NppSignalLength, Npp16s, cudaStream_t);
template NppStatus thresholdLT<Npp16sc, Npp16s>(const Npp16sc*, Npp16sc*, NppSignalLength, Npp16s, cudaStream_t);
template NppStatus thresholdLT<Npp32f, Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, Npp32f, cudaStream_t);
template NppStatus thresholdLT<Npp32fc, Npp32f>(const Npp32fc*, Npp32fc*, NppSignalLength, Npp32f, cudaStream_t);
template NppStatus thresholdLT<Npp64f, Npp64f>(const Npp64f*, Npp64f*, NppSignalLength, Npp64f, cudaStream_t);
template NppStatus thresholdLT<Npp64fc, Npp64f>(const Npp64fc*, Npp64fc*, NppSignalLength, Npp64f, cudaStream_t);

// thresholdGTVal / LTVal: value is the sample type T.
template NppStatus thresholdGTVal<Npp16s, Npp16s>(const Npp16s*, Npp16s*, NppSignalLength, Npp16s, Npp16s, cudaStream_t);
template NppStatus thresholdGTVal<Npp16sc, Npp16s>(const Npp16sc*, Npp16sc*, NppSignalLength, Npp16s, Npp16sc, cudaStream_t);
template NppStatus thresholdGTVal<Npp32f, Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, Npp32f, Npp32f, cudaStream_t);
template NppStatus thresholdGTVal<Npp32fc, Npp32f>(const Npp32fc*, Npp32fc*, NppSignalLength, Npp32f, Npp32fc, cudaStream_t);
template NppStatus thresholdGTVal<Npp64f, Npp64f>(const Npp64f*, Npp64f*, NppSignalLength, Npp64f, Npp64f, cudaStream_t);
template NppStatus thresholdGTVal<Npp64fc, Npp64f>(const Npp64fc*, Npp64fc*, NppSignalLength, Npp64f, Npp64fc, cudaStream_t);

template NppStatus thresholdLTVal<Npp16s, Npp16s>(const Npp16s*, Npp16s*, NppSignalLength, Npp16s, Npp16s, cudaStream_t);
template NppStatus thresholdLTVal<Npp16sc, Npp16s>(const Npp16sc*, Npp16sc*, NppSignalLength, Npp16s, Npp16sc, cudaStream_t);
template NppStatus thresholdLTVal<Npp32f, Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, Npp32f, Npp32f, cudaStream_t);
template NppStatus thresholdLTVal<Npp32fc, Npp32f>(const Npp32fc*, Npp32fc*, NppSignalLength, Npp32f, Npp32fc, cudaStream_t);
template NppStatus thresholdLTVal<Npp64f, Npp64f>(const Npp64f*, Npp64f*, NppSignalLength, Npp64f, Npp64f, cudaStream_t);
template NppStatus thresholdLTVal<Npp64fc, Npp64f>(const Npp64fc*, Npp64fc*, NppSignalLength, Npp64f, Npp64fc, cudaStream_t);

}  // namespace threshold
}  // namespace npps
