// Scalar / bitwise / shift / reduction 1D executors for the npps unary
// arithmetic families (p1.1c). Mirrors npps_binary_ops.cu / npps_constant_ops.cu.
#include "npp.h"
#include "npp_version_compat.h"
#include "npps_unary_executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace npps {
namespace unary {

namespace ed = npps::elementwise;
namespace ek = npps::elementwise::detail;

namespace {

// Kernel-launch error check moved to the shared header npp_kernel_utils.h
// (package p0.1); re-exported here so the executors below keep using the
// unqualified name.
using npp_kernel::checkCudaKernelError;

}  // namespace

namespace kernel {

template <typename F, typename A, typename D>
__global__ void unary_sfs_kernel(const A* src, D* dst, NppSignalLength n, F op, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) dst[i] = op.template applyS<A, D>(src[i], sf);
}
template <typename F, typename A>
__global__ void unary_sfs_inplace_kernel(A* d, NppSignalLength n, F op, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) d[i] = op.template applyS<A, A>(d[i], sf);
}
template <typename F, typename A, typename D>
__global__ void unary_f_kernel(const A* src, D* dst, NppSignalLength n, F op) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) dst[i] = op.template applyF<A, D>(src[i]);
}
template <typename F, typename A>
__global__ void unary_f_inplace_kernel(A* d, NppSignalLength n, F op) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) d[i] = op.template applyF<A, A>(d[i]);
}
template <typename A>
__global__ void shift_kernel(const A* src, int shift, A* dst, NppSignalLength n, char dir) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n) return;
  unsigned long long u = (unsigned long long)src[i];
  dst[i] = dir ? (A)(u << shift) : (A)(u >> shift);
}
template <typename A>
__global__ void shift_inplace_kernel(A* d, int shift, NppSignalLength n, char dir) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n) return;
  unsigned long long u = (unsigned long long)d[i];
  d[i] = dir ? (A)(u << shift) : (A)(u >> shift);
}
template <typename A>
__global__ void bitwise_bin_kernel(const A* a, const A* b, A* d, NppSignalLength n, int op) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n) return;
  unsigned long long x = (unsigned long long)a[i], y = (unsigned long long)b[i];
  if (op == 0) d[i] = (A)(x & y);
  else if (op == 1) d[i] = (A)(x | y);
  else d[i] = (A)(x ^ y);
}
template <typename A>
__global__ void bitwise_bin_inplace_kernel(const A* a, A* d, NppSignalLength n, int op) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n) return;
  unsigned long long x = (unsigned long long)a[i], y = (unsigned long long)d[i];
  if (op == 0) d[i] = (A)(x & y);
  else if (op == 1) d[i] = (A)(x | y);
  else d[i] = (A)(x ^ y);
}
template <typename A>
__global__ void not_kernel(const A* src, A* dst, NppSignalLength n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) dst[i] = (A)(~(unsigned long long)src[i]);
}
template <typename A>
__global__ void not_inplace_kernel(A* d, NppSignalLength n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)n) d[i] = (A)(~(unsigned long long)d[i]);
}
__global__ void mullow_kernel(const Npp32s* a, const Npp32s* b, Npp32s* d, NppSignalLength n, int sf) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n) return;
  long long p = (long long)a[i] * (long long)b[i];
  // NPP 13.4: round the full 64-bit product by 2^sf (banker's), then take the
  // low 32 bits (validated against goldens).
  long long scaled = ek::scale_sat<Npp64s>(p, sf);
  d[i] = (Npp32s)(scaled & 0xFFFFFFFFLL);
}

// -- reduction (SumLn) --
template <typename SRC, typename PT>
__global__ void sumln_partial(const SRC* src, NppSignalLength n, PT* partial) {
  __shared__ PT s[256];
  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = (idx < (size_t)n) ? (PT)log((double)src[idx]) : (PT)0;
  __syncthreads();
  for (unsigned int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (tid < st) s[tid] += s[tid + st];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = s[0];
}
template <typename PT>
__global__ void sumln_final(const PT* partial, size_t nblk, PT* dst) {
  __shared__ PT s[256];
  size_t tid = threadIdx.x;
  s[tid] = (PT)0;
  size_t i = tid;
  while (i < nblk) { s[tid] += partial[i]; i += blockDim.x; }
  __syncthreads();
  for (unsigned int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) s[tid] += s[tid + off];
    __syncthreads();
  }
  if (tid == 0) *dst = s[0];
}

// --npps Cauchy --
__global__ void cauchy_kernel(Npp32f* d, NppSignalLength n, float p) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n) return;
  float x = d[i];
  float v = x / p;
  d[i] = 0.5f * logf(1.0f + v * v);
}
__global__ void cauchyD_kernel(Npp32f* d, NppSignalLength n, float p) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n) return;
  float x = d[i];
  d[i] = x / (p * p + x * x);
}
__global__ void cauchyDD2_kernel(Npp32f* d, Npp32f* d2, NppSignalLength n, float p) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n) return;
  float x = d[i];
  float pp = p * p;
  float den = pp + x * x;
  d[i] = x / den;
  d2[i] = (pp - x * x) / (den * den);
}

}  // namespace kernel

namespace {

template <typename F, typename A, typename D>
NppStatus run_scaled(const A* src, D* dst, NppSignalLength nLength, int sf, cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  if (nLength < 0) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  F op; (void)cudaGetLastError();
  kernel::unary_sfs_kernel<F, A, D><<<(unsigned)grid, ek::kBlock, 0, stream>>>(src, dst, nLength, op, sf);
  return checkCudaKernelError();
}
template <typename F, typename A>
NppStatus run_scaled_i(A* d, NppSignalLength nLength, int sf, cudaStream_t stream) {
  if (!d) return NPP_NULL_POINTER_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  if (nLength < 0) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  F op; (void)cudaGetLastError();
  kernel::unary_sfs_inplace_kernel<F, A><<<(unsigned)grid, ek::kBlock, 0, stream>>>(d, nLength, op, sf);
  return checkCudaKernelError();
}
template <typename F, typename A, typename D>
NppStatus run_float(const A* src, D* dst, NppSignalLength nLength, cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  if (nLength < 0) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  F op; (void)cudaGetLastError();
  kernel::unary_f_kernel<F, A, D><<<(unsigned)grid, ek::kBlock, 0, stream>>>(src, dst, nLength, op);
  return checkCudaKernelError();
}
template <typename F, typename A>
NppStatus run_float_inplace(A* src, NppSignalLength nLength, cudaStream_t stream) {
  if (!src) return NPP_NULL_POINTER_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  if (nLength < 0) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  F op; (void)cudaGetLastError();
  kernel::unary_f_inplace_kernel<F, A><<<(unsigned)grid, ek::kBlock, 0, stream>>>(src, nLength, op);
  return checkCudaKernelError();
}
template <typename A>
NppStatus run_shift(const A* src, int sh, A* dst, NppSignalLength nLength, char dir, cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (nLength < 0) return NPP_SIZE_ERROR;
  if (nLength == 0) return NPP_NO_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  (void)cudaGetLastError();
  kernel::shift_kernel<A><<<(unsigned)grid, ek::kBlock, 0, stream>>>(src, sh, dst, nLength, dir);
  return checkCudaKernelError();
}
template <typename A>
NppStatus run_shift_inplace(int sh, A* src, NppSignalLength nLength, char dir, cudaStream_t stream) {
  if (!src) return NPP_NULL_POINTER_ERROR;
  if (nLength < 0) return NPP_SIZE_ERROR;
  if (nLength == 0) return NPP_NO_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  (void)cudaGetLastError();
  kernel::shift_inplace_kernel<A><<<(unsigned)grid, ek::kBlock, 0, stream>>>(src, sh, nLength, dir);
  return checkCudaKernelError();
}
template <typename A>
NppStatus run_bitwise(const A* a, const A* b, A* d, NppSignalLength nLength, int op, cudaStream_t stream) {
  if (!a || !b || !d) return NPP_NULL_POINTER_ERROR;
  if (nLength < 0) return NPP_SIZE_ERROR;
  if (nLength == 0) return NPP_NO_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  (void)cudaGetLastError();
  kernel::bitwise_bin_kernel<A><<<(unsigned)grid, ek::kBlock, 0, stream>>>(a, b, d, nLength, op);
  return checkCudaKernelError();
}
template <typename A>
NppStatus run_bitwise_inplace(const A* a, A* d, NppSignalLength nLength, int op, cudaStream_t stream) {
  if (!a || !d) return NPP_NULL_POINTER_ERROR;
  if (nLength < 0) return NPP_SIZE_ERROR;
  if (nLength == 0) return NPP_NO_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  (void)cudaGetLastError();
  kernel::bitwise_bin_inplace_kernel<A><<<(unsigned)grid, ek::kBlock, 0, stream>>>(a, d, nLength, op);
  return checkCudaKernelError();
}
template <typename A>
NppStatus run_not(const A* src, A* dst, NppSignalLength nLength, int inplace, cudaStream_t stream) {
  if (!src || !dst) return NPP_NULL_POINTER_ERROR;
  if (nLength < 0) return NPP_SIZE_ERROR;
  if (nLength == 0) return NPP_NO_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  (void)cudaGetLastError();
  if (inplace) kernel::not_inplace_kernel<A><<<(unsigned)grid, ek::kBlock, 0, stream>>>(dst, nLength);
  else kernel::not_kernel<A><<<(unsigned)grid, ek::kBlock, 0, stream>>>(src, dst, nLength);
  return checkCudaKernelError();
}
template <typename A, typename D>
NppStatus run_sumln(const A* src, NppSignalLength nLength, D* dst, Npp8u* buf, cudaStream_t stream) {
  if (!src || !dst || !buf) return NPP_NULL_POINTER_ERROR;
  if (nLength < 0) return NPP_SIZE_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  D* partial = reinterpret_cast<D*>(buf);
  (void)cudaGetLastError();
  kernel::sumln_partial<A, D><<<(unsigned)grid, ek::kBlock, 0, stream>>>(src, nLength, partial);
  if (checkCudaKernelError() != NPP_NO_ERROR) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  kernel::sumln_final<D><<<1, ek::kBlock, 0, stream>>>(partial, grid, dst);
  return checkCudaKernelError();
}

}  // anonymous namespace

// --- public entry points ---
template <typename F, typename A, typename D>
NppStatus runSfs(const A* src, D* dst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream) {
  return run_scaled<F, A, D>(src, dst, nLength, nScaleFactor, stream);
}
template <typename F, typename A>
NppStatus runSfsI(A* pSrcDst, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream) {
  return run_scaled_i<F, A>(pSrcDst, nLength, nScaleFactor, stream);
}
template <typename F, typename A, typename D>
NppStatus runF(const A* src, D* dst, NppSignalLength nLength, cudaStream_t stream) {
  return run_float<F, A, D>(src, dst, nLength, stream);
}
template <typename F, typename A>
NppStatus runFI(A* pSrcDst, NppSignalLength nLength, cudaStream_t stream) {
  return run_float_inplace<F, A>(pSrcDst, nLength, stream);
}

template <typename A>
NppStatus lShiftC(const A* src, int nValue, A* dst, NppSignalLength nLength, cudaStream_t stream) {
  return run_shift(src, nValue, dst, nLength, 1, stream);
}
template <typename A>
NppStatus rShiftC(const A* src, int nValue, A* dst, NppSignalLength nLength, cudaStream_t stream) {
  return run_shift(src, nValue, dst, nLength, 0, stream);
}
template <typename A>
NppStatus lShiftCI(int nValue, A* pSrcDst, NppSignalLength nLength, cudaStream_t stream) {
  return run_shift_inplace(nValue, pSrcDst, nLength, 1, stream);
}
template <typename A>
NppStatus rShiftCI(int nValue, A* pSrcDst, NppSignalLength nLength, cudaStream_t stream) {
  return run_shift_inplace(nValue, pSrcDst, nLength, 0, stream);
}

template <typename A>
NppStatus bitwiseAnd(const A* a, const A* b, A* d, NppSignalLength nLength, cudaStream_t stream) { return run_bitwise(a, b, d, nLength, 0, stream); }
template <typename A>
NppStatus bitwiseOr(const A* a, const A* b, A* d, NppSignalLength nLength, cudaStream_t stream) { return run_bitwise(a, b, d, nLength, 1, stream); }
template <typename A>
NppStatus bitwiseXor(const A* a, const A* b, A* d, NppSignalLength nLength, cudaStream_t stream) { return run_bitwise(a, b, d, nLength, 2, stream); }
template <typename A>
NppStatus bitwiseAndI(const A* a, A* d, NppSignalLength nLength, cudaStream_t stream) { return run_bitwise_inplace(a, d, nLength, 0, stream); }
template <typename A>
NppStatus bitwiseOrI(const A* a, A* d, NppSignalLength nLength, cudaStream_t stream) { return run_bitwise_inplace(a, d, nLength, 1, stream); }
template <typename A>
NppStatus bitwiseXorI(const A* a, A* d, NppSignalLength nLength, cudaStream_t stream) { return run_bitwise_inplace(a, d, nLength, 2, stream); }

template <typename A> NppStatus notOp(const A* src, A* dst, NppSignalLength n, cudaStream_t st) { return run_not(src, dst, n, 0, st); }
template <typename A> NppStatus notOpI(A* d, NppSignalLength n, cudaStream_t st) { return run_not(d, d, n, 1, st); }

NppStatus mulLow(const Npp32s* a, const Npp32s* b, Npp32s* d, NppSignalLength nLength, int sf, cudaStream_t stream) {
  if (!a || !b || !d) return NPP_NULL_POINTER_ERROR;
  if (nLength < 0) return NPP_SIZE_ERROR;
  if (nLength == 0) return NPP_NO_ERROR;
  const size_t n = (size_t)nLength;
  const size_t grid = (n + ek::kBlock - 1) / ek::kBlock;
  (void)cudaGetLastError();
  kernel::mullow_kernel<<<(unsigned)grid, ek::kBlock, 0, stream>>>(a, b, d, nLength, sf);
  return checkCudaKernelError();
}

template <typename A, typename D>
NppStatus sumLn(const A* src, NppSignalLength nLength, D* dst, Npp8u* buf, cudaStream_t stream) {
  return run_sumln<A, D>(src, nLength, dst, buf, stream);
}
template <typename A>
NppStatus sumLnGetBufferSize(NppSignalLength nLength, int* hpBufferSize) {
  if (!hpBufferSize) return NPP_NULL_POINTER_ERROR;
  if (nLength < 0) return NPP_SIZE_ERROR;
  if (nLength == 0) return NPP_SIZE_ERROR;
  size_t nb = ((size_t)nLength + ek::kBlock - 1) / ek::kBlock;
  *hpBufferSize = (int)(nb * sizeof(Npp64f));
  return NPP_NO_ERROR;
}

NppStatus cauchy(Npp32f* d, NppSignalLength n, Npp32f p, cudaStream_t stream) {
  if (!d) return NPP_NULL_POINTER_ERROR;
  if (n < 0) return NPP_SIZE_ERROR; if (n == 0) return NPP_NO_ERROR;
  const size_t g = ((size_t)n + ek::kBlock - 1) / ek::kBlock;
  (void)cudaGetLastError();
  kernel::cauchy_kernel<<<(unsigned)g, ek::kBlock, 0, stream>>>(d, n, p);
  return checkCudaKernelError();
}
NppStatus cauchyD(Npp32f* d, NppSignalLength n, Npp32f p, cudaStream_t stream) {
  if (!d) return NPP_NULL_POINTER_ERROR;
  if (n < 0) return NPP_SIZE_ERROR; if (n == 0) return NPP_NO_ERROR;
  const size_t g = ((size_t)n + ek::kBlock - 1) / ek::kBlock;
  (void)cudaGetLastError();
  kernel::cauchyD_kernel<<<(unsigned)g, ek::kBlock, 0, stream>>>(d, n, p);
  return checkCudaKernelError();
}
NppStatus cauchyDD2(Npp32f* d, Npp32f* d2, NppSignalLength n, Npp32f p, cudaStream_t stream) {
  if (!d || !d2) return NPP_NULL_POINTER_ERROR;
  if (n < 0) return NPP_SIZE_ERROR; if (n == 0) return NPP_NO_ERROR;
  const size_t g = ((size_t)n + ek::kBlock - 1) / ek::kBlock;
  (void)cudaGetLastError();
  kernel::cauchyDD2_kernel<<<(unsigned)g, ek::kBlock, 0, stream>>>(d, d2, n, p);
  return checkCudaKernelError();
}

// --- explicit instantiations ---
template NppStatus runSfs<SqrtOp,Npp8u,Npp8u>(const Npp8u*, Npp8u*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<SqrtOp,Npp16u,Npp16u>(const Npp16u*, Npp16u*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<SqrtOp,Npp16s,Npp16s>(const Npp16s*, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<SqrtOp,Npp16sc,Npp16sc>(const Npp16sc*, Npp16sc*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<SqrtOp,Npp32s,Npp16s>(const Npp32s*, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<SqrtOp,Npp64s,Npp64s>(const Npp64s*, Npp64s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<SqrtOp,Npp64s,Npp16s>(const Npp64s*, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<SqrtOp,Npp8u>(Npp8u*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<SqrtOp,Npp16u>(Npp16u*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<SqrtOp,Npp16s>(Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<SqrtOp,Npp16sc>(Npp16sc*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<SqrtOp,Npp64s>(Npp64s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<SqrOp,Npp8u,Npp8u>(const Npp8u*, Npp8u*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<SqrOp,Npp16u,Npp16u>(const Npp16u*, Npp16u*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<SqrOp,Npp16s,Npp16s>(const Npp16s*, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<SqrOp,Npp16sc,Npp16sc>(const Npp16sc*, Npp16sc*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<SqrOp,Npp8u>(Npp8u*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<SqrOp,Npp16u>(Npp16u*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<SqrOp,Npp16s>(Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<SqrOp,Npp16sc>(Npp16sc*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<ExpOp,Npp16s,Npp16s>(const Npp16s*, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<ExpOp,Npp32s,Npp32s>(const Npp32s*, Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<ExpOp,Npp64s,Npp64s>(const Npp64s*, Npp64s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<ExpOp,Npp16s>(Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<ExpOp,Npp32s>(Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<ExpOp,Npp64s>(Npp64s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<LnOp,Npp16s,Npp16s>(const Npp16s*, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<LnOp,Npp32s,Npp32s>(const Npp32s*, Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<LnOp,Npp32s,Npp16s>(const Npp32s*, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<LnOp,Npp16s>(Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<LnOp,Npp32s>(Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<Log10Op,Npp32s,Npp32s>(const Npp32s*, Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<Log10Op,Npp32s>(Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<CubrtOp,Npp32s,Npp16s>(const Npp32s*, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<AbsOp,Npp16s,Npp16s>(const Npp16s*, Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfs<AbsOp,Npp32s,Npp32s>(const Npp32s*, Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<AbsOp,Npp16s>(Npp16s*, NppSignalLength, int, cudaStream_t);
template NppStatus runSfsI<AbsOp,Npp32s>(Npp32s*, NppSignalLength, int, cudaStream_t);
template NppStatus runF<SqrtOp,Npp32f,Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runF<SqrtOp,Npp64f,Npp64f>(const Npp64f*, Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runF<SqrtOp,Npp32fc,Npp32fc>(const Npp32fc*, Npp32fc*, NppSignalLength, cudaStream_t);
template NppStatus runF<SqrtOp,Npp64fc,Npp64fc>(const Npp64fc*, Npp64fc*, NppSignalLength, cudaStream_t);
template NppStatus runFI<SqrtOp,Npp32f>(Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<SqrtOp,Npp64f>(Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<SqrtOp,Npp32fc>(Npp32fc*, NppSignalLength, cudaStream_t);
template NppStatus runFI<SqrtOp,Npp64fc>(Npp64fc*, NppSignalLength, cudaStream_t);
template NppStatus runF<SqrOp,Npp32f,Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runF<SqrOp,Npp32fc,Npp32fc>(const Npp32fc*, Npp32fc*, NppSignalLength, cudaStream_t);
template NppStatus runF<SqrOp,Npp64f,Npp64f>(const Npp64f*, Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runF<SqrOp,Npp64fc,Npp64fc>(const Npp64fc*, Npp64fc*, NppSignalLength, cudaStream_t);
template NppStatus runFI<SqrOp,Npp32f>(Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<SqrOp,Npp32fc>(Npp32fc*, NppSignalLength, cudaStream_t);
template NppStatus runFI<SqrOp,Npp64f>(Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<SqrOp,Npp64fc>(Npp64fc*, NppSignalLength, cudaStream_t);
template NppStatus runF<ExpOp,Npp32f,Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runF<ExpOp,Npp32f,Npp64f>(const Npp32f*, Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runF<ExpOp,Npp64f,Npp64f>(const Npp64f*, Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<ExpOp,Npp32f>(Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<ExpOp,Npp64f>(Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runF<LnOp,Npp32f,Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runF<LnOp,Npp64f,Npp64f>(const Npp64f*, Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runF<LnOp,Npp64f,Npp32f>(const Npp64f*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<LnOp,Npp32f>(Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<LnOp,Npp64f>(Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runF<AbsOp,Npp32f,Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runF<AbsOp,Npp64f,Npp64f>(const Npp64f*, Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<AbsOp,Npp32f>(Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<AbsOp,Npp64f>(Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runF<ArctanOp,Npp32f,Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runF<ArctanOp,Npp64f,Npp64f>(const Npp64f*, Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<ArctanOp,Npp32f>(Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus runFI<ArctanOp,Npp64f>(Npp64f*, NppSignalLength, cudaStream_t);
template NppStatus runF<CubrtOp,Npp32f,Npp32f>(const Npp32f*, Npp32f*, NppSignalLength, cudaStream_t);
template NppStatus lShiftC<Npp8u>(const Npp8u*, int, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus rShiftC<Npp8u>(const Npp8u*, int, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus lShiftCI<Npp8u>(int, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus rShiftCI<Npp8u>(int, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus lShiftC<Npp16u>(const Npp16u*, int, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus rShiftC<Npp16u>(const Npp16u*, int, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus lShiftCI<Npp16u>(int, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus rShiftCI<Npp16u>(int, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus lShiftC<Npp16s>(const Npp16s*, int, Npp16s*, NppSignalLength, cudaStream_t);
template NppStatus rShiftC<Npp16s>(const Npp16s*, int, Npp16s*, NppSignalLength, cudaStream_t);
template NppStatus lShiftCI<Npp16s>(int, Npp16s*, NppSignalLength, cudaStream_t);
template NppStatus rShiftCI<Npp16s>(int, Npp16s*, NppSignalLength, cudaStream_t);
template NppStatus lShiftC<Npp32u>(const Npp32u*, int, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus rShiftC<Npp32u>(const Npp32u*, int, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus lShiftCI<Npp32u>(int, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus rShiftCI<Npp32u>(int, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus lShiftC<Npp32s>(const Npp32s*, int, Npp32s*, NppSignalLength, cudaStream_t);
template NppStatus rShiftC<Npp32s>(const Npp32s*, int, Npp32s*, NppSignalLength, cudaStream_t);
template NppStatus lShiftCI<Npp32s>(int, Npp32s*, NppSignalLength, cudaStream_t);
template NppStatus rShiftCI<Npp32s>(int, Npp32s*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseAnd<Npp8u>(const Npp8u*, const Npp8u*, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseOr<Npp8u>(const Npp8u*, const Npp8u*, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseXor<Npp8u>(const Npp8u*, const Npp8u*, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseAndI<Npp8u>(const Npp8u*, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseOrI<Npp8u>(const Npp8u*, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseXorI<Npp8u>(const Npp8u*, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus notOp<Npp8u>(const Npp8u*, Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus notOpI<Npp8u>(Npp8u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseAnd<Npp16u>(const Npp16u*, const Npp16u*, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseOr<Npp16u>(const Npp16u*, const Npp16u*, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseXor<Npp16u>(const Npp16u*, const Npp16u*, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseAndI<Npp16u>(const Npp16u*, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseOrI<Npp16u>(const Npp16u*, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseXorI<Npp16u>(const Npp16u*, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus notOp<Npp16u>(const Npp16u*, Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus notOpI<Npp16u>(Npp16u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseAnd<Npp32u>(const Npp32u*, const Npp32u*, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseOr<Npp32u>(const Npp32u*, const Npp32u*, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseXor<Npp32u>(const Npp32u*, const Npp32u*, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseAndI<Npp32u>(const Npp32u*, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseOrI<Npp32u>(const Npp32u*, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus bitwiseXorI<Npp32u>(const Npp32u*, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus notOp<Npp32u>(const Npp32u*, Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus notOpI<Npp32u>(Npp32u*, NppSignalLength, cudaStream_t);
template NppStatus sumLn<Npp16s,Npp32f>(const Npp16s*, NppSignalLength, Npp32f*, Npp8u*, cudaStream_t);
template NppStatus sumLn<Npp32f,Npp32f>(const Npp32f*, NppSignalLength, Npp32f*, Npp8u*, cudaStream_t);
template NppStatus sumLn<Npp32f,Npp64f>(const Npp32f*, NppSignalLength, Npp64f*, Npp8u*, cudaStream_t);
template NppStatus sumLn<Npp64f,Npp64f>(const Npp64f*, NppSignalLength, Npp64f*, Npp8u*, cudaStream_t);
template NppStatus sumLnGetBufferSize<Npp16s>(NppSignalLength, int*);
template NppStatus sumLnGetBufferSize<Npp32f>(NppSignalLength, int*);
template NppStatus sumLnGetBufferSize<Npp64f>(NppSignalLength, int*);
}  // namespace unary
}  // namespace npps
