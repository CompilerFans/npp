// Templated kernels + executors for the npps initialization families (Set/Copy/
// Zero).  Entry points are declared in npps_initialization_executor.h; each
// template is explicitly instantiated below for every dtype the families use.

#include "npp.h"
#include "npp_version_compat.h"
#include "npps_initialization_executor.h"
#include "npp_kernel_utils.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace npps {
namespace init {

namespace {

const int kBlockSize = 256;

template <typename T>
__global__ void set_kernel(T value, T *dst, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = value;
}

template <typename T>
__global__ void copy_kernel(const T *src, T *dst, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

// Kernel-launch error check moved to the shared header npp_kernel_utils.h
// (package p0.1); re-exported here so the executors below keep using the
// unqualified name.
using npp_kernel::checkCudaKernelError;

// Shared validation for all three families.  nLength is a signed length on the
// MPP side (NppSignalLength == int) and size_t on the reference NPP 13; the
// negative guard below is a compile-time no-op for the size_t case.
// A zero / negative length returns an error; this is re-validated against the
// 5090 golden records.
inline NppStatus validate(const void *ptr, NppSignalLength nLength) {
  if (ptr == nullptr) return NPP_NULL_POINTER_ERROR;
  if (static_cast<long long>(nLength) <= 0) return NPP_SIZE_ERROR;
  return NPP_NO_ERROR;
}

}  // namespace

template <typename T>
NppStatus set(T value, T *pDst, NppSignalLength nLength, cudaStream_t stream) {
  NppStatus st = validate(pDst, nLength);
  if (st != NPP_NO_ERROR) return st;
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + kBlockSize - 1) / kBlockSize;
  set_kernel<<<grid, kBlockSize, 0, stream>>>(value, pDst, n);
  return checkCudaKernelError();
}

template <typename T>
NppStatus copy(const T *pSrc, T *pDst, NppSignalLength nLength, cudaStream_t stream) {
  NppStatus st = validate(pSrc, nLength);
  if (st != NPP_NO_ERROR) return st;
  st = validate(pDst, nLength);
  if (st != NPP_NO_ERROR) return st;
  const size_t n = static_cast<size_t>(nLength);
  const size_t grid = (n + kBlockSize - 1) / kBlockSize;
  copy_kernel<<<grid, kBlockSize, 0, stream>>>(pSrc, pDst, n);
  return checkCudaKernelError();
}

template <typename T>
NppStatus zero(T *pDst, NppSignalLength nLength, cudaStream_t stream) {
  NppStatus st = validate(pDst, nLength);
  if (st != NPP_NO_ERROR) return st;
  const size_t n = static_cast<size_t>(nLength);
  cudaError_t e = cudaMemsetAsync(pDst, 0, n * sizeof(T), stream);
  return (e == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// nppsSet dtypes: 16s 16sc 16u 32s 32sc 32u 64f 64fc 64s 64sc 8s
template NppStatus set<Npp16s>(Npp16s, Npp16s *, NppSignalLength, cudaStream_t);
template NppStatus set<Npp16sc>(Npp16sc, Npp16sc *, NppSignalLength, cudaStream_t);
template NppStatus set<Npp16u>(Npp16u, Npp16u *, NppSignalLength, cudaStream_t);
template NppStatus set<Npp32s>(Npp32s, Npp32s *, NppSignalLength, cudaStream_t);
template NppStatus set<Npp32sc>(Npp32sc, Npp32sc *, NppSignalLength, cudaStream_t);
template NppStatus set<Npp32u>(Npp32u, Npp32u *, NppSignalLength, cudaStream_t);
template NppStatus set<Npp64f>(Npp64f, Npp64f *, NppSignalLength, cudaStream_t);
template NppStatus set<Npp64fc>(Npp64fc, Npp64fc *, NppSignalLength, cudaStream_t);
template NppStatus set<Npp64s>(Npp64s, Npp64s *, NppSignalLength, cudaStream_t);
template NppStatus set<Npp64sc>(Npp64sc, Npp64sc *, NppSignalLength, cudaStream_t);
template NppStatus set<Npp8s>(Npp8s, Npp8s *, NppSignalLength, cudaStream_t);

// nppsCopy dtypes: 8u 16s 16sc 32f 32fc 32s 32sc 64fc 64s 64sc
template NppStatus copy<Npp8u>(const Npp8u *, Npp8u *, NppSignalLength, cudaStream_t);
template NppStatus copy<Npp16s>(const Npp16s *, Npp16s *, NppSignalLength, cudaStream_t);
template NppStatus copy<Npp16sc>(const Npp16sc *, Npp16sc *, NppSignalLength, cudaStream_t);
template NppStatus copy<Npp32f>(const Npp32f *, Npp32f *, NppSignalLength, cudaStream_t);
template NppStatus copy<Npp32fc>(const Npp32fc *, Npp32fc *, NppSignalLength, cudaStream_t);
template NppStatus copy<Npp32s>(const Npp32s *, Npp32s *, NppSignalLength, cudaStream_t);
template NppStatus copy<Npp32sc>(const Npp32sc *, Npp32sc *, NppSignalLength, cudaStream_t);
template NppStatus copy<Npp64fc>(const Npp64fc *, Npp64fc *, NppSignalLength, cudaStream_t);
template NppStatus copy<Npp64s>(const Npp64s *, Npp64s *, NppSignalLength, cudaStream_t);
template NppStatus copy<Npp64sc>(const Npp64sc *, Npp64sc *, NppSignalLength, cudaStream_t);

// nppsZero dtypes: 16s 16sc 32fc 32s 32sc 64f 64fc 64s 64sc
template NppStatus zero<Npp16s>(Npp16s *, NppSignalLength, cudaStream_t);
template NppStatus zero<Npp16sc>(Npp16sc *, NppSignalLength, cudaStream_t);
template NppStatus zero<Npp32fc>(Npp32fc *, NppSignalLength, cudaStream_t);
template NppStatus zero<Npp32s>(Npp32s *, NppSignalLength, cudaStream_t);
template NppStatus zero<Npp32sc>(Npp32sc *, NppSignalLength, cudaStream_t);
template NppStatus zero<Npp64f>(Npp64f *, NppSignalLength, cudaStream_t);
template NppStatus zero<Npp64fc>(Npp64fc *, NppSignalLength, cudaStream_t);
template NppStatus zero<Npp64s>(Npp64s *, NppSignalLength, cudaStream_t);
template NppStatus zero<Npp64sc>(Npp64sc *, NppSignalLength, cudaStream_t);

}  // namespace init
}  // namespace npps
