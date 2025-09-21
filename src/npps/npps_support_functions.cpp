#include "npp.h"
#include <cstdio>
#include <cuda_runtime.h>

template <typename T> static T *nppsMallocTemplate(size_t nSize) {
  if (nSize == 0) {
    return nullptr;
  }

  void *devPtr = nullptr;
  size_t sizeInBytes = nSize * sizeof(T);

  cudaError_t result = cudaMalloc(&devPtr, sizeInBytes);

  if (result != cudaSuccess) {
    return nullptr;
  }

  return static_cast<T *>(devPtr);
}

// 8-bit signal memory allocators
Npp8u *nppsMalloc_8u(size_t nSize) { return nppsMallocTemplate<Npp8u>(nSize); }

Npp8s *nppsMalloc_8s(size_t nSize) { return nppsMallocTemplate<Npp8s>(nSize); }

// 16-bit signal memory allocators
Npp16u *nppsMalloc_16u(size_t nSize) { return nppsMallocTemplate<Npp16u>(nSize); }

Npp16s *nppsMalloc_16s(size_t nSize) { return nppsMallocTemplate<Npp16s>(nSize); }

Npp16sc *nppsMalloc_16sc(size_t nSize) { return nppsMallocTemplate<Npp16sc>(nSize); }

// 32-bit signal memory allocators
Npp32u *nppsMalloc_32u(size_t nSize) { return nppsMallocTemplate<Npp32u>(nSize); }

Npp32s *nppsMalloc_32s(size_t nSize) { return nppsMallocTemplate<Npp32s>(nSize); }

Npp32sc *nppsMalloc_32sc(size_t nSize) { return nppsMallocTemplate<Npp32sc>(nSize); }

Npp32f *nppsMalloc_32f(size_t nSize) { return nppsMallocTemplate<Npp32f>(nSize); }

Npp32fc *nppsMalloc_32fc(size_t nSize) { return nppsMallocTemplate<Npp32fc>(nSize); }

// 64-bit signal memory allocators
Npp64s *nppsMalloc_64s(size_t nSize) { return nppsMallocTemplate<Npp64s>(nSize); }

Npp64sc *nppsMalloc_64sc(size_t nSize) { return nppsMallocTemplate<Npp64sc>(nSize); }

Npp64f *nppsMalloc_64f(size_t nSize) { return nppsMallocTemplate<Npp64f>(nSize); }

Npp64fc *nppsMalloc_64fc(size_t nSize) { return nppsMallocTemplate<Npp64fc>(nSize); }

void nppsFree(void *pValues) {
  if (pValues) {
    cudaFree(pValues);
  }
}
