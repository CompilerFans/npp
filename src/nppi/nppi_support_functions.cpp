#include "npp.h"
#include <cstdio>
#include <cuda_runtime.h>
template <typename T>

static T *nppiMallocTemplate(int nWidthPixels, int nHeightPixels, int nChannels, int *pStepBytes) {
  if (nWidthPixels <= 0 || nHeightPixels <= 0 || nChannels <= 0 || !pStepBytes) {
    return nullptr;
  }

  void *devPtr = nullptr;
  size_t pitch = 0;

  size_t widthInBytes = nWidthPixels * nChannels * sizeof(T);
  cudaError_t result = cudaMallocPitch(&devPtr, &pitch, widthInBytes, nHeightPixels);

  if (result != cudaSuccess) {
    return nullptr;
  }

  // Initialize allocated memory to zero to prevent NaN/Inf values
  cudaMemset2D(devPtr, pitch, 0, widthInBytes, nHeightPixels);

  *pStepBytes = static_cast<int>(pitch);
  return static_cast<T *>(devPtr);
}

// 8-bit unsigned image memory allocators

Npp8u *nppiMalloc_8u_C1(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp8u>(nWidthPixels, nHeightPixels, 1, pStepBytes);
}

Npp8u *nppiMalloc_8u_C2(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp8u>(nWidthPixels, nHeightPixels, 2, pStepBytes);
}

Npp8u *nppiMalloc_8u_C3(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp8u>(nWidthPixels, nHeightPixels, 3, pStepBytes);
}

Npp8u *nppiMalloc_8u_C4(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp8u>(nWidthPixels, nHeightPixels, 4, pStepBytes);
}

// 16-bit unsigned image memory allocators

Npp16u *nppiMalloc_16u_C1(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp16u>(nWidthPixels, nHeightPixels, 1, pStepBytes);
}

Npp16u *nppiMalloc_16u_C2(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp16u>(nWidthPixels, nHeightPixels, 2, pStepBytes);
}

Npp16u *nppiMalloc_16u_C3(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp16u>(nWidthPixels, nHeightPixels, 3, pStepBytes);
}

Npp16u *nppiMalloc_16u_C4(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp16u>(nWidthPixels, nHeightPixels, 4, pStepBytes);
}

// 16-bit signed image memory allocators

Npp16s *nppiMalloc_16s_C1(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp16s>(nWidthPixels, nHeightPixels, 1, pStepBytes);
}

Npp16s *nppiMalloc_16s_C2(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp16s>(nWidthPixels, nHeightPixels, 2, pStepBytes);
}

Npp16s *nppiMalloc_16s_C4(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp16s>(nWidthPixels, nHeightPixels, 4, pStepBytes);
}

// 16-bit signed complex image memory allocators

Npp16sc *nppiMalloc_16sc_C1(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp16sc>(nWidthPixels, nHeightPixels, 1, pStepBytes);
}

Npp16sc *nppiMalloc_16sc_C2(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp16sc>(nWidthPixels, nHeightPixels, 2, pStepBytes);
}

Npp16sc *nppiMalloc_16sc_C3(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp16sc>(nWidthPixels, nHeightPixels, 3, pStepBytes);
}

Npp16sc *nppiMalloc_16sc_C4(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp16sc>(nWidthPixels, nHeightPixels, 4, pStepBytes);
}

// 32-bit signed image memory allocators

Npp32s *nppiMalloc_32s_C1(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32s>(nWidthPixels, nHeightPixels, 1, pStepBytes);
}

Npp32s *nppiMalloc_32s_C3(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32s>(nWidthPixels, nHeightPixels, 3, pStepBytes);
}

Npp32s *nppiMalloc_32s_C4(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32s>(nWidthPixels, nHeightPixels, 4, pStepBytes);
}

// 32-bit signed complex image memory allocators

Npp32sc *nppiMalloc_32sc_C1(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32sc>(nWidthPixels, nHeightPixels, 1, pStepBytes);
}

Npp32sc *nppiMalloc_32sc_C2(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32sc>(nWidthPixels, nHeightPixels, 2, pStepBytes);
}

Npp32sc *nppiMalloc_32sc_C3(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32sc>(nWidthPixels, nHeightPixels, 3, pStepBytes);
}

Npp32sc *nppiMalloc_32sc_C4(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32sc>(nWidthPixels, nHeightPixels, 4, pStepBytes);
}

// 32-bit float image memory allocators

Npp32f *nppiMalloc_32f_C1(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32f>(nWidthPixels, nHeightPixels, 1, pStepBytes);
}

Npp32f *nppiMalloc_32f_C2(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32f>(nWidthPixels, nHeightPixels, 2, pStepBytes);
}

Npp32f *nppiMalloc_32f_C3(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32f>(nWidthPixels, nHeightPixels, 3, pStepBytes);
}

Npp32f *nppiMalloc_32f_C4(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32f>(nWidthPixels, nHeightPixels, 4, pStepBytes);
}

// 32-bit float complex image memory allocators

Npp32fc *nppiMalloc_32fc_C1(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32fc>(nWidthPixels, nHeightPixels, 1, pStepBytes);
}

Npp32fc *nppiMalloc_32fc_C2(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32fc>(nWidthPixels, nHeightPixels, 2, pStepBytes);
}

Npp32fc *nppiMalloc_32fc_C3(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32fc>(nWidthPixels, nHeightPixels, 3, pStepBytes);
}

Npp32fc *nppiMalloc_32fc_C4(int nWidthPixels, int nHeightPixels, int *pStepBytes) {
  return nppiMallocTemplate<Npp32fc>(nWidthPixels, nHeightPixels, 4, pStepBytes);
}

void nppiFree(void *pData) {
  if (pData) {
    cudaFree(pData);
  }
}
