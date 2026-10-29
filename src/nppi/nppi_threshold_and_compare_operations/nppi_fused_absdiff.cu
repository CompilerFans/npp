// Fused absdiff + Greater-Than-Value threshold entry points (package p2.3).
// Runtime-dispatched over NppDataType (8U/16U/16S/32F) and NppiChannels
// (CH_1 / CH_3 / CH_A4).  Semantics baseline: NVIDIA NPP 13.4 on RTX 5090.
//
//   out = absdiff(a, b);   if (out > thr[c]) out = val[c];  else out = absdiff
// For CH_A4 the operation is applied to RGB and the alpha element is copied
// through from the first source (matches the AC4 sibling conventions).
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace {

template <typename T> __device__ __forceinline__ T fabsDiff(T a, T b) {
  return (T)(a >= b ? a - b : b - a);
}

// Threshold / value arrays are copied by value into the kernel.
template <typename T> struct FusedData { T thr[4]; T val[4]; };

// PHY  = physical channels / pixel (1, 3 or 4)
// ARR  = array elements for threshold and value (CH_1 -> 1, CH_3 -> 3, CH_A4 -> 3)
template <typename T, int PHY, int ARR>
__global__ void kFused(const T* p1, int s1, const T* p2, int s2, T* pDst, int sD,
                       int w, int h, FusedData<T> fd) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const T* a = (const T*)((const char*)p1 + (size_t)y * s1);
  const T* b = (const T*)((const char*)p2 + (size_t)y * s2);
  T* const d = (T*)((char*)pDst + (size_t)y * sD);
  for (int c = 0; c < PHY; ++c) {
    if (PHY == 4 && c == 3) continue;  // A4 alpha element is left untouched
    const int ti = c;
    const T ad = fabsDiff(a[x * PHY + c], b[x * PHY + c]);
    d[x * PHY + c] = (ad > fd.thr[ti]) ? fd.val[ti] : ad;
  }
}

template <typename T, int PHY, int ARR>
inline NppStatus fusedRun(const void* p1, int s1, const void* p2, int s2, void* pDst, int sD,
                          int w, int h, const void* thr, const void* val, cudaStream_t st) {
  if (w <= 0 || h <= 0) return NPP_NO_ERROR;
  FusedData<T> fd;
  for (int i = 0; i < ARR; ++i) { fd.thr[i] = ((const T*)thr)[i]; fd.val[i] = ((const T*)val)[i]; }
  dim3 b(16, 16), g((unsigned)((w + 15) / 16), (unsigned)((h + 15) / 16));
  (void)cudaGetLastError();
  kFused<T, PHY, ARR><<<g, b, 0, st>>>((const T*)p1, s1, (const T*)p2, s2, (T*)pDst, sD,
                                       w, h, fd);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <typename T, int PHY, int ARR>
inline NppStatus fusedRunI(const void* pD, int sD, const void* p2, int s2,
                           int w, int h, const void* thr, const void* val, cudaStream_t st) {
  if (w <= 0 || h <= 0) return NPP_NO_ERROR;
  FusedData<T> fd;
  for (int i = 0; i < ARR; ++i) { fd.thr[i] = ((const T*)thr)[i]; fd.val[i] = ((const T*)val)[i]; }
  dim3 b(16, 16), g((unsigned)((w + 15) / 16), (unsigned)((h + 15) / 16));
  (void)cudaGetLastError();
  kFused<T, PHY, ARR><<<g, b, 0, st>>>((const T*)pD, sD, (const T*)p2, s2, (T*)pD, sD,
                                       w, h, fd);
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

struct ChConfig { int phy; int arr; };

inline bool channelsOk(NppiChannels ch, ChConfig& out) {
  switch (ch) {
    case NPP_CH_1:  out = ChConfig{1, 1}; return true;
    case NPP_CH_3:  out = ChConfig{3, 3}; return true;
    case NPP_CH_A4: out = ChConfig{4, 3}; return true;
    default: return false;
  }
}

}  // namespace

extern "C" {

NppStatus nppiFusedAbsDiff_Threshold_GTVal_Ctx_impl(NppDataType eSrcDstType, NppiChannels eSrcDstChannels,
                                                    const void* pSrc1, int nSrc1Step,
                                                    const void* pSrc2, int nSrc2Step,
                                                    void* pDst, int nDstStep, NppiSize oSizeROI,
                                                    const void* pThreshold, const void* pvalue,
                                                    NppStreamContext nppStreamCtx) {
  ChConfig cc;
  if (!channelsOk(eSrcDstChannels, cc))
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  auto run = [&](auto TTag) {
    using T = decltype(TTag);
    switch (cc.arr) {
      case 1:  return fusedRun<T, 1, 1>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                                        oSizeROI.width, oSizeROI.height, pThreshold, pvalue, nppStreamCtx.hStream);
      default:
        if (cc.phy == 3) return fusedRun<T, 3, 3>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                                                  oSizeROI.width, oSizeROI.height, pThreshold, pvalue, nppStreamCtx.hStream);
        return fusedRun<T, 4, 3>(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep,
                                 oSizeROI.width, oSizeROI.height, pThreshold, pvalue, nppStreamCtx.hStream);
    }
  };
  switch (eSrcDstType) {
    case NPP_8U:  return run(Npp8u{});
    case NPP_16U: return run(Npp16u{});
    case NPP_16S: return run(Npp16s{});
    case NPP_32F: return run(Npp32f{});
    default: return NPP_NOT_SUPPORTED_MODE_ERROR;
  }
}

NppStatus nppiFusedAbsDiff_Threshold_GTVal_I_Ctx_impl(NppDataType eSrcDstType, NppiChannels eSrcDstChannels,
                                                      void* pSrcDst, int nSrcDstStep,
                                                      const void* pSrc2, int nSrc2Step,
                                                      NppiSize oSizeROI,
                                                      const void* pThreshold, const void* pvalue,
                                                      NppStreamContext nppStreamCtx) {
  ChConfig cc;
  if (!channelsOk(eSrcDstChannels, cc))
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  auto run = [&](auto TTag) {
    using T = decltype(TTag);
    switch (cc.arr) {
      case 1:  return fusedRunI<T, 1, 1>(pSrcDst, nSrcDstStep, pSrc2, nSrc2Step,
                                         oSizeROI.width, oSizeROI.height, pThreshold, pvalue, nppStreamCtx.hStream);
      default:
        if (cc.phy == 3) return fusedRunI<T, 3, 3>(pSrcDst, nSrcDstStep, pSrc2, nSrc2Step,
                                                   oSizeROI.width, oSizeROI.height, pThreshold, pvalue, nppStreamCtx.hStream);
        return fusedRunI<T, 4, 3>(pSrcDst, nSrcDstStep, pSrc2, nSrc2Step,
                                  oSizeROI.width, oSizeROI.height, pThreshold, pvalue, nppStreamCtx.hStream);
    }
  };
  switch (eSrcDstType) {
    case NPP_8U:  return run(Npp8u{});
    case NPP_16U: return run(Npp16u{});
    case NPP_16S: return run(Npp16s{});
    case NPP_32F: return run(Npp32f{});
    default: return NPP_NOT_SUPPORTED_MODE_ERROR;
  }
}

}  // extern "C"
