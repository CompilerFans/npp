// Morphological operations executor (package m1).  See nppi_morph_executor.h
// for the measured NVIDIA NPP 13.4 semantics reproduced here.
#include "nppi_morph_executor.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace nppi {
namespace morph {
namespace {

__constant__ Npp8u c_box3[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

inline int divUp(int a, int b) { return (a + b - 1) / b; }

// Replicate clamp of an image coordinate, mapped back to the pSrc-relative
// index used by the kernels (origin = pSrc - oSrcOffset).
__device__ __forceinline__ int clampImage(int v, int lo, int hi) {
  v = v < lo ? lo : v;
  return v > hi ? hi : v;
}

// 8u square odd masks >= 5 rows with a centred anchor are read with an
// effective anchor of (-1,-1) by NPP's 8u kernels (measured 5x5 and 7x7).
__host__ inline void effectiveAnchor(int elemBytes, int mw, int mh, int *ax, int *ay) {
  if (elemBytes == 1 && mw == mh && (mw & 1) && mw >= 5 && *ax == mw / 2 && *ay == mh / 2) {
    *ax = -1;
    *ay = -1;
  }
}

// Row clamp bound of the plain (non-border) read path.
__host__ inline int plainRowMax(int roiH, int mh, int ay) {
  const int bound = (mh <= 3) ? (roiH - mh + ay) : (roiH - 1 + ay);
  return bound;
}

// ---------------------------------------------------------------------------
// Masked min/max kernel (plain and border paths).
// ---------------------------------------------------------------------------
template <typename T, int C, bool AC4, bool BORDER>
__global__ void morphMaskKernel(const T *__restrict__ pSrc, int nSrcStep, T *__restrict__ pDst, int nDstStep,
                                int W, int H, const Npp8u *__restrict__ pMask, int mw, int mh, int ax, int ay,
                                int srcW, int srcH, int offX, int offY, int rowMax, int op) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;

  enum { NCH = AC4 ? (C > 1 ? C - 1 : 1) : C };
  T acc[NCH];
  bool any = false;
  T *drow = (T *)((char *)pDst + (size_t)y * nDstStep) + (size_t)x * C;

  for (int my = 0; my < mh; ++my) {
    int cy;
    if (BORDER) {
      cy = clampImage(y + offY + my - ay, 0, srcH - 1) - offY;
    } else {
      cy = y + my - ay;
      cy = cy > rowMax ? rowMax : cy;
      cy = cy < -1 ? -1 : cy;
    }
    const char *row = (const char *)pSrc + (ptrdiff_t)cy * nSrcStep;
    for (int mx = 0; mx < mw; ++mx) {
      if (!pMask[my * mw + mx]) continue;
      int cx;
      if (BORDER) {
        cx = clampImage(x + offX + mx - ax, 0, srcW - 1) - offX;
      } else {
        cx = x + mx - ax;
        const int hi = W - 1 + ax;
        cx = cx > hi ? hi : cx;
        cx = cx < -1 ? -1 : cx;
      }
      const T *p = (const T *)row + (ptrdiff_t)cx * C;
#pragma unroll
      for (int k = 0; k < NCH; ++k) {
        const T v = p[k];
        if (!any) {
          acc[k] = v;
        } else if (op == 0) {
          acc[k] = (v > acc[k]) ? v : acc[k];
        } else {
          acc[k] = (v < acc[k]) ? v : acc[k];
        }
      }
      any = true;
    }
  }
  if (!any) {
    // An empty structuring element zero-fills the processed channels; AC4 keeps
    // its alpha plane untouched.
#pragma unroll
    for (int k = 0; k < NCH; ++k) drow[k] = (T)0;
    return;
  }
#pragma unroll
  for (int k = 0; k < NCH; ++k) drow[k] = acc[k];
}

// ---------------------------------------------------------------------------
// Gray (signed / float structuring element) border kernel.
// ---------------------------------------------------------------------------
template <typename T, typename M, bool SAT>
__global__ void grayKernel(const T *__restrict__ pSrc, int nSrcStep, T *__restrict__ pDst, int nDstStep, int W, int H,
                           const M *__restrict__ pMask, int mw, int mh, int ax, int ay, int srcW, int srcH, int offX,
                           int offY, int op) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;

  double best = 0.0;
  bool any = false;
  for (int my = 0; my < mh; ++my) {
    const int cy = clampImage(y + offY + my - ay, 0, srcH - 1) - offY;
    const char *row = (const char *)pSrc + (ptrdiff_t)cy * nSrcStep;
    for (int mx = 0; mx < mw; ++mx) {
      const int cx = clampImage(x + offX + mx - ax, 0, srcW - 1) - offX;
      const double v = (double)(((const T *)row)[cx]) + (double)pMask[my * mw + mx];
      if (!any) {
        best = v;
      } else if (op == 0) {
        best = v > best ? v : best;
      } else {
        best = v < best ? v : best;
      }
      any = true;
    }
  }
  T *d = (T *)((char *)pDst + (size_t)y * nDstStep) + (size_t)x;
  if (!any) {
    *d = (T)0;
    return;
  }
  double r = best;
  if (SAT) {
    r = r < 0.0 ? 0.0 : r;
    r = r > 255.0 ? 255.0 : r;
  }
  *d = (T)r;
}

// ---------------------------------------------------------------------------
// Residual kernel for TopHat / BlackHat / Gradient.
//   mode 0: dst = src - morph  (TopHat)
//   mode 1: dst = morph - src  (BlackHat, Gradient)
// The arithmetic is the destination type's modular arithmetic; float types use
// plain subtraction.
// ---------------------------------------------------------------------------
template <typename T, int C, bool AC4>
__global__ void residualKernel(const T *__restrict__ pSrc, int nSrcStep, const T *__restrict__ pMorph, int nMorphStep,
                               T *__restrict__ pDst, int nDstStep, int W, int H, int mode) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  enum { NCH = AC4 ? (C > 1 ? C - 1 : 1) : C };
  const T *s = (const T *)((const char *)pSrc + (size_t)y * nSrcStep) + (size_t)x * C;
  const T *m = (const T *)((const char *)pMorph + (size_t)y * nMorphStep) + (size_t)x * C;
  T *d = (T *)((char *)pDst + (size_t)y * nDstStep) + (size_t)x * C;
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    const T a = mode == 0 ? s[k] : m[k];
    const T b = mode == 0 ? m[k] : s[k];
    d[k] = (T)(a - b);
  }
}

// ---------------------------------------------------------------------------
// Validation helpers (order matches the recorded NVIDIA goldens).
// ---------------------------------------------------------------------------
inline bool isBadRoi(NppiSize roi) { return roi.width <= 0 || roi.height <= 0; }

inline bool dstStepBad(int nDstStep, int rowBytes) { return nDstStep < rowBytes; }

inline bool stepNotEven(int nStep, int elemBytes) { return elemBytes > 1 && (nStep % elemBytes) != 0; }

inline bool anchorOutOfRange(NppiPoint a, NppiSize m) {
  // Measured acceptance window of the masked plain variants: one past the mask
  // box on the high side is accepted, five is rejected.
  return a.x < -(m.width + 1) || a.x > m.width + 1 || a.y < -(m.height + 1) || a.y > m.height + 1;
}

inline bool borderUnsupported(int elemBytes, NppiBorderType t) {
  // 8u masked border variants reject anything but replicate; 16u/32f accept
  // (and ignore) other modes, as measured.
  return elemBytes == 1 && t != NPP_BORDER_REPLICATE;
}

inline bool srcGeometryBad(NppiSize srcSize, NppiPoint off) {
  return srcSize.width <= 0 || srcSize.height <= 0 || off.x < 0 || off.y < 0 || off.x >= srcSize.width ||
         off.y >= srcSize.height;
}

template <typename T, int C, bool AC4>
inline NppStatus validateMasked(const void *pSrc, const void *pDst, NppiSize roi, const void *pMask, NppiSize ms,
                                NppiPoint a, int nDstStep) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (isBadRoi(roi)) return NPP_SIZE_ERROR;
  if (!pMask) return NPP_NULL_POINTER_ERROR;
  if (ms.width < 0 || ms.height < 0) return NPP_OUT_OFF_RANGE_ERROR;
  if (anchorOutOfRange(a, ms)) return NPP_OUT_OFF_RANGE_ERROR;
  const int rowBytes = roi.width * C * (int)sizeof(T);
  if (dstStepBad(nDstStep, rowBytes)) return NPP_STEP_ERROR;
  if (stepNotEven(nDstStep, (int)sizeof(T))) return NPP_NOT_EVEN_STEP_ERROR;
  return NPP_NO_ERROR;
}

// Scratch buffer geometry: NPP returns align(roi.width*C*elem,64)*roi.height+8.
inline size_t bufferPitch(NppiSize roi, int elemBytes) {
  const size_t raw = (size_t)roi.width * elemBytes;
  return (raw + 63) & ~(size_t)63;
}

template <typename T, int C, bool AC4>
inline NppStatus launchMasked(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int W, int H, const Npp8u *pMask,
                              int mw, int mh, int ax, int ay, int srcW, int srcH, int offX, int offY, int rowMax,
                              int op, bool border, cudaStream_t stream) {
  dim3 block(16, 8);
  dim3 grid(divUp(W, block.x), divUp(H, block.y));
  if (border) {
    morphMaskKernel<T, C, AC4, true><<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, W, H, pMask, mw, mh,
                                                                ax, ay, srcW, srcH, offX, offY, rowMax, op);
  } else {
    morphMaskKernel<T, C, AC4, false><<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, W, H, pMask, mw, mh,
                                                                 ax, ay, srcW, srcH, offX, offY, rowMax, op);
  }
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <typename T, int C, bool AC4>
inline NppStatus launchGray(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int W, int H, const void *pMask,
                            int mw, int mh, int ax, int ay, int srcW, int srcH, int offX, int offY, int op,
                            cudaStream_t stream) {
  dim3 block(16, 8);
  dim3 grid(divUp(W, block.x), divUp(H, block.y));
  const bool sat = sizeof(T) == 1;
  if (sizeof(T) == 4) {
    grayKernel<T, float, false><<<grid, block, 0, stream>>>((const T *)pSrc, nSrcStep, pDst, nDstStep, W, H,
                                                            (const float *)pMask, mw, mh, ax, ay, srcW, srcH, offX,
                                                            offY, op);
  } else if (sat) {
    grayKernel<T, Npp32s, true><<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, W, H,
                                                           (const Npp32s *)pMask, mw, mh, ax, ay, srcW, srcH, offX,
                                                           offY, op);
  } else {
    grayKernel<T, Npp32s, false><<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, W, H,
                                                            (const Npp32s *)pMask, mw, mh, ax, ay, srcW, srcH, offX,
                                                            offY, op);
  }
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------------
template <typename T, int C, bool AC4>
NppStatus runPlain(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                   NppiSize oMaskSize, NppiPoint oAnchor, int op, cudaStream_t stream) {
  (void)nSrcStep;
  NppStatus st = validateMasked<T, C, AC4>(pSrc, pDst, oSizeROI, pMask, oMaskSize, oAnchor, nDstStep);
  if (st != NPP_NO_ERROR) return st;
  int ax = oAnchor.x, ay = oAnchor.y;
  effectiveAnchor((int)sizeof(T), oMaskSize.width, oMaskSize.height, &ax, &ay);
  // An empty mask (zero area or all-zero taps) zero-fills the ROI; the kernel
  // handles both uniformly.
  const int mw = oMaskSize.width > 0 ? oMaskSize.width : 0;
  const int mh = oMaskSize.height > 0 ? oMaskSize.height : 0;
  if (mw == 0 || mh == 0) {
    const size_t rowBytes = (size_t)oSizeROI.width * C * sizeof(T);
    cudaMemset2DAsync(pDst, (size_t)nDstStep, 0, rowBytes, oSizeROI.height, stream);
    return NPP_SUCCESS;
  }
  const int rowMax = plainRowMax(oSizeROI.height, mh, ay);
  return launchMasked<T, C, AC4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, mw, mh, ax,
                                 ay, oSizeROI.width, oSizeROI.height, 0, 0, rowMax, op, false, stream);
}

template <typename T, int C, bool AC4>
NppStatus runBorder(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, T *pDst, int nDstStep,
                    NppiSize oSizeROI, const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                    NppiBorderType eBorderType, int op, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (isBadRoi(oSizeROI)) return NPP_SIZE_ERROR;
  if (srcGeometryBad(oSrcSize, oSrcOffset)) return NPP_OUT_OFF_RANGE_ERROR;
  if (!pMask) return NPP_NULL_POINTER_ERROR;
  if (oMaskSize.width < 0 || oMaskSize.height < 0) return NPP_OUT_OFF_RANGE_ERROR;
  if (anchorOutOfRange(oAnchor, oMaskSize)) return NPP_OUT_OFF_RANGE_ERROR;
  if (borderUnsupported((int)sizeof(T), eBorderType)) return NPP_NOT_SUPPORTED_MODE_ERROR;
  const int rowBytes = oSizeROI.width * C * (int)sizeof(T);
  if (dstStepBad(nDstStep, rowBytes)) return NPP_STEP_ERROR;
  if (stepNotEven(nDstStep, (int)sizeof(T))) return NPP_NOT_EVEN_STEP_ERROR;

  const int mw = oMaskSize.width > 0 ? oMaskSize.width : 0;
  const int mh = oMaskSize.height > 0 ? oMaskSize.height : 0;
  if (mw == 0 || mh == 0) {
    cudaMemset2DAsync(pDst, (size_t)nDstStep, 0, (size_t)oSizeROI.width * C * sizeof(T), oSizeROI.height, stream);
    return NPP_SUCCESS;
  }
  return launchMasked<T, C, AC4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, mw, mh,
                                 oAnchor.x, oAnchor.y, oSrcSize.width, oSrcSize.height, oSrcOffset.x,
                                 oSrcOffset.y, 0, op, true, stream);
}

template <typename T, int C, bool AC4>
NppStatus run3x3Plain(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, NppiSize oSizeROI, int op,
                      cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (isBadRoi(oSizeROI)) return NPP_SIZE_ERROR;
  const int rowBytes = oSizeROI.width * C * (int)sizeof(T);
  if (dstStepBad(nDstStep, rowBytes)) return NPP_STEP_ERROR;
  if (stepNotEven(nDstStep, (int)sizeof(T))) return NPP_NOT_EVEN_STEP_ERROR;
  return launchMasked<T, C, AC4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, c_box3, 3, 3, 1, 1,
                                 oSizeROI.width, oSizeROI.height, 0, 0, plainRowMax(oSizeROI.height, 3, 1), op,
                                 false, stream);
}

template <typename T, int C, bool AC4>
NppStatus run3x3Border(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, T *pDst, int nDstStep,
                       NppiSize oSizeROI, NppiBorderType eBorderType, int op, cudaStream_t stream) {
  (void)eBorderType;  // measured: the 3x3 border family never validates the mode
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (isBadRoi(oSizeROI)) return NPP_SIZE_ERROR;
  if (srcGeometryBad(oSrcSize, oSrcOffset)) return NPP_OUT_OFF_RANGE_ERROR;
  const int rowBytes = oSizeROI.width * C * (int)sizeof(T);
  if (dstStepBad(nDstStep, rowBytes)) return NPP_STEP_ERROR;
  if (stepNotEven(nDstStep, (int)sizeof(T))) return NPP_NOT_EVEN_STEP_ERROR;
  return launchMasked<T, C, AC4>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, c_box3, 3, 3, 1, 1,
                                 oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, 0, op, true, stream);
}

template <typename T, int C, bool AC4>
NppStatus runComposite(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, T *pDst, int nDstStep,
                       NppiSize oSizeROI, const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
                       NppiBorderType eBorderType, int kind, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (!pBuffer) return NPP_NULL_POINTER_ERROR;
  if (isBadRoi(oSizeROI)) return NPP_SIZE_ERROR;
  if (srcGeometryBad(oSrcSize, oSrcOffset)) return NPP_OUT_OFF_RANGE_ERROR;
  if (!pMask) return NPP_NULL_POINTER_ERROR;
  if (oMaskSize.width < 0 || oMaskSize.height < 0) return NPP_OUT_OFF_RANGE_ERROR;
  if (anchorOutOfRange(oAnchor, oMaskSize)) return NPP_OUT_OFF_RANGE_ERROR;
  if (borderUnsupported((int)sizeof(T), eBorderType)) return NPP_NOT_SUPPORTED_MODE_ERROR;
  const int rowBytes = oSizeROI.width * C * (int)sizeof(T);
  if (dstStepBad(nDstStep, rowBytes)) return NPP_STEP_ERROR;
  if (stepNotEven(nDstStep, (int)sizeof(T))) return NPP_NOT_EVEN_STEP_ERROR;

  const int W = oSizeROI.width, H = oSizeROI.height;
  const int mw = oMaskSize.width > 0 ? oMaskSize.width : 0;
  const int mh = oMaskSize.height > 0 ? oMaskSize.height : 0;
  if (mw == 0 || mh == 0) {
    cudaMemset2DAsync(pDst, (size_t)nDstStep, 0, (size_t)rowBytes, H, stream);
    return NPP_SUCCESS;
  }
  // First pass writes into the caller's scratch buffer (ROI geometry).
  T *tmp = (T *)pBuffer;
  const int tmpStep = (int)(bufferPitch(oSizeROI, C * (int)sizeof(T)));
  const int firstOp = (kind == 0 || kind == 3) ? kErode : kDilate;  // Open/TopHat erode first
  NppStatus st = launchMasked<T, C, AC4>(pSrc, nSrcStep, tmp, tmpStep, W, H, pMask, mw, mh, oAnchor.x, oAnchor.y,
                                         oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y, 0, firstOp,
                                         true, stream);
  if (st != NPP_NO_ERROR) return st;
  const int secondOp = 1 - firstOp;
  T *second = (kind == 0 || kind == 1) ? pDst : tmp;
  st = launchMasked<T, C, AC4>(tmp, tmpStep, second, (kind == 0 || kind == 1) ? nDstStep : tmpStep, W, H, pMask, mw,
                               mh, oAnchor.x, oAnchor.y, W, H, 0, 0, 0, secondOp, true, stream);
  if (st != NPP_NO_ERROR) return st;
  if (kind == 0 || kind == 1) return NPP_SUCCESS;  // Open / Close already in pDst

  const int mode = (kind == 3) ? 0 : 1;  // TopHat: src - morph; else morph - src
  dim3 block(16, 8);
  dim3 grid(divUp(W, block.x), divUp(H, block.y));
  residualKernel<T, C, AC4><<<grid, block, 0, stream>>>(pSrc, nSrcStep, tmp, tmpStep, pDst, nDstStep, W, H, mode);
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <typename T, typename M>
NppStatus runGray(const T *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, T *pDst, int nDstStep,
                  NppiSize oSizeROI, const M *pMask, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType,
                  int op, cudaStream_t stream) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (isBadRoi(oSizeROI)) return NPP_SIZE_ERROR;
  if (srcGeometryBad(oSrcSize, oSrcOffset)) return NPP_OUT_OFF_RANGE_ERROR;
  if (!pMask) return NPP_NULL_POINTER_ERROR;
  if (oMaskSize.width < 0 || oMaskSize.height < 0) return NPP_OUT_OFF_RANGE_ERROR;
  if (eBorderType != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  const int rowBytes = oSizeROI.width * (int)sizeof(T);
  if (dstStepBad(nDstStep, rowBytes)) return NPP_STEP_ERROR;
  if (stepNotEven(nDstStep, (int)sizeof(T))) return NPP_NOT_EVEN_STEP_ERROR;

  const int mw = oMaskSize.width > 0 ? oMaskSize.width : 0;
  const int mh = oMaskSize.height > 0 ? oMaskSize.height : 0;
  if (mw == 0 || mh == 0) {
    cudaMemset2DAsync(pDst, (size_t)nDstStep, 0, (size_t)rowBytes, oSizeROI.height, stream);
    return NPP_SUCCESS;
  }
  return launchGray<T, 1, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, mw, mh,
                                 oAnchor.x, oAnchor.y, oSrcSize.width, oSrcSize.height, oSrcOffset.x, oSrcOffset.y,
                                 op, stream);
}

NppStatus getBufferSize(NppiSize oSizeROI, int nElementBytes, int *pBufferSize) {
  if (!pBufferSize) return NPP_NULL_POINTER_ERROR;
  if (isBadRoi(oSizeROI)) return NPP_SIZE_ERROR;
  const size_t pitch = bufferPitch(oSizeROI, nElementBytes);
  *pBufferSize = (int)(pitch * (size_t)oSizeROI.height + 8);
  return NPP_SUCCESS;
}

// ---------------------------------------------------------------------------
// Explicit instantiations used by the generated wrappers.
// ---------------------------------------------------------------------------
#define NPPI_MORPH_INST(T, C, AC4)                                                                         \
  template NppStatus runPlain<T, C, AC4>(const T *, int, T *, int, NppiSize, const Npp8u *, NppiSize,       \
                                         NppiPoint, int, cudaStream_t);                                    \
  template NppStatus runBorder<T, C, AC4>(const T *, int, NppiSize, NppiPoint, T *, int, NppiSize,          \
                                          const Npp8u *, NppiSize, NppiPoint, NppiBorderType, int,         \
                                          cudaStream_t);                                                   \
  template NppStatus run3x3Plain<T, C, AC4>(const T *, int, T *, int, NppiSize, int, cudaStream_t);          \
  template NppStatus run3x3Border<T, C, AC4>(const T *, int, NppiSize, NppiPoint, T *, int, NppiSize,        \
                                             NppiBorderType, int, cudaStream_t);                            \
  template NppStatus runComposite<T, C, AC4>(const T *, int, NppiSize, NppiPoint, T *, int, NppiSize,        \
                                             const Npp8u *, NppiSize, NppiPoint, Npp8u *, NppiBorderType,   \
                                             int, cudaStream_t);

NPPI_MORPH_INST(Npp8u, 1, false)
NPPI_MORPH_INST(Npp8u, 3, false)
NPPI_MORPH_INST(Npp8u, 4, false)
NPPI_MORPH_INST(Npp8u, 4, true)
NPPI_MORPH_INST(Npp16u, 1, false)
NPPI_MORPH_INST(Npp16u, 3, false)
NPPI_MORPH_INST(Npp16u, 4, false)
NPPI_MORPH_INST(Npp16u, 4, true)
NPPI_MORPH_INST(Npp16s, 1, false)
NPPI_MORPH_INST(Npp16s, 3, false)
NPPI_MORPH_INST(Npp16s, 4, false)
NPPI_MORPH_INST(Npp32f, 1, false)
NPPI_MORPH_INST(Npp32f, 3, false)
NPPI_MORPH_INST(Npp32f, 4, false)
NPPI_MORPH_INST(Npp32f, 4, true)
NPPI_MORPH_INST(Npp64f, 1, false)

#undef NPPI_MORPH_INST

template NppStatus runGray<Npp8u, Npp32s>(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize,
                                          const Npp32s *, NppiSize, NppiPoint, NppiBorderType, int, cudaStream_t);
template NppStatus runGray<Npp32f, Npp32f>(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize,
                                           const Npp32f *, NppiSize, NppiPoint, NppiBorderType, int, cudaStream_t);

}  // namespace morph
}  // namespace nppi
