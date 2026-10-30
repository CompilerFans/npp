#include "npp.h"
#include <cuda_runtime.h>
#include <cstdint>


// Morphological operations kernels

//=============================================================================
// Vectorized morphology kernels (128-bit / vector unit access paths)
//
// The kernels below implement exactly the same semantics as the reference
// kernels in this file (replicated / clamped borders, arbitrary structuring
// elements with per-tap mask bits, channel plan untouched).  The host
// implementations select them only when alignment / mask-shape constraints
// are met; every other case keeps using the original kernels, so results and
// API behaviour never change.
//
// Common design:
//  * one thread produces a vector-wide run of adjacent pixels
//    (4 pixels for 8u C1, 4 pixels for 32f C1) fed by 128-bit row loads,
//  * structuring element rows are streamed (one mask row at a time), so the
//    register footprint is independent of the mask height and arbitrary mask
//    shapes are supported,
//  * 8u families reduce whole words with a SWAR byte-parallel min/max
//    (morphSwMin8 / morphSwMax8 below): on the MXC600-U the per-lane compare
//    instructions (ternary 32-bit min/max, __vminu4 / __vsetltu4 class) are
//    an order of magnitude slower than plain integer ALU ops, while the
//    ~15-op SWAR reduction of four bytes at once runs at plain-ALU speed,
//  * the only shared memory used is a one-word flag telling whether the
//    structuring element is fully populated (all mask bytes non-zero); in
//    that case the reduction is separable (row fold first, then the
//    horizontal neighbours), otherwise each tap is applied individually,
//  * threads whose aligned access window would leave the row (borders,
//    unaligned tails, tiny ROIs) fall back to the per-pixel reference loop
//    below (identical to erodeGeneral / dilateGeneral).
//=============================================================================

__device__ __forceinline__ int morphVecClamp(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

// Reference per-pixel morphology (borders clamped, mask taps honoured).
// pMask == nullptr means "all taps active" (used by the dedicated 3x3 APIs).
template <bool DILATE>
__device__ __forceinline__ Npp8u morphVecRef8u(const Npp8u *__restrict__ pSrc, int nSrcStep, int x, int y, int width,
                                               int height, const Npp8u *__restrict__ pMask, int maskW, int maskH,
                                               int ax, int ay) {
  Npp8u val = DILATE ? (Npp8u)0 : (Npp8u)255;
  for (int my = 0; my < maskH; ++my) {
    const Npp8u *__restrict__ row =
        (const Npp8u *)((const char *)pSrc + (size_t)morphVecClamp(y + my - ay, height - 1) * nSrcStep);
    const Npp8u *__restrict__ mrow = pMask ? pMask + (size_t)my * maskW : nullptr;
    for (int mx = 0; mx < maskW; ++mx) {
      if (mrow && !mrow[mx])
        continue;
      const Npp8u v = row[morphVecClamp(x + mx - ax, width - 1)];
      if (DILATE) {
        if (v > val)
          val = v;
      } else if (v < val) {
        val = v;
      }
    }
  }
  return val;
}

//-----------------------------------------------------------------------------
// SWAR byte-parallel min / max for 8u data (four pixels per 32-bit word).
//
// On the MXC600-U every per-lane compare / select instruction (plain 32-bit
// ternary min-max, __vminu4, __vsetltu4 ...) costs an order of magnitude more
// than a plain integer ALU op, so reducing four bytes with ~15 ALU ops is far
// cheaper than four per-byte compares.  The identity below never borrows
// across byte lanes ((a|H) - (b&L) stays within [0x01,0xFF] per lane), and
// expands the per-lane "a < b" bit to a full-lane select mask.
//-----------------------------------------------------------------------------
__device__ __forceinline__ unsigned int morphSwMin8(unsigned int a, unsigned int b) {
  const unsigned int H = 0x80808080u, L = 0x7f7f7f7fu;
  const unsigned int d = ((a | H) - (b & L)) & H; // lane-wise a >= b indicator
  const unsigned int x = a ^ b;
  const unsigned int lt = ((~a) & b & H) | ((~x & H) & ~d); // lane-wise a < b
  const unsigned int m = (lt >> 7) * 0xFFu;                 // full-lane select mask
  return b ^ (x & m);                                       // m == 0xFF ? a : b
}
__device__ __forceinline__ unsigned int morphSwMax8(unsigned int a, unsigned int b) {
  return ~morphSwMin8(~a, ~b);
}

// sw[j] := the 32-bit word covering window bytes 4j..4j+3, extracted from the
// aligned load span w[0..7] where the window starts at byte (K * 4 + r8 / 8).
// K is a compile-time case, r8 a launch constant (both stay uniform).
template <int K, int NW>
__device__ __forceinline__ void morphVecFunnelWords(unsigned int *sw, const unsigned int *w, int r8) {
  if (r8 == 0) {
#pragma unroll
    for (int j = 0; j < NW; ++j)
      sw[j] = w[K + j];
  } else {
#pragma unroll
    for (int j = 0; j < NW; ++j)
      sw[j] = (w[K + j] >> r8) | (w[K + j + 1] << (32 - r8));
  }
}

//-----------------------------------------------------------------------------
// 8u C1R vectorized kernel: 4 output pixels (one 32-bit word) per thread.
//
// Load stage (shared by both reduction paths):
//  * each mask row is fetched with two uint4 (128-bit) loads covering the
//    32-byte span [lo, lo+32) with lo = (x0 - ax) - loadShift; loadShift
//    (= (pSrc - outAlign - ax) & 15) realigns the window to the loads, which
//    stay 4-byte aligned whenever nSrcStep % 4 == 0 (checked on the host),
//  * a funnel shift rebuilds the window words sw[0..SW_WORDS): word j covers
//    the output pixels x0..x0+3 and their MW-1 right neighbours, shifted by
//    j words,
//  * rows above / below the image are clamped (replicate border), exactly
//    like the reference kernels.
//
// Reduction:
//  * fully populated structuring element (every mask byte non-zero, or the
//    dedicated 3x3 APIs' nullptr mask): separable - the vertical stage folds
//    every row into the window words, the horizontal stage then folds the
//    MW-1 horizontal neighbours into the output word.  O(maskH*SW_WORDS + MW)
//    morphSwMin8/morphSwMax8 calls per 4 pixels.
//  * arbitrary mask: every tap is applied on its own shifted window word and
//    gated by its mask byte.  O(maskH * MW) calls per 4 pixels.
//
// Threads whose window would leave the row (borders, unaligned tails, tiny
// ROIs) fall back to morphVecRef8u (identical to erodeGeneral/dilateGeneral).
//-----------------------------------------------------------------------------
template <bool DILATE, int MW>
__global__ void nppiMorph8u_C1R_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                       int nDstStep, int width, int height, const Npp8u *__restrict__ pMask,
                                       int maskH, int ax, int ay, int outAlign, int loadShift) {
  enum { SW_WORDS = (4 + MW - 1 + 3) >> 2 };

  // Cooperative mask scan (the mask lives in device memory, so the host
  // cannot inspect it): blocks whose structuring element is fully populated
  // take the separable reduction below.  Writing 0 is idempotent, so no
  // atomics are needed; both barriers are reached by every thread of the
  // block (the pMask condition is launch-uniform).
  __shared__ int s_allOnes;
  if (pMask != nullptr) {
    const int tid = (int)(threadIdx.y * blockDim.x + threadIdx.x);
    const int nthreads = (int)(blockDim.x * blockDim.y);
    const int maskBytes = MW * maskH;
    if (tid == 0)
      s_allOnes = 1;
    __syncthreads();
    for (int i = tid; i < maskBytes; i += nthreads)
      if (pMask[i] == 0)
        s_allOnes = 0;
    __syncthreads();
  }
  const bool allOnes = (pMask == nullptr) || (s_allOnes != 0);

  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  const int x0 = chunk * 4 - outAlign; // output span [x0, x0+4), 4-byte aligned store
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y >= height || x0 >= width)
    return;

  Npp8u *__restrict__ dstRow = (Npp8u *)((char *)pDst + (size_t)y * nDstStep);
  const int lo = (x0 - ax) - loadShift; // aligned start of the 32B load span

  if (x0 < 0 || x0 + 4 > width || lo < 0 || lo + 32 > width) {
    // Border / tail thread: per-pixel reference path (clamped borders).
    const int xs = x0 > 0 ? x0 : 0;
    const int xe = x0 + 4 < width ? x0 + 4 : width;
    for (int x = xs; x < xe; ++x)
      dstRow[x] = morphVecRef8u<DILATE>(pSrc, nSrcStep, x, y, width, height, pMask, MW, maskH, ax, ay);
    return;
  }

  const int q = loadShift >> 2;       // word offset of the window in the loads
  const int r8 = (loadShift & 3) * 8; // byte rotation of the window in the loads

  if (allOnes) {
    // ---- separable reduction: vertical fold of every row ----
    unsigned int v[SW_WORDS];
    {
      const Npp8u *__restrict__ row =
          (const Npp8u *)((const char *)pSrc + (size_t)morphVecClamp(y - ay, height - 1) * nSrcStep) + lo;
      const uint4 a = ((const uint4 *)row)[0];
      const uint4 b = ((const uint4 *)row)[1];
      const unsigned int w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      switch (q) {
      case 0:
        morphVecFunnelWords<0, SW_WORDS>(v, w, r8);
        break;
      case 1:
        morphVecFunnelWords<1, SW_WORDS>(v, w, r8);
        break;
      case 2:
        morphVecFunnelWords<2, SW_WORDS>(v, w, r8);
        break;
      default:
        morphVecFunnelWords<3, SW_WORDS>(v, w, r8);
        break;
      }
    }
    for (int my = 1; my < maskH; ++my) {
      const Npp8u *__restrict__ row =
          (const Npp8u *)((const char *)pSrc + (size_t)morphVecClamp(y + my - ay, height - 1) * nSrcStep) + lo;
      const uint4 a = ((const uint4 *)row)[0];
      const uint4 b = ((const uint4 *)row)[1];
      const unsigned int w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      unsigned int sw[SW_WORDS];
      switch (q) {
      case 0:
        morphVecFunnelWords<0, SW_WORDS>(sw, w, r8);
        break;
      case 1:
        morphVecFunnelWords<1, SW_WORDS>(sw, w, r8);
        break;
      case 2:
        morphVecFunnelWords<2, SW_WORDS>(sw, w, r8);
        break;
      default:
        morphVecFunnelWords<3, SW_WORDS>(sw, w, r8);
        break;
      }
#pragma unroll
      for (int j = 0; j < SW_WORDS; ++j)
        v[j] = DILATE ? morphSwMax8(v[j], sw[j]) : morphSwMin8(v[j], sw[j]);
    }
    // ---- horizontal fold of the MW-1 neighbours ----
    unsigned int out = v[0];
#pragma unroll
    for (int mx = 1; mx < MW; ++mx) {
      const int wi = mx >> 2;
      const int sh = (mx & 3) * 8;
      const unsigned int s = sh == 0 ? v[wi] : ((v[wi] >> sh) | (v[wi + 1] << (32 - sh)));
      out = DILATE ? morphSwMax8(out, s) : morphSwMin8(out, s);
    }
    *(unsigned int *)(dstRow + x0) = out;
  } else {
    // ---- per-tap reduction, each tap gated by its mask byte ----
    unsigned int out = DILATE ? 0u : 0xFFFFFFFFu;
    for (int my = 0; my < maskH; ++my) {
      const Npp8u *__restrict__ row =
          (const Npp8u *)((const char *)pSrc + (size_t)morphVecClamp(y + my - ay, height - 1) * nSrcStep) + lo;
      const uint4 a = ((const uint4 *)row)[0];
      const uint4 b = ((const uint4 *)row)[1];
      const unsigned int w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      unsigned int sw[SW_WORDS];
      switch (q) {
      case 0:
        morphVecFunnelWords<0, SW_WORDS>(sw, w, r8);
        break;
      case 1:
        morphVecFunnelWords<1, SW_WORDS>(sw, w, r8);
        break;
      case 2:
        morphVecFunnelWords<2, SW_WORDS>(sw, w, r8);
        break;
      default:
        morphVecFunnelWords<3, SW_WORDS>(sw, w, r8);
        break;
      }
      const Npp8u *__restrict__ mrow = pMask + (size_t)my * MW;
#pragma unroll
      for (int mx = 0; mx < MW; ++mx) {
        if (!mrow[mx])
          continue;
        const int wi = mx >> 2;
        const int sh = (mx & 3) * 8;
        const unsigned int s = sh == 0 ? sw[wi] : ((sw[wi] >> sh) | (sw[wi + 1] << (32 - sh)));
        out = DILATE ? morphSwMax8(out, s) : morphSwMin8(out, s);
      }
    }
    *(unsigned int *)(dstRow + x0) = out;
  }
}

// Host-side launch helper for the 8u C1R vectorized kernel.  Returns true
// when the kernel was launched; the caller falls back to the original path
// otherwise.
template <bool DILATE>
static bool morphVecLaunch8uC1(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width, int height,
                               const Npp8u *pMask, int maskW, int maskH, int ax, int ay, cudaStream_t stream) {
  if (width <= 0 || height <= 0 || nSrcStep <= 0 || nDstStep <= 0)
    return false;
  // 4-byte aligned row strides keep the uint4 loads and the word store
  // aligned for every chunk (loadShift construction above).
  if ((nSrcStep & 3) != 0 || (nDstStep & 3) != 0)
    return false;
  if (maskW != 3 && maskW != 5 && maskW != 7 && maskW != 9)
    return false;
  if (pMask != nullptr && (size_t)maskW * (size_t)maskH > 1024)
    return false; // keep the cooperative mask scan trivial
  const int outAlign = (int)(((uintptr_t)pDst) & 15);
  const int loadShift = (int)((((size_t)pSrc) - (size_t)outAlign - (size_t)ax) & 15);
  dim3 blockSize(32, 16);
  dim3 gridSize((unsigned)((width + outAlign + 127) / 128), (unsigned)((height + 15) / 16));
  switch (maskW) {
  case 3:
    nppiMorph8u_C1R_kernel<DILATE, 3>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, width, height, pMask, maskH, ax, ay,
                                             outAlign, loadShift);
    break;
  case 5:
    nppiMorph8u_C1R_kernel<DILATE, 5>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, width, height, pMask, maskH, ax, ay,
                                             outAlign, loadShift);
    break;
  case 7:
    nppiMorph8u_C1R_kernel<DILATE, 7>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, width, height, pMask, maskH, ax, ay,
                                             outAlign, loadShift);
    break;
  case 9:
    nppiMorph8u_C1R_kernel<DILATE, 9>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, width, height, pMask, maskH, ax, ay,
                                             outAlign, loadShift);
    break;
  default:
    return false;
  }
  return cudaGetLastError() == cudaSuccess;
}

// Device function for 3x3 erosion (minimum value in neighborhood)

template <typename T> __device__ inline T erode3x3(const T *pSrc, int nSrcStep, int x, int y, int width, int height) {
  T minVal = T(255); // Default maximum value for 8u

  // Apply 3x3 structuring element
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int srcY = y + dy;
      int srcX = x + dx;

      // Handle boundaries by clamping
      srcY = max(0, min(srcY, height - 1));
      srcX = max(0, min(srcX, width - 1));

      const T *src_row = (const T *)((const char *)pSrc + srcY * nSrcStep);
      T pixelValue = src_row[srcX];

      if (pixelValue < minVal) {
        minVal = pixelValue;
      }
    }
  }

  return minVal;
}

// Template specialization for Npp32f (float)
template <>

__device__ inline Npp32f erode3x3<Npp32f>(const Npp32f *pSrc, int nSrcStep, int x, int y, int width, int height) {
  Npp32f minVal = 1e10f; // Large value for float

  // Apply 3x3 structuring element
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int srcY = y + dy;
      int srcX = x + dx;

      // Handle boundaries by clamping
      srcY = max(0, min(srcY, height - 1));
      srcX = max(0, min(srcX, width - 1));

      const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + srcY * nSrcStep);
      Npp32f pixelValue = src_row[srcX];

      if (pixelValue < minVal) {
        minVal = pixelValue;
      }
    }
  }

  return minVal;
}

// Device function for 3x3 dilation (maximum value in neighborhood)

template <typename T> __device__ inline T dilate3x3(const T *pSrc, int nSrcStep, int x, int y, int width, int height) {
  T maxVal = (T)0; // Start with minimum value

  // Apply 3x3 structuring element
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int srcY = y + dy;
      int srcX = x + dx;

      // Handle boundaries by clamping
      srcY = max(0, min(srcY, height - 1));
      srcX = max(0, min(srcX, width - 1));

      const T *src_row = (const T *)((const char *)pSrc + srcY * nSrcStep);
      T pixelValue = src_row[srcX];

      if (pixelValue > maxVal) {
        maxVal = pixelValue;
      }
    }
  }

  return maxVal;
}

// Template specialization for Npp32f (float) dilation
template <>

__device__ inline Npp32f dilate3x3<Npp32f>(const Npp32f *pSrc, int nSrcStep, int x, int y, int width, int height) {
  Npp32f maxVal = -1e10f; // Very small value for float

  // Apply 3x3 structuring element
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int srcY = y + dy;
      int srcX = x + dx;

      // Handle boundaries by clamping
      srcY = max(0, min(srcY, height - 1));
      srcX = max(0, min(srcX, width - 1));

      const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + srcY * nSrcStep);
      Npp32f pixelValue = src_row[srcX];

      if (pixelValue > maxVal) {
        maxVal = pixelValue;
      }
    }
  }

  return maxVal;
}

//=============================================================================
// Erosion Kernels
//=============================================================================

// Kernel for 8-bit unsigned single channel 3x3 erosion
__global__ void nppiErode3x3_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                           int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    dst_row[x] = erode3x3(pSrc, nSrcStep, x, y, width, height);
  }
}

// Kernel for 32-bit float single channel 3x3 erosion
__global__ void nppiErode3x3_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);
    dst_row[x] = erode3x3(pSrc, nSrcStep, x, y, width, height);
  }
}

//=============================================================================
// Dilation Kernels
//=============================================================================

// Kernel for 8-bit unsigned single channel 3x3 dilation
__global__ void nppiDilate3x3_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    dst_row[x] = dilate3x3(pSrc, nSrcStep, x, y, width, height);
  }
}

// Kernel for 32-bit float single channel 3x3 dilation
__global__ void nppiDilate3x3_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                             int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);
    dst_row[x] = dilate3x3(pSrc, nSrcStep, x, y, width, height);
  }
}

//=============================================================================
// Host Functions
//=============================================================================

extern "C" {

// 8-bit unsigned single channel 3x3 erosion implementation
NppStatus nppiErode3x3_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
  // Vectorized path (128-bit loads, SWAR byte-parallel min/max); the original
  // kernel below stays as the fallback for unaligned steps / launch failures.
  if (morphVecLaunch8uC1<false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nullptr, 3, 3, 1, 1,
                                nppStreamCtx.hStream))
    return NPP_SUCCESS;

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiErode3x3_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                               oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel 3x3 erosion implementation
NppStatus nppiErode3x3_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiErode3x3_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned single channel 3x3 dilation implementation
NppStatus nppiDilate3x3_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx) {
  // Vectorized path (128-bit loads, SWAR byte-parallel min/max); the original
  // kernel below stays as the fallback for unaligned steps / launch failures.
  if (morphVecLaunch8uC1<true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, nullptr, 3, 3, 1, 1,
                               nppStreamCtx.hStream))
    return NPP_SUCCESS;

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiDilate3x3_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// 32-bit float single channel 3x3 dilation implementation
NppStatus nppiDilate3x3_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiDilate3x3_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                 oSizeROI.width, oSizeROI.height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}

//=============================================================================
// General Morphology Kernels with Arbitrary Structure Elements
//=============================================================================

// Generic erosion device function with arbitrary kernel
template <typename T>
__device__ inline T erodeGeneral(const T *pSrc, int nSrcStep, int x, int y, int width, int height, const Npp8u *pMask,
                                 int maskWidth, int maskHeight, int anchorX, int anchorY) {
  T minVal = T(255); // Default maximum value for integer types

  for (int my = 0; my < maskHeight; my++) {
    for (int mx = 0; mx < maskWidth; mx++) {
      // Check if mask element is active (non-zero)
      if (pMask[my * maskWidth + mx] != 0) {
        int srcY = y + my - anchorY;
        int srcX = x + mx - anchorX;

        // Handle boundaries by clamping
        srcY = max(0, min(srcY, height - 1));
        srcX = max(0, min(srcX, width - 1));

        const T *src_row = (const T *)((const char *)pSrc + srcY * nSrcStep);
        T pixelValue = src_row[srcX];

        if (pixelValue < minVal) {
          minVal = pixelValue;
        }
      }
    }
  }

  return minVal;
}

// Specialization for float
template <>
__device__ inline Npp32f erodeGeneral<Npp32f>(const Npp32f *pSrc, int nSrcStep, int x, int y, int width, int height,
                                              const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                              int anchorY) {
  Npp32f minVal = 1e10f; // Large value for float

  for (int my = 0; my < maskHeight; my++) {
    for (int mx = 0; mx < maskWidth; mx++) {
      if (pMask[my * maskWidth + mx] != 0) {
        int srcY = y + my - anchorY;
        int srcX = x + mx - anchorX;

        srcY = max(0, min(srcY, height - 1));
        srcX = max(0, min(srcX, width - 1));

        const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + srcY * nSrcStep);
        Npp32f pixelValue = src_row[srcX];

        if (pixelValue < minVal) {
          minVal = pixelValue;
        }
      }
    }
  }

  return minVal;
}

// Generic dilation device function with arbitrary kernel
template <typename T>
__device__ inline T dilateGeneral(const T *pSrc, int nSrcStep, int x, int y, int width, int height, const Npp8u *pMask,
                                  int maskWidth, int maskHeight, int anchorX, int anchorY) {
  T maxVal = T(0); // Default minimum value for integer types

  for (int my = 0; my < maskHeight; my++) {
    for (int mx = 0; mx < maskWidth; mx++) {
      if (pMask[my * maskWidth + mx] != 0) {
        int srcY = y + my - anchorY;
        int srcX = x + mx - anchorX;

        srcY = max(0, min(srcY, height - 1));
        srcX = max(0, min(srcX, width - 1));

        const T *src_row = (const T *)((const char *)pSrc + srcY * nSrcStep);
        T pixelValue = src_row[srcX];

        if (pixelValue > maxVal) {
          maxVal = pixelValue;
        }
      }
    }
  }

  return maxVal;
}

// Specialization for float
template <>
__device__ inline Npp32f dilateGeneral<Npp32f>(const Npp32f *pSrc, int nSrcStep, int x, int y, int width, int height,
                                               const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                               int anchorY) {
  Npp32f maxVal = -1e10f; // Small value for float

  for (int my = 0; my < maskHeight; my++) {
    for (int mx = 0; mx < maskWidth; mx++) {
      if (pMask[my * maskWidth + mx] != 0) {
        int srcY = y + my - anchorY;
        int srcX = x + mx - anchorX;

        srcY = max(0, min(srcY, height - 1));
        srcX = max(0, min(srcX, width - 1));

        const Npp32f *src_row = (const Npp32f *)((const char *)pSrc + srcY * nSrcStep);
        Npp32f pixelValue = src_row[srcX];

        if (pixelValue > maxVal) {
          maxVal = pixelValue;
        }
      }
    }
  }

  return maxVal;
}

// General erosion kernels
__global__ void nppiErode_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                        int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                        int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u result = erodeGeneral(pSrc, nSrcStep, x, y, width, height, pMask, maskWidth, maskHeight, anchorX, anchorY);

    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    dst_row[x] = result;
  }
}

__global__ void nppiErode_8u_C1R_kernel_tiled(const Npp8u *__restrict__ pSrc, int nSrcStep,
                                                  Npp8u *__restrict__ pDst, int nDstStep, int width, int height,
                                                  const Npp8u *__restrict__ pMask, int maskW, int maskH, int anchorX,
                                                  int anchorY, int tileH, int tilePitch) {
  extern __shared__ unsigned char smem[];

  unsigned char *tile = smem;
  unsigned char *mskSh = tile + tilePitch * tileH;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int tcount = blockDim.x * blockDim.y;

  // Output pixel start (4 pixels per thread horizontally)
  const int outX0 = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
  const int outY = blockIdx.y * blockDim.y + threadIdx.y;

  // Tile origin in source image (include anchor offset)
  const int tileOrgX = blockIdx.x * blockDim.x * 4 - anchorX;
  const int tileOrgY = blockIdx.y * blockDim.y - anchorY;

  // 1) Load tile from global memory (with border clamping)
  const int tileElems = tilePitch * tileH;
  for (int idx = tid; idx < tileElems; idx += tcount) {
    int ly = idx / tilePitch;
    int lx = idx % tilePitch;

    int sx = tileOrgX + lx;
    int sy = tileOrgY + ly;

    int clamped_x = min(max(sx, 0), width - 1);
    int clamped_y = min(max(sy, 0), height - 1);

    const Npp8u *srow = (const Npp8u *)((const char *)pSrc + clamped_y * nSrcStep);
    tile[ly * tilePitch + lx] = srow[clamped_x];
  }

  // 2) Load mask to shared memory
  const int msize = maskW * maskH;
  for (int i = tid; i < msize; i += tcount) {
    mskSh[i] = pMask[i];
  }

  __syncthreads();

  // 3) Process output pixels
  if (outY >= height)
    return;

  const int baseTileX = threadIdx.x * 4;
  const int baseTileY = threadIdx.y;

  for (int dx = 0; dx < 4; ++dx) {
    int outX = outX0 + dx;
    if (outX >= width)
      continue;

    unsigned char minVal = 255;

    // Iterate mask window
    for (int my = 0; my < maskH; ++my) {
      const unsigned char *trow = tile + (baseTileY + my) * tilePitch;
      const unsigned char *mrow = mskSh + my * maskW;

#pragma unroll
      for (int mx = 0; mx < maskW; ++mx) {
        if (mrow[mx]) {
          unsigned char pixelVal = trow[baseTileX + dx + mx];
          if (pixelVal < minVal) {
            minVal = pixelVal;
          }
        }
      }
    }

    Npp8u *drow = (Npp8u *)((char *)pDst + outY * nDstStep);
    drow[outX] = minVal;
  }
}
// Specialized 3x3 erosion kernel, anchor = (1,1), 8u C1
template <int BLOCK_X, int BLOCK_Y>
__global__ void nppiErode_8u_C1R_3x3_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                            int nDstStep, int width, int height, const Npp8u *__restrict__ pMask) {

  // 3x3 mask in shared (9 bytes)
  __shared__ unsigned char maskSh[9];

  // Image tile in shared: (BLOCK_Y + 2) x (BLOCK_X + 2)
  __shared__ unsigned char tile[BLOCK_Y + 2][BLOCK_X + 2];

  // Load mask: 9 elements, one-time per block
  if (threadIdx.y == 0 && threadIdx.x < 9) {
    maskSh[threadIdx.x] = pMask[threadIdx.x];
  }
  __syncthreads();

  const int maxX = width - 1;
  const int maxY = height - 1;

  // Top-left corner of tile in the source image (including 1-pixel halo)
  const int x0 = blockIdx.x * BLOCK_X - 1;
  const int y0 = blockIdx.y * BLOCK_Y - 1;

  // 1) Load tile from global memory (replicate border)
  for (int ty = threadIdx.y; ty < BLOCK_Y + 2; ty += blockDim.y) {
    int gy = y0 + ty;
    int cy = gy;
    if (cy < 0)
      cy = 0;
    else if (cy > maxY)
      cy = maxY;

    const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + cy * nSrcStep);

    for (int tx = threadIdx.x; tx < BLOCK_X + 2; tx += blockDim.x) {
      int gx = x0 + tx;
      int cx = gx;
      if (cx < 0)
        cx = 0;
      else if (cx > maxX)
        cx = maxX;

      tile[ty][tx] = srcRow[cx];
    }
  }

  __syncthreads();

  // 2) Compute one output pixel per thread
  const int x = blockIdx.x * BLOCK_X + threadIdx.x;
  const int y = blockIdx.y * BLOCK_Y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  // Position of the center in tile
  const int sx = threadIdx.x + 1;
  const int sy = threadIdx.y + 1;

  unsigned char minVal = 255;
  unsigned char v;

#define APPLY(maskIndex, dy, dx)                                                                                       \
  if (maskSh[maskIndex]) {                                                                                             \
    v = tile[sy + (dy)][sx + (dx)];                                                                                    \
    if (v < minVal)                                                                                                    \
      minVal = v;                                                                                                      \
  }

  // 3x3 neighborhood with maskSh[0..8] in row-major order
  // row -1
  APPLY(0, -1, -1);
  APPLY(1, -1, 0);
  APPLY(2, -1, 1);
  // row 0
  APPLY(3, 0, -1);
  APPLY(4, 0, 0);
  APPLY(5, 0, 1);
  // row +1
  APPLY(6, 1, -1);
  APPLY(7, 1, 0);
  APPLY(8, 1, 1);

#undef APPLY

  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);
  dstRow[x] = minVal;
}

__global__ void nppiErode_8u_C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                        int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                        int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    // Process each channel separately
    for (int c = 0; c < 4; c++) {
      // Create temporary single-channel views
      const Npp8u *src_chan = pSrc + c;
      Npp8u *dst_chan = dst_row + x * 4 + c;

      Npp8u minVal = 255;
      for (int my = 0; my < maskHeight; my++) {
        for (int mx = 0; mx < maskWidth; mx++) {
          if (pMask[my * maskWidth + mx] != 0) {
            int srcY = y + my - anchorY;
            int srcX = x + mx - anchorX;

            srcY = max(0, min(srcY, height - 1));
            srcX = max(0, min(srcX, width - 1));

            const Npp8u *src_row = (const Npp8u *)((const char *)src_chan + srcY * nSrcStep);
            Npp8u pixelValue = src_row[srcX * 4];

            if (pixelValue < minVal) {
              minVal = pixelValue;
            }
          }
        }
      }
      *dst_chan = minVal;
    }
  }
}
__global__ void nppiErode_8u_C4R_kernel_tiled(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                              int nDstStep, int width, int height, const Npp8u *__restrict__ pMask,
                                              int maskWidth, int maskHeight, int anchorX, int anchorY) {
  extern __shared__ unsigned char shared[];

  constexpr int PIXELS_PER_THREAD_X = 4;
  const int blockSpanX = blockDim.x * PIXELS_PER_THREAD_X;

  // tile width covered the entire block pixel range
  const int tileWidth = blockSpanX + maskWidth - 1;
  const int tileHeight = blockDim.y + maskHeight - 1;

  uchar4 *tile = reinterpret_cast<uchar4 *>(shared);
  unsigned char *maskShared = reinterpret_cast<unsigned char *>(tile + tileWidth * tileHeight);

  const int tileOriginX = blockIdx.x * blockSpanX - anchorX;
  const int tileOriginY = blockIdx.y * blockDim.y - anchorY;

  const int linearThreadId = threadIdx.y * blockDim.x + threadIdx.x;
  const int threadsPerBlock = blockDim.x * blockDim.y;

  // load tile
  const int tileSize = tileWidth * tileHeight;
  for (int idx = linearThreadId; idx < tileSize; idx += threadsPerBlock) {
    int localY = idx / tileWidth;
    int localX = idx % tileWidth;

    int srcX = tileOriginX + localX;
    int srcY = tileOriginY + localY;

    srcX = max(0, min(srcX, width - 1));
    srcY = max(0, min(srcY, height - 1));

    const uchar4 *src_row = reinterpret_cast<const uchar4 *>((const char *)pSrc + srcY * nSrcStep);
    tile[idx] = src_row[srcX];
  }

  // load mask
  const int maskSize = maskWidth * maskHeight;
  for (int idx = linearThreadId; idx < maskSize; idx += threadsPerBlock) {
    maskShared[idx] = pMask[idx];
  }

  __syncthreads();

  const int outY = blockIdx.y * blockDim.y + threadIdx.y;
  const int outXBase = (blockIdx.x * blockDim.x + threadIdx.x) * PIXELS_PER_THREAD_X;
  if (outY >= height || outXBase >= width)
    return;

  uchar4 *dst_row = reinterpret_cast<uchar4 *>((char *)pDst + outY * nDstStep);

#pragma unroll
  for (int i = 0; i < PIXELS_PER_THREAD_X; ++i) {
    int outX = outXBase + i;
    if (outX >= width)
      break;

    uchar4 minVal = make_uchar4(255, 255, 255, 255);

    for (int my = 0; my < maskHeight; ++my) {
      const unsigned char *maskRow = maskShared + my * maskWidth;
      const uchar4 *tileRow = tile + (threadIdx.y + my) * tileWidth;
#pragma unroll 4
      for (int mx = 0; mx < maskWidth; ++mx) {
        if (maskRow[mx]) {
          int tileX = threadIdx.x * PIXELS_PER_THREAD_X + i + mx;
          uchar4 pixelValue = tileRow[tileX];

          minVal.x = min(minVal.x, pixelValue.x);
          minVal.y = min(minVal.y, pixelValue.y);
          minVal.z = min(minVal.z, pixelValue.z);
          minVal.w = min(minVal.w, pixelValue.w);
        }
      }
    }

    dst_row[outX] = minVal;
  }
}
// opt for masksize 3x3
__global__ void nppiErode_8u_C4R_3x3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                            int height) {

  const int BLOCK_WIDTH = 32; // blockSize.x
  const int BLOCK_HEIGHT = 8; // blockSize.y
  const int SHARED_WIDTH = 34;
  const int SHARED_HEIGHT = 10;

  // fixed shared mem
  __shared__ Npp8u shared_mem[10][136]; // [blockSize.y + 2][(blockSize.x + 2)*4]

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int blockX = blockIdx.x * BLOCK_WIDTH;
  int blockY = blockIdx.y * BLOCK_HEIGHT;

  // load
  for (int pos = ty * blockDim.x + tx; pos < SHARED_HEIGHT * SHARED_WIDTH; pos += blockDim.x * blockDim.y) {
    int loadY = pos / SHARED_WIDTH;
    int loadX = pos % SHARED_WIDTH;

    int srcY = blockY + loadY - 1;
    int srcX = blockX + loadX - 1;

    if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
      const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + srcY * nSrcStep);
      const Npp8u *src_pixel = &src_row[srcX * 4];

      shared_mem[loadY][loadX * 4 + 0] = src_pixel[0];
      shared_mem[loadY][loadX * 4 + 1] = src_pixel[1];
      shared_mem[loadY][loadX * 4 + 2] = src_pixel[2];
      shared_mem[loadY][loadX * 4 + 3] = src_pixel[3];
    } else {
      shared_mem[loadY][loadX * 4 + 0] = 0;
      shared_mem[loadY][loadX * 4 + 1] = 0;
      shared_mem[loadY][loadX * 4 + 2] = 0;
      shared_mem[loadY][loadX * 4 + 3] = 0;
    }
  }

  __syncthreads();

  int x = blockX + tx;
  int y = blockY + ty;

  if (x < width && y < height) {
    int shared_y = ty + 1;
    int shared_x = tx + 1;
    int idx = shared_x * 4;

    Npp8u minVal[4];

    // init as center pixel
    minVal[0] = shared_mem[shared_y][idx + 0];
    minVal[1] = shared_mem[shared_y][idx + 1];
    minVal[2] = shared_mem[shared_y][idx + 2];
    minVal[3] = shared_mem[shared_y][idx + 3];

    // 3x3 field compare
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0)
          continue;

        int neighbor_y = shared_y + dy;
        int neighbor_x = shared_x + dx;
        int neighbor_idx = neighbor_x * 4;

        minVal[0] = min(minVal[0], shared_mem[neighbor_y][neighbor_idx + 0]);
        minVal[1] = min(minVal[1], shared_mem[neighbor_y][neighbor_idx + 1]);
        minVal[2] = min(minVal[2], shared_mem[neighbor_y][neighbor_idx + 2]);
        minVal[3] = min(minVal[3], shared_mem[neighbor_y][neighbor_idx + 3]);
      }
    }

    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    Npp8u *dst_pixel = &dst_row[x * 4];

    dst_pixel[0] = minVal[0];
    dst_pixel[1] = minVal[1];
    dst_pixel[2] = minVal[2];
    dst_pixel[3] = minVal[3];
  }
}

__global__ void nppiErode_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                         int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                         int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f result = erodeGeneral(pSrc, nSrcStep, x, y, width, height, pMask, maskWidth, maskHeight, anchorX, anchorY);

    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);
    dst_row[x] = result;
  }
}

__global__ void nppiErode_32f_C1R_kernel_opt(const Npp32f* pSrc, int nSrcStep,
                                                  Npp32f* pDst, int nDstStep, int width, int height,
                                                  const Npp8u *__restrict__ pMask, int maskW, int maskH, int anchorX,
                                                  int anchorY, int tileH, int tilePitch) {
  extern __shared__ Npp32f smem_32f_c1r[];

  Npp32f *tile = smem_32f_c1r;
  Npp32f *mskSh = tile + tilePitch * tileH;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int tcount = blockDim.x * blockDim.y;

  // Output pixel start (4 pixels per thread horizontally)
  const int outX0 = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
  const int outY = blockIdx.y * blockDim.y + threadIdx.y;

  // Tile origin in source image (include anchor offset)
  const int tileOrgX = blockIdx.x * blockDim.x * 4 - anchorX;
  const int tileOrgY = blockIdx.y * blockDim.y - anchorY;

  // 1) Load tile from global memory (with border clamping)
  const int tileElems = tilePitch * tileH;
  for (int idx = tid; idx < tileElems; idx += tcount) {
    int ly = idx / tilePitch;
    int lx = idx % tilePitch;

    int sx = tileOrgX + lx;
    int sy = tileOrgY + ly;

    int clamped_x = min(max(sx, 0), width - 1);
    int clamped_y = min(max(sy, 0), height - 1);

    const Npp32f *srow = (const Npp32f *)((const char *)pSrc + clamped_y * nSrcStep);
    tile[ly * tilePitch + lx] = srow[clamped_x];
  }

  // 2) Load mask to shared memory
  const int msize = maskW * maskH;
  for (int i = tid; i < msize; i += tcount) {
    mskSh[i] = pMask[i];
  }

  __syncthreads();

  // 3) Process output pixels
  if (outY >= height)
    return;

  const int baseTileX = threadIdx.x * 4;
  const int baseTileY = threadIdx.y;

  for (int dx = 0; dx < 4; ++dx) {
    int outX = outX0 + dx;
    if (outX >= width)
      continue;

    Npp32f minVal = 1e10f;

    // Iterate mask window
    for (int my = 0; my < maskH; ++my) {
      const Npp32f *trow = tile + (baseTileY + my) * tilePitch;
      const Npp32f *mrow = mskSh + my * maskW;

#pragma unroll
      for (int mx = 0; mx < maskW; ++mx) {
        if (mrow[mx]) {
          Npp32f pixelVal = trow[baseTileX + dx + mx];
          if (pixelVal < minVal) {
            minVal = pixelVal;
          }
        }
      }
    }

    Npp32f *drow = (Npp32f *)((char *)pDst + outY * nDstStep);
    drow[outX] = minVal;
  }
}

__global__ void nppiErode_32f_C4R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                         int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                         int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

    // Process each channel separately
    for (int c = 0; c < 4; c++) {
      const Npp32f *src_chan = pSrc + c;
      Npp32f *dst_chan = dst_row + x * 4 + c;

      Npp32f minVal = 1e10f;
      for (int my = 0; my < maskHeight; my++) {
        for (int mx = 0; mx < maskWidth; mx++) {
          if (pMask[my * maskWidth + mx] != 0) {
            int srcY = y + my - anchorY;
            int srcX = x + mx - anchorX;

            srcY = max(0, min(srcY, height - 1));
            srcX = max(0, min(srcX, width - 1));

            const Npp32f *src_row = (const Npp32f *)((const char *)src_chan + srcY * nSrcStep);
            Npp32f pixelValue = src_row[srcX * 4];

            if (pixelValue < minVal) {
              minVal = pixelValue;
            }
          }
        }
      }
      *dst_chan = minVal;
    }
  }
}

__global__ void nppiErode_32f_C4R_kernel_opt(const Npp32f *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                             int nDstStep, int width, int height, const Npp8u *__restrict__ pMask,
                                             int maskWidth, int maskHeight, int anchorX, int anchorY) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tile_x = blockIdx.x * blockDim.x;
  const int tile_y = blockIdx.y * blockDim.y;
  const int out_x = tile_x + tx;
  const int out_y = tile_y + ty;
  if (out_x >= width || out_y >= height)
    return;

  const int TILE_WIDTH = blockDim.x + maskWidth - 1;
  const int TILE_HEIGHT = blockDim.y + maskHeight - 1;
  const int maskSize = maskWidth * maskHeight;

  extern __shared__ unsigned char sdata[];

  float4 *tile = reinterpret_cast<float4 *>(sdata);
  Npp8u *sMask = reinterpret_cast<Npp8u *>(tile + TILE_WIDTH * TILE_HEIGHT);

  const int tid = ty * blockDim.x + tx;
  const int numThreads = blockDim.x * blockDim.y;

  // 1) mask -> shared
  for (int idx = tid; idx < maskSize; idx += numThreads) {
    sMask[idx] = pMask[idx];
  }
  __syncthreads();

  // 2) tile + halo -> shared
  for (int j = ty; j < TILE_HEIGHT; j += blockDim.y) {
    int gy = tile_y + j - anchorY;
    gy = max(0, min(gy, height - 1));
    const float4 *src_row = reinterpret_cast<const float4 *>((const char *)pSrc + gy * nSrcStep);

    for (int i = tx; i < TILE_WIDTH; i += blockDim.x) {
      int gx = tile_x + i - anchorX;
      gx = max(0, min(gx, width - 1));
      tile[j * TILE_WIDTH + i] = src_row[gx];
    }
  }
  __syncthreads();

  float4 minVal = make_float4(1e10f, 1e10f, 1e10f, 1e10f);

  for (int my = 0; my < maskHeight; ++my) {
    const int sy = ty + my;
    const int maskRow = my * maskWidth;

    for (int mx = 0; mx < maskWidth; ++mx) {
      if (!sMask[maskRow + mx])
        continue;

      const int sx = tx + mx;
      const float4 v = tile[sy * TILE_WIDTH + sx];

      minVal.x = fminf(minVal.x, v.x);
      minVal.y = fminf(minVal.y, v.y);
      minVal.z = fminf(minVal.z, v.z);
      minVal.w = fminf(minVal.w, v.w);
    }
  }

  float4 *dst_row = reinterpret_cast<float4 *>((char *)pDst + out_y * nDstStep);
  dst_row[out_x] = minVal;
}

__global__ void nppiDilate_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                         int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                         int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u result = dilateGeneral(pSrc, nSrcStep, x, y, width, height, pMask, maskWidth, maskHeight, anchorX, anchorY);

    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    dst_row[x] = result;
  }
}

__global__ void nppiDilate_8u_C1R_kernel_tiled(const Npp8u *__restrict__ pSrc, int nSrcStep,
                                                   Npp8u *__restrict__ pDst, int nDstStep, int width, int height,
                                                   const Npp8u *__restrict__ pMask, int maskW, int maskH, int anchorX,
                                                   int anchorY, int tileH, int tilePitch) {
  extern __shared__ unsigned char smem[];

  unsigned char *tile = smem;
  unsigned char *mskSh = tile + tilePitch * tileH;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int tcount = blockDim.x * blockDim.y;

  // Output pixel start (4 pixels per thread horizontally)
  const int outX0 = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
  const int outY = blockIdx.y * blockDim.y + threadIdx.y;

  // Tile origin in source image (include anchor offset)
  const int tileOrgX = blockIdx.x * blockDim.x * 4 - anchorX;
  const int tileOrgY = blockIdx.y * blockDim.y - anchorY;

  // 1) Load tile from global memory (with border clamping)
  const int tileElems = tilePitch * tileH;
  for (int idx = tid; idx < tileElems; idx += tcount) {
    int ly = idx / tilePitch;
    int lx = idx % tilePitch;

    int sx = tileOrgX + lx;
    int sy = tileOrgY + ly;

    int clamped_x = min(max(sx, 0), width - 1);
    int clamped_y = min(max(sy, 0), height - 1);

    const Npp8u *srow = (const Npp8u *)((const char *)pSrc + clamped_y * nSrcStep);
    tile[ly * tilePitch + lx] = srow[clamped_x];
  }

  // 2) Load mask to shared memory
  const int msize = maskW * maskH;
  for (int i = tid; i < msize; i += tcount) {
    mskSh[i] = pMask[i];
  }

  __syncthreads();

  // 3) Process output pixels
  if (outY >= height)
    return;

  const int baseTileX = threadIdx.x * 4;
  const int baseTileY = threadIdx.y;

  for (int dx = 0; dx < 4; ++dx) {
    int outX = outX0 + dx;
    if (outX >= width)
      continue;

    unsigned char maxVal = 0;

    // Iterate mask window
    for (int my = 0; my < maskH; ++my) {
      const unsigned char *trow = tile + (baseTileY + my) * tilePitch;
      const unsigned char *mrow = mskSh + my * maskW;

#pragma unroll
      for (int mx = 0; mx < maskW; ++mx) {
        if (mrow[mx]) {
          unsigned char pixelVal = trow[baseTileX + dx + mx];
          if (pixelVal > maxVal) {
            maxVal = pixelVal;
          }
        }
      }
    }

    Npp8u *drow = (Npp8u *)((char *)pDst + outY * nDstStep);
    drow[outX] = maxVal;
  }
}

template <int BLOCK_X, int BLOCK_Y>
__global__ void nppiDilate_8u_C1R_3x3_kernel(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                             int nDstStep, int width, int height, const Npp8u *__restrict__ pMask) {
  // 3x3 mask in shared (9 bytes)
  __shared__ unsigned char maskSh[9];

  // Image tile in shared: (BLOCK_Y + 2) x (BLOCK_X + 2)
  __shared__ unsigned char tile[BLOCK_Y + 2][BLOCK_X + 2];

  // Load mask: 9 elements, one-time per block
  if (threadIdx.y == 0 && threadIdx.x < 9) {
    maskSh[threadIdx.x] = pMask[threadIdx.x];
  }
  __syncthreads();

  const int maxX = width - 1;
  const int maxY = height - 1;

  // Top-left corner of tile in the source image (including 1-pixel halo)
  const int x0 = blockIdx.x * BLOCK_X - 1;
  const int y0 = blockIdx.y * BLOCK_Y - 1;

  // 1) Load tile from global memory (replicate border)
  for (int ty = threadIdx.y; ty < BLOCK_Y + 2; ty += blockDim.y) {
    int gy = y0 + ty;
    int cy = gy;
    if (cy < 0)
      cy = 0;
    else if (cy > maxY)
      cy = maxY;

    const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + cy * nSrcStep);

    for (int tx = threadIdx.x; tx < BLOCK_X + 2; tx += blockDim.x) {
      int gx = x0 + tx;
      int cx = gx;
      if (cx < 0)
        cx = 0;
      else if (cx > maxX)
        cx = maxX;

      tile[ty][tx] = srcRow[cx];
    }
  }

  __syncthreads();

  // 2) Compute one output pixel per thread
  const int x = blockIdx.x * BLOCK_X + threadIdx.x;
  const int y = blockIdx.y * BLOCK_Y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  // Position of the center in tile
  const int sx = threadIdx.x + 1;
  const int sy = threadIdx.y + 1;

  unsigned char maxVal = 0;
  unsigned char v;

#define APPLY(maskIndex, dy, dx)                                                                                       \
  if (maskSh[maskIndex]) {                                                                                             \
    v = tile[sy + (dy)][sx + (dx)];                                                                                    \
    if (v > maxVal)                                                                                                    \
      maxVal = v;                                                                                                      \
  }

  // 3x3 neighborhood with maskSh[0..8] in row-major order
  // row -1
  APPLY(0, -1, -1);
  APPLY(1, -1, 0);
  APPLY(2, -1, 1);
  // row 0
  APPLY(3, 0, -1);
  APPLY(4, 0, 0);
  APPLY(5, 0, 1);
  // row +1
  APPLY(6, 1, -1);
  APPLY(7, 1, 0);
  APPLY(8, 1, 1);

#undef APPLY

  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);
  dstRow[x] = maxVal;
}

__global__ void nppiDilate_8u_C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                         int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                         int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);

    // Process each channel separately
    for (int c = 0; c < 4; c++) {
      const Npp8u *src_chan = pSrc + c;
      Npp8u *dst_chan = dst_row + x * 4 + c;

      Npp8u maxVal = 0;
      for (int my = 0; my < maskHeight; my++) {
        for (int mx = 0; mx < maskWidth; mx++) {
          if (pMask[my * maskWidth + mx] != 0) {
            int srcY = y + my - anchorY;
            int srcX = x + mx - anchorX;

            srcY = max(0, min(srcY, height - 1));
            srcX = max(0, min(srcX, width - 1));

            const Npp8u *src_row = (const Npp8u *)((const char *)src_chan + srcY * nSrcStep);
            Npp8u pixelValue = src_row[srcX * 4];

            if (pixelValue > maxVal) {
              maxVal = pixelValue;
            }
          }
        }
      }
      *dst_chan = maxVal;
    }
  }
}
__global__ void nppiDilate_8u_C4R_kernel_tiled(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                               int nDstStep, int width, int height, const Npp8u *__restrict__ pMask,
                                               int maskWidth, int maskHeight, int anchorX, int anchorY) {
  extern __shared__ unsigned char shared[];

  constexpr int PIXELS_PER_THREAD_X = 4;
  const int blockSpanX = blockDim.x * PIXELS_PER_THREAD_X;

  // tile width covered the entire block pixel range
  const int tileWidth = blockSpanX + maskWidth - 1;
  const int tileHeight = blockDim.y + maskHeight - 1;

  uchar4 *tile = reinterpret_cast<uchar4 *>(shared);
  unsigned char *maskShared = reinterpret_cast<unsigned char *>(tile + tileWidth * tileHeight);

  const int tileOriginX = blockIdx.x * blockSpanX - anchorX;
  const int tileOriginY = blockIdx.y * blockDim.y - anchorY;

  const int linearThreadId = threadIdx.y * blockDim.x + threadIdx.x;
  const int threadsPerBlock = blockDim.x * blockDim.y;

  // load tile
  const int tileSize = tileWidth * tileHeight;
  for (int idx = linearThreadId; idx < tileSize; idx += threadsPerBlock) {
    int localY = idx / tileWidth;
    int localX = idx % tileWidth;

    int srcX = tileOriginX + localX;
    int srcY = tileOriginY + localY;

    srcX = max(0, min(srcX, width - 1));
    srcY = max(0, min(srcY, height - 1));

    const uchar4 *src_row = reinterpret_cast<const uchar4 *>((const char *)pSrc + srcY * nSrcStep);
    tile[idx] = src_row[srcX];
  }

  // load mask
  const int maskSize = maskWidth * maskHeight;
  for (int idx = linearThreadId; idx < maskSize; idx += threadsPerBlock) {
    maskShared[idx] = pMask[idx];
  }

  __syncthreads();

  const int outY = blockIdx.y * blockDim.y + threadIdx.y;
  const int outXBase = (blockIdx.x * blockDim.x + threadIdx.x) * PIXELS_PER_THREAD_X;
  if (outY >= height || outXBase >= width)
    return;

  uchar4 *dst_row = reinterpret_cast<uchar4 *>((char *)pDst + outY * nDstStep);

#pragma unroll
  for (int i = 0; i < PIXELS_PER_THREAD_X; ++i) {
    int outX = outXBase + i;
    if (outX >= width)
      break;

    uchar4 minVal = make_uchar4(0, 0, 0, 0);

    for (int my = 0; my < maskHeight; ++my) {
      const unsigned char *maskRow = maskShared + my * maskWidth;
      const uchar4 *tileRow = tile + (threadIdx.y + my) * tileWidth;
#pragma unroll 4
      for (int mx = 0; mx < maskWidth; ++mx) {
        if (maskRow[mx]) {
          int tileX = threadIdx.x * PIXELS_PER_THREAD_X + i + mx;
          uchar4 pixelValue = tileRow[tileX];

          minVal.x = max(minVal.x, pixelValue.x);
          minVal.y = max(minVal.y, pixelValue.y);
          minVal.z = max(minVal.z, pixelValue.z);
          minVal.w = max(minVal.w, pixelValue.w);
        }
      }
    }

    dst_row[outX] = minVal;
  }
}
// opt for masksize 3x3
__global__ void nppiDilate_8u_C4R_3x3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                             int height) {

  const int BLOCK_WIDTH = 32; // blockSize.x
  const int BLOCK_HEIGHT = 8; // blockSize.y
  const int SHARED_WIDTH = 34;
  const int SHARED_HEIGHT = 10;

  // fixed shared mem
  __shared__ Npp8u shared_mem[10][136]; // [blockSize.y + 2][(blockSize.x + 2)*4]

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int blockX = blockIdx.x * BLOCK_WIDTH;
  int blockY = blockIdx.y * BLOCK_HEIGHT;

  // load
  for (int pos = ty * blockDim.x + tx; pos < SHARED_HEIGHT * SHARED_WIDTH; pos += blockDim.x * blockDim.y) {
    int loadY = pos / SHARED_WIDTH;
    int loadX = pos % SHARED_WIDTH;

    int srcY = blockY + loadY - 1;
    int srcX = blockX + loadX - 1;

    if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
      const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + srcY * nSrcStep);
      const Npp8u *src_pixel = &src_row[srcX * 4];

      shared_mem[loadY][loadX * 4 + 0] = src_pixel[0];
      shared_mem[loadY][loadX * 4 + 1] = src_pixel[1];
      shared_mem[loadY][loadX * 4 + 2] = src_pixel[2];
      shared_mem[loadY][loadX * 4 + 3] = src_pixel[3];
    } else {
      shared_mem[loadY][loadX * 4 + 0] = 0;
      shared_mem[loadY][loadX * 4 + 1] = 0;
      shared_mem[loadY][loadX * 4 + 2] = 0;
      shared_mem[loadY][loadX * 4 + 3] = 0;
    }
  }

  __syncthreads();

  int x = blockX + tx;
  int y = blockY + ty;

  if (x < width && y < height) {
    int shared_y = ty + 1;
    int shared_x = tx + 1;
    int idx = shared_x * 4;

    Npp8u maxVal[4];

    // init as center pixel
    maxVal[0] = shared_mem[shared_y][idx + 0];
    maxVal[1] = shared_mem[shared_y][idx + 1];
    maxVal[2] = shared_mem[shared_y][idx + 2];
    maxVal[3] = shared_mem[shared_y][idx + 3];

    // 3x3 field compare
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0)
          continue;

        int neighbor_y = shared_y + dy;
        int neighbor_x = shared_x + dx;
        int neighbor_idx = neighbor_x * 4;

        maxVal[0] = max(maxVal[0], shared_mem[neighbor_y][neighbor_idx + 0]);
        maxVal[1] = max(maxVal[1], shared_mem[neighbor_y][neighbor_idx + 1]);
        maxVal[2] = max(maxVal[2], shared_mem[neighbor_y][neighbor_idx + 2]);
        maxVal[3] = max(maxVal[3], shared_mem[neighbor_y][neighbor_idx + 3]);
      }
    }

    Npp8u *dst_row = (Npp8u *)((char *)pDst + y * nDstStep);
    Npp8u *dst_pixel = &dst_row[x * 4];

    dst_pixel[0] = maxVal[0];
    dst_pixel[1] = maxVal[1];
    dst_pixel[2] = maxVal[2];
    dst_pixel[3] = maxVal[3];
  }
}

__global__ void nppiDilate_32f_C1R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                          int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                          int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f result = dilateGeneral(pSrc, nSrcStep, x, y, width, height, pMask, maskWidth, maskHeight, anchorX, anchorY);

    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);
    dst_row[x] = result;
  }
}

__global__ void nppiDilate_32f_C1R_kernel_opt(const Npp32f* pSrc, int nSrcStep,
                                                   Npp32f* pDst, int nDstStep, int width, int height,
                                                   const Npp8u* pMask, int maskW, int maskH, int anchorX,
                                                   int anchorY, int tileH, int tilePitch) {
  extern __shared__ Npp32f smem_32f_c1r[];

  Npp32f *tile = smem_32f_c1r;
  Npp32f *mskSh = tile + tilePitch * tileH;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int tcount = blockDim.x * blockDim.y;

  // Output pixel start (4 pixels per thread horizontally)
  const int outX0 = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
  const int outY = blockIdx.y * blockDim.y + threadIdx.y;

  // Tile origin in source image (include anchor offset)
  const int tileOrgX = blockIdx.x * blockDim.x * 4 - anchorX;
  const int tileOrgY = blockIdx.y * blockDim.y - anchorY;

  // 1) Load tile from global memory (with border clamping)
  const int tileElems = tilePitch * tileH;
  for (int idx = tid; idx < tileElems; idx += tcount) {
    int ly = idx / tilePitch;
    int lx = idx % tilePitch;

    int sx = tileOrgX + lx;
    int sy = tileOrgY + ly;

    int clamped_x = min(max(sx, 0), width - 1);
    int clamped_y = min(max(sy, 0), height - 1);

    const Npp32f *srow = (const Npp32f *)((const char *)pSrc + clamped_y * nSrcStep);
    tile[ly * tilePitch + lx] = srow[clamped_x];
  }

  // 2) Load mask to shared memory
  const int msize = maskW * maskH;
  for (int i = tid; i < msize; i += tcount) {
    mskSh[i] = pMask[i];
  }

  __syncthreads();

  // 3) Process output pixels
  if (outY >= height)
    return;

  const int baseTileX = threadIdx.x * 4;
  const int baseTileY = threadIdx.y;

  for (int dx = 0; dx < 4; ++dx) {
    int outX = outX0 + dx;
    if (outX >= width)
      continue;

    Npp32f maxVal = -1e10f;

    // Iterate mask window
    for (int my = 0; my < maskH; ++my) {
      const Npp32f *trow = tile + (baseTileY + my) * tilePitch;
      const Npp32f *mrow = mskSh + my * maskW;

#pragma unroll
      for (int mx = 0; mx < maskW; ++mx) {
        if (mrow[mx]) {
          Npp32f pixelVal = trow[baseTileX + dx + mx];
          if (pixelVal > maxVal) {
            maxVal = pixelVal;
          }
        }
      }
    }

    Npp32f *drow = (Npp32f *)((char *)pDst + outY * nDstStep);
    drow[outX] = maxVal;
  }
}

__global__ void nppiDilate_32f_C4R_kernel(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, int width,
                                          int height, const Npp8u *pMask, int maskWidth, int maskHeight, int anchorX,
                                          int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    Npp32f *dst_row = (Npp32f *)((char *)pDst + y * nDstStep);

    // Process each channel separately
    for (int c = 0; c < 4; c++) {
      const Npp32f *src_chan = pSrc + c;
      Npp32f *dst_chan = dst_row + x * 4 + c;

      Npp32f maxVal = -1e10f;
      for (int my = 0; my < maskHeight; my++) {
        for (int mx = 0; mx < maskWidth; mx++) {
          if (pMask[my * maskWidth + mx] != 0) {
            int srcY = y + my - anchorY;
            int srcX = x + mx - anchorX;

            srcY = max(0, min(srcY, height - 1));
            srcX = max(0, min(srcX, width - 1));

            const Npp32f *src_row = (const Npp32f *)((const char *)src_chan + srcY * nSrcStep);
            Npp32f pixelValue = src_row[srcX * 4];

            if (pixelValue > maxVal) {
              maxVal = pixelValue;
            }
          }
        }
      }
      *dst_chan = maxVal;
    }
  }
}

__global__ void nppiDilate_32f_C4R_kernel_opt(const Npp32f *__restrict__ pSrc, int nSrcStep, Npp32f *__restrict__ pDst,
                                              int nDstStep, int width, int height, const Npp8u *__restrict__ pMask,
                                              int maskWidth, int maskHeight, int anchorX, int anchorY) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tile_x = blockIdx.x * blockDim.x;
  const int tile_y = blockIdx.y * blockDim.y;
  const int out_x = tile_x + tx;
  const int out_y = tile_y + ty;
  if (out_x >= width || out_y >= height)
    return;

  const int TILE_WIDTH = blockDim.x + maskWidth - 1;
  const int TILE_HEIGHT = blockDim.y + maskHeight - 1;
  const int maskSize = maskWidth * maskHeight;

  extern __shared__ unsigned char sdata[];

  float4 *tile = reinterpret_cast<float4 *>(sdata);
  Npp8u *sMask = reinterpret_cast<Npp8u *>(tile + TILE_WIDTH * TILE_HEIGHT);

  const int tid = ty * blockDim.x + tx;
  const int numThreads = blockDim.x * blockDim.y;

  // mask -> shared
  for (int idx = tid; idx < maskSize; idx += numThreads) {
    sMask[idx] = pMask[idx];
  }
  __syncthreads();

  // tile + halo -> shared
  for (int j = ty; j < TILE_HEIGHT; j += blockDim.y) {
    int gy = tile_y + j - anchorY;
    gy = max(0, min(gy, height - 1));
    const float4 *src_row = reinterpret_cast<const float4 *>((const char *)pSrc + gy * nSrcStep);

    for (int i = tx; i < TILE_WIDTH; i += blockDim.x) {
      int gx = tile_x + i - anchorX;
      gx = max(0, min(gx, width - 1));
      tile[j * TILE_WIDTH + i] = src_row[gx];
    }
  }
  __syncthreads();

  float4 maxVal = make_float4(-1e10f, -1e10f, -1e10f, -1e10f);

  for (int my = 0; my < maskHeight; ++my) {
    const int sy = ty + my;
    const int maskRow = my * maskWidth;

    for (int mx = 0; mx < maskWidth; ++mx) {
      if (!sMask[maskRow + mx])
        continue;

      const int sx = tx + mx;
      const float4 v = tile[sy * TILE_WIDTH + sx];

      maxVal.x = fmaxf(maxVal.x, v.x);
      maxVal.y = fmaxf(maxVal.y, v.y);
      maxVal.z = fmaxf(maxVal.z, v.z);
      maxVal.w = fmaxf(maxVal.w, v.w);
    }
  }

  float4 *dst_row = reinterpret_cast<float4 *>((char *)pDst + out_y * nDstStep);
  dst_row[out_x] = maxVal;
}

//=============================================================================
// Host functions for general morphology operations
//=============================================================================

extern "C" {

NppStatus nppiErode_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {

  // Vectorized path (128-bit loads, SWAR byte-parallel min/max; separable
  // reduction for fully populated structuring elements, per-tap otherwise);
  // the original kernels below stay as the fallback for unaligned steps /
  // other mask widths.
  if (morphVecLaunch8uC1<false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask,
                                oMaskSize.width, oMaskSize.height, oAnchor.x, oAnchor.y, nppStreamCtx.hStream))
    return NPP_SUCCESS;

  dim3 blockSize(32, 8);
  if (oMaskSize.width == 3 && oMaskSize.height == 3 && oAnchor.x == 1 && oAnchor.y == 1) {
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    nppiErode_8u_C1R_3x3_kernel<32, 8><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask);
    cudaError_t cudaStatus = cudaGetLastError();
    return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // general opt
  dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Tile geometry (must match kernel)
  const int tileW_noPad = blockSize.x * 4 + oMaskSize.width - 1;
  const int tileH = blockSize.y + oMaskSize.height - 1;
  const int tilePitch = (tileW_noPad + 3) & ~3;

  const size_t tileSharedElements = size_t(tilePitch) * tileH;
  const size_t maskSharedElements = size_t(oMaskSize.width) * oMaskSize.height;
  const size_t sharedBytes = (tileSharedElements + maskSharedElements) * sizeof(Npp8u);

  nppiErode_8u_C1R_kernel_tiled<<<gridSize, blockSize, sharedBytes, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y, tileH, tilePitch);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiErode_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppStreamContext nppStreamCtx) {
  if (oMaskSize.width == 3 && oMaskSize.height == 3 && oAnchor.x == 1 && oAnchor.y == 1) {
    dim3 blockSize(32, 8);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y); // special for 3x3
    nppiErode_8u_C4R_3x3_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                  oSizeROI.width, oSizeROI.height);
  } else {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    const int tileWidth = blockSize.x * 4 + oMaskSize.width - 1;
    const int tileHeight = blockSize.y + oMaskSize.height - 1;
    const size_t tileSharedElements = static_cast<size_t>(tileWidth) * tileHeight;
    const size_t maskSharedElements = static_cast<size_t>(oMaskSize.width) * oMaskSize.height;
    const size_t sharedBytes = tileSharedElements * sizeof(uchar4) + maskSharedElements * sizeof(Npp8u);

    nppiErode_8u_C4R_kernel_tiled<<<gridSize, blockSize, sharedBytes, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
        oAnchor.x, oAnchor.y);
  }
  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiErode_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(32, 4);
  dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Tile geometry (must match kernel)
  const int tileW_noPad = blockSize.x * 4 + oMaskSize.width - 1;
  const int tileH = blockSize.y + oMaskSize.height - 1;
  const int tilePitch = (tileW_noPad + 3) & ~3;

  const size_t tileSharedElements = size_t(tilePitch) * tileH;
  const size_t maskSharedElements = size_t(oMaskSize.width) * oMaskSize.height;
  const size_t sharedBytes = (tileSharedElements + maskSharedElements) * sizeof(Npp32f);

  nppiErode_32f_C1R_kernel_opt<<<gridSize, blockSize, sharedBytes, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y, tileH, tilePitch);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiErode_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  const int tileW = blockSize.x + oMaskSize.width - 1;
  const int tileH = blockSize.y + oMaskSize.height - 1;

  size_t sharedSize = static_cast<size_t>(tileW) * tileH * sizeof(float4) +
                      static_cast<size_t>(oMaskSize.width) * oMaskSize.height * sizeof(Npp8u);

  nppiErode_32f_C4R_kernel_opt<<<gridSize, blockSize, sharedSize, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDilate_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {

  // Vectorized path (128-bit loads, SWAR byte-parallel min/max; separable
  // reduction for fully populated structuring elements, per-tap otherwise);
  // the original kernels below stay as the fallback for unaligned steps /
  // other mask widths.
  if (morphVecLaunch8uC1<true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask,
                               oMaskSize.width, oMaskSize.height, oAnchor.x, oAnchor.y, nppStreamCtx.hStream))
    return NPP_SUCCESS;

  dim3 blockSize(32, 8);

  if (oMaskSize.width == 3 && oMaskSize.height == 3 && oAnchor.x == 1 && oAnchor.y == 1) {
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    nppiDilate_8u_C1R_3x3_kernel<32, 8><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask);
    cudaError_t cudaStatus = cudaGetLastError();
    return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  const int tileW_noPad = blockSize.x * 4 + oMaskSize.width - 1;
  const int tileH = blockSize.y + oMaskSize.height - 1;
  const int tilePitch = (tileW_noPad + 3) & ~3;

  const size_t tileSharedElements = size_t(tilePitch) * tileH;
  const size_t maskSharedElements = size_t(oMaskSize.width) * oMaskSize.height;
  const size_t sharedBytes = (tileSharedElements + maskSharedElements) * sizeof(Npp8u);

  nppiDilate_8u_C1R_kernel_tiled<<<gridSize, blockSize, sharedBytes, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y, tileH, tilePitch);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDilate_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppStreamContext nppStreamCtx) {

  if (oMaskSize.width == 3 && oMaskSize.height == 3 && oAnchor.x == 1 && oAnchor.y == 1) {
    dim3 blockSize(32, 8);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y); // special for 3x3
    nppiDilate_8u_C4R_3x3_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                                                   oSizeROI.width, oSizeROI.height);
  } else {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    const int tileWidth = blockSize.x * 4 + oMaskSize.width - 1;
    const int tileHeight = blockSize.y + oMaskSize.height - 1;
    const size_t tileSharedElements = static_cast<size_t>(tileWidth) * tileHeight;
    const size_t maskSharedElements = static_cast<size_t>(oMaskSize.width) * oMaskSize.height;
    const size_t sharedBytes = tileSharedElements * sizeof(uchar4) + maskSharedElements * sizeof(Npp8u);

    nppiDilate_8u_C4R_kernel_tiled<<<gridSize, blockSize, sharedBytes, nppStreamCtx.hStream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
        oAnchor.x, oAnchor.y);
  }
  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDilate_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppStreamContext nppStreamCtx) {
  dim3 blockSize(32, 4);
  dim3 gridSize((oSizeROI.width + blockSize.x * 4 - 1) / (blockSize.x * 4),
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  const int tileW_noPad = blockSize.x * 4 + oMaskSize.width - 1;
  const int tileH = blockSize.y + oMaskSize.height - 1;
  const int tilePitch = (tileW_noPad + 3) & ~3;

  const size_t tileSharedElements = size_t(tilePitch) * tileH;
  const size_t maskSharedElements = size_t(oMaskSize.width) * oMaskSize.height;
  const size_t sharedBytes = (tileSharedElements + maskSharedElements) * sizeof(Npp32f);

  nppiDilate_32f_C1R_kernel_opt<<<gridSize, blockSize, sharedBytes, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y, tileH, tilePitch);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiDilate_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp8u *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppStreamContext nppStreamCtx) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  const int tileW = blockSize.x + oMaskSize.width - 1;
  const int tileH = blockSize.y + oMaskSize.height - 1;

  size_t sharedSize = static_cast<size_t>(tileW) * tileH * sizeof(float4) +
                      static_cast<size_t>(oMaskSize.width) * oMaskSize.height * sizeof(Npp8u);

  nppiDilate_32f_C4R_kernel_opt<<<gridSize, blockSize, sharedSize, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, pMask, oMaskSize.width, oMaskSize.height,
      oAnchor.x, oAnchor.y);

  cudaError_t cudaStatus = cudaGetLastError();
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}
}
