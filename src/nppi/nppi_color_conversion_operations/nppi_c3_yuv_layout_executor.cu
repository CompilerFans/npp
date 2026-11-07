// c3 YUV/YCbCr layout executor kernels.
//
// Every conversion here is a pure byte permutation / decimation of the
// source samples: no color-space arithmetic happens in this file.  The
// mapping was measured in the reference implementation with NVIDIA NPP 13.4 (reference outputs are
// recorded by reference comparisons and is
// reproduced verbatim, including the 4:1:1 -> 4:2:0 row-interleaving quirk.

#include "nppi_c3_yuv_layout_executor.h"

#include <cuda_runtime.h>

namespace nppi {
namespace c3yuv {
namespace {

constexpr int kBlockX = 128;
constexpr int kBlockY = 8;

inline dim3 gridFor(int w, int h) {
  return dim3((w + kBlockX - 1) / kBlockX, (h + kBlockY - 1) / kBlockY);
}

// NVIDIA compares the step as unsigned against the minimum row bytes.
inline bool step_ok(int nStep, int nMinRowBytes) {
  return static_cast<unsigned>(nStep) >= static_cast<unsigned>(nMinRowBytes);
}
inline bool roi_valid(NppiSize roi) { return roi.width >= 0 && roi.height >= 0; }
inline NppStatus launch_ok() {
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

// Packed C2 pair layouts (measured):
//   0 YCbCr422: [Y0 Cb Y1 Cr]   (luma in slots 0/2)
//   1 YCrCb422: [Y0 Cr Y1 Cb]   (luma in slots 0/2, chroma swapped)
//   2 CbYCr422: [Cb Y0 Cr Y1]   (luma in slots 1/3)
__device__ __forceinline__ int c2_y_offset(int x, int layout) {
  return ((x & 1) << 1) + (layout == 2 ? 1: 0);
}
__device__ __forceinline__ int c2_cb_offset(int layout) { return layout == 0 ? 1: (layout == 1 ? 3: 0); }
__device__ __forceinline__ int c2_cr_offset(int layout) { return layout == 0 ? 3: (layout == 1 ? 1: 2); }

__global__ void k_y_from_c2(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst, int nDstStep,
                            int width, int height, int yw, int layout) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= yw || y >= height) {
    return;
  }
  const Npp8u *s = pSrc + (size_t)y * nSrcStep + (size_t)(x >> 1) * 4;
  pDst[(size_t)y * nDstStep + x] = s[c2_y_offset(x, layout)];
}

// Chroma planes of a packed C2 image: destination sample (cx, cy) reads
// packed pair (cx * px, cy * py).
__global__ void k_chroma_from_c2(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst0, int nDst0Step,
                                 Npp8u *__restrict__ pDst1, int nDst1Step, int cw, int ch, int px, int py,
                                 int layout) {
  const int cx = blockIdx.x * blockDim.x + threadIdx.x;
  const int cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cx >= cw || cy >= ch) {
    return;
  }
  const Npp8u *s = pSrc + (size_t)(cy * py) * nSrcStep + (size_t)(cx * px) * 4;
  pDst0[(size_t)cy * nDst0Step + cx] = s[c2_cb_offset(layout)];
  pDst1[(size_t)cy * nDst1Step + cx] = s[c2_cr_offset(layout)];
}

// Same, but the chroma samples go to one interleaved (Cb,Cr) plane.
__global__ void k_chroma_from_c2_p2(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst,
                                    int nDstStep, int cw, int ch, int px, int py, int layout) {
  const int cx = blockIdx.x * blockDim.x + threadIdx.x;
  const int cy = blockIdx.y * blockDim.y + threadIdx.y;
  if (cx >= cw || cy >= ch) {
    return;
  }
  const Npp8u *s = pSrc + (size_t)(cy * py) * nSrcStep + (size_t)(cx * px) * 4;
  Npp8u *d = pDst + (size_t)cy * nDstStep + (size_t)cx * 2;
  d[0] = s[c2_cb_offset(layout)];
  d[1] = s[c2_cr_offset(layout)];
}

// NVIDIA writes luma in whole pixel groups determined by the destination
// chroma subsampling: 4-pixel groups for 4:1:1 destinations, 2-pixel groups
// otherwise.  The trailing partial group is never written.  `height` already
// carries the 4:2:0 row rounding.
__global__ void k_copy_y(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst, int nDstStep, int width,
                         int height, int yw) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= yw || y >= height) {
    return;
  }
  pDst[(size_t)y * nDstStep + x] = pSrc[(size_t)y * nSrcStep + x];
}

// Planar chroma -> packed C2 pairs.
// srcMode 0: chroma column == pair index; 1: pair / 2 (4:1:1); 2: row / 2 (4:2:0).
template <int SRCSWAP, int SWAPDST>
__global__ void k_p3_to_c2(const Npp8u *__restrict__ pSrc0, const Npp8u *__restrict__ pSrc1,
                           const Npp8u *__restrict__ pSrc2, int s0, int s1, int s2, Npp8u *__restrict__ pDst,
                           int nDstStep, int width, int height, int srcMode) {
  // The three plane steps are passed by value: a host int[3] cannot be
  // dereferenced from device code (ATU fault on MACA).
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (p >= (width >> 1) || y >= height) {
    return;
  }
  const int col = (srcMode == 1) ? (p >> 1): p;
  const int row = (srcMode == 2) ? (y >> 1): y;
  const Npp8u *planeCb = SRCSWAP ? pSrc2: pSrc1;
  const Npp8u *planeCr = SRCSWAP ? pSrc1: pSrc2;
  const int cbStep = SRCSWAP ? s2: s1;
  const int crStep = SRCSWAP ? s1: s2;
  const Npp8u cb = planeCb[(size_t)row * cbStep + col];
  const Npp8u cr = planeCr[(size_t)row * crStep + col];
  const Npp8u *sy = pSrc0 + (size_t)y * s0;
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)p * 4;
  d[c2_y_offset(2 * p, SWAPDST)] = sy[2 * p];
  d[c2_y_offset(2 * p + 1, SWAPDST)] = sy[2 * p + 1];
  d[c2_cb_offset(SWAPDST)] = cb;
  d[c2_cr_offset(SWAPDST)] = cr;
}

// Planar Y + interleaved 4:1:1 CbCr -> packed C2 4:2:2.
template <int SWAPDST>
__global__ void k_p2_to_c2_411(const Npp8u *__restrict__ pSrcY, int nSrcYStep, const Npp8u *__restrict__ pSrcC,
                               int nSrcCStep, Npp8u *__restrict__ pDst, int nDstStep, int width, int height) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (p >= (width >> 1) || y >= height) {
    return;
  }
  const int col = p >> 1;
  const Npp8u *s = pSrcC + (size_t)y * nSrcCStep;
  const Npp8u *sy = pSrcY + (size_t)y * nSrcYStep;
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)p * 4;
  d[c2_y_offset(2 * p, SWAPDST)] = sy[2 * p];
  d[c2_y_offset(2 * p + 1, SWAPDST)] = sy[2 * p + 1];
  d[c2_cb_offset(SWAPDST)] = s[2 * col];
  d[c2_cr_offset(SWAPDST)] = s[2 * col + 1];
}

// Planar Y + interleaved 4:1:1 CbCr -> planar P3 4:1:1.
__global__ void k_p2_to_p3_411(const Npp8u *__restrict__ pSrcY, int nSrcYStep, const Npp8u *__restrict__ pSrcC,
                               int nSrcCStep, Npp8u *__restrict__ pDstY, int nDstYStep, Npp8u *__restrict__ pDstCb,
                               int nDstCbStep, Npp8u *__restrict__ pDstCr, int nDstCrStep, int width, int height,
                               int yw) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= yw || y >= height) {
    return;
  }
  pDstY[(size_t)y * nDstYStep + x] = pSrcY[(size_t)y * nSrcYStep + x];
  if (x < (width >> 2)) {
    const Npp8u *s = pSrcC + (size_t)y * nSrcCStep;
    pDstCb[(size_t)y * nDstCbStep + x] = s[2 * x];
    pDstCr[(size_t)y * nDstCrStep + x] = s[2 * x + 1];
  }
}

// Planar Y + interleaved 4:1:1 CbCr -> planar P3 4:2:2 (chroma replicated
// across the two pixel pairs that share a 4:1:1 sample).
__global__ void k_p2_to_p3_422(const Npp8u *__restrict__ pSrcY, int nSrcYStep, const Npp8u *__restrict__ pSrcC,
                               int nSrcCStep, Npp8u *__restrict__ pDstY, int nDstYStep, Npp8u *__restrict__ pDstCb,
                               int nDstCbStep, Npp8u *__restrict__ pDstCr, int nDstCrStep, int width, int height,
                               int yw) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= yw || y >= height) {
    return;
  }
  pDstY[(size_t)y * nDstYStep + x] = pSrcY[(size_t)y * nSrcYStep + x];
  if (x < (width >> 1)) {
    const Npp8u *s = pSrcC + (size_t)y * nSrcCStep;
    pDstCb[(size_t)y * nDstCbStep + x] = s[2 * (x >> 1)];
    pDstCr[(size_t)y * nDstCrStep + x] = s[2 * (x >> 1) + 1];
  }
}

// Reorder packed C2 chroma/Y slots between layouts.
__global__ void k_c2_reorder(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDst, int nDstStep,
                             int width, int height, int layoutIn, int layoutOut) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (p >= (width >> 1) || y >= height) {
    return;
  }
  const Npp8u *s = pSrc + (size_t)y * nSrcStep + (size_t)p * 4;
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)p * 4;
  d[c2_y_offset(2 * p, layoutOut)] = s[c2_y_offset(2 * p, layoutIn)];
  d[c2_y_offset(2 * p + 1, layoutOut)] = s[c2_y_offset(2 * p + 1, layoutIn)];
  d[c2_cb_offset(layoutOut)] = s[c2_cb_offset(layoutIn)];
  d[c2_cr_offset(layoutOut)] = s[c2_cr_offset(layoutIn)];
}

// Planar -> planar chroma resampling (MODE documented in the header).
template <int MODE>
__global__ void k_p3_resample(const Npp8u *__restrict__ pSrcCb, const Npp8u *__restrict__ pSrcCr, int nSrcCStep,
                              Npp8u *__restrict__ pDstCb, Npp8u *__restrict__ pDstCr, int nDstCStep, int width,
                              int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  int cw = width >> 1;
  int ch = height;
  if (MODE == 1) {
    cw = width >> 2;
  } else if (MODE == 2 || MODE == 4) {
    ch = height >> 1;
  } else if (MODE == 3) {
    // Measured: the 4:2:0 -> 4:2:2 destination keeps only height & ~1 rows.
    ch = height & ~1;
  }
  if (x >= cw || y >= ch) {
    return;
  }
  int sx = x;
  int sy = y;
  if (MODE == 0) {
    sx = x >> 1;
  } else if (MODE == 1) {
    sx = 2 * x;
  } else if (MODE == 2) {
    sy = 2 * y;
  } else if (MODE == 3) {
    sy = y >> 1;
  } else if (MODE == 4) {
    sx = x >> 1;
    sy = 2 * y + (x & 1);
  }
  pDstCb[(size_t)y * nDstCStep + x] = pSrcCb[(size_t)sy * nSrcCStep + sx];
  pDstCr[(size_t)y * nDstCStep + x] = pSrcCr[(size_t)sy * nSrcCStep + sx];
}

// Planar Y + chroma -> planar Y + interleaved CbCr (MODE as in header).
template <int MODE>
__global__ void k_p3_to_p2(const Npp8u *__restrict__ pSrcCb, const Npp8u *__restrict__ pSrcCr, int nSrcCStep,
                           Npp8u *__restrict__ pDst, int nDstStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (MODE == 3) {
    // 4:1:1 -> 4:2:0: byte index 2 * x with x = 2c + dy (a full-width row).
    if (x >= (width >> 1) || y >= ((height & ~1) >> 1)) {
      return;
    }
    const int c = x >> 1;
    const int dy = x & 1;
    const int sy = 2 * y + dy;
    Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)(2 * x);
    d[0] = pSrcCb[(size_t)sy * nSrcCStep + c];
    d[1] = pSrcCr[(size_t)sy * nSrcCStep + c];
    return;
  }
  int cw = width >> 2;
  int ch = height;
  if (MODE == 1) {
    cw = width >> 1;
    ch = height >> 1;
  }
  if (x >= cw || y >= ch) {
    return;
  }
  int sx = x;
  int sy = y;
  if (MODE == 0) {
    sx = 2 * x;
  } else if (MODE == 1) {
    sy = 2 * y;
  }
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)x * 2;
  d[0] = pSrcCb[(size_t)sy * nSrcCStep + sx];
  d[1] = pSrcCr[(size_t)sy * nSrcCStep + sx];
}

// Interleaved 4:1:1 CbCr -> planar 4:2:0 chroma planes (NVIDIA quirk).
template <int SWAPDST>
__global__ void k_p2_to_p3_420(const Npp8u *__restrict__ pSrcC, int nSrcCStep, Npp8u *__restrict__ pDstCb,
                               int nDstCbStep, Npp8u *__restrict__ pDstCr, int nDstCrStep, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (width >> 1) || j >= ((height & ~1) >> 1)) {
    return;
  }
  const int sy = 2 * j + (x & 1);
  const int c = x >> 1;
  const Npp8u *s = pSrcC + (size_t)sy * nSrcCStep;
  pDstCb[(size_t)j * nDstCbStep + x] = s[2 * c];
  pDstCr[(size_t)j * nDstCrStep + x] = s[2 * c + 1];
}

} // namespace

NppStatus c2ToP3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], const int aDstStep[3], NppiSize oSizeROI,
                 cudaStream_t stream, int layoutSrc, int px, int py, int swapDst) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 2 * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  const int yh = (py == 2) ? (h & ~1): h;
  const int yw = w & ~((px == 2) ? 3: 1);
  k_y_from_c2<<<gridFor(yw, yh), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst[0], aDstStep[0], w, yh, yw,
                                                                    layoutSrc);
  const int cw = (px == 2) ? (w >> 2): (w >> 1);
  const int ch = (py == 2) ? (h >> 1): h;
  Npp8u *planeCb = swapDst ? pDst[2]: pDst[1];
  Npp8u *planeCr = swapDst ? pDst[1]: pDst[2];
  const int stepCb = swapDst ? aDstStep[2]: aDstStep[1];
  const int stepCr = swapDst ? aDstStep[1]: aDstStep[2];
  if (cw > 0 && ch > 0) {
    k_chroma_from_c2<<<gridFor(cw, ch), dim3(kBlockX, kBlockY), 0, stream>>>(
        pSrc, nSrcStep, planeCb, stepCb, planeCr, stepCr, cw, ch, px, py, layoutSrc);
  }
  return launch_ok();
}

NppStatus c2ToP2(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep, Npp8u *pDstCbCr, int nDstCbCrStep,
                 NppiSize oSizeROI, cudaStream_t stream, int layoutSrc, int px, int py) {
  if (!pSrc || !pDstY) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 2 * oSizeROI.width) || !step_ok(nDstYStep, oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  const int yh = (py == 2) ? (h & ~1): h;
  const int yw = w & ~((px == 2) ? 3: 1);
  k_y_from_c2<<<gridFor(yw, yh), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDstY, nDstYStep, w, yh, yw,
                                                                    layoutSrc);
  const int cw = (px == 2) ? (w >> 2): (w >> 1);
  const int ch = (py == 2) ? (h >> 1): h;
  if (cw > 0 && ch > 0 && pDstCbCr) {
    k_chroma_from_c2_p2<<<gridFor(cw, ch), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDstCbCr, nDstCbCrStep,
                                                                                cw, ch, px, py, layoutSrc);
  }
  return launch_ok();
}

NppStatus p3ToC2(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                 cudaStream_t stream, int swapSrc, int srcMode, int swapDst) {
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nDstStep, 2 * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int w = oSizeROI.width;
  const int h = (srcMode == 2) ? (oSizeROI.height & ~1): oSizeROI.height;
  const int pairs = w >> 1;
  if (pairs == 0 || h == 0) {
    return launch_ok();
  }
  dim3 g = gridFor(pairs, h);
#define LAUNCH_P3C2(SS, SD)                                                                                            \
  k_p3_to_c2<SS, SD><<<g, dim3(kBlockX, kBlockY), 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], aSrcStep[0], aSrcStep[1],  \
                                                               aSrcStep[2], pDst, nDstStep, w, h, srcMode)
  if (swapSrc) {
    if (swapDst) {
      LAUNCH_P3C2(1, 1);
    } else {
      LAUNCH_P3C2(1, 0);
    }
  } else {
    if (swapDst) {
      LAUNCH_P3C2(0, 1);
    } else {
      LAUNCH_P3C2(0, 0);
    }
  }
#undef LAUNCH_P3C2
  return launch_ok();
}

NppStatus p2ToC2_411(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr, int nSrcCbCrStep, Npp8u *pDst,
                     int nDstStep, NppiSize oSizeROI, cudaStream_t stream, int swapDst) {
  if (!pSrcY) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  // The reference validates the luma source step here (p2c2_small_src_y_step).
  if (!step_ok(nSrcYStep, oSizeROI.width) || !step_ok(nDstStep, 2 * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if ((w >> 1) == 0) {
    return launch_ok();
  }
  dim3 g = gridFor(w >> 1, h);
  if (swapDst) {
    k_p2_to_c2_411<1><<<g, dim3(kBlockX, kBlockY), 0, stream>>>(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst,
                                                               nDstStep, w, h);
  } else {
    k_p2_to_c2_411<0><<<g, dim3(kBlockX, kBlockY), 0, stream>>>(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst,
                                                               nDstStep, w, h);
  }
  return launch_ok();
}

NppStatus p2ToP3_411(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr, int nSrcCbCrStep, Npp8u *pDst[3],
                     const int aDstStep[3], NppiSize oSizeROI, cudaStream_t stream) {
  if (!pSrcY) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcYStep, oSizeROI.width) || !step_ok(aDstStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  const int yw = w & ~3;
  k_p2_to_p3_411<<<gridFor(yw, h), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst[0], aDstStep[0], pDst[1], aDstStep[1], pDst[2], aDstStep[2], w, h,
      yw);
  return launch_ok();
}

NppStatus p2ToP3_422(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr, int nSrcCbCrStep, Npp8u *pDst[3],
                     const int aDstStep[3], NppiSize oSizeROI, cudaStream_t stream) {
  if (!pSrcY) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcYStep, oSizeROI.width) || !step_ok(aDstStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  const int yw = w & ~1;
  k_p2_to_p3_422<<<gridFor(yw, h), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst[0], aDstStep[0], pDst[1], aDstStep[1], pDst[2], aDstStep[2], w, h,
      yw);
  return launch_ok();
}

NppStatus p3ToP2_411(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDstY, int nDstYStep, Npp8u *pDstCbCr,
                     int nDstCbCrStep, NppiSize oSizeROI, cudaStream_t stream) {
  return p3ToP2(pSrc, aSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI, stream, 0, 2);
}

NppStatus c2Reorder(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                    cudaStream_t stream, int layoutSrc, int layoutDst) {
  if (!pSrc) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 2 * oSizeROI.width) || !step_ok(nDstStep, 2 * oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  if ((w >> 1) == 0) {
    return launch_ok();
  }
  k_c2_reorder<<<gridFor(w >> 1, h), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, w, h,
                                                                         layoutSrc, layoutDst);
  return launch_ok();
}

NppStatus p3Resample(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], const int aDstStep[3],
                     NppiSize oSizeROI, cudaStream_t stream, int swapSrc, int swapDst, int mode) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width) || !step_ok(aDstStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  const int yh = (mode == 2 || mode == 3 || mode == 4) ? (h & ~1): h;
  const int yw = w & ~((mode == 1) ? 3: 1);
  k_copy_y<<<gridFor(yw, yh), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc[0], aSrcStep[0], pDst[0], aDstStep[0], w, yh,
                                                                 yw);
  const Npp8u *srcCb = swapSrc ? pSrc[2]: pSrc[1];
  const Npp8u *srcCr = swapSrc ? pSrc[1]: pSrc[2];
  Npp8u *dstCb = swapDst ? pDst[2]: pDst[1];
  Npp8u *dstCr = swapDst ? pDst[1]: pDst[2];
  const int srcCStep = swapSrc ? aSrcStep[2]: aSrcStep[1];
  const int dstCStep = swapDst ? aDstStep[2]: aDstStep[1];
  int cw = w >> 1;
  int ch = h;
  if (mode == 1) {
    cw = w >> 2;
  } else if (mode == 2 || mode == 4) {
    ch = (h & ~1) >> 1;
  }
  if (cw <= 0 || ch <= 0) {
    return launch_ok();
  }
  dim3 g = gridFor(cw, ch);
#define LAUNCH_RES(M)                                                                                                  \
  k_p3_resample<M><<<g, dim3(kBlockX, kBlockY), 0, stream>>>(srcCb, srcCr, srcCStep, dstCb, dstCr, dstCStep, w, h)
  switch (mode) {
  case 0:
    LAUNCH_RES(0);
    break;
  case 1:
    LAUNCH_RES(1);
    break;
  case 2:
    LAUNCH_RES(2);
    break;
  case 3:
    LAUNCH_RES(3);
    break;
  case 4:
    LAUNCH_RES(4);
    break;
  default:
    return NPP_BAD_ARGUMENT_ERROR;
  }
#undef LAUNCH_RES
  return launch_ok();
}

NppStatus p3ToP2(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDstY, int nDstYStep, Npp8u *pDstCbCr,
                 int nDstCbCrStep, NppiSize oSizeROI, cudaStream_t stream, int swapSrc, int mode) {
  if (!pSrc || !pDstY) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(aSrcStep[0], oSizeROI.width) || !step_ok(nDstYStep, oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  const int yh = (mode == 1 || mode == 3) ? (h & ~1): h;
  const int yw = w & ~((mode == 0 || mode == 2) ? 3: 1);
  k_copy_y<<<gridFor(yw, yh), dim3(kBlockX, kBlockY), 0, stream>>>(pSrc[0], aSrcStep[0], pDstY, nDstYStep, w, yh, yw);
  const Npp8u *srcCb = swapSrc ? pSrc[2]: pSrc[1];
  const Npp8u *srcCr = swapSrc ? pSrc[1]: pSrc[2];
  const int srcCStep = swapSrc ? aSrcStep[2]: aSrcStep[1];
  int cw = w >> 2;
  int ch = h;
  if (mode == 1) {
    cw = w >> 1;
    ch = (h & ~1) >> 1;
  } else if (mode == 3) {
    cw = w >> 1;
    ch = (h & ~1) >> 1;
  }
  if (cw <= 0 || ch <= 0) {
    return launch_ok();
  }
  dim3 g = gridFor(cw, ch);
#define LAUNCH_P3P2(M)                                                                                                 \
  k_p3_to_p2<M><<<g, dim3(kBlockX, kBlockY), 0, stream>>>(srcCb, srcCr, srcCStep, pDstCbCr, nDstCbCrStep, w, h)
  switch (mode) {
  case 0:
    LAUNCH_P3P2(0);
    break;
  case 1:
    LAUNCH_P3P2(1);
    break;
  case 2:
    LAUNCH_P3P2(2);
    break;
  case 3:
    LAUNCH_P3P2(3);
    break;
  default:
    return NPP_BAD_ARGUMENT_ERROR;
  }
#undef LAUNCH_P3P2
  return launch_ok();
}

NppStatus p2ToP3_420(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr, int nSrcCbCrStep, Npp8u *pDst[3],
                     const int aDstStep[3], NppiSize oSizeROI, cudaStream_t stream, int swapDst) {
  if (!pSrcY) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcYStep, oSizeROI.width) || !step_ok(aDstStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  const int w = oSizeROI.width, h = oSizeROI.height;
  const int yw = w & ~1;
  k_copy_y<<<gridFor(yw, h & ~1), dim3(kBlockX, kBlockY), 0, stream>>>(pSrcY, nSrcYStep, pDst[0], aDstStep[0], w,
                                                                      h & ~1, yw);
  const int cw = w >> 1, ch = (h & ~1) >> 1;
  if (cw > 0 && ch > 0) {
    dim3 g = gridFor(cw, ch);
    if (swapDst) {
      k_p2_to_p3_420<1><<<g, dim3(kBlockX, kBlockY), 0, stream>>>(pSrcCbCr, nSrcCbCrStep, pDst[2], aDstStep[2],
                                                                 pDst[1], aDstStep[1], w, h);
    } else {
      k_p2_to_p3_420<0><<<g, dim3(kBlockX, kBlockY), 0, stream>>>(pSrcCbCr, nSrcCbCrStep, pDst[1], aDstStep[1],
                                                                 pDst[2], aDstStep[2], w, h);
    }
  }
  return launch_ok();
}

} // namespace c3yuv
} // namespace nppi
