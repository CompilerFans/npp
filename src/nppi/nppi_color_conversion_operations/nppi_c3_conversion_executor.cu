// c3 package executor kernels (see nppi_c3_conversion_executor.h).
//
// All 8u pixel math uses single-precision intrinsics (__fmul_rn/__fadd_rn/
// __fsub_rn) so the device code cannot be re-associated or FMA-contracted by
// the compiler.  The exact operation order is part of the contract: it is
// bit-exact with the NVIDIA NPP 13.4 reference (validated against recorded
// goldens) and is mirrored by the CPU reference used in the diff tests.

#include "nppi_c3_conversion_executor.h"

#include <cuda_runtime.h>

namespace nppi {
namespace c3conv {
namespace {

constexpr int kBlockX = 128;
constexpr int kBlockY = 8;

__device__ __forceinline__ int trunc_u8(float v) {
  // C-style truncation toward zero, then saturate.
  int t = static_cast<int>(v);
  return t < 0 ? 0 : (t > 255 ? 255 : t);
}

// --- JPEG forward: order validated against NVIDIA goldens ---------------
__device__ __forceinline__ void rgb_to_ycbcr_jpeg(float r, float g, float b, int &y, int &cb, int &cr) {
  const float fy = __fadd_rn(__fadd_rn(__fmul_rn(0.299f, r), __fmul_rn(0.587f, g)), __fmul_rn(0.114f, b));
  const float fcb =
      __fadd_rn(__fadd_rn(__fsub_rn(__fmul_rn(-0.168736f, r), __fmul_rn(0.331264f, g)), __fmul_rn(0.5f, b)), 128.0f);
  const float fcr =
      __fadd_rn(__fsub_rn(__fsub_rn(__fmul_rn(0.5f, r), __fmul_rn(0.418688f, g)), __fmul_rn(0.081312f, b)), 128.0f);
  y = trunc_u8(fy);
  cb = trunc_u8(fcb);
  cr = trunc_u8(fcr);
}

// --- JPEG inverse ---------------------------------------------------------
__device__ __forceinline__ void ycbcr_to_rgb_jpeg(float y, float cb, float cr, int &r, int &g, int &b) {
  const float dcb = __fsub_rn(cb, 128.0f);
  const float dcr = __fsub_rn(cr, 128.0f);
  const float fr = __fadd_rn(y, __fmul_rn(1.402f, dcr));
  const float fg = __fsub_rn(__fsub_rn(y, __fmul_rn(0.344136f, dcb)), __fmul_rn(0.714136f, dcr));
  const float fb = __fadd_rn(y, __fmul_rn(1.772f, dcb));
  r = trunc_u8(fr);
  g = trunc_u8(fg);
  b = trunc_u8(fb);
}

// dst = (R, G, B) planes; src packed C3 (bgr swaps r/b reads).
__global__ void k_rgb444jpeg_packed_to_p3(const Npp8u *__restrict__ pSrc, int nSrcStep, Npp8u *__restrict__ pDstY,
                                          int nDstYStep, Npp8u *__restrict__ pDstCb, int nDstCbStep,
                                          Npp8u *__restrict__ pDstCr, int nDstCrStep, int width, int height,
                                          int bgr) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const Npp8u *s = pSrc + (size_t)y * nSrcStep + (size_t)x * 3;
  const float r = bgr ? s[2] : s[0];
  const float g = s[1];
  const float b = bgr ? s[0] : s[2];
  int yv, cb, cr;
  rgb_to_ycbcr_jpeg(r, g, b, yv, cb, cr);
  pDstY[(size_t)y * nDstYStep + x] = (Npp8u)yv;
  pDstCb[(size_t)y * nDstCbStep + x] = (Npp8u)cb;
  pDstCr[(size_t)y * nDstCrStep + x] = (Npp8u)cr;
}

// dst = (R, G, B) planes; src planar (R, G, B) or (B, G, R) planes.
__global__ void k_rgb444jpeg_p3_to_p3(const Npp8u *__restrict__ pSrc0, const Npp8u *__restrict__ pSrc1,
                                      const Npp8u *__restrict__ pSrc2, int nSrcStep, Npp8u *__restrict__ pDstY,
                                      int nDstYStep, Npp8u *__restrict__ pDstCb, int nDstCbStep,
                                      Npp8u *__restrict__ pDstCr, int nDstCrStep, int width, int height, int bgr) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const size_t i = (size_t)y * nSrcStep + x;
  const float r = bgr ? pSrc2[i] : pSrc0[i];
  const float g = pSrc1[i];
  const float b = bgr ? pSrc0[i] : pSrc2[i];
  int yv, cb, cr;
  rgb_to_ycbcr_jpeg(r, g, b, yv, cb, cr);
  pDstY[(size_t)y * nDstYStep + x] = (Npp8u)yv;
  pDstCb[(size_t)y * nDstCbStep + x] = (Npp8u)cb;
  pDstCr[(size_t)y * nDstCrStep + x] = (Npp8u)cr;
}

// src planar YCbCr -> dst packed RGB/BGR (bgr swaps r/b writes).
__global__ void k_ycbcr444jpeg_p3_to_packed(const Npp8u *__restrict__ pSrcY, const Npp8u *__restrict__ pSrcCb,
                                            const Npp8u *__restrict__ pSrcCr, int nSrcStep, Npp8u *__restrict__ pDst,
                                            int nDstStep, int width, int height, int bgr) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const size_t i = (size_t)y * nSrcStep + x;
  int r, g, b;
  ycbcr_to_rgb_jpeg((float)pSrcY[i], (float)pSrcCb[i], (float)pSrcCr[i], r, g, b);
  Npp8u *d = pDst + (size_t)y * nDstStep + (size_t)x * 3;
  if (bgr) {
    d[0] = (Npp8u)b;
    d[1] = (Npp8u)g;
    d[2] = (Npp8u)r;
  } else {
    d[0] = (Npp8u)r;
    d[1] = (Npp8u)g;
    d[2] = (Npp8u)b;
  }
}

// src planar YCbCr -> dst planar RGB/BGR planes.
__global__ void k_ycbcr444jpeg_p3_to_p3(const Npp8u *__restrict__ pSrcY, const Npp8u *__restrict__ pSrcCb,
                                        const Npp8u *__restrict__ pSrcCr, int nSrcStep, Npp8u *__restrict__ pDst0,
                                        int nDst0Step, Npp8u *__restrict__ pDst1, int nDst1Step,
                                        Npp8u *__restrict__ pDst2, int nDst2Step, int width, int height, int bgr) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const size_t i = (size_t)y * nSrcStep + x;
  int r, g, b;
  ycbcr_to_rgb_jpeg((float)pSrcY[i], (float)pSrcCb[i], (float)pSrcCr[i], r, g, b);
  const size_t o = (size_t)y * nDst0Step + x;
  if (bgr) {
    pDst0[o] = (Npp8u)b;
    pDst1[(size_t)y * nDst1Step + x] = (Npp8u)g;
    pDst2[(size_t)y * nDst2Step + x] = (Npp8u)r;
  } else {
    pDst0[o] = (Npp8u)r;
    pDst1[(size_t)y * nDst1Step + x] = (Npp8u)g;
    pDst2[(size_t)y * nDst2Step + x] = (Npp8u)b;
  }
}

// Validation semantics measured against NVIDIA NPP 13.4 on an RTX 5090
// (see test/unit/diff/test_nppi_c3_jpeg444_generated.cpp goldens):
//   1. null plane pointers            -> NPP_NULL_POINTER_ERROR
//      (a null plane-array pointer segfaults in NVIDIA NPP; we return
//       NPP_NULL_POINTER_ERROR instead - such calls are never recorded)
//   2. negative ROI width or height   -> NPP_SIZE_ERROR (checked before steps)
//   3. step smaller than the row bytes (unsigned compare: negative steps pass)
//                                      -> NPP_STEP_ERROR
//   4. width == 0 or height == 0      -> NPP_NO_ERROR (no kernel launched)
//   5. negative step passes validation; NVIDIA reads out of bounds, we simply
//      do not touch memory and return NPP_NO_ERROR (no outputs recorded).
bool roi_valid(const NppiSize roi) { return roi.width >= 0 && roi.height >= 0; }

// NVIDIA compares the step as unsigned against the minimum row bytes.
bool step_ok(const int nStep, const int nMinRowBytes) {
  return static_cast<unsigned>(nStep) >= static_cast<unsigned>(nMinRowBytes);
}

template <class SrcPtrs, class DstPtrs>
NppStatus validate3plane(const SrcPtrs planes, int nSrcStep, DstPtrs dst, int nDstStep, NppiSize roi) {
  if (!planes || !planes[0] || !planes[1] || !planes[2] || !dst || !dst[0] || !dst[1] || !dst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(roi)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, roi.width) || !step_ok(nDstStep, roi.width)) {
    return NPP_STEP_ERROR;
  }
  return NPP_NO_ERROR;
}

template <class DstPtrs>
NppStatus validatePackedToP3(const void *pSrc, int nSrcStep, DstPtrs dst, int nDstStep, NppiSize roi) {
  if (!pSrc || !dst || !dst[0] || !dst[1] || !dst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(roi)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 3 * roi.width) || !step_ok(nDstStep, roi.width)) {
    return NPP_STEP_ERROR;
  }
  return NPP_NO_ERROR;
}

template <class SrcPtrs>
NppStatus validateP3ToPacked(const SrcPtrs planes, int nSrcStep, const void *pDst, int nDstStep, NppiSize roi) {
  if (!planes || !planes[0] || !planes[1] || !planes[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(roi)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, roi.width) || !step_ok(nDstStep, 3 * roi.width)) {
    return NPP_STEP_ERROR;
  }
  return NPP_NO_ERROR;
}

NppStatus launch_ok() {
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

} // namespace

NppStatus rgbToYCbCr444JpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream) {
  NppStatus st = validate3plane(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
  }
  // Degenerate ROIs and negative steps: NVIDIA returns NPP_NO_ERROR without
  // producing defined output (negative steps read out of bounds there); we
  // skip the launch entirely.
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || nSrcStep < 0 || nDstStep < 0) {
    return NPP_NO_ERROR;
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_rgb444jpeg_p3_to_p3<<<grid, block, 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst[0], nDstStep, pDst[1],
                                                    nDstStep, pDst[2], nDstStep, oSizeROI.width, oSizeROI.height, 0);
  return launch_ok();
}

NppStatus rgbToYCbCr444JpegC3P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI,
                                cudaStream_t stream) {
  NppStatus st = validatePackedToP3(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
  }
  // Degenerate ROIs and negative steps: NVIDIA returns NPP_NO_ERROR without
  // producing defined output (negative steps read out of bounds there); we
  // skip the launch entirely.
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || nSrcStep < 0 || nDstStep < 0) {
    return NPP_NO_ERROR;
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_rgb444jpeg_packed_to_p3<<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst[0], nDstStep, pDst[1], nDstStep, pDst[2],
                                                        nDstStep, oSizeROI.width, oSizeROI.height, 0);
  return launch_ok();
}

NppStatus bgrToYCbCr444JpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream) {
  NppStatus st = validate3plane(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
  }
  // Degenerate ROIs and negative steps: NVIDIA returns NPP_NO_ERROR without
  // producing defined output (negative steps read out of bounds there); we
  // skip the launch entirely.
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || nSrcStep < 0 || nDstStep < 0) {
    return NPP_NO_ERROR;
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_rgb444jpeg_p3_to_p3<<<grid, block, 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst[0], nDstStep, pDst[1],
                                                    nDstStep, pDst[2], nDstStep, oSizeROI.width, oSizeROI.height, 1);
  return launch_ok();
}

NppStatus bgrToYCbCr444JpegC3P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI,
                                cudaStream_t stream) {
  NppStatus st = validatePackedToP3(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
  }
  // Degenerate ROIs and negative steps: NVIDIA returns NPP_NO_ERROR without
  // producing defined output (negative steps read out of bounds there); we
  // skip the launch entirely.
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || nSrcStep < 0 || nDstStep < 0) {
    return NPP_NO_ERROR;
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_rgb444jpeg_packed_to_p3<<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst[0], nDstStep, pDst[1], nDstStep, pDst[2],
                                                        nDstStep, oSizeROI.width, oSizeROI.height, 1);
  return launch_ok();
}

NppStatus yCbCr444ToRgbJpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream) {
  NppStatus st = validate3plane(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
  }
  // Degenerate ROIs and negative steps: NVIDIA returns NPP_NO_ERROR without
  // producing defined output (negative steps read out of bounds there); we
  // skip the launch entirely.
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || nSrcStep < 0 || nDstStep < 0) {
    return NPP_NO_ERROR;
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_ycbcr444jpeg_p3_to_p3<<<grid, block, 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst[0], nDstStep, pDst[1],
                                                      nDstStep, pDst[2], nDstStep, oSizeROI.width, oSizeROI.height,
                                                      0);
  return launch_ok();
}

NppStatus yCbCr444ToRgbJpegP3C3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI, cudaStream_t stream) {
  NppStatus st = validateP3ToPacked(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
  }
  // Degenerate ROIs and negative steps: NVIDIA returns NPP_NO_ERROR without
  // producing defined output (negative steps read out of bounds there); we
  // skip the launch entirely.
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || nSrcStep < 0 || nDstStep < 0) {
    return NPP_NO_ERROR;
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_ycbcr444jpeg_p3_to_packed<<<grid, block, 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst, nDstStep,
                                                          oSizeROI.width, oSizeROI.height, 0);
  return launch_ok();
}

NppStatus yCbCr444ToBgrJpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream) {
  NppStatus st = validate3plane(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
  }
  // Degenerate ROIs and negative steps: NVIDIA returns NPP_NO_ERROR without
  // producing defined output (negative steps read out of bounds there); we
  // skip the launch entirely.
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || nSrcStep < 0 || nDstStep < 0) {
    return NPP_NO_ERROR;
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_ycbcr444jpeg_p3_to_p3<<<grid, block, 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst[0], nDstStep, pDst[1],
                                                      nDstStep, pDst[2], nDstStep, oSizeROI.width, oSizeROI.height,
                                                      1);
  return launch_ok();
}

NppStatus yCbCr444ToBgrJpegP3C3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI, cudaStream_t stream) {
  NppStatus st = validateP3ToPacked(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
  }
  // Degenerate ROIs and negative steps: NVIDIA returns NPP_NO_ERROR without
  // producing defined output (negative steps read out of bounds there); we
  // skip the launch entirely.
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || nSrcStep < 0 || nDstStep < 0) {
    return NPP_NO_ERROR;
  }
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_ycbcr444jpeg_p3_to_packed<<<grid, block, 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst, nDstStep,
                                                          oSizeROI.width, oSizeROI.height, 1);
  return launch_ok();
}

// === JPEG 4:2:2 / 4:2:0 subsampled conversions ==============================
//
// Measured NVIDIA semantics (validated bit-exact against 5090 goldens):
//   forward: luma is computed per pixel, chroma is the integer average
//            (sum >> N) of the *truncated* per-pixel 8-bit chroma values.
//   processed extent: columns < width & ~1; rows < height & ~1 for 420,
//            all rows for 422 forward.  The trailing odd column/row is left
//            untouched (keeps its previous buffer content).
//   inverse: chroma is replicated to the covered pixels.  The inverse
//            converts the FULL width (the odd trailing column uses chroma
//            sample x/2) but only height & ~1 rows; the trailing odd row
//            keeps its previous content (both for 420 and 422).
//   planar-source forward (P3R): measured NO-OP -- validates arguments,
//            never writes, never warns (see the entry points below).
//   odd width/height -> NPP_DOUBLE_SIZE_WARNING (still converts the even
//            part) -- but only for the packed-source forward and for every
//            inverse conversion; the planar-source forward never warns.

struct SrcPacked {
  const Npp8u *p;
  int step;
  int bgr;
  __device__ __forceinline__ void pixel(int x, int y, int &r, int &g, int &b) const {
    const Npp8u *s = p + (size_t)y * step + (size_t)x * 3;
    r = bgr ? s[2] : s[0];
    g = s[1];
    b = bgr ? s[0] : s[2];
  }
};

// Forward to 420: one thread per luma pixel of the even extent; the thread at
// the top-left corner of each 2x2 block also emits the block chroma.
template <class Src>
__global__ void k_fwd_420(const Src src, Npp8u *__restrict__ pDstY, int stepY, Npp8u *__restrict__ pDstCb,
                          int stepCb, Npp8u *__restrict__ pDstCr, int stepCr, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  // Measured: the inverse covers the FULL width (the trailing odd column is
  // converted, using chroma sample x/2) but only height & ~1 rows.
  const int w = width;
  const int h = height & ~1;
  if (x >= w || y >= h) {
    return;
  }
  int r, g, b;
  src.pixel(x, y, r, g, b);
  int yv, cbv, crv;
  rgb_to_ycbcr_jpeg((float)r, (float)g, (float)b, yv, cbv, crv);
  pDstY[(size_t)y * stepY + x] = (Npp8u)yv;
  if ((x & 1) == 0 && (y & 1) == 0) {
    int r0, g0, b0, r1, g1, b1, r2, g2, b2, r3, g3, b3;
    src.pixel(x, y, r0, g0, b0);
    src.pixel(x + 1, y, r1, g1, b1);
    src.pixel(x, y + 1, r2, g2, b2);
    src.pixel(x + 1, y + 1, r3, g3, b3);
    int yc, cb0, cr0, cb1, cr1, cb2, cr2, cb3, cr3;
    rgb_to_ycbcr_jpeg((float)r0, (float)g0, (float)b0, yc, cb0, cr0);
    rgb_to_ycbcr_jpeg((float)r1, (float)g1, (float)b1, yc, cb1, cr1);
    rgb_to_ycbcr_jpeg((float)r2, (float)g2, (float)b2, yc, cb2, cr2);
    rgb_to_ycbcr_jpeg((float)r3, (float)g3, (float)b3, yc, cb3, cr3);
    const size_t o = (size_t)(y >> 1) * stepCb + (x >> 1);
    pDstCb[o] = (Npp8u)((cb0 + cb1 + cb2 + cb3) >> 2);
    pDstCr[o] = (Npp8u)((cr0 + cr1 + cr2 + cr3) >> 2);
  }
}

// Forward to 422: chroma from horizontal pairs; all rows are processed.
template <class Src>
__global__ void k_fwd_422(const Src src, Npp8u *__restrict__ pDstY, int stepY, Npp8u *__restrict__ pDstCb,
                          int stepCb, Npp8u *__restrict__ pDstCr, int stepCr, int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  // Measured: the inverse covers the FULL width but only height & ~1 rows.
  const int w = width;
  const int h = height & ~1;
  if (x >= w || y >= h) {
    return;
  }
  int r, g, b;
  src.pixel(x, y, r, g, b);
  int yv, cbv, crv;
  rgb_to_ycbcr_jpeg((float)r, (float)g, (float)b, yv, cbv, crv);
  pDstY[(size_t)y * stepY + x] = (Npp8u)yv;
  if ((x & 1) == 0) {
    int r0, g0, b0, r1, g1, b1;
    src.pixel(x, y, r0, g0, b0);
    src.pixel(x + 1, y, r1, g1, b1);
    int yc, cb0, cr0, cb1, cr1;
    rgb_to_ycbcr_jpeg((float)r0, (float)g0, (float)b0, yc, cb0, cr0);
    rgb_to_ycbcr_jpeg((float)r1, (float)g1, (float)b1, yc, cb1, cr1);
    const size_t o = (size_t)y * stepCb + (x >> 1);
    pDstCb[o] = (Npp8u)((cb0 + cb1) >> 1);
    pDstCr[o] = (Npp8u)((cr0 + cr1) >> 1);
  }
}

// Inverse from 420: one thread per output pixel of the even extent.
__global__ void k_inv_420(const Npp8u *__restrict__ pSrcY, int stepY, const Npp8u *__restrict__ pSrcCb, int stepCb,
                          const Npp8u *__restrict__ pSrcCr, int stepCr, Npp8u *__restrict__ pDst0, int step0,
                          Npp8u *__restrict__ pDst1, int step1, Npp8u *__restrict__ pDst2, int step2, int width,
                          int height, int bgr, int packed3) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int w = width & ~1;
  const int h = height & ~1;
  if (x >= w || y >= h) {
    return;
  }
  const Npp8u Y = pSrcY[(size_t)y * stepY + x];
  const Npp8u Cb = pSrcCb[(size_t)(y >> 1) * stepCb + (x >> 1)];
  const Npp8u Cr = pSrcCr[(size_t)(y >> 1) * stepCr + (x >> 1)];
  int r, g, b;
  ycbcr_to_rgb_jpeg((float)Y, (float)Cb, (float)Cr, r, g, b);
  if (packed3) {
    Npp8u *d = pDst0 + (size_t)y * step0 + (size_t)x * 3;
    d[0] = (Npp8u)(bgr ? b : r);
    d[1] = (Npp8u)g;
    d[2] = (Npp8u)(bgr ? r : b);
  } else {
    pDst0[(size_t)y * step0 + x] = (Npp8u)(bgr ? b : r);
    pDst1[(size_t)y * step1 + x] = (Npp8u)g;
    pDst2[(size_t)y * step2 + x] = (Npp8u)(bgr ? r : b);
  }
}

// Inverse from 422: chroma replicated horizontally, all rows processed.
__global__ void k_inv_422(const Npp8u *__restrict__ pSrcY, int stepY, const Npp8u *__restrict__ pSrcCb, int stepCb,
                          const Npp8u *__restrict__ pSrcCr, int stepCr, Npp8u *__restrict__ pDst0, int step0,
                          Npp8u *__restrict__ pDst1, int step1, Npp8u *__restrict__ pDst2, int step2, int width,
                          int height, int bgr, int packed3) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int w = width & ~1;
  if (x >= w || y >= height) {
    return;
  }
  const Npp8u Y = pSrcY[(size_t)y * stepY + x];
  const Npp8u Cb = pSrcCb[(size_t)y * stepCb + (x >> 1)];
  const Npp8u Cr = pSrcCr[(size_t)y * stepCr + (x >> 1)];
  int r, g, b;
  ycbcr_to_rgb_jpeg((float)Y, (float)Cb, (float)Cr, r, g, b);
  if (packed3) {
    Npp8u *d = pDst0 + (size_t)y * step0 + (size_t)x * 3;
    d[0] = (Npp8u)(bgr ? b : r);
    d[1] = (Npp8u)g;
    d[2] = (Npp8u)(bgr ? r : b);
  } else {
    pDst0[(size_t)y * step0 + x] = (Npp8u)(bgr ? b : r);
    pDst1[(size_t)y * step1 + x] = (Npp8u)g;
    pDst2[(size_t)y * step2 + x] = (Npp8u)(bgr ? r : b);
  }
}


// --- shared 420/422 helpers -----------------------------------------------
// Measured NVIDIA validation for the subsampled JPEG conversions:
//   - the Y-plane / packed-source row stride is validated (unsigned) against
//     the full row width; the CHROMA plane strides are NOT validated at all
//     (a chroma step of 0 is accepted and the call still returns success).
//   - only the packed-source forward and every inverse conversion report
//     NPP_DOUBLE_SIZE_WARNING for odd dimensions; the planar-source forward
//     never warns.  420 checks width and height parity, the 422 forward
//     checks width only, the 422 inverse checks both.
//   - the BGR planar-destination inverse validates nDstStep against
//     3 * roi.width (packed-style), unlike the RGB variant (width).

bool steps_neg(const int *steps, int n) {
  for (int i = 0; i < n; ++i) {
    if (steps[i] < 0) {
      return true;
    }
  }
  return false;
}

// Planar-destination inverse: RGB variants validate against the plane width,
// BGR variants (NVIDIA quirk) against the packed 3-channel row width.
int dstMinRowBytes(int bgr, const NppiSize roi) { return bgr ? 3 * roi.width : roi.width; }

// Zero-sized ROI: NVIDIA performs no work and reports success.
NppStatus degenerate(const NppiSize roi) { return NPP_NO_ERROR; }

// Odd dimensions on subsampled formats produce a warning (work still done).
NppStatus warn_even(const NppiSize roi, int checkFlags) {
  if ((checkFlags & 1) && (roi.width & 1)) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  if ((checkFlags & 2) && (roi.height & 1)) {
    return NPP_DOUBLE_SIZE_WARNING;
  }
  return NPP_NO_ERROR;
}

dim3 grid2(const NppiSize roi) {
  return dim3((roi.width + kBlockX - 1) / kBlockX, (roi.height + kBlockY - 1) / kBlockY);
}

template <class Src>
void launch_fwd_420(const Src src, Npp8u *const *pDst, const int *aDstStep, NppiSize roi, cudaStream_t stream) {
  k_fwd_420<<<grid2(roi), dim3(kBlockX, kBlockY), 0, stream>>>(src, pDst[0], aDstStep[0], pDst[1], aDstStep[1],
                                                               pDst[2], aDstStep[2], roi.width, roi.height);
}

template <class Src>
void launch_fwd_422(const Src src, Npp8u *const *pDst, const int *aDstStep, NppiSize roi, cudaStream_t stream) {
  k_fwd_422<<<grid2(roi), dim3(kBlockX, kBlockY), 0, stream>>>(src, pDst[0], aDstStep[0], pDst[1], aDstStep[1],
                                                               pDst[2], aDstStep[2], roi.width, roi.height);
}


// --- JPEG 420/422 forward -------------------------------------------------
NppStatus rgbToYCbCr420JpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], const int aDstStep[3],
                              NppiSize oSizeROI, cudaStream_t stream, int bgr) {
  // Measured NVIDIA behavior: the planar-source forward validates its
  // arguments and then returns NPP_NO_ERROR WITHOUT WRITING ANY OUTPUT and
  // without ever reporting NPP_DOUBLE_SIZE_WARNING (verified on 5090
  // goldens and on the upstream reference goldens, with several different
  // step combinations).
  (void)stream;
  (void)bgr;
  NppStatus st = validate3plane(pSrc, nSrcStep, pDst, aDstStep[0], oSizeROI);
  return st == NPP_NO_ERROR ? NPP_NO_ERROR : st;
}

NppStatus rgbToYCbCr420JpegC3P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], const int aDstStep[3],
                                NppiSize oSizeROI, cudaStream_t stream, int bgr) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 3 * oSizeROI.width) || !step_ok(aDstStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || nSrcStep < 0 || steps_neg(aDstStep, 3)) {
    return degenerate(oSizeROI);
  }
  SrcPacked src{pSrc, nSrcStep, bgr};
  launch_fwd_420(src, pDst, aDstStep, oSizeROI, stream);
  return warn_even(oSizeROI, 3);
}

NppStatus rgbToYCbCr422JpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], const int aDstStep[3],
                              NppiSize oSizeROI, cudaStream_t stream, int bgr) {
  // Measured NVIDIA behavior: the planar-source forward validates its
  // arguments and then returns NPP_NO_ERROR WITHOUT WRITING ANY OUTPUT and
  // without ever reporting NPP_DOUBLE_SIZE_WARNING (verified on 5090
  // goldens and on the upstream reference goldens, with several different
  // step combinations).
  (void)stream;
  (void)bgr;
  NppStatus st = validate3plane(pSrc, nSrcStep, pDst, aDstStep[0], oSizeROI);
  return st == NPP_NO_ERROR ? NPP_NO_ERROR : st;
}

NppStatus rgbToYCbCr422JpegC3P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], const int aDstStep[3],
                                NppiSize oSizeROI, cudaStream_t stream, int bgr) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nSrcStep, 3 * oSizeROI.width) || !step_ok(aDstStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || nSrcStep < 0 || steps_neg(aDstStep, 3)) {
    return degenerate(oSizeROI);
  }
  SrcPacked src{pSrc, nSrcStep, bgr};
  launch_fwd_422(src, pDst, aDstStep, oSizeROI, stream);
  return warn_even(oSizeROI, 1);
}

// --- JPEG 420/422 inverse -------------------------------------------------
NppStatus yCbCr420ToRgbJpegP3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream, int bgr) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nDstStep, dstMinRowBytes(bgr, oSizeROI)) || !step_ok(aSrcStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || steps_neg(aSrcStep, 3) || nDstStep < 0) {
    return degenerate(oSizeROI);
  }
  k_inv_420<<<grid2(oSizeROI), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1], pSrc[2], aSrcStep[2], pDst[0], nDstStep, pDst[1], nDstStep,
      pDst[2], nDstStep, oSizeROI.width, oSizeROI.height, bgr, 0);
  return launch_ok() == NPP_NO_ERROR ? warn_even(oSizeROI, 3) : launch_ok();
}

NppStatus yCbCr420ToRgbJpegP3C3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI, cudaStream_t stream, int bgr) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nDstStep, 3 * oSizeROI.width) || !step_ok(aSrcStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || steps_neg(aSrcStep, 3) || nDstStep < 0) {
    return degenerate(oSizeROI);
  }
  k_inv_420<<<grid2(oSizeROI), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1], pSrc[2], aSrcStep[2], pDst, nDstStep, pDst, nDstStep, pDst,
      nDstStep, oSizeROI.width, oSizeROI.height, bgr, 1);
  return launch_ok() == NPP_NO_ERROR ? warn_even(oSizeROI, 3) : launch_ok();
}

NppStatus yCbCr422ToRgbJpegP3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream, int bgr) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nDstStep, dstMinRowBytes(bgr, oSizeROI)) || !step_ok(aSrcStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || steps_neg(aSrcStep, 3) || nDstStep < 0) {
    return degenerate(oSizeROI);
  }
  k_inv_422<<<grid2(oSizeROI), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1], pSrc[2], aSrcStep[2], pDst[0], nDstStep, pDst[1], nDstStep,
      pDst[2], nDstStep, oSizeROI.width, oSizeROI.height, bgr, 0);
  return launch_ok() == NPP_NO_ERROR ? warn_even(oSizeROI, 3) : launch_ok();
}

NppStatus yCbCr422ToRgbJpegP3C3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI, cudaStream_t stream, int bgr) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!roi_valid(oSizeROI)) {
    return NPP_SIZE_ERROR;
  }
  if (!step_ok(nDstStep, 3 * oSizeROI.width) || !step_ok(aSrcStep[0], oSizeROI.width)) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || steps_neg(aSrcStep, 3) || nDstStep < 0) {
    return degenerate(oSizeROI);
  }
  k_inv_422<<<grid2(oSizeROI), dim3(kBlockX, kBlockY), 0, stream>>>(
      pSrc[0], aSrcStep[0], pSrc[1], aSrcStep[1], pSrc[2], aSrcStep[2], pDst, nDstStep, pDst, nDstStep, pDst,
      nDstStep, oSizeROI.width, oSizeROI.height, bgr, 1);
  return launch_ok() == NPP_NO_ERROR ? warn_even(oSizeROI, 3) : launch_ok();
}

} // namespace c3conv
} // namespace nppi
