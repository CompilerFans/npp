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

NppStatus validate3plane(const void *const *planes, int nSrcStep, void *const *dst, int nDstStep, NppiSize roi) {
  if (!planes || !planes[0] || !planes[1] || !planes[2] || !dst || !dst[0] || !dst[1] || !dst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (roi.width < 0 || roi.height < 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  return NPP_NO_ERROR;
}

NppStatus validatePackedToP3(const void *pSrc, int nSrcStep, void *const *dst, int nDstStep, NppiSize roi) {
  if (!pSrc || !dst || !dst[0] || !dst[1] || !dst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (roi.width < 0 || roi.height < 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  return NPP_NO_ERROR;
}

NppStatus validateP3ToPacked(const void *const *planes, int nSrcStep, const void *pDst, int nDstStep, NppiSize roi) {
  if (!planes || !planes[0] || !planes[1] || !planes[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (roi.width < 0 || roi.height < 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  return NPP_NO_ERROR;
}

NppStatus launch_ok() {
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

} // namespace

NppStatus rgbToYCbCr444JpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream) {
  NppStatus st = validate3plane((const void *const *)pSrc, nSrcStep, (void *const *)pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
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
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_rgb444jpeg_packed_to_p3<<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst[0], nDstStep, pDst[1], nDstStep, pDst[2],
                                                        nDstStep, oSizeROI.width, oSizeROI.height, 0);
  return launch_ok();
}

NppStatus bgrToYCbCr444JpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream) {
  NppStatus st = validate3plane((const void *const *)pSrc, nSrcStep, (void *const *)pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
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
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_rgb444jpeg_packed_to_p3<<<grid, block, 0, stream>>>(pSrc, nSrcStep, pDst[0], nDstStep, pDst[1], nDstStep, pDst[2],
                                                        nDstStep, oSizeROI.width, oSizeROI.height, 1);
  return launch_ok();
}

NppStatus yCbCr444ToRgbJpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream) {
  NppStatus st = validate3plane((const void *const *)pSrc, nSrcStep, (void *const *)pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
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
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_ycbcr444jpeg_p3_to_packed<<<grid, block, 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst, nDstStep,
                                                          oSizeROI.width, oSizeROI.height, 0);
  return launch_ok();
}

NppStatus yCbCr444ToBgrJpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream) {
  NppStatus st = validate3plane((const void *const *)pSrc, nSrcStep, (void *const *)pDst, nDstStep, oSizeROI);
  if (st != NPP_NO_ERROR) {
    return st;
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
  dim3 block(kBlockX, kBlockY);
  dim3 grid((oSizeROI.width + block.x - 1) / block.x, (oSizeROI.height + block.y - 1) / block.y);
  k_ycbcr444jpeg_p3_to_packed<<<grid, block, 0, stream>>>(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst, nDstStep,
                                                          oSizeROI.width, oSizeROI.height, 1);
  return launch_ok();
}

} // namespace c3conv
} // namespace nppi
