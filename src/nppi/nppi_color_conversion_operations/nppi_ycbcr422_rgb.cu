#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline Npp8u clamp_u8_double(double v) {
  if (v < 0.0) {
    v = 0.0;
  } else if (v > 255.0) {
    v = 255.0;
  }
  return static_cast<Npp8u>(v);
}

__device__ inline void rgb_to_ycbcr_pixel(Npp8u r, Npp8u g, Npp8u b, Npp8u &y, Npp8u &cb, Npp8u &cr) {
  double R = r;
  double G = g;
  double B = b;
  double Y = 0.257 * R + 0.504 * G + 0.098 * B + 16.0;
  double Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128.0;
  double Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128.0;

  y = clamp_u8_double(Y);
  cb = clamp_u8_double(Cb);
  cr = clamp_u8_double(Cr);
}

__device__ inline void ycbcr_to_rgb_pixel(Npp8u y, Npp8u cb, Npp8u cr, Npp8u &r, Npp8u &g, Npp8u &b) {
  double Y = 1.164 * (static_cast<double>(y) - 16.0);
  double Cr = static_cast<double>(cr) - 128.0;
  double Cb = static_cast<double>(cb) - 128.0;

  double R = Y + 1.596 * Cr;
  double G = Y - 0.813 * Cr - 0.392 * Cb;
  double B = Y + 2.017 * Cb;

  r = clamp_u8_double(R);
  g = clamp_u8_double(G);
  b = clamp_u8_double(B);
}

__global__ void rgb_to_ycbcr422_c2_kernel(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, int width,
                                          int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int pairWidth = width >> 1;
  if (x >= pairWidth || y >= height) {
    return;
  }

  int baseX = x << 1;
  const Npp8u *srcRow = (const Npp8u *)((const char *)src + y * srcStep);
  Npp8u *dstRow = (Npp8u *)((char *)dst + y * dstStep);

  Npp8u y0, cb0, cr0;
  Npp8u y1, cb1, cr1;
  int idx0 = baseX * 3;
  int idx1 = (baseX + 1) * 3;
  rgb_to_ycbcr_pixel(srcRow[idx0 + 0], srcRow[idx0 + 1], srcRow[idx0 + 2], y0, cb0, cr0);
  rgb_to_ycbcr_pixel(srcRow[idx1 + 0], srcRow[idx1 + 1], srcRow[idx1 + 2], y1, cb1, cr1);

  Npp8u cb = static_cast<Npp8u>((cb0 + cb1) / 2);
  Npp8u cr = static_cast<Npp8u>((cr0 + cr1) / 2);

  int outIdx = x * 4;
  dstRow[outIdx + 0] = y0;
  dstRow[outIdx + 1] = cb;
  dstRow[outIdx + 2] = y1;
  dstRow[outIdx + 3] = cr;
}

__global__ void bgr_to_ycbcr422_c2_kernel(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, int width,
                                          int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int pairWidth = width >> 1;
  if (x >= pairWidth || y >= height) {
    return;
  }

  int baseX = x << 1;
  const Npp8u *srcRow = (const Npp8u *)((const char *)src + y * srcStep);
  Npp8u *dstRow = (Npp8u *)((char *)dst + y * dstStep);

  Npp8u y0, cb0, cr0;
  Npp8u y1, cb1, cr1;
  int idx0 = baseX * 3;
  int idx1 = (baseX + 1) * 3;
  rgb_to_ycbcr_pixel(srcRow[idx0 + 2], srcRow[idx0 + 1], srcRow[idx0 + 0], y0, cb0, cr0);
  rgb_to_ycbcr_pixel(srcRow[idx1 + 2], srcRow[idx1 + 1], srcRow[idx1 + 0], y1, cb1, cr1);

  Npp8u cb = static_cast<Npp8u>((cb0 + cb1) / 2);
  Npp8u cr = static_cast<Npp8u>((cr0 + cr1) / 2);

  int outIdx = x * 4;
  dstRow[outIdx + 0] = y0;
  dstRow[outIdx + 1] = cb;
  dstRow[outIdx + 2] = y1;
  dstRow[outIdx + 3] = cr;
}

__global__ void bgra_to_ycbcr422_c2_kernel(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, int width,
                                           int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int pairWidth = width >> 1;
  if (x >= pairWidth || y >= height) {
    return;
  }

  int baseX = x << 1;
  const Npp8u *srcRow = (const Npp8u *)((const char *)src + y * srcStep);
  Npp8u *dstRow = (Npp8u *)((char *)dst + y * dstStep);

  Npp8u y0, cb0, cr0;
  Npp8u y1, cb1, cr1;
  int idx0 = baseX * 4;
  int idx1 = (baseX + 1) * 4;
  rgb_to_ycbcr_pixel(srcRow[idx0 + 2], srcRow[idx0 + 1], srcRow[idx0 + 0], y0, cb0, cr0);
  rgb_to_ycbcr_pixel(srcRow[idx1 + 2], srcRow[idx1 + 1], srcRow[idx1 + 0], y1, cb1, cr1);

  Npp8u cb = static_cast<Npp8u>((cb0 + cb1) / 2);
  Npp8u cr = static_cast<Npp8u>((cr0 + cr1) / 2);

  int outIdx = x * 4;
  dstRow[outIdx + 0] = y0;
  dstRow[outIdx + 1] = cb;
  dstRow[outIdx + 2] = y1;
  dstRow[outIdx + 3] = cr;
}

__global__ void ycbcr422_to_rgb_c3_kernel(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, int width,
                                          int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)src + y * srcStep);
  Npp8u *dstRow = (Npp8u *)((char *)dst + y * dstStep);

  int pairIdx = (x >> 1) * 4;
  Npp8u y0 = srcRow[pairIdx + 0];
  Npp8u cb = srcRow[pairIdx + 1];
  Npp8u y1 = srcRow[pairIdx + 2];
  Npp8u cr = srcRow[pairIdx + 3];

  Npp8u r, g, b;
  if ((x & 1) == 0) {
    ycbcr_to_rgb_pixel(y0, cb, cr, r, g, b);
  } else {
    ycbcr_to_rgb_pixel(y1, cb, cr, r, g, b);
  }

  int outIdx = x * 3;
  dstRow[outIdx + 0] = r;
  dstRow[outIdx + 1] = g;
  dstRow[outIdx + 2] = b;
}

__global__ void ycbcr422_to_bgr_c3_kernel(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, int width,
                                          int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)src + y * srcStep);
  Npp8u *dstRow = (Npp8u *)((char *)dst + y * dstStep);

  int pairIdx = (x >> 1) * 4;
  Npp8u y0 = srcRow[pairIdx + 0];
  Npp8u cb = srcRow[pairIdx + 1];
  Npp8u y1 = srcRow[pairIdx + 2];
  Npp8u cr = srcRow[pairIdx + 3];

  Npp8u r, g, b;
  if ((x & 1) == 0) {
    ycbcr_to_rgb_pixel(y0, cb, cr, r, g, b);
  } else {
    ycbcr_to_rgb_pixel(y1, cb, cr, r, g, b);
  }

  int outIdx = x * 3;
  dstRow[outIdx + 0] = b;
  dstRow[outIdx + 1] = g;
  dstRow[outIdx + 2] = r;
}

__global__ void ycbcr422_to_bgr_c4_kernel(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, int width,
                                          int height, Npp8u alpha) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)src + y * srcStep);
  Npp8u *dstRow = (Npp8u *)((char *)dst + y * dstStep);

  int pairIdx = (x >> 1) * 4;
  Npp8u y0 = srcRow[pairIdx + 0];
  Npp8u cb = srcRow[pairIdx + 1];
  Npp8u y1 = srcRow[pairIdx + 2];
  Npp8u cr = srcRow[pairIdx + 3];

  Npp8u r, g, b;
  if ((x & 1) == 0) {
    ycbcr_to_rgb_pixel(y0, cb, cr, r, g, b);
  } else {
    ycbcr_to_rgb_pixel(y1, cb, cr, r, g, b);
  }

  int outIdx = x * 4;
  dstRow[outIdx + 0] = b;
  dstRow[outIdx + 1] = g;
  dstRow[outIdx + 2] = r;
  dstRow[outIdx + 3] = alpha;
}

extern "C" {

cudaError_t nppiRGBToYCbCr422_8u_C3C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize(((oSizeROI.width / 2) + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  rgb_to_ycbcr422_c2_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYCbCr422_8u_C3C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize(((oSizeROI.width / 2) + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_ycbcr422_c2_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiBGRToYCbCr422_8u_AC4C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize(((oSizeROI.width / 2) + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgra_to_ycbcr422_c2_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                 oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCr422ToRGB_8u_C2C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr422_to_rgb_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCr422ToBGR_8u_C2C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr422_to_bgr_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiYCbCr422ToBGR_8u_C2C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, Npp8u alpha, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  ycbcr422_to_bgr_c4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                oSizeROI.height, alpha);
  return cudaGetLastError();
}

} // extern "C"
