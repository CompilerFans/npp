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

__global__ void rgb_to_ycbcr420_y_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep, int width,
                                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  int idx = x * 3;
  Npp8u r = srcRow[idx + 0];
  Npp8u g = srcRow[idx + 1];
  Npp8u b = srcRow[idx + 2];

  Npp8u yv, cbv, crv;
  rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

  Npp8u *dstRowY = (Npp8u *)((char *)pDstY + y * nDstYStep);
  dstRowY[x] = yv;
}

__global__ void bgr_to_ycbcr420_y_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep, int width,
                                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  int idx = x * 3;
  Npp8u b = srcRow[idx + 0];
  Npp8u g = srcRow[idx + 1];
  Npp8u r = srcRow[idx + 2];

  Npp8u yv, cbv, crv;
  rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

  Npp8u *dstRowY = (Npp8u *)((char *)pDstY + y * nDstYStep);
  dstRowY[x] = yv;
}

__global__ void rgb_to_ycbcr420_cbcr_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstCb, int nDstCbStep,
                                               Npp8u *pDstCr, int nDstCrStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int cbWidth = width >> 1;
  int cbHeight = height >> 1;
  if (x >= cbWidth || y >= cbHeight) {
    return;
  }

  int baseX = x << 1;
  int baseY = y << 1;

  int sumCb = 0;
  int sumCr = 0;

  for (int dy = 0; dy < 2; ++dy) {
    const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + (baseY + dy) * nSrcStep);
    for (int dx = 0; dx < 2; ++dx) {
      int idx = (baseX + dx) * 3;
      Npp8u r = srcRow[idx + 0];
      Npp8u g = srcRow[idx + 1];
      Npp8u b = srcRow[idx + 2];

      Npp8u yv, cbv, crv;
      rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);
      sumCb += cbv;
      sumCr += crv;
    }
  }

  Npp8u *dstRowCb = (Npp8u *)((char *)pDstCb + y * nDstCbStep);
  Npp8u *dstRowCr = (Npp8u *)((char *)pDstCr + y * nDstCrStep);
  dstRowCb[x] = static_cast<Npp8u>(sumCb / 4);
  dstRowCr[x] = static_cast<Npp8u>(sumCr / 4);
}

__global__ void bgr_to_ycbcr420_cbcr_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstCb, int nDstCbStep,
                                               Npp8u *pDstCr, int nDstCrStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int cbWidth = width >> 1;
  int cbHeight = height >> 1;
  if (x >= cbWidth || y >= cbHeight) {
    return;
  }

  int baseX = x << 1;
  int baseY = y << 1;

  int sumCb = 0;
  int sumCr = 0;

  for (int dy = 0; dy < 2; ++dy) {
    const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + (baseY + dy) * nSrcStep);
    for (int dx = 0; dx < 2; ++dx) {
      int idx = (baseX + dx) * 3;
      Npp8u b = srcRow[idx + 0];
      Npp8u g = srcRow[idx + 1];
      Npp8u r = srcRow[idx + 2];

      Npp8u yv, cbv, crv;
      rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);
      sumCb += cbv;
      sumCr += crv;
    }
  }

  Npp8u *dstRowCb = (Npp8u *)((char *)pDstCb + y * nDstCbStep);
  Npp8u *dstRowCr = (Npp8u *)((char *)pDstCr + y * nDstCrStep);
  dstRowCb[x] = static_cast<Npp8u>(sumCb / 4);
  dstRowCr[x] = static_cast<Npp8u>(sumCr / 4);
}

__global__ void bgr_to_ycbcr420_y_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep, int width,
                                             int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  int idx = x * 4;
  Npp8u b = srcRow[idx + 0];
  Npp8u g = srcRow[idx + 1];
  Npp8u r = srcRow[idx + 2];

  Npp8u yv, cbv, crv;
  rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);

  Npp8u *dstRowY = (Npp8u *)((char *)pDstY + y * nDstYStep);
  dstRowY[x] = yv;
}

__global__ void bgr_to_ycbcr420_cbcr_ac4_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstCb, int nDstCbStep,
                                                Npp8u *pDstCr, int nDstCrStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int cbWidth = width >> 1;
  int cbHeight = height >> 1;
  if (x >= cbWidth || y >= cbHeight) {
    return;
  }

  int baseX = x << 1;
  int baseY = y << 1;

  int sumCb = 0;
  int sumCr = 0;

  for (int dy = 0; dy < 2; ++dy) {
    const Npp8u *srcRow = (const Npp8u *)((const char *)pSrc + (baseY + dy) * nSrcStep);
    for (int dx = 0; dx < 2; ++dx) {
      int idx = (baseX + dx) * 4;
      Npp8u b = srcRow[idx + 0];
      Npp8u g = srcRow[idx + 1];
      Npp8u r = srcRow[idx + 2];

      Npp8u yv, cbv, crv;
      rgb_to_ycbcr_pixel(r, g, b, yv, cbv, crv);
      sumCb += cbv;
      sumCr += crv;
    }
  }

  Npp8u *dstRowCb = (Npp8u *)((char *)pDstCb + y * nDstCbStep);
  Npp8u *dstRowCr = (Npp8u *)((char *)pDstCr + y * nDstCrStep);
  dstRowCb[x] = static_cast<Npp8u>(sumCb / 4);
  dstRowCr[x] = static_cast<Npp8u>(sumCr / 4);
}

__global__ void ycbcr420_to_rgb_c3_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                          const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep, int width,
                                          int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *rowY = (const Npp8u *)((const char *)pSrcY + y * nSrcYStep);
  const Npp8u *rowCb = (const Npp8u *)((const char *)pSrcCb + (y >> 1) * nSrcCbStep);
  const Npp8u *rowCr = (const Npp8u *)((const char *)pSrcCr + (y >> 1) * nSrcCrStep);

  Npp8u yv = rowY[x];
  Npp8u cbv = rowCb[x >> 1];
  Npp8u crv = rowCr[x >> 1];

  Npp8u r, g, b;
  ycbcr_to_rgb_pixel(yv, cbv, crv, r, g, b);

  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);
  int idx = x * 3;
  dstRow[idx + 0] = r;
  dstRow[idx + 1] = g;
  dstRow[idx + 2] = b;
}

extern "C" {

cudaError_t nppiRGBToYCbCr420_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  rgb_to_ycbcr420_y_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, nDstYStep, oSizeROI.width,
                                                                  oSizeROI.height);

  dim3 gridSizeCb((oSizeROI.width / 2 + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height / 2 + blockSize.y - 1) / blockSize.y);
  rgb_to_ycbcr420_cbcr_c3_kernel<<<gridSizeCb, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstCb, nDstCbStep, pDstCr,
                                                                       nDstCrStep, oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

cudaError_t nppiBGRToYCbCr420_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  bgr_to_ycbcr420_y_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, nDstYStep, oSizeROI.width,
                                                                  oSizeROI.height);

  dim3 gridSizeCb((oSizeROI.width / 2 + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height / 2 + blockSize.y - 1) / blockSize.y);
  bgr_to_ycbcr420_cbcr_c3_kernel<<<gridSizeCb, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstCb, nDstCbStep, pDstCr,
                                                                       nDstCrStep, oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

cudaError_t nppiBGRToYCbCr420_8u_AC4P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                               Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                               NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  bgr_to_ycbcr420_y_ac4_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, nDstYStep, oSizeROI.width,
                                                                   oSizeROI.height);

  dim3 gridSizeCb((oSizeROI.width / 2 + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height / 2 + blockSize.y - 1) / blockSize.y);
  bgr_to_ycbcr420_cbcr_ac4_kernel<<<gridSizeCb, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstCb, nDstCbStep, pDstCr,
                                                                        nDstCrStep, oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

cudaError_t nppiYCbCr420ToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  ycbcr420_to_rgb_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcCb, nSrcCbStep, pSrcCr,
                                                                nSrcCrStep, pDst, nDstStep, oSizeROI.width,
                                                                oSizeROI.height);
  return cudaGetLastError();
}

} // extern "C"
