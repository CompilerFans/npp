#include "nppdefs.h"
#include <cuda_runtime.h>

__device__ inline Npp8u clamp_u8(float v) { return (Npp8u)fmaxf(0.0f, fminf(255.0f, v)); }

__device__ inline void rgb_to_yuv(Npp8u r, Npp8u g, Npp8u b, Npp8u &y, Npp8u &u, Npp8u &v) {
  float fy = 0.299f * r + 0.587f * g + 0.114f * b;
  float fu = 0.492f * (b - fy) + 128.0f;
  float fv = 0.877f * (r - fy) + 128.0f;
  y = clamp_u8(fy + 0.5f);
  u = clamp_u8(fu + 0.5f);
  v = clamp_u8(fv + 0.5f);
}

// RGB到YUV转换的kernelimplementation
// 使用NPP头文件说明的转换公式
__global__ void nppiRGBToYUV_8u_C3R_kernel_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                                int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;

    Npp8u r = pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = pSrcRow[idx + 2];

    Npp8u y, u, v;
    rgb_to_yuv(r, g, b, y, u, v);

    pDstRow[idx + 0] = y;
    pDstRow[idx + 1] = u;
    pDstRow[idx + 2] = v;
  }
}

__global__ void nppiRGBToYUV_8u_AC4R_kernel_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                                 int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 4;

    Npp8u r = pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = pSrcRow[idx + 2];
    Npp8u a = pSrcRow[idx + 3];

    Npp8u yv, uv, vv;
    rgb_to_yuv(r, g, b, yv, uv, vv);

    pDstRow[idx + 0] = yv;
    pDstRow[idx + 1] = uv;
    pDstRow[idx + 2] = vv;
    pDstRow[idx + 3] = a;
  }
}

__global__ void nppiRGBToYUV_8u_P3R_kernel_impl(const Npp8u *pSrcR, const Npp8u *pSrcG, const Npp8u *pSrcB, int nSrcStep,
                                                Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV, int nDstStep, int width,
                                                int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *srcRRow = (const Npp8u *)((const char *)pSrcR + y * nSrcStep);
    const Npp8u *srcGRow = (const Npp8u *)((const char *)pSrcG + y * nSrcStep);
    const Npp8u *srcBRow = (const Npp8u *)((const char *)pSrcB + y * nSrcStep);
    Npp8u *dstYRow = (Npp8u *)((char *)pDstY + y * nDstStep);
    Npp8u *dstURow = (Npp8u *)((char *)pDstU + y * nDstStep);
    Npp8u *dstVRow = (Npp8u *)((char *)pDstV + y * nDstStep);

    Npp8u yv, uv, vv;
    rgb_to_yuv(srcRRow[x], srcGRow[x], srcBRow[x], yv, uv, vv);

    dstYRow[x] = yv;
    dstURow[x] = uv;
    dstVRow[x] = vv;
  }
}

__global__ void nppiRGBToYUV_8u_C3P3R_kernel_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU,
                                                  Npp8u *pDstV, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dstYRow = (Npp8u *)((char *)pDstY + y * nDstStep);
    Npp8u *dstURow = (Npp8u *)((char *)pDstU + y * nDstStep);
    Npp8u *dstVRow = (Npp8u *)((char *)pDstV + y * nDstStep);

    int idx = x * 3;
    Npp8u r = pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = pSrcRow[idx + 2];

    Npp8u yv, uv, vv;
    rgb_to_yuv(r, g, b, yv, uv, vv);

    dstYRow[x] = yv;
    dstURow[x] = uv;
    dstVRow[x] = vv;
  }
}

__global__ void nppiRGBToYUV_8u_AC4P4R_kernel_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU,
                                                   Npp8u *pDstV, Npp8u *pDstA, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *dstYRow = (Npp8u *)((char *)pDstY + y * nDstStep);
    Npp8u *dstURow = (Npp8u *)((char *)pDstU + y * nDstStep);
    Npp8u *dstVRow = (Npp8u *)((char *)pDstV + y * nDstStep);
    Npp8u *dstARow = (Npp8u *)((char *)pDstA + y * nDstStep);

    int idx = x * 4;
    Npp8u r = pSrcRow[idx + 0];
    Npp8u g = pSrcRow[idx + 1];
    Npp8u b = pSrcRow[idx + 2];
    Npp8u a = pSrcRow[idx + 3];

    Npp8u yv, uv, vv;
    rgb_to_yuv(r, g, b, yv, uv, vv);

    dstYRow[x] = yv;
    dstURow[x] = uv;
    dstVRow[x] = vv;
    dstARow[x] = a;
  }
}

// YUV到RGB转换的kernelimplementation
__global__ void nppiYUVToRGB_8u_C3R_kernel_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                                int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;

    float Y = pSrcRow[idx + 0];
    float U = pSrcRow[idx + 1] - 128.0f;
    float V = pSrcRow[idx + 2] - 128.0f;

    float R = Y + 1.140f * V;
    float G = Y - 0.394f * U - 0.581f * V;
    float B = Y + 2.032f * U;

    pDstRow[idx + 0] = clamp_u8(R + 0.5f);
    pDstRow[idx + 1] = clamp_u8(G + 0.5f);
    pDstRow[idx + 2] = clamp_u8(B + 0.5f);
  }
}

extern "C" {
// RGB到YUV转换
cudaError_t nppiRGBToYUV_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToYUV_8u_C3R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                      oSizeROI.height);

  return cudaGetLastError();
}

cudaError_t nppiRGBToYUV_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToYUV_8u_AC4R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                       oSizeROI.height);

  return cudaGetLastError();
}

cudaError_t nppiRGBToYUV_8u_P3R_kernel(const Npp8u *pSrcR, const Npp8u *pSrcG, const Npp8u *pSrcB, int nSrcStep,
                                       Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToYUV_8u_P3R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrcR, pSrcG, pSrcB, nSrcStep, pDstY, pDstU,
                                                                      pDstV, nDstStep, oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

cudaError_t nppiRGBToYUV_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV,
                                         int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToYUV_8u_C3P3R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, pDstU, pDstV, nDstStep,
                                                                        oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

cudaError_t nppiRGBToYUV_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV,
                                          Npp8u *pDstA, int nDstStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiRGBToYUV_8u_AC4P4R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDstY, pDstU, pDstV, pDstA,
                                                                         nDstStep, oSizeROI.width, oSizeROI.height);

  return cudaGetLastError();
}

// YUV到RGB转换
cudaError_t nppiYUVToRGB_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiYUVToRGB_8u_C3R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                      oSizeROI.height);

  return cudaGetLastError();
}
}