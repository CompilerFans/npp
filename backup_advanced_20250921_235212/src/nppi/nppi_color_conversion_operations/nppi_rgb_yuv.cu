#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// RGB到YUV转换的kernel实现
// 使用ITU-R BT.601标准转换公式
__global__ void nppiRGBToYUV_8u_C3R_kernel_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                                int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;

    // 获取RGB值
    float R = pSrcRow[idx + 0];
    float G = pSrcRow[idx + 1];
    float B = pSrcRow[idx + 2];

    // RGB到YUV转换公式
    // Y = 0.299*R + 0.587*G + 0.114*B
    // U = -0.147*R - 0.289*G + 0.436*B + 128
    // V = 0.615*R - 0.515*G - 0.100*B + 128
    float Y = 0.299f * R + 0.587f * G + 0.114f * B;
    float U = -0.147f * R - 0.289f * G + 0.436f * B + 128.0f;
    float V = 0.615f * R - 0.515f * G - 0.100f * B + 128.0f;

    // 饱和到[0,255]
    pDstRow[idx + 0] = (Npp8u)fminf(fmaxf(Y, 0.0f), 255.0f);
    pDstRow[idx + 1] = (Npp8u)fminf(fmaxf(U, 0.0f), 255.0f);
    pDstRow[idx + 2] = (Npp8u)fminf(fmaxf(V, 0.0f), 255.0f);
  }
}

// YUV到RGB转换的kernel实现
__global__ void nppiYUVToRGB_8u_C3R_kernel_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                                int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

    int idx = x * 3;

    // 获取YUV值
    float Y = pSrcRow[idx + 0];
    float U = pSrcRow[idx + 1] - 128.0f;
    float V = pSrcRow[idx + 2] - 128.0f;

    // YUV到RGB转换公式
    // R = Y + 1.140*V
    // G = Y - 0.395*U - 0.581*V
    // B = Y + 2.032*U
    float R = Y + 1.140f * V;
    float G = Y - 0.395f * U - 0.581f * V;
    float B = Y + 2.032f * U;

    // 饱和到[0,255]
    pDstRow[idx + 0] = (Npp8u)fminf(fmaxf(R, 0.0f), 255.0f);
    pDstRow[idx + 1] = (Npp8u)fminf(fmaxf(G, 0.0f), 255.0f);
    pDstRow[idx + 2] = (Npp8u)fminf(fmaxf(B, 0.0f), 255.0f);
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