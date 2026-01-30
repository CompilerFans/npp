#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__device__ inline Npp8u clamp_u8(float v) {
  if (v < 0.0f) {
    v = 0.0f;
  } else if (v > 255.0f) {
    v = 255.0f;
  }
  return static_cast<Npp8u>(v);
}

__device__ inline void bgr_to_lab_pixel(Npp8u b, Npp8u g, Npp8u r, Npp8u &l, Npp8u &a, Npp8u &bb) {
  const float nCIE_LAB_D65_xn = 0.950455f;
  const float nCIE_LAB_D65_zn = 1.088753f;

  float nNormalizedR = static_cast<float>(r) * 0.003921569f;
  float nNormalizedG = static_cast<float>(g) * 0.003921569f;
  float nNormalizedB = static_cast<float>(b) * 0.003921569f;

  float nX = 0.412453f * nNormalizedR + 0.35758f * nNormalizedG + 0.180423f * nNormalizedB;
  float nY = 0.212671f * nNormalizedR + 0.71516f * nNormalizedG + 0.072169f * nNormalizedB;
  float nZ = 0.019334f * nNormalizedR + 0.119193f * nNormalizedG + 0.950227f * nNormalizedB;

  float nL = cbrtf(nY);
  float nfX = nX * (1.0f / nCIE_LAB_D65_xn);
  float nfY = nY;
  float nfZ = nZ * (1.0f / nCIE_LAB_D65_zn);

  nfY = nL - 16.0f;
  nL = 116.0f * nL - 16.0f;
  float nA = cbrtf(nfX) - 16.0f;
  nA = 500.0f * (nA - nfY);
  float nB = cbrtf(nfZ) - 16.0f;
  nB = 200.0f * (nfY - nB);

  nL = nL * 255.0f * 0.01f;
  nA = nA + 128.0f;
  nB = nB + 128.0f;

  l = clamp_u8(nL);
  a = clamp_u8(nA);
  bb = clamp_u8(nB);
}

__device__ inline void lab_to_bgr_pixel(Npp8u l, Npp8u a, Npp8u bb, Npp8u &b, Npp8u &g, Npp8u &r) {
  const float nCIE_LAB_D65_xn = 0.950455f;
  const float nCIE_LAB_D65_zn = 1.088753f;

  float nL = static_cast<float>(l) * 100.0f * 0.003921569f;
  float nA = static_cast<float>(a) - 128.0f;
  float nB = static_cast<float>(bb) - 128.0f;

  float nP = (nL + 16.0f) * 0.008621f;
  float nNormalizedY = nP * nP * nP;
  float nNormalizedX = nCIE_LAB_D65_xn * powf((nP + nA * 0.002f), 3.0f);
  float nNormalizedZ = nCIE_LAB_D65_zn * powf((nP - nB * 0.005f), 3.0f);

  float nR = 3.240479f * nNormalizedX - 1.53715f * nNormalizedY - 0.498535f * nNormalizedZ;
  float nG = -0.969256f * nNormalizedX + 1.875991f * nNormalizedY + 0.041556f * nNormalizedZ;
  float nBB = 0.055648f * nNormalizedX - 0.204043f * nNormalizedY + 1.057311f * nNormalizedZ;

  if (nR > 1.0f) {
    nR = 1.0f;
  }
  if (nG > 1.0f) {
    nG = 1.0f;
  }
  if (nBB > 1.0f) {
    nBB = 1.0f;
  }
  if (nR < 0.0f) {
    nR = 0.0f;
  }
  if (nG < 0.0f) {
    nG = 0.0f;
  }
  if (nBB < 0.0f) {
    nBB = 0.0f;
  }

  r = clamp_u8(nR * 255.0f);
  g = clamp_u8(nG * 255.0f);
  b = clamp_u8(nBB * 255.0f);
}

__global__ void bgr_to_lab_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                     int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

  int idx = x * 3;
  Npp8u b = pSrcRow[idx + 0];
  Npp8u g = pSrcRow[idx + 1];
  Npp8u r = pSrcRow[idx + 2];

  Npp8u l, a, bb;
  bgr_to_lab_pixel(b, g, r, l, a, bb);

  pDstRow[idx + 0] = l;
  pDstRow[idx + 1] = a;
  pDstRow[idx + 2] = bb;
}

__global__ void lab_to_bgr_c3_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                     int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);

  int idx = x * 3;
  Npp8u l = pSrcRow[idx + 0];
  Npp8u a = pSrcRow[idx + 1];
  Npp8u bb = pSrcRow[idx + 2];

  Npp8u b, g, r;
  lab_to_bgr_pixel(l, a, bb, b, g, r);

  pDstRow[idx + 0] = b;
  pDstRow[idx + 1] = g;
  pDstRow[idx + 2] = r;
}

extern "C" {

cudaError_t nppiBGRToLab_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  bgr_to_lab_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                           oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiLabToBGR_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  lab_to_bgr_c3_kernel<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                           oSizeROI.height);
  return cudaGetLastError();
}
}
