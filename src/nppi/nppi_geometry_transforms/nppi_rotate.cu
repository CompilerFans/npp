#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define M_PI 3.14159265358979323846
__global__ void rotate_8u_C1R_kernel(const Npp8u *__restrict__ pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp8u *__restrict__ pDst, int nDstStep, NppiRect oDstROI, double nAngle,
                                     double nShiftX, double nShiftY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= oDstROI.width || y >= oDstROI.height)
    return;

  // Convert angle to radians
  double angleRad = nAngle * M_PI / 180.0;
  double cosA = cos(angleRad);
  double sinA = sin(angleRad);

  // Destination coordinates relative to ROI
  int dstX = x + oDstROI.x;
  int dstY = y + oDstROI.y;

  // Apply reverse transformation to find source coordinates
  // For rotation, the inverse transformation is: rotate by -angle
  double srcXf = (dstX - nShiftX) * cosA - (dstY - nShiftY) * sinA;
  double srcYf = (dstX - nShiftX) * sinA + (dstY - nShiftY) * cosA;

  int srcX = (int)(srcXf + 0.5);
  int srcY = (int)(srcYf + 0.5);

  // Check if source coordinates are within source ROI
  if (srcX >= oSrcROI.x && srcX < oSrcROI.x + oSrcROI.width && srcY >= oSrcROI.y && srcY < oSrcROI.y + oSrcROI.height) {

    const Npp8u *srcRow = (const Npp8u *)(((const char *)pSrc) + srcY * nSrcStep);
    Npp8u *dstRow = (Npp8u *)(((char *)pDst) + dstY * nDstStep);

    dstRow[dstX] = srcRow[srcX];
  } else {
    // Set to black for pixels outside source ROI
    Npp8u *dstRow = (Npp8u *)(((char *)pDst) + dstY * nDstStep);
    dstRow[dstX] = 0;
  }
}
__global__ void rotate_8u_C3R_kernel(const Npp8u *__restrict__ pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp8u *__restrict__ pDst, int nDstStep, NppiRect oDstROI, double nAngle,
                                     double nShiftX, double nShiftY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= oDstROI.width || y >= oDstROI.height)
    return;

  // Convert angle to radians
  double angleRad = nAngle * M_PI / 180.0;
  double cosA = cos(angleRad);
  double sinA = sin(angleRad);

  // Destination coordinates relative to ROI
  int dstX = x + oDstROI.x;
  int dstY = y + oDstROI.y;

  // Apply reverse transformation to find source coordinates
  // For rotation, the inverse transformation is: rotate by -angle
  double srcXf = (dstX - nShiftX) * cosA - (dstY - nShiftY) * sinA;
  double srcYf = (dstX - nShiftX) * sinA + (dstY - nShiftY) * cosA;

  int srcX = (int)(srcXf + 0.5);
  int srcY = (int)(srcYf + 0.5);

  // Check if source coordinates are within source ROI
  if (srcX >= oSrcROI.x && srcX < oSrcROI.x + oSrcROI.width && srcY >= oSrcROI.y && srcY < oSrcROI.y + oSrcROI.height) {

    const Npp8u *srcRow = (const Npp8u *)(((const char *)pSrc) + srcY * nSrcStep);
    Npp8u *dstRow = (Npp8u *)(((char *)pDst) + dstY * nDstStep);

    // Copy 3 channels
    dstRow[dstX * 3 + 0] = srcRow[srcX * 3 + 0];
    dstRow[dstX * 3 + 1] = srcRow[srcX * 3 + 1];
    dstRow[dstX * 3 + 2] = srcRow[srcX * 3 + 2];
  } else {
    // Set to black for pixels outside source ROI
    Npp8u *dstRow = (Npp8u *)(((char *)pDst) + dstY * nDstStep);
    dstRow[dstX * 3 + 0] = 0;
    dstRow[dstX * 3 + 1] = 0;
    dstRow[dstX * 3 + 2] = 0;
  }
}
__global__ void rotate_32f_C1R_kernel(const Npp32f *__restrict__ pSrc, NppiSize oSrcSize, int nSrcStep,
                                      NppiRect oSrcROI, Npp32f *__restrict__ pDst, int nDstStep, NppiRect oDstROI,
                                      double nAngle, double nShiftX, double nShiftY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= oDstROI.width || y >= oDstROI.height)
    return;

  // Convert angle to radians
  double angleRad = nAngle * M_PI / 180.0;
  double cosA = cos(angleRad);
  double sinA = sin(angleRad);

  // Destination coordinates relative to ROI
  int dstX = x + oDstROI.x;
  int dstY = y + oDstROI.y;

  // Apply reverse transformation to find source coordinates
  // For rotation, the inverse transformation is: rotate by -angle
  double srcXf = (dstX - nShiftX) * cosA - (dstY - nShiftY) * sinA;
  double srcYf = (dstX - nShiftX) * sinA + (dstY - nShiftY) * cosA;

  int srcX = (int)(srcXf + 0.5);
  int srcY = (int)(srcYf + 0.5);

  // Check if source coordinates are within source ROI
  if (srcX >= oSrcROI.x && srcX < oSrcROI.x + oSrcROI.width && srcY >= oSrcROI.y && srcY < oSrcROI.y + oSrcROI.height) {

    const Npp32f *srcRow = (const Npp32f *)(((const char *)pSrc) + srcY * nSrcStep);
    Npp32f *dstRow = (Npp32f *)(((char *)pDst) + dstY * nDstStep);

    dstRow[dstX] = srcRow[srcX];
  } else {
    // Set to zero for pixels outside source ROI
    Npp32f *dstRow = (Npp32f *)(((char *)pDst) + dstY * nDstStep);
    dstRow[dstX] = 0.0f;
  }
}

extern "C" {
NppStatus nppiRotate_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                                     int eInterpolation, NppStreamContext nppStreamCtx) {

  // Setup kernel launch parameters
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  rotate_8u_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst,
                                                                         nDstStep, oDstROI, nAngle, nShiftX, nShiftY);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}
NppStatus nppiRotate_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                                     int eInterpolation, NppStreamContext nppStreamCtx) {

  // Setup kernel launch parameters
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  rotate_8u_C3R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst,
                                                                         nDstStep, oDstROI, nAngle, nShiftX, nShiftY);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}
NppStatus nppiRotate_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX,
                                      double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx) {

  // Setup kernel launch parameters
  dim3 blockSize(16, 16);
  dim3 gridSize((oDstROI.width + blockSize.x - 1) / blockSize.x, (oDstROI.height + blockSize.y - 1) / blockSize.y);

  // Launch kernel with the specified GPU stream
  rotate_32f_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst,
                                                                          nDstStep, oDstROI, nAngle, nShiftX, nShiftY);

  // Check for kernel launch errors
  cudaError_t cudaErr = cudaGetLastError();
  if (cudaErr != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}
}
