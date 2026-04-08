#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {

__global__ void copy_y_plane_kernel(const Npp8u *pSrcY, int nSrcYStep, Npp8u *pDstY, int nDstYStep, int width,
                                    int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrcY + y * nSrcYStep);
  Npp8u *dstRow = (Npp8u *)((char *)pDstY + y * nDstYStep);
  dstRow[x] = srcRow[x];
}

__global__ void planar420_interleave_chroma_pair_kernel(const Npp8u *pSrcChroma0, int nSrcChroma0Step,
                                                        const Npp8u *pSrcChroma1, int nSrcChroma1Step,
                                                        Npp8u *pDstChroma01, int nDstChroma01Step, int width,
                                                        int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int chromaWidth = width >> 1;
  int chromaHeight = height >> 1;
  if (x >= chromaWidth || y >= chromaHeight) {
    return;
  }

  const Npp8u *srcRowChroma0 = (const Npp8u *)((const char *)pSrcChroma0 + y * nSrcChroma0Step);
  const Npp8u *srcRowChroma1 = (const Npp8u *)((const char *)pSrcChroma1 + y * nSrcChroma1Step);
  Npp8u *dstRow = (Npp8u *)((char *)pDstChroma01 + y * nDstChroma01Step);
  int dstIdx = x << 1;
  dstRow[dstIdx] = srcRowChroma0[x];
  dstRow[dstIdx + 1] = srcRowChroma1[x];
}

__global__ void semiplanar420_split_chroma_pair_kernel(const Npp8u *pSrcChroma01, int nSrcChroma01Step,
                                                       Npp8u *pDstChroma0, int nDstChroma0Step, Npp8u *pDstChroma1,
                                                       int nDstChroma1Step, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int chromaWidth = width >> 1;
  int chromaHeight = height >> 1;
  if (x >= chromaWidth || y >= chromaHeight) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrcChroma01 + y * nSrcChroma01Step);
  Npp8u *dstRowChroma0 = (Npp8u *)((char *)pDstChroma0 + y * nDstChroma0Step);
  Npp8u *dstRowChroma1 = (Npp8u *)((char *)pDstChroma1 + y * nDstChroma1Step);
  int srcIdx = x << 1;
  dstRowChroma0[x] = srcRow[srcIdx];
  dstRowChroma1[x] = srcRow[srcIdx + 1];
}

__global__ void planar420_to_planar422_chroma_kernel(const Npp8u *pSrcChroma, int nSrcChromaStep, Npp8u *pDstChroma,
                                                     int nDstChromaStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int chromaWidth = width >> 1;
  if (x >= chromaWidth || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrcChroma + (y >> 1) * nSrcChromaStep);
  Npp8u *dstRow = (Npp8u *)((char *)pDstChroma + y * nDstChromaStep);
  dstRow[x] = srcRow[x];
}

__global__ void semiplanar420_to_planar422_chroma_pair_kernel(const Npp8u *pSrcChroma01, int nSrcChroma01Step,
                                                              Npp8u *pDstChroma0, int nDstChroma0Step,
                                                              Npp8u *pDstChroma1, int nDstChroma1Step, int width,
                                                              int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int chromaWidth = width >> 1;
  if (x >= chromaWidth || y >= height) {
    return;
  }

  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrcChroma01 + (y >> 1) * nSrcChroma01Step);
  Npp8u *dstRowChroma0 = (Npp8u *)((char *)pDstChroma0 + y * nDstChroma0Step);
  Npp8u *dstRowChroma1 = (Npp8u *)((char *)pDstChroma1 + y * nDstChroma1Step);
  int srcIdx = x << 1;
  dstRowChroma0[x] = srcRow[srcIdx];
  dstRowChroma1[x] = srcRow[srcIdx + 1];
}

__global__ void semiplanar420_to_packed422_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcChroma01,
                                                  int nSrcChroma01Step, Npp8u *pDst, int nDstStep, int width,
                                                  int height, int packedMode) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int chromaWidth = width >> 1;
  if (x >= chromaWidth || y >= height) {
    return;
  }

  const Npp8u *srcYRow = (const Npp8u *)((const char *)pSrcY + y * nSrcYStep);
  const Npp8u *srcChromaRow = (const Npp8u *)((const char *)pSrcChroma01 + (y >> 1) * nSrcChroma01Step);
  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);
  int chromaIdx = x << 1;
  int dstIdx = x << 2;
  Npp8u y0 = srcYRow[x << 1];
  Npp8u y1 = srcYRow[(x << 1) + 1];
  Npp8u cb = srcChromaRow[chromaIdx];
  Npp8u cr = srcChromaRow[chromaIdx + 1];

  if (packedMode == 0) {
    dstRow[dstIdx] = y0;
    dstRow[dstIdx + 1] = cb;
    dstRow[dstIdx + 2] = y1;
    dstRow[dstIdx + 3] = cr;
  } else {
    dstRow[dstIdx] = cb;
    dstRow[dstIdx + 1] = y0;
    dstRow[dstIdx + 2] = cr;
    dstRow[dstIdx + 3] = y1;
  }
}

__global__ void planar_ycrcb420_to_packed_cbycr422_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCr,
                                                          int nSrcCrStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                                          Npp8u *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int chromaWidth = width >> 1;
  if (x >= chromaWidth || y >= height) {
    return;
  }

  const Npp8u *srcYRow = (const Npp8u *)((const char *)pSrcY + y * nSrcYStep);
  const Npp8u *srcCrRow = (const Npp8u *)((const char *)pSrcCr + (y >> 1) * nSrcCrStep);
  const Npp8u *srcCbRow = (const Npp8u *)((const char *)pSrcCb + (y >> 1) * nSrcCbStep);
  Npp8u *dstRow = (Npp8u *)((char *)pDst + y * nDstStep);
  int dstIdx = x << 2;
  dstRow[dstIdx] = srcCbRow[x];
  dstRow[dstIdx + 1] = srcYRow[x << 1];
  dstRow[dstIdx + 2] = srcCrRow[x];
  dstRow[dstIdx + 3] = srcYRow[(x << 1) + 1];
}

__global__ void planar_ycrcb420_to_semiplanar411_kernel(const Npp8u *pSrcCr, int nSrcCrStep, const Npp8u *pSrcCb,
                                                        int nSrcCbStep, Npp8u *pDstCbCr, int nDstCbCrStep, int width,
                                                        int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int chromaWidth411 = width >> 2;
  if (x >= chromaWidth411 || y >= height) {
    return;
  }

  int srcY = y >> 1;
  int srcX = (x << 1) + (y & 1);
  const Npp8u *srcCbRow = (const Npp8u *)((const char *)pSrcCb + srcY * nSrcCbStep);
  const Npp8u *srcCrRow = (const Npp8u *)((const char *)pSrcCr + srcY * nSrcCrStep);
  Npp8u *dstRow = (Npp8u *)((char *)pDstCbCr + y * nDstCbCrStep);
  int dstIdx = x << 1;
  dstRow[dstIdx] = srcCbRow[srcX];
  dstRow[dstIdx + 1] = srcCrRow[srcX];
}

__global__ void semiplanar420_to_planar411_chroma_pair_kernel(const Npp8u *pSrcCbCr, int nSrcCbCrStep, Npp8u *pDstCb,
                                                              int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                                              int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int chromaWidth411 = width >> 2;
  if (x >= chromaWidth411 || y >= height) {
    return;
  }

  int srcY = y >> 1;
  int srcX = (x << 1) + (y & 1);
  const Npp8u *srcRow = (const Npp8u *)((const char *)pSrcCbCr + srcY * nSrcCbCrStep);
  Npp8u *dstCbRow = (Npp8u *)((char *)pDstCb + y * nDstCbStep);
  Npp8u *dstCrRow = (Npp8u *)((char *)pDstCr + y * nDstCrStep);
  int srcIdx = srcX << 1;
  dstCbRow[x] = srcRow[srcIdx];
  dstCrRow[x] = srcRow[srcIdx + 1];
}

} // namespace

extern "C" {

cudaError_t nppiPlanar420ToSemiPlanar420_8u_P3P2R_kernel(const Npp8u *pSrcY, int nSrcYStep,
                                                         const Npp8u *pSrcChroma0, int nSrcChroma0Step,
                                                         const Npp8u *pSrcChroma1, int nSrcChroma1Step,
                                                         Npp8u *pDstY, int nDstYStep, Npp8u *pDstChroma01,
                                                         int nDstChroma01Step, NppiSize oSizeROI,
                                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  copy_y_plane_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pDstY, nDstYStep, oSizeROI.width,
                                                           oSizeROI.height);

  dim3 chromaGrid((oSizeROI.width / 2 + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height / 2 + blockSize.y - 1) / blockSize.y);
  planar420_interleave_chroma_pair_kernel<<<chromaGrid, blockSize, 0, stream>>>(
      pSrcChroma0, nSrcChroma0Step, pSrcChroma1, nSrcChroma1Step, pDstChroma01, nDstChroma01Step, oSizeROI.width,
      oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiSemiPlanar420ToPlanar420_8u_P2P3R_kernel(const Npp8u *pSrcY, int nSrcYStep,
                                                         const Npp8u *pSrcChroma01, int nSrcChroma01Step,
                                                         Npp8u *pDstY, int nDstYStep, Npp8u *pDstChroma0,
                                                         int nDstChroma0Step, Npp8u *pDstChroma1,
                                                         int nDstChroma1Step, NppiSize oSizeROI,
                                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  copy_y_plane_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pDstY, nDstYStep, oSizeROI.width,
                                                           oSizeROI.height);

  dim3 chromaGrid((oSizeROI.width / 2 + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height / 2 + blockSize.y - 1) / blockSize.y);
  semiplanar420_split_chroma_pair_kernel<<<chromaGrid, blockSize, 0, stream>>>(
      pSrcChroma01, nSrcChroma01Step, pDstChroma0, nDstChroma0Step, pDstChroma1, nDstChroma1Step, oSizeROI.width,
      oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiPlanar420ToPlanar422_8u_P3P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb,
                                                     int nSrcCbStep, const Npp8u *pSrcCr, int nSrcCrStep,
                                                     Npp8u *pDstY, int nDstYStep, Npp8u *pDstCb, int nDstCbStep,
                                                     Npp8u *pDstCr, int nDstCrStep, NppiSize oSizeROI,
                                                     cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 lumaGrid((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  copy_y_plane_kernel<<<lumaGrid, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pDstY, nDstYStep, oSizeROI.width,
                                                           oSizeROI.height);

  dim3 chromaGrid((oSizeROI.width / 2 + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  planar420_to_planar422_chroma_kernel<<<chromaGrid, blockSize, 0, stream>>>(pSrcCb, nSrcCbStep, pDstCb, nDstCbStep,
                                                                              oSizeROI.width, oSizeROI.height);
  planar420_to_planar422_chroma_kernel<<<chromaGrid, blockSize, 0, stream>>>(pSrcCr, nSrcCrStep, pDstCr, nDstCrStep,
                                                                              oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiSemiPlanar420ToPlanar422_8u_P2P3R_kernel(const Npp8u *pSrcY, int nSrcYStep,
                                                         const Npp8u *pSrcCbCr, int nSrcCbCrStep, Npp8u *pDstY,
                                                         int nDstYStep, Npp8u *pDstCb, int nDstCbStep,
                                                         Npp8u *pDstCr, int nDstCrStep, NppiSize oSizeROI,
                                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 lumaGrid((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  copy_y_plane_kernel<<<lumaGrid, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pDstY, nDstYStep, oSizeROI.width,
                                                           oSizeROI.height);

  dim3 chromaGrid((oSizeROI.width / 2 + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  semiplanar420_to_planar422_chroma_pair_kernel<<<chromaGrid, blockSize, 0, stream>>>(
      pSrcCbCr, nSrcCbCrStep, pDstCb, nDstCbStep, pDstCr, nDstCrStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiSemiPlanar420ToPacked422_8u_P2C2R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                                         int nSrcCbCrStep, Npp8u *pDst, int nDstStep,
                                                         NppiSize oSizeROI, int packedMode, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width / 2 + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  semiplanar420_to_packed422_kernel<<<gridSize, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep,
                                                                         pDst, nDstStep, oSizeROI.width,
                                                                         oSizeROI.height, packedMode);
  return cudaGetLastError();
}

cudaError_t nppiPlanarYCrCb420ToPackedCbYCr422_8u_P3C2R_kernel(const Npp8u *pSrcY, int nSrcYStep,
                                                               const Npp8u *pSrcCr, int nSrcCrStep,
                                                               const Npp8u *pSrcCb, int nSrcCbStep, Npp8u *pDst,
                                                               int nDstStep, NppiSize oSizeROI,
                                                               cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width / 2 + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  planar_ycrcb420_to_packed_cbycr422_kernel<<<gridSize, blockSize, 0, stream>>>(
      pSrcY, nSrcYStep, pSrcCr, nSrcCrStep, pSrcCb, nSrcCbStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiPlanarYCrCb420ToSemiPlanar411_8u_P3P2R_kernel(const Npp8u *pSrcY, int nSrcYStep,
                                                              const Npp8u *pSrcCr, int nSrcCrStep,
                                                              const Npp8u *pSrcCb, int nSrcCbStep, Npp8u *pDstY,
                                                              int nDstYStep, Npp8u *pDstCbCr, int nDstCbCrStep,
                                                              NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 lumaGrid((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  copy_y_plane_kernel<<<lumaGrid, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pDstY, nDstYStep, oSizeROI.width,
                                                           oSizeROI.height);

  dim3 chromaGrid((oSizeROI.width / 4 + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  planar_ycrcb420_to_semiplanar411_kernel<<<chromaGrid, blockSize, 0, stream>>>(
      pSrcCr, nSrcCrStep, pSrcCb, nSrcCbStep, pDstCbCr, nDstCbCrStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

cudaError_t nppiSemiPlanar420ToPlanar411_8u_P2P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                                         int nSrcCbCrStep, Npp8u *pDstY, int nDstYStep,
                                                         Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr,
                                                         int nDstCrStep, NppiSize oSizeROI, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 lumaGrid((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  copy_y_plane_kernel<<<lumaGrid, blockSize, 0, stream>>>(pSrcY, nSrcYStep, pDstY, nDstYStep, oSizeROI.width,
                                                           oSizeROI.height);

  dim3 chromaGrid((oSizeROI.width / 4 + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
  semiplanar420_to_planar411_chroma_pair_kernel<<<chromaGrid, blockSize, 0, stream>>>(
      pSrcCbCr, nSrcCbCrStep, pDstCb, nDstCbStep, pDstCr, nDstCrStep, oSizeROI.width, oSizeROI.height);
  return cudaGetLastError();
}

} // extern "C"
