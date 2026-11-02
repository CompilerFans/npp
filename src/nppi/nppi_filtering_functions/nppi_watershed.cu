// nppiSegmentWatershed_16u: best-effort deterministic watershed.
//
// NPP 13.4's implementation is a cell-state machine (Initialize / Intra /
// Inter state updates and several output kernels).  Its pSrcDst output is an
// internal state image rather than the filtered source, and the exact
// flooding order could not be reproduced; this executor therefore implements
// a standard topographic watershed:
//   * every pixel descends to the neighbour with the smallest
//     (height, linear index) until it reaches a local minimum (the seed);
//     pointer jumping converges the labels;
//   * the marker label of a pixel is the linear index of its seed;
//   * pixels adjacent (4-way for nppiNormL1, 8-way for nppiNormInf) to a
//     different label are segment boundaries;
//   * pSrcDst keeps the source image and applies the requested boundary
//     policy: NONE = unchanged, BLACK = 0 on boundaries, WHITE = 65535,
//     ONLY = 0 on boundaries / 65535 elsewhere, CONTRAST = 0/65535 chosen
//     from the relative root indices.
// pMarkerLabels may be NULL.  Validation order matches the probe: null
// src/buffer -8, zero or negative ROI -6, unsupported norm -5, unsupported
// boundary type -2.
#include "npp.h"

#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ int linearIndex(int x, int y, int w) { return y * w + x; }

__global__ void watershedParentKernel(const Npp16u *pSrc, int nSrcStep, int width, int height, int connectivity,
                                      int *pParent) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int self = linearIndex(x, y, width);
  const Npp16u *row = (const Npp16u *)((const char *)pSrc + (size_t)y * nSrcStep);
  Npp16u bestValue = row[x];
  int bestIndex = self;
  const int radius = connectivity == 8 ? 1 : 1;
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx == 0 && dy == 0) continue;
      if (connectivity == 4 && dx != 0 && dy != 0) continue;
      const int nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
      const Npp16u v = *(const Npp16u *)((const char *)pSrc + (size_t)ny * nSrcStep + (size_t)nx * sizeof(Npp16u));
      const int idx = linearIndex(nx, ny, width);
      if (v < bestValue || (v == bestValue && idx < bestIndex)) {
        bestValue = v;
        bestIndex = idx;
      }
    }
  }
  pParent[self] = bestIndex;
}

__global__ void watershedJumpKernel(int *pLabel, int count, int *pChanged) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int next = pLabel[pLabel[i]];
  if (next != pLabel[i]) {
    pLabel[i] = next;
    *pChanged = 1;
  }
}

__global__ void watershedOutputKernel(const Npp16u *pSrc, int nSrcStep, int width, int height, int connectivity,
                                      int boundaryType, const int *pLabel, Npp32u *pMarkerLabels,
                                      int nMarkerLabelsStep, Npp16u *pSrcDst, int nSrcDstStep) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int self = linearIndex(x, y, width);
  const int label = pLabel[self];
  bool boundary = false;
  int minNeighborLabel = label;
  const int radius = connectivity == 8 ? 1 : 1;
  for (int dy = -radius; dy <= radius && !boundary; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx == 0 && dy == 0) continue;
      if (connectivity == 4 && dx != 0 && dy != 0) continue;
      const int nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
      const int other = pLabel[linearIndex(nx, ny, width)];
      if (other < minNeighborLabel) minNeighborLabel = other;
      if (other != label) boundary = true;
    }
  }

  const Npp16u value =
      *(const Npp16u *)((const char *)pSrc + (size_t)y * nSrcStep + (size_t)x * sizeof(Npp16u));
  Npp16u out = value;
  switch (boundaryType) {
    case NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE: out = value; break;
    case NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK: out = boundary ? (Npp16u)0 : value; break;
    case NPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE: out = boundary ? (Npp16u)65535 : value; break;
    case NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY: out = boundary ? (Npp16u)0 : (Npp16u)65535; break;
    default:  // CONTRAST: the higher root index side becomes black.
      out = boundary ? (label > minNeighborLabel ? (Npp16u)0 : (Npp16u)65535) : value;
      break;
  }
  *(Npp16u *)((char *)pSrcDst + (size_t)y * nSrcDstStep + (size_t)x * sizeof(Npp16u)) = out;
  if (pMarkerLabels != nullptr) {
    pMarkerLabels[(size_t)y * (nMarkerLabelsStep / sizeof(Npp32u)) + x] = (Npp32u)label;
  }
}

NppStatus watershedCommon(Npp16u *pSrcDst, int nSrcDstStep, Npp32u *pMarkerLabels, int nMarkerLabelsStep,
                          NppiNorm eNorm, NppiWatershedSegmentBoundaryType eSegmentBoundaryType, NppiSize oSizeROI,
                          Npp8u *pDeviceMemoryBuffer, cudaStream_t stream) {
  if (pSrcDst == nullptr || pDeviceMemoryBuffer == nullptr) return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0 || oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_SIZE_ERROR;
  }
  if (eNorm != nppiNormL1 && eNorm != nppiNormInf) return NPP_BAD_ARGUMENT_ERROR;
  if (eSegmentBoundaryType != NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE &&
      eSegmentBoundaryType != NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK &&
      eSegmentBoundaryType != NPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE &&
      eSegmentBoundaryType != NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY &&
      eSegmentBoundaryType != NPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST) {
    return NPP_ERROR;  // measured: generic -2, not NPP_NOT_SUPPORTED_MODE_ERROR
  }

  const int connectivity = eNorm == nppiNormInf ? 8 : 4;
  const size_t count = (size_t)oSizeROI.width * oSizeROI.height;
  int *pParent = nullptr, *pLabel = nullptr, *pChanged = nullptr;
  if (cudaMalloc(&pParent, count * sizeof(int)) != cudaSuccess) return NPP_MEMORY_ALLOCATION_ERR;
  if (cudaMalloc(&pLabel, count * sizeof(int)) != cudaSuccess) {
    cudaFree(pParent);
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  if (cudaMalloc(&pChanged, sizeof(int)) != cudaSuccess) {
    cudaFree(pParent);
    cudaFree(pLabel);
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  dim3 block(16, 16);
  dim3 grid((oSizeROI.width + 15) / 16, (oSizeROI.height + 15) / 16);
  watershedParentKernel<<<grid, block, 0, stream>>>(pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height,
                                                    connectivity, pParent);
  cudaMemcpyAsync(pLabel, pParent, count * sizeof(int), cudaMemcpyDeviceToDevice, stream);

  for (int iteration = 0; iteration < 64; ++iteration) {
    cudaMemsetAsync(pChanged, 0, sizeof(int), stream);
    watershedJumpKernel<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(pLabel, (int)count, pChanged);
    int changed = 0;
    cudaError_t err = cudaMemcpyAsync(&changed, pChanged, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      cudaFree(pParent);
      cudaFree(pLabel);
      cudaFree(pChanged);
      return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    if (changed == 0) break;
  }

  watershedOutputKernel<<<grid, block, 0, stream>>>(pSrcDst, nSrcDstStep, oSizeROI.width, oSizeROI.height,
                                                    connectivity, (int)eSegmentBoundaryType, pLabel, pMarkerLabels,
                                                    nMarkerLabelsStep, pSrcDst, nSrcDstStep);
  cudaError_t err = cudaGetLastError();
  cudaFree(pParent);
  cudaFree(pLabel);
  cudaFree(pChanged);
  return err == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}  // namespace

NppStatus nppiSegmentWatershed_16u_C1IR_Ctx(Npp16u *pSrcDst, Npp32s nSrcDstStep, Npp32u *pMarkerLabels,
                                            Npp32s nMarkerLabelsStep, NppiNorm eNorm,
                                            NppiWatershedSegmentBoundaryType eSegmentBoundaryType, NppiSize oSizeROI,
                                            Npp8u *pDeviceMemoryBuffer, NppStreamContext nppStreamCtx) {
  return watershedCommon(pSrcDst, nSrcDstStep, pMarkerLabels, nMarkerLabelsStep, eNorm, eSegmentBoundaryType,
                         oSizeROI, pDeviceMemoryBuffer, nppStreamCtx.hStream);
}

NppStatus nppiSegmentWatershed_16u_C1IR(Npp16u *pSrcDst, Npp32s nSrcDstStep, Npp32u *pMarkerLabels,
                                        Npp32s nMarkerLabelsStep, NppiNorm eNorm,
                                        NppiWatershedSegmentBoundaryType eSegmentBoundaryType, NppiSize oSizeROI,
                                        Npp8u *pDeviceMemoryBuffer) {
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  return nppiSegmentWatershed_16u_C1IR_Ctx(pSrcDst, nSrcDstStep, pMarkerLabels, nMarkerLabelsStep, eNorm,
                                           eSegmentBoundaryType, oSizeROI, pDeviceMemoryBuffer, ctx);
}
