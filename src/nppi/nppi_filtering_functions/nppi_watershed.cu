#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define WATERSHED_MASK -2
#define WATERSHED_WSHED -1
#define WATERSHED_INIT 0

// Pixel structure for priority queue
struct WatershedPixel {
  int x, y;
  Npp8u intensity;

  __device__ bool operator<(const WatershedPixel &other) const { return intensity < other.intensity; }
};

// Compute gradient magnitude
__global__ void computeGradient_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pGradient, int nGradStep, int width,
                                       int height, Npp8u eNorm) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u *grad_row = (Npp8u *)((char *)pGradient + y * nGradStep);

  float gx = 0.0f, gy = 0.0f;

  // Compute gradient using Sobel operator
  if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
    const Npp8u *prev_row = (const Npp8u *)((const char *)pSrc + (y - 1) * nSrcStep);
    const Npp8u *next_row = (const Npp8u *)((const char *)pSrc + (y + 1) * nSrcStep);

    // X direction gradient
    gx = (float)prev_row[x + 1] + 2.0f * src_row[x + 1] + (float)next_row[x + 1] - (float)prev_row[x - 1] -
         2.0f * src_row[x - 1] - (float)next_row[x - 1];

    // Y direction gradient
    gy = (float)next_row[x - 1] + 2.0f * next_row[x] + (float)next_row[x + 1] - (float)prev_row[x - 1] -
         2.0f * prev_row[x] - (float)prev_row[x + 1];
  }

  float magnitude;
  if (eNorm == 1) {
    magnitude = fabs(gx) + fabs(gy); // L1 norm
  } else {
    magnitude = sqrtf(gx * gx + gy * gy); // L2 norm
  }

  grad_row[x] = (Npp8u)min(magnitude, 255.0f);
}

// Initialize markers

__global__ void initializeMarkers_kernel(Npp32s *pMarkers, int nMarkersStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  Npp32s *marker_row = (Npp32s *)((char *)pMarkers + y * nMarkersStep);

  if (marker_row[x] > 0) {
    // Keep marked seed points unchanged
    return;
  } else {
    // Initialize unmarked points as MASK
    marker_row[x] = WATERSHED_MASK;
  }
}

// Find boundary pixels (neighbors of seed points)
__global__ void findBoundaryPixels_kernel(const Npp32s *pMarkers, int nMarkersStep, const Npp8u *pGradient,
                                          int nGradStep, WatershedPixel *pBoundaryPixels, int *pBoundaryCount,
                                          int width, int height, int maxPixels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32s *marker_row = (const Npp32s *)((const char *)pMarkers + y * nMarkersStep);
  const Npp8u *grad_row = (const Npp8u *)((const char *)pGradient + y * nGradStep);

  // If current pixel is MASK, check if adjacent to marked region
  if (marker_row[x] == WATERSHED_MASK) {
    bool isBoundary = false;

    // Check 4-connected neighborhood
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int i = 0; i < 4; i++) {
      int nx = x + dx[i];
      int ny = y + dy[i];

      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        const Npp32s *neighbor_row = (const Npp32s *)((const char *)pMarkers + ny * nMarkersStep);
        if (neighbor_row[nx] > 0) { // Neighbor is marked seed point
          isBoundary = true;
          break;
        }
      }
    }

    if (isBoundary) {
      int idx = atomicAdd(pBoundaryCount, 1);
      if (idx < maxPixels) {
        pBoundaryPixels[idx].x = x;
        pBoundaryPixels[idx].y = y;
        pBoundaryPixels[idx].intensity = grad_row[x];
      }
    }
  }
}

// Process pixels in queue
__global__ void processWatershedPixels_kernel(const WatershedPixel *pPixels, int pixelCount, Npp32s *pMarkers,
                                              int nMarkersStep, int width, int height, WatershedPixel *pNewPixels,
                                              int *pNewCount, const Npp8u *pGradient, int nGradStep) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= pixelCount)
    return;

  WatershedPixel pixel = pPixels[idx];
  int x = pixel.x;
  int y = pixel.y;

  Npp32s *marker_row = (Npp32s *)((char *)pMarkers + y * nMarkersStep);

  if (marker_row[x] != WATERSHED_MASK)
    return; // Already processed

  // Check neighborhood labels
  int dx[] = {-1, 1, 0, 0};
  int dy[] = {0, 0, -1, 1};

  Npp32s neighborLabel = 0;
  bool hasMultipleLabels = false;

  for (int i = 0; i < 4; i++) {
    int nx = x + dx[i];
    int ny = y + dy[i];

    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
      const Npp32s *neighbor_row = (const Npp32s *)((const char *)pMarkers + ny * nMarkersStep);
      Npp32s label = neighbor_row[nx];

      if (label > 0) { // Valid label
        if (neighborLabel == 0) {
          neighborLabel = label;
        } else if (neighborLabel != label) {
          hasMultipleLabels = true;
          break;
        }
      }
    }
  }

  if (hasMultipleLabels) {
    marker_row[x] = WATERSHED_WSHED; // Watershed line
  } else if (neighborLabel > 0) {
    marker_row[x] = neighborLabel; // Extend label

    // Add MASK pixels in neighborhood to next processing round
    for (int i = 0; i < 4; i++) {
      int nx = x + dx[i];
      int ny = y + dy[i];

      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        const Npp32s *neighbor_row = (const Npp32s *)((const char *)pMarkers + ny * nMarkersStep);
        if (neighbor_row[nx] == WATERSHED_MASK) {
          int newIdx = atomicAdd(pNewCount, 1);
          if (newIdx < width * height) { // Prevent overflow
            const Npp8u *neighbor_grad_row = (const Npp8u *)((const char *)pGradient + ny * nGradStep);
            pNewPixels[newIdx].x = nx;
            pNewPixels[newIdx].y = ny;
            pNewPixels[newIdx].intensity = neighbor_grad_row[nx];
          }
        }
      }
    }
  }
}

extern "C" {

// Get required buffer size

NppStatus nppiSegmentWatershedGetBufferSize_8u_C1R_Ctx_impl(NppiSize oSizeROI, size_t *hpBufferSize) {
  size_t imageSize = (size_t)oSizeROI.width * oSizeROI.height;

  // Required buffers:
  // 1. Gradient image (Npp8u)
  // 2. Pixel queue 1 (WatershedPixel)
  // 3. Pixel queue 2 (WatershedPixel)
  // 4. Counter (int)

  size_t gradientSize = imageSize * sizeof(Npp8u);
  size_t queueSize = imageSize * sizeof(WatershedPixel) * 2;
  size_t counterSize = sizeof(int) * 2;

  size_t totalSize = gradientSize + queueSize + counterSize;
  size_t alignedSize = (totalSize + 511) & ~511; // 512byte alignment

  *hpBufferSize = alignedSize;
  return NPP_SUCCESS;
}

// WatershedSegmentation main function
NppStatus nppiSegmentWatershed_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, Npp32s nSrcDstStep, Npp32u *pMarkerLabels,
                                                Npp32s nMarkerLabelsStep, NppiNorm eNorm, NppiSize oSizeROI,
                                                Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx) {
  int width = oSizeROI.width;
  int height = oSizeROI.height;
  size_t imageSize = width * height;

  // Setup buffers
  Npp8u *pGradient = pDeviceBuffer;
  WatershedPixel *pPixelQueue1 = (WatershedPixel *)(pGradient + imageSize);
  WatershedPixel *pPixelQueue2 = pPixelQueue1 + imageSize;
  int *pQueueCount1 = (int *)(pPixelQueue2 + imageSize);
  int *pQueueCount2 = pQueueCount1 + 1;

  int gradStep = width * sizeof(Npp8u);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  // Step 1: Compute gradient
  computeGradient_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pSrcDst, nSrcDstStep, pGradient, gradStep,
                                                                           width, height, (Npp8u)eNorm);

  // Step 2: Initialize markers
  initializeMarkers_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>((Npp32s *)pMarkerLabels, nMarkerLabelsStep,
                                                                             width, height);

  // Step 3: Find initial boundary pixels
  cudaMemsetAsync(pQueueCount1, 0, sizeof(int), nppStreamCtx.hStream);

  findBoundaryPixels_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      (Npp32s *)pMarkerLabels, nMarkerLabelsStep, pGradient, gradStep, pPixelQueue1, pQueueCount1, width, height,
      (int)imageSize);

  // Step 4: Iterative processing (simplified priority queue)
  WatershedPixel *currentQueue = pPixelQueue1;
  WatershedPixel *nextQueue = pPixelQueue2;
  int *currentCount = pQueueCount1;
  int *nextCount = pQueueCount2;

  for (int iteration = 0; iteration < 255; iteration++) { // Maximum 255 iterations
    int h_count;
    cudaMemcpyAsync(&h_count, currentCount, sizeof(int), cudaMemcpyDeviceToHost, nppStreamCtx.hStream);
    cudaStreamSynchronize(nppStreamCtx.hStream);

    if (h_count == 0)
      break; // No more pixels to process

    // Clear next queue
    cudaMemsetAsync(nextCount, 0, sizeof(int), nppStreamCtx.hStream);

    // Process all pixels at current intensity level
    dim3 linearBlockSize(256);
    dim3 linearGridSize((h_count + linearBlockSize.x - 1) / linearBlockSize.x);

    processWatershedPixels_kernel<<<linearGridSize, linearBlockSize, 0, nppStreamCtx.hStream>>>(
        currentQueue, h_count, (Npp32s *)pMarkerLabels, nMarkerLabelsStep, width, height, nextQueue, nextCount,
        pGradient, gradStep);

    // Swap queues
    WatershedPixel *tempQueue = currentQueue;
    currentQueue = nextQueue;
    nextQueue = tempQueue;

    int *tempCount = currentCount;
    currentCount = nextCount;
    nextCount = tempCount;
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}
