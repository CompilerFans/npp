#include "npp.h"
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// PBA (Parallel Banding Algorithm) for Euclidean Distance Transform
// Reference: Cao et al. "Parallel Banding Algorithm to compute exact distance transform with the GPU"

#define BLOCK_SIZE 16
#define INF_SITE (-1)

// Check if a pixel is a site (foreground)
template <typename T> __device__ __forceinline__ bool isSite(T value, T minSite, T maxSite) {
  return value >= minSite && value <= maxSite;
}

// Squared Euclidean distance
__device__ __forceinline__ float squaredDistanceF(int x1, int y1, int x2, int y2) {
  float dx = (float)(x1 - x2);
  float dy = (float)(y1 - y2);
  return dx * dx + dy * dy;
}

__device__ __forceinline__ double squaredDistanceD(int x1, int y1, int x2, int y2) {
  double dx = (double)(x1 - x2);
  double dy = (double)(y1 - y2);
  return dx * dx + dy * dy;
}

// Phase 1: Initialize sites - mark foreground pixels
// Store site coordinates as (x, y) encoded in int2
template <typename SrcType>
__global__ void initSites_kernel(const SrcType *pSrc, int nSrcStep, int width, int height, SrcType nMinSiteValue,
                                 SrcType nMaxSiteValue, int *pSitesX, int *pSitesY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const SrcType *srcRow = (const SrcType *)((const char *)pSrc + y * nSrcStep);
  SrcType value = srcRow[x];

  int idx = y * width + x;
  if (isSite(value, nMinSiteValue, nMaxSiteValue)) {
    pSitesX[idx] = x;
    pSitesY[idx] = y;
  } else {
    pSitesX[idx] = INF_SITE;
    pSitesY[idx] = INF_SITE;
  }
}

// Phase 2: Column propagation (vertical flood fill)
// Each thread handles one column
__global__ void columnFlood_kernel(int *pSitesX, int *pSitesY, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= width)
    return;

  // Forward pass: propagate from top to bottom
  int lastX = INF_SITE;
  int lastY = INF_SITE;

  for (int y = 0; y < height; y++) {
    int idx = y * width + x;
    if (pSitesX[idx] != INF_SITE) {
      lastX = pSitesX[idx];
      lastY = pSitesY[idx];
    } else if (lastX != INF_SITE) {
      pSitesX[idx] = lastX;
      pSitesY[idx] = lastY;
    }
  }

  // Backward pass: propagate from bottom to top, keeping closer site
  lastX = INF_SITE;
  lastY = INF_SITE;

  for (int y = height - 1; y >= 0; y--) {
    int idx = y * width + x;
    int curX = pSitesX[idx];
    int curY = pSitesY[idx];

    if (curX != INF_SITE && curY == y && curX == x) {
      // Original site
      lastX = curX;
      lastY = curY;
    } else if (lastX != INF_SITE) {
      if (curX == INF_SITE) {
        pSitesX[idx] = lastX;
        pSitesY[idx] = lastY;
      } else {
        // Compare distances and keep closer
        int distCur = abs(y - curY);
        int distLast = abs(y - lastY);
        if (distLast < distCur) {
          pSitesX[idx] = lastX;
          pSitesY[idx] = lastY;
        } else {
          lastX = curX;
          lastY = curY;
        }
      }
    }
  }
}

// Phase 3: Row propagation using PBA
// Each thread handles one row
__global__ void rowPropagate_kernel(int *pSitesX, int *pSitesY, int width, int height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= height)
    return;

  int *rowX = pSitesX + y * width;
  int *rowY = pSitesY + y * width;

  // Multiple passes to ensure convergence
  for (int pass = 0; pass < 2; pass++) {
    // Forward pass
    for (int x = 1; x < width; x++) {
      int prevSiteX = rowX[x - 1];
      int prevSiteY = rowY[x - 1];
      int curSiteX = rowX[x];
      int curSiteY = rowY[x];

      if (prevSiteX != INF_SITE) {
        if (curSiteX == INF_SITE) {
          rowX[x] = prevSiteX;
          rowY[x] = prevSiteY;
        } else {
          float distPrev = squaredDistanceF(x, y, prevSiteX, prevSiteY);
          float distCur = squaredDistanceF(x, y, curSiteX, curSiteY);
          if (distPrev < distCur) {
            rowX[x] = prevSiteX;
            rowY[x] = prevSiteY;
          }
        }
      }
    }

    // Backward pass
    for (int x = width - 2; x >= 0; x--) {
      int nextSiteX = rowX[x + 1];
      int nextSiteY = rowY[x + 1];
      int curSiteX = rowX[x];
      int curSiteY = rowY[x];

      if (nextSiteX != INF_SITE) {
        if (curSiteX == INF_SITE) {
          rowX[x] = nextSiteX;
          rowY[x] = nextSiteY;
        } else {
          float distNext = squaredDistanceF(x, y, nextSiteX, nextSiteY);
          float distCur = squaredDistanceF(x, y, curSiteX, curSiteY);
          if (distNext < distCur) {
            rowX[x] = nextSiteX;
            rowY[x] = nextSiteY;
          }
        }
      }
    }
  }
}

// Compute final distance transform output (16u truncated)
__global__ void computeDistanceTransform16u_kernel(const int *pSitesX, const int *pSitesY, int width, int height,
                                                   Npp16u *pDstTransform, int nDstTransformStep) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  int siteX = pSitesX[idx];
  int siteY = pSitesY[idx];

  Npp16u *dstRow = (Npp16u *)((char *)pDstTransform + y * nDstTransformStep);

  if (siteX == INF_SITE) {
    dstRow[x] = 65535;
  } else {
    float dist = sqrtf(squaredDistanceF(x, y, siteX, siteY));
    dstRow[x] = (Npp16u)fminf(dist, 65535.0f);
  }
}

// Compute final distance transform output (32f)
__global__ void computeDistanceTransform32f_kernel(const int *pSitesX, const int *pSitesY, int width, int height,
                                                   Npp32f *pDstTransform, int nDstTransformStep) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  int siteX = pSitesX[idx];
  int siteY = pSitesY[idx];

  Npp32f *dstRow = (Npp32f *)((char *)pDstTransform + y * nDstTransformStep);

  if (siteX == INF_SITE) {
    dstRow[x] = INFINITY;
  } else {
    dstRow[x] = sqrtf(squaredDistanceF(x, y, siteX, siteY));
  }
}

// Compute final distance transform output (64f)
__global__ void computeDistanceTransform64f_kernel(const int *pSitesX, const int *pSitesY, int width, int height,
                                                   Npp64f *pDstTransform, int nDstTransformStep) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  int siteX = pSitesX[idx];
  int siteY = pSitesY[idx];

  Npp64f *dstRow = (Npp64f *)((char *)pDstTransform + y * nDstTransformStep);

  if (siteX == INF_SITE) {
    dstRow[x] = INFINITY;
  } else {
    dstRow[x] = sqrt(squaredDistanceD(x, y, siteX, siteY));
  }
}

// Compute Voronoi diagram output
__global__ void computeVoronoi_kernel(const int *pSitesX, const int *pSitesY, int width, int height,
                                      Npp16s *pDstVoronoi, int nDstVoronoiStep) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  int siteX = pSitesX[idx];
  int siteY = pSitesY[idx];

  Npp16s *dstRow = (Npp16s *)((char *)pDstVoronoi + y * nDstVoronoiStep);

  if (siteX == INF_SITE) {
    dstRow[x * 2] = -1;
    dstRow[x * 2 + 1] = -1;
  } else {
    dstRow[x * 2] = (Npp16s)siteX;
    dstRow[x * 2 + 1] = (Npp16s)siteY;
  }
}

// Compute Manhattan distances (absolute)
__global__ void computeAbsoluteManhattanDistances_kernel(const int *pSitesX, const int *pSitesY, int width, int height,
                                                         Npp16u *pDstManhattan, int nDstManhattanStep) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  int siteX = pSitesX[idx];
  int siteY = pSitesY[idx];

  Npp16u *dstRow = (Npp16u *)((char *)pDstManhattan + y * nDstManhattanStep);

  if (siteX == INF_SITE) {
    dstRow[x * 2] = 65535;
    dstRow[x * 2 + 1] = 65535;
  } else {
    dstRow[x * 2] = (Npp16u)abs(x - siteX);
    dstRow[x * 2 + 1] = (Npp16u)abs(y - siteY);
  }
}

// Compute Manhattan distances (relative/signed)
__global__ void computeRelativeManhattanDistances_kernel(const int *pSitesX, const int *pSitesY, int width, int height,
                                                         Npp16s *pDstManhattan, int nDstManhattanStep) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  int siteX = pSitesX[idx];
  int siteY = pSitesY[idx];

  Npp16s *dstRow = (Npp16s *)((char *)pDstManhattan + y * nDstManhattanStep);

  if (siteX == INF_SITE) {
    dstRow[x * 2] = 32767;
    dstRow[x * 2 + 1] = 32767;
  } else {
    dstRow[x * 2] = (Npp16s)(x - siteX);
    dstRow[x * 2 + 1] = (Npp16s)(y - siteY);
  }
}

// Signed distance transform: Initialize boundary sites
// For signed DT, sites are boundary pixels (transition between inside/outside)
template <typename T>
__global__ void initSignedBoundarySites_kernel(const T *pSrc, int nSrcStep, int width, int height, T nCutoffValue,
                                               int *pSitesExtX, int *pSitesExtY, int *pSitesIntX, int *pSitesIntY,
                                               bool *pIsInside) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const T *srcRow = (const T *)((const char *)pSrc + y * nSrcStep);
  T value = srcRow[x];

  int idx = y * width + x;

  // Determine if current pixel is inside (value > cutoff) or outside
  bool inside = value > nCutoffValue;
  pIsInside[idx] = inside;

  // Initialize all sites as invalid
  pSitesExtX[idx] = INF_SITE;
  pSitesExtY[idx] = INF_SITE;
  pSitesIntX[idx] = INF_SITE;
  pSitesIntY[idx] = INF_SITE;
}

// Mark boundary pixels as sites
template <typename T>
__global__ void markBoundarySites_kernel(const T *pSrc, int nSrcStep, int width, int height, T nCutoffValue,
                                         const bool *pIsInside, int *pSitesExtX, int *pSitesExtY, int *pSitesIntX,
                                         int *pSitesIntY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  bool inside = pIsInside[idx];

  // Check if this is a boundary pixel (has neighbor with different state)
  bool isBoundary = false;

  // Check 4-connected neighbors
  if (x > 0) {
    int nidx = y * width + (x - 1);
    if (pIsInside[nidx] != inside)
      isBoundary = true;
  }
  if (x < width - 1) {
    int nidx = y * width + (x + 1);
    if (pIsInside[nidx] != inside)
      isBoundary = true;
  }
  if (y > 0) {
    int nidx = (y - 1) * width + x;
    if (pIsInside[nidx] != inside)
      isBoundary = true;
  }
  if (y < height - 1) {
    int nidx = (y + 1) * width + x;
    if (pIsInside[nidx] != inside)
      isBoundary = true;
  }

  if (isBoundary) {
    if (inside) {
      // Interior boundary - site for exterior distance
      pSitesExtX[idx] = x;
      pSitesExtY[idx] = y;
    } else {
      // Exterior boundary - site for interior distance
      pSitesIntX[idx] = x;
      pSitesIntY[idx] = y;
    }
  }
}

// Merge exterior and interior distances for signed distance transform
__global__ void mergeSignedDistances_kernel(const int *pSitesExtX, const int *pSitesExtY, const int *pSitesIntX,
                                            const int *pSitesIntY, const bool *pIsInside, int width, int height,
                                            Npp64f nSubPixelXShift, Npp64f nSubPixelYShift, Npp64f *pDstTransform,
                                            int nDstTransformStep) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int idx = y * width + x;

  int siteExtX = pSitesExtX[idx];
  int siteExtY = pSitesExtY[idx];
  int siteIntX = pSitesIntX[idx];
  int siteIntY = pSitesIntY[idx];
  bool inside = pIsInside[idx];

  Npp64f *dstRow = (Npp64f *)((char *)pDstTransform + y * nDstTransformStep);

  double distExt = (siteExtX != INF_SITE) ? sqrt(squaredDistanceD(x, y, siteExtX, siteExtY)) : INFINITY;
  double distInt = (siteIntX != INF_SITE) ? sqrt(squaredDistanceD(x, y, siteIntX, siteIntY)) : INFINITY;

  // Signed distance: positive inside, negative outside
  if (inside) {
    // Inside region: distance to exterior boundary (positive)
    dstRow[x] = distExt;
  } else {
    // Outside region: distance to interior boundary (negative)
    dstRow[x] = -distInt;
  }
}

extern "C" {

// Buffer size calculation
NppStatus nppiDistanceTransformPBAGetBufferSize_impl(NppiSize oSizeROI, size_t *hpBufferSize) {
  size_t numPixels = (size_t)oSizeROI.width * oSizeROI.height;
  // Two arrays for X and Y coordinates
  size_t sitesSize = numPixels * sizeof(int) * 2;
  // Align to 256 bytes
  *hpBufferSize = (sitesSize + 255) & ~255;
  return NPP_SUCCESS;
}

NppStatus nppiDistanceTransformPBAGetAntialiasingBufferSize_impl(NppiSize oSizeROI, size_t *hpAntialiasingBufferSize) {
  size_t numPixels = (size_t)oSizeROI.width * oSizeROI.height;
  size_t tempSize = numPixels * sizeof(float);
  *hpAntialiasingBufferSize = (tempSize + 255) & ~255;
  return NPP_SUCCESS;
}

NppStatus nppiSignedDistanceTransformPBAGetBufferSize_impl(NppiSize oSizeROI, size_t *hpBufferSize) {
  size_t numPixels = (size_t)oSizeROI.width * oSizeROI.height;
  // 4 site arrays (ext X, ext Y, int X, int Y) + 1 bool array for inside/outside
  size_t sitesSize = numPixels * sizeof(int) * 4;
  size_t boolSize = numPixels * sizeof(bool);
  *hpBufferSize = ((sitesSize + boolSize) + 255) & ~255;
  return NPP_SUCCESS;
}

NppStatus nppiSignedDistanceTransformPBAGet64fBufferSize_impl(NppiSize oSizeROI, size_t *hpBufferSize) {
  return nppiSignedDistanceTransformPBAGetBufferSize_impl(oSizeROI, hpBufferSize);
}

// Helper function to run PBA distance transform
static void runPBA(int *pSitesX, int *pSitesY, int width, int height, cudaStream_t stream) {
  dim3 colBlockSize(256);
  dim3 colGridSize((width + colBlockSize.x - 1) / colBlockSize.x);
  columnFlood_kernel<<<colGridSize, colBlockSize, 0, stream>>>(pSitesX, pSitesY, width, height);

  dim3 rowBlockSize(256);
  dim3 rowGridSize((height + rowBlockSize.x - 1) / rowBlockSize.x);
  rowPropagate_kernel<<<rowGridSize, rowBlockSize, 0, stream>>>(pSitesX, pSitesY, width, height);
}

// Distance transform implementations
NppStatus nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx_impl(
    Npp8u *pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue, Npp16s *pDstVoronoi, int nDstVoronoiStep,
    Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u *pDstVoronoiAbsoluteManhattanDistances,
    int nDstVoronoiAbsoluteManhattanDistancesStep, Npp16u *pDstTransform, int nDstTransformStep, NppiSize oSizeROI,
    Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx) {

  int width = oSizeROI.width;
  int height = oSizeROI.height;
  cudaStream_t stream = nppStreamCtx.hStream;
  size_t numPixels = (size_t)width * height;

  int *pSitesX = (int *)pDeviceBuffer;
  int *pSitesY = pSitesX + numPixels;

  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  // Phase 1: Initialize sites
  initSites_kernel<Npp8u><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, width, height, nMinSiteValue,
                                                              nMaxSiteValue, pSitesX, pSitesY);

  // Phase 2 & 3: PBA propagation
  runPBA(pSitesX, pSitesY, width, height, stream);

  // Output Voronoi diagram if requested
  if (pDstVoronoi != nullptr) {
    computeVoronoi_kernel<<<gridSize, blockSize, 0, stream>>>(pSitesX, pSitesY, width, height, pDstVoronoi,
                                                              nDstVoronoiStep);
  }

  // Output absolute Manhattan distances if requested
  if (pDstVoronoiAbsoluteManhattanDistances != nullptr) {
    computeAbsoluteManhattanDistances_kernel<<<gridSize, blockSize, 0, stream>>>(
        pSitesX, pSitesY, width, height, pDstVoronoiAbsoluteManhattanDistances,
        nDstVoronoiAbsoluteManhattanDistancesStep);
  }

  // Output distance transform if requested
  if (pDstTransform != nullptr) {
    computeDistanceTransform16u_kernel<<<gridSize, blockSize, 0, stream>>>(pSitesX, pSitesY, width, height,
                                                                           pDstTransform, nDstTransformStep);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiDistanceTransformPBA_8u32f_C1R_Ctx_impl(Npp8u *pSrc, int nSrcStep, Npp8u nMinSiteValue,
                                                      Npp8u nMaxSiteValue, Npp16s *pDstVoronoi, int nDstVoronoiStep,
                                                      Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
                                                      Npp16s *pDstVoronoiRelativeManhattanDistances,
                                                      int nDstVoronoiRelativeManhattanDistancesStep,
                                                      Npp32f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI,
                                                      Npp8u *pDeviceBuffer, NppStreamContext nppStreamCtx) {

  int width = oSizeROI.width;
  int height = oSizeROI.height;
  cudaStream_t stream = nppStreamCtx.hStream;
  size_t numPixels = (size_t)width * height;

  int *pSitesX = (int *)pDeviceBuffer;
  int *pSitesY = pSitesX + numPixels;

  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  // Phase 1: Initialize sites
  initSites_kernel<Npp8u><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, width, height, nMinSiteValue,
                                                              nMaxSiteValue, pSitesX, pSitesY);

  // Phase 2 & 3: PBA propagation
  runPBA(pSitesX, pSitesY, width, height, stream);

  // Output Voronoi diagram if requested
  if (pDstVoronoi != nullptr) {
    computeVoronoi_kernel<<<gridSize, blockSize, 0, stream>>>(pSitesX, pSitesY, width, height, pDstVoronoi,
                                                              nDstVoronoiStep);
  }

  // Output relative Manhattan distances if requested
  if (pDstVoronoiRelativeManhattanDistances != nullptr) {
    computeRelativeManhattanDistances_kernel<<<gridSize, blockSize, 0, stream>>>(
        pSitesX, pSitesY, width, height, pDstVoronoiRelativeManhattanDistances,
        nDstVoronoiRelativeManhattanDistancesStep);
  }

  // Output distance transform if requested
  if (pDstTransform != nullptr) {
    computeDistanceTransform32f_kernel<<<gridSize, blockSize, 0, stream>>>(pSitesX, pSitesY, width, height,
                                                                           pDstTransform, nDstTransformStep);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiDistanceTransformPBA_8u64f_C1R_Ctx_impl(
    Npp8u *pSrc, int nSrcStep, Npp8u nMinSiteValue, Npp8u nMaxSiteValue, Npp16s *pDstVoronoi, int nDstVoronoiStep,
    Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16s *pDstVoronoiRelativeManhattanDistances,
    int nDstVoronoiRelativeManhattanDistancesStep, Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI,
    Npp8u *pDeviceBuffer, Npp8u *pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {

  int width = oSizeROI.width;
  int height = oSizeROI.height;
  cudaStream_t stream = nppStreamCtx.hStream;
  size_t numPixels = (size_t)width * height;

  int *pSitesX = (int *)pDeviceBuffer;
  int *pSitesY = pSitesX + numPixels;

  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  // Phase 1: Initialize sites
  initSites_kernel<Npp8u><<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, width, height, nMinSiteValue,
                                                              nMaxSiteValue, pSitesX, pSitesY);

  // Phase 2 & 3: PBA propagation
  runPBA(pSitesX, pSitesY, width, height, stream);

  // Output Voronoi diagram if requested
  if (pDstVoronoi != nullptr) {
    computeVoronoi_kernel<<<gridSize, blockSize, 0, stream>>>(pSitesX, pSitesY, width, height, pDstVoronoi,
                                                              nDstVoronoiStep);
  }

  // Output relative Manhattan distances if requested
  if (pDstVoronoiRelativeManhattanDistances != nullptr) {
    computeRelativeManhattanDistances_kernel<<<gridSize, blockSize, 0, stream>>>(
        pSitesX, pSitesY, width, height, pDstVoronoiRelativeManhattanDistances,
        nDstVoronoiRelativeManhattanDistancesStep);
  }

  // Output distance transform if requested
  if (pDstTransform != nullptr) {
    computeDistanceTransform64f_kernel<<<gridSize, blockSize, 0, stream>>>(pSitesX, pSitesY, width, height,
                                                                           pDstTransform, nDstTransformStep);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiSignedDistanceTransformPBA_32f64f_C1R_Ctx_impl(
    Npp32f *pSrc, int nSrcStep, Npp32f nCutoffValue, Npp64f nSubPixelXShift, Npp64f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f *pDstTransform,
    int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u *pAntialiasingDeviceBuffer,
    NppStreamContext nppStreamCtx) {

  int width = oSizeROI.width;
  int height = oSizeROI.height;
  cudaStream_t stream = nppStreamCtx.hStream;
  size_t numPixels = (size_t)width * height;

  // Buffer layout: ExtX, ExtY, IntX, IntY, IsInside
  int *pSitesExtX = (int *)pDeviceBuffer;
  int *pSitesExtY = pSitesExtX + numPixels;
  int *pSitesIntX = pSitesExtY + numPixels;
  int *pSitesIntY = pSitesIntX + numPixels;
  bool *pIsInside = (bool *)(pSitesIntY + numPixels);

  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  // Step 1: Initialize and determine inside/outside
  initSignedBoundarySites_kernel<Npp32f><<<gridSize, blockSize, 0, stream>>>(
      pSrc, nSrcStep, width, height, nCutoffValue, pSitesExtX, pSitesExtY, pSitesIntX, pSitesIntY, pIsInside);

  // Step 2: Mark boundary pixels as sites
  markBoundarySites_kernel<Npp32f><<<gridSize, blockSize, 0, stream>>>(
      pSrc, nSrcStep, width, height, nCutoffValue, pIsInside, pSitesExtX, pSitesExtY, pSitesIntX, pSitesIntY);

  // Step 3: Run PBA for exterior sites (boundary of inside region)
  runPBA(pSitesExtX, pSitesExtY, width, height, stream);

  // Step 4: Run PBA for interior sites (boundary of outside region)
  runPBA(pSitesIntX, pSitesIntY, width, height, stream);

  // Output Voronoi diagram if requested (use exterior sites)
  if (pDstVoronoi != nullptr) {
    computeVoronoi_kernel<<<gridSize, blockSize, 0, stream>>>(pSitesExtX, pSitesExtY, width, height, pDstVoronoi,
                                                              nDstVoronoiStep);
  }

  // Output relative Manhattan distances if requested
  if (pDstVoronoiRelativeManhattanDistances != nullptr) {
    computeRelativeManhattanDistances_kernel<<<gridSize, blockSize, 0, stream>>>(
        pSitesExtX, pSitesExtY, width, height, pDstVoronoiRelativeManhattanDistances,
        nDstVoronoiRelativeManhattanDistancesStep);
  }

  // Merge distances for signed output
  if (pDstTransform != nullptr) {
    mergeSignedDistances_kernel<<<gridSize, blockSize, 0, stream>>>(pSitesExtX, pSitesExtY, pSitesIntX, pSitesIntY,
                                                                    pIsInside, width, height, nSubPixelXShift,
                                                                    nSubPixelYShift, pDstTransform, nDstTransformStep);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiSignedDistanceTransformPBA_64f_C1R_Ctx_impl(
    Npp64f *pSrc, int nSrcStep, Npp64f nCutoffValue, Npp64f nSubPixelXShift, Npp64f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep, Npp64f *pDstTransform,
    int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer, Npp8u *pAntialiasingDeviceBuffer,
    NppStreamContext nppStreamCtx) {

  int width = oSizeROI.width;
  int height = oSizeROI.height;
  cudaStream_t stream = nppStreamCtx.hStream;
  size_t numPixels = (size_t)width * height;

  // Buffer layout: ExtX, ExtY, IntX, IntY, IsInside
  int *pSitesExtX = (int *)pDeviceBuffer;
  int *pSitesExtY = pSitesExtX + numPixels;
  int *pSitesIntX = pSitesExtY + numPixels;
  int *pSitesIntY = pSitesIntX + numPixels;
  bool *pIsInside = (bool *)(pSitesIntY + numPixels);

  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  // Step 1: Initialize and determine inside/outside
  initSignedBoundarySites_kernel<Npp64f><<<gridSize, blockSize, 0, stream>>>(
      pSrc, nSrcStep, width, height, nCutoffValue, pSitesExtX, pSitesExtY, pSitesIntX, pSitesIntY, pIsInside);

  // Step 2: Mark boundary pixels as sites
  markBoundarySites_kernel<Npp64f><<<gridSize, blockSize, 0, stream>>>(
      pSrc, nSrcStep, width, height, nCutoffValue, pIsInside, pSitesExtX, pSitesExtY, pSitesIntX, pSitesIntY);

  // Step 3: Run PBA for exterior sites
  runPBA(pSitesExtX, pSitesExtY, width, height, stream);

  // Step 4: Run PBA for interior sites
  runPBA(pSitesIntX, pSitesIntY, width, height, stream);

  // Output Voronoi diagram if requested
  if (pDstVoronoi != nullptr) {
    computeVoronoi_kernel<<<gridSize, blockSize, 0, stream>>>(pSitesExtX, pSitesExtY, width, height, pDstVoronoi,
                                                              nDstVoronoiStep);
  }

  // Output relative Manhattan distances if requested
  if (pDstVoronoiRelativeManhattanDistances != nullptr) {
    computeRelativeManhattanDistances_kernel<<<gridSize, blockSize, 0, stream>>>(
        pSitesExtX, pSitesExtY, width, height, pDstVoronoiRelativeManhattanDistances,
        nDstVoronoiRelativeManhattanDistancesStep);
  }

  // Merge distances for signed output
  if (pDstTransform != nullptr) {
    mergeSignedDistances_kernel<<<gridSize, blockSize, 0, stream>>>(pSitesExtX, pSitesExtY, pSitesIntX, pSitesIntY,
                                                                    pIsInside, width, height, nSubPixelXShift,
                                                                    nSubPixelYShift, pDstTransform, nDstTransformStep);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"
