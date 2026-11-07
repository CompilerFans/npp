// Shared kernels for the signed PBA distance transforms
// (nppiSignedDistanceTransformPBA_* and the Abs variants).
//
// The reference semantics were decoded from NPP 13.4 in the reference implementation and are
// documented in test/unit/num/num_nppi_signed_distance_transform.h:
//   * inside pixels are those with src >= 1; outside sites have src <= 0 or
//     src < nCutoffValue;
//   * the folded value for a displacement (dx,dy) and C = 30 + cutoff is
//     sqrt((max(a,b) - C)^2 + min(a,b)^2) with a = |dx| + sx, b = |dy| + sy;
//   * the signed flavour stores +folded inside and -folded outside, using the
//     nearest opposite-class pixel; the Abs flavour folds the distance to the
//     nearest inside pixel;
//   * the 32f64f entry points switch to an antialiasing path when the AA
//     buffer is supplied: the signed one outputs the positive Euclidean
//     distance to the nearest opposite-class pixel, the Abs one outputs
//     sqrt((d+sx)^2 + (d+sy)^2) with d the horizontal distance to the inside
//     column span (both measured);
//   * the 64f entry points ignore the AA buffer and always use the folded
//     signed transform (measured: the Abs 64f transform equals the signed one).
#pragma once

#include <cmath>

namespace nppi_sdt {

enum Mode {
  kSignedFolded = 0,    // signed folded transform (inside +t, outside -t)
  kAbsFolded = 1,       // folded magnitude to the nearest inside pixel
  kSignedAbsRel = 2,    // signed folded, absolute relative Manhattan outputs
  kEuclidOpposite = 3,  // positive Euclidean distance to the nearest opposite-class pixel
  kAaAbs = 4,           // antialiased Abs 32f64f: sqrt((d+sx)^2 + (d+sy)^2)
};

constexpr int kMinRoi = 64;

template <typename T> __device__ __forceinline__ bool insideClass(T v) { return v >= (T)1; }

template <typename T> __device__ __forceinline__ bool outsideSite(T v, T cutoff) {
  return v <= (T)0 || v < cutoff;
}

// counters[0] = outside count, [1] = inside count, [2] = min inside x,
// [3] = max inside x.
static __global__ void initCountersKernel(int *counters) {
  counters[0] = 0;
  counters[1] = 0;
  counters[2] = 0x7fffffff;
  counters[3] = -1;
}

// Build compact site lists in the caller-provided scratch buffer.
template <typename T>
__global__ void classifySitesKernel(const T *pSrc, int nSrcStep, int width, int height, T cutoff, int *outX,
                                    int *outY, int *inX, int *inY, int *counters) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const T v = *(const T *)((const char *)pSrc + (size_t)y * nSrcStep + (size_t)x * sizeof(T));
  if (insideClass(v)) {
    const int i = atomicAdd(&counters[1], 1);
    inX[i] = x;
    inY[i] = y;
    atomicMin(&counters[2], x);
    atomicMax(&counters[3], x);
  } else if (outsideSite(v, cutoff)) {
    const int i = atomicAdd(&counters[0], 1);
    outX[i] = x;
    outY[i] = y;
  }
}

// Reference tie-break measured on the unsigned PBA: smaller x, then larger y.
__device__ __forceinline__ bool closerSite(long long d2, int x, int y, long long bestD2, int bestX, int bestY) {
  if (d2 != bestD2) return d2 < bestD2;
  if (x != bestX) return x < bestX;
  return y > bestY;
}

template <typename CalcT>
__device__ __forceinline__ CalcT foldedDistance(int x, int y, int sx, int sy, CalcT subX, CalcT subY, CalcT C) {
  const CalcT a = (CalcT)abs(x - sx) + subX;
  const CalcT b = (CalcT)abs(y - sy) + subY;
  const CalcT major = a >= b ? a: b;
  const CalcT minor = a >= b ? b: a;
  const CalcT d = major - C;
  return sqrt(d * d + minor * minor);
}

__device__ __forceinline__ void writeVoronoi(Npp16s *pDst, int step, int x, int y, int sx, int sy) {
  if (pDst) {
    Npp16s *row = (Npp16s *)((char *)pDst + (size_t)y * step);
    row[2 * x] = (Npp16s)sx;
    row[2 * x + 1] = (Npp16s)sy;
  }
}

__device__ __forceinline__ void writeRelative(Npp16u *pDst, int step, int x, int y, int rx, int ry) {
  if (pDst) {
    Npp16u *row = (Npp16u *)((char *)pDst + (size_t)y * step);
    row[2 * x] = (Npp16u)rx;
    row[2 * x + 1] = (Npp16u)ry;
  }
}

template <typename T, typename OutT, bool Abs, bool AbsRel>
__global__ void signedDistanceKernel(const T *pSrc, int nSrcStep, int width, int height, T cutoff,
                                     OutT subX, OutT subY, const int *outXc, const int *outYc,
                                     const int *inXc, const int *inYc, const int *counters,
                                     Npp16s *pDstVoronoi, int nDstVoronoiStep,
                                     Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
                                     Npp16u *pDstRelative, int nDstRelativeStep, OutT *pDstTransform,
                                     int nDstTransformStep) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int insideCount = counters[1];
  const int outsideCount = counters[0];
  const OutT C = (OutT)30 + (OutT)cutoff;

  if ((!Abs && insideCount == 0) || (Abs && outsideCount == 0)) {
    // Signed reference zeroes every requested output when there is no inside;
    // the abs variant also zeroes when there is no outside.
    writeVoronoi(pDstVoronoi, nDstVoronoiStep, x, y, 0, 0);
    writeVoronoi(pDstVoronoiIndices, nDstVoronoiIndicesStep, x, y, 0, 0);
    writeRelative(pDstRelative, nDstRelativeStep, x, y, 0, 0);
    if (pDstTransform) {
      *(OutT *)((char *)pDstTransform + (size_t)y * nDstTransformStep + (size_t)x * sizeof(OutT)) = (OutT)0;
    }
    return;
  }

  // Nearest site of each class (brute force; exact and order independent).
  int outX = -1, outY = 0;
  long long outD2 = -1;
  if (outsideCount > 0) {
    for (int i = 0; i < outsideCount; ++i) {
      const int sx = outXc[i], sy = outYc[i];
      const long long dx = (long long)x - sx, dy = (long long)y - sy;
      const long long d2 = dx * dx + dy * dy;
      if (outD2 < 0 || closerSite(d2, sx, sy, outD2, outX, outY)) {
        outD2 = d2;
        outX = sx;
        outY = sy;
      }
    }
  }

  int inX = 0, inY = 0;
  long long inD2 = -1;
  for (int i = 0; i < insideCount; ++i) {
    const int sx = inXc[i], sy = inYc[i];
    const long long dx = (long long)x - sx, dy = (long long)y - sy;
    const long long d2 = dx * dx + dy * dy;
    if (inD2 < 0 || closerSite(d2, sx, sy, inD2, inX, inY)) {
      inD2 = d2;
      inX = sx;
      inY = sy;
    }
  }

  const T v = *(const T *)((const char *)pSrc + (size_t)y * nSrcStep + (size_t)x * sizeof(T));
  const bool inside = insideClass(v);

  int siteX, siteY, vorX, vorY;
  OutT value;
  int relX, relY;
  if (Abs) {
    siteX = inX;
    siteY = inY;
    vorX = inX;
    vorY = inY;
    value = foldedDistance<OutT>(x, y, siteX, siteY, subX, subY, C);
    // The third abs output is the *absolute* Manhattan distance.
    relX = abs(siteX - x);
    relY = abs(siteY - y);
  } else {
    vorX = outX;
    vorY = outY;
    if (inside) {
      siteX = outX;
      siteY = outY;
      value = foldedDistance<OutT>(x, y, siteX, siteY, subX, subY, C);
    } else {
      siteX = inX;
      siteY = inY;
      value = -foldedDistance<OutT>(x, y, siteX, siteY, subX, subY, C);
    }
    relX = x - siteX;
    relY = y - siteY;
    if (AbsRel) {
      relX = abs(relX);
      relY = abs(relY);
    }
  }

  writeVoronoi(pDstVoronoi, nDstVoronoiStep, x, y, vorX, vorY);
  writeVoronoi(pDstVoronoiIndices, nDstVoronoiIndicesStep, x, y, siteX, siteY);
  writeRelative(pDstRelative, nDstRelativeStep, x, y, relX, relY);
  if (pDstTransform) {
    *(OutT *)((char *)pDstTransform + (size_t)y * nDstTransformStep + (size_t)x * sizeof(OutT)) = value;
  }
}

// Positive Euclidean distance to the nearest opposite-class pixel, with the
// sub-pixel shifts applied per axis (NPP's antialiased 32f64f signed path).
template <typename T, typename OutT>
__global__ void euclidOppositeKernel(const T *pSrc, int nSrcStep, int width, int height, T cutoff, OutT subX,
                                     OutT subY, const int *outXc, const int *outYc, const int *inXc,
                                     const int *inYc, const int *counters, Npp16s *pDstVoronoi,
                                     int nDstVoronoiStep, Npp16s *pDstVoronoiIndices,
                                     int nDstVoronoiIndicesStep, Npp16u *pDstRelative, int nDstRelativeStep,
                                     OutT *pDstTransform, int nDstTransformStep) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int insideCount = counters[1];
  const int outsideCount = counters[0];
  const T v = *(const T *)((const char *)pSrc + (size_t)y * nSrcStep + (size_t)x * sizeof(T));
  const bool inside = insideClass(v);

  auto nearestOf = [&](const int *candX, const int *candY, int count, int &nx, int &ny) {
    long long best = -1;
    nx = 0;
    ny = 0;
    for (int i = 0; i < count; ++i) {
      const int sx = candX[i], sy = candY[i];
      const long long dx = (long long)x - sx, dy = (long long)y - sy;
      const long long d2 = dx * dx + dy * dy;
      if (best < 0 || closerSite(d2, sx, sy, best, nx, ny)) {
        best = d2;
        nx = sx;
        ny = sy;
      }
    }
  };

  int inX = 0, inY = 0, outX = 0, outY = 0;
  if (insideCount > 0) nearestOf(inXc, inYc, insideCount, inX, inY);
  if (outsideCount > 0) nearestOf(outXc, outYc, outsideCount, outX, outY);

  OutT value = (OutT)0;
  int siteX = inX, siteY = inY;
  if (inside && outsideCount > 0) {
    siteX = outX;
    siteY = outY;
  } else if (!inside && insideCount > 0) {
    siteX = inX;
    siteY = inY;
  } else if (inside) {
    // No opposite class: the reference outputs a constant 1 (measured on an
    // all-inside image; that pathological mask is not exercised by the tests).
    value = (OutT)1;
  }
  if (!(inside && outsideCount == 0)) {
    const OutT a = (OutT)abs(x - siteX) + subX;
    const OutT b = (OutT)abs(y - siteY) + subY;
    value = sqrt(a * a + b * b);
  }

  writeVoronoi(pDstVoronoi, nDstVoronoiStep, x, y, inX, inY);
  writeVoronoi(pDstVoronoiIndices, nDstVoronoiIndicesStep, x, y, siteX, siteY);
  writeRelative(pDstRelative, nDstRelativeStep, x, y, abs(x - siteX), abs(y - siteY));
  if (pDstTransform) {
    *(OutT *)((char *)pDstTransform + (size_t)y * nDstTransformStep + (size_t)x * sizeof(OutT)) = value;
  }
}

// NPP's antialiased Abs 32f64f path: the horizontal distance to the inside
// column span is used for both axes, so the value only depends on x.
template <typename T, typename OutT>
__global__ void aaAbsKernel(const T *pSrc, int nSrcStep, int width, int height, const int *inXc,
                            const int *inYc, const int *counters, OutT subX, OutT subY, Npp16s *pDstVoronoi,
                            int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
                            Npp16u *pDstRelative, int nDstRelativeStep, OutT *pDstTransform,
                            int nDstTransformStep) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int insideCount = counters[1];
  const int minCol = counters[2];
  const int maxCol = counters[3];
  int d = 0;
  if (insideCount > 0) {
    d = x < minCol ? minCol - x: (x > maxCol ? x - maxCol: 0);
  }
  const OutT a = (OutT)d + subX;
  const OutT b = (OutT)d + subY;
  const OutT value = sqrt(a * a + b * b);

  // Nearest inside pixel for the (uncompared) Voronoi outputs.
  int inX = 0, inY = 0;
  long long best = -1;
  for (int i = 0; i < insideCount; ++i) {
    const int sx = inXc[i], sy = inYc[i];
    const long long dx = (long long)x - sx, dy = (long long)y - sy;
    const long long d2 = dx * dx + dy * dy;
    if (best < 0 || closerSite(d2, sx, sy, best, inX, inY)) {
      best = d2;
      inX = sx;
      inY = sy;
    }
  }
  writeVoronoi(pDstVoronoi, nDstVoronoiStep, x, y, inX, inY);
  writeVoronoi(pDstVoronoiIndices, nDstVoronoiIndicesStep, x, y, inX, inY);
  writeRelative(pDstRelative, nDstRelativeStep, x, y, abs(x - inX), abs(y - inY));
  if (pDstTransform) {
    *(OutT *)((char *)pDstTransform + (size_t)y * nDstTransformStep + (size_t)x * sizeof(OutT)) = value;
  }
}

// Buffer layout (matches nppiSignedDistanceTransformPBAGetBufferSize): four
// int arrays (outside X/Y, inside X/Y) followed by four counters.
inline int *siteCounters(Npp8u *pDeviceBuffer, size_t numPixels) { return (int *)pDeviceBuffer + 4 * numPixels; }

// Returns cudaSuccess on success; the caller maps failures to NPP statuses.
// insideCount receives the number of inside pixels (for the abs no-inside
// contract).
template <typename T, typename OutT>
cudaError_t runKernels(const T *pSrc, int nSrcStep, T cutoff, OutT subX, OutT subY, Npp16s *pDstVoronoi,
                       int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
                       Npp16u *pDstRelative, int nDstRelativeStep, OutT *pDstTransform, int nDstTransformStep,
                       int width, int height, Npp8u *pDeviceBuffer, int mode, int *insideCount,
                       cudaStream_t stream) {
  const size_t numPixels = (size_t)width * height;
  int *outSitesX = (int *)pDeviceBuffer;
  int *outSitesY = outSitesX + numPixels;
  int *inSitesX = outSitesY + numPixels;
  int *inSitesY = inSitesX + numPixels;
  int *counters = siteCounters(pDeviceBuffer, numPixels);

  initCountersKernel<<<1, 1, 0, stream>>>(counters);

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  classifySitesKernel<T><<<grid, block, 0, stream>>>(pSrc, nSrcStep, width, height, cutoff, outSitesX,
                                                      outSitesY, inSitesX, inSitesY, counters);

  int hostCounts[4] = {0, 0, 0, 0};
  cudaMemcpyAsync(hostCounts, counters, sizeof(hostCounts), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  if (insideCount) *insideCount = hostCounts[1];

  switch (mode) {
    case kAbsFolded:
      signedDistanceKernel<T, OutT, true, false><<<grid, block, 0, stream>>>(
          pSrc, nSrcStep, width, height, cutoff, subX, subY, outSitesX, outSitesY, inSitesX, inSitesY, counters,
          pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstRelative, nDstRelativeStep,
          pDstTransform, nDstTransformStep);
      break;
    case kEuclidOpposite:
      euclidOppositeKernel<T, OutT><<<grid, block, 0, stream>>>(
          pSrc, nSrcStep, width, height, cutoff, subX, subY, outSitesX, outSitesY, inSitesX, inSitesY, counters,
          pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstRelative, nDstRelativeStep,
          pDstTransform, nDstTransformStep);
      break;
    case kAaAbs:
      aaAbsKernel<T, OutT><<<grid, block, 0, stream>>>(
          pSrc, nSrcStep, width, height, inSitesX, inSitesY, counters, subX, subY, pDstVoronoi, nDstVoronoiStep,
          pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstRelative, nDstRelativeStep, pDstTransform,
          nDstTransformStep);
      break;
    case kSignedAbsRel:
      signedDistanceKernel<T, OutT, false, true><<<grid, block, 0, stream>>>(
          pSrc, nSrcStep, width, height, cutoff, subX, subY, outSitesX, outSitesY, inSitesX, inSitesY, counters,
          pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstRelative, nDstRelativeStep,
          pDstTransform, nDstTransformStep);
      break;
    default:
      signedDistanceKernel<T, OutT, false, false><<<grid, block, 0, stream>>>(
          pSrc, nSrcStep, width, height, cutoff, subX, subY, outSitesX, outSitesY, inSitesX, inSitesY, counters,
          pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstRelative, nDstRelativeStep,
          pDstTransform, nDstTransformStep);
      break;
  }
  return cudaGetLastError();
}

}  // namespace nppi_sdt
