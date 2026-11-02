// Signed distance transform (PBA family) — exact mask semantics decoded from
// NPP 13.4 libnppif.
//
// The reference runs the two PBA passes on outside (source <= 0, or < cutoff)
// and inside (source >= 1) pixels and then merges them with
// generateAdditionalRelativeShiftOutput. That output kernel does NOT emit the
// Euclidean distance: with the site displacement (dx,dy), the raw shift
// (sx,sy) and C = 30 + cutoff it computes
//
//   a = |dx| + sx, b = |dy| + sy
//   t = sqrt((max(a,b) - C)^2 + min(a,b)^2)
//
// and stores +t for pixels inside a region and -t outside. The 30 is a fixed
// band-reconstruction constant in the reference, independent of the ROI.
// Voronoi outputs:
//   signed: pDstVoronoi      = nearest outside pixel (self for outside pixels)
//           pDstVoronoiIndices = nearest opposite-class pixel (the one used
//                                by the transform)
//           relative manhattan = pixel - that opposite site
//   abs 32f: voronoi and indices = nearest inside pixel; relative = absolute
//            Manhattan distance; the transform is positive.
//   abs 64f without an AA buffer: same sites and signed transform as the
//            signed flavour, but the relative output is absolute.
// When there are no outside pixels the signed transform uses the dummy site
// (-1,0); when there are no inside pixels the signed outputs are all zero and
// the abs 32f/AA paths return NPP_NULL_POINTER_ERROR, as measured.
//
// Known limitations:
//   * fractional source values (0 < v < 1, v >= cutoff) feed a sub-pixel
//     antialiasing path in the reference; this executor treats them as
//     outside-class non-sites (0/1 masks are exact);
//   * the 32f64f abs entry point requires the antialiasing workspace, and the
//     reference AA values are input-sensitive and not a plain distance field.
//     This executor does not reproduce them: it emits the Euclidean distance
//     to the nearest inside pixel (0 inside) and the absolute nearest-inside
//     outputs.  The 64f abs variant uses the same approximation when an AA
//     buffer is supplied and is exact when the buffer is null.
#include "npp.h"

#include <cmath>
#include <cuda_runtime.h>

namespace {

constexpr int kMinROI = 64;

template <typename T> __device__ __forceinline__ bool insideClass(T v) { return v >= (T)1; }

template <typename T> __device__ __forceinline__ bool outsideSite(T v, T cutoff) {
  return v <= (T)0 || v < cutoff;
}

// Build compact site lists in the caller-provided scratch buffer.
template <typename T>
__global__ void classifySitesKernel(const T *pSrc, int nSrcStep, int width, int height, T cutoff, int2 *outSites,
                                    int2 *inSites, int *counters) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const T v = *(const T *)((const char *)pSrc + (size_t)y * nSrcStep + (size_t)x * sizeof(T));
  if (insideClass(v)) {
    const int i = atomicAdd(&counters[1], 1);
    inSites[i] = make_int2(x, y);
  } else if (outsideSite(v, cutoff)) {
    const int i = atomicAdd(&counters[0], 1);
    outSites[i] = make_int2(x, y);
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
  const CalcT major = a >= b ? a : b;
  const CalcT minor = a >= b ? b : a;
  const CalcT d = major - C;
  return sqrt(d * d + minor * minor);
}

template <typename T, typename OutT, bool Abs, bool AbsRel>
__global__ void signedDistanceKernel(const T *pSrc, int nSrcStep, int width, int height, T cutoff,
                                     OutT subX, OutT subY, const int2 *outSites, const int2 *inSites,
                                     const int *counters, Npp16s *pDstVoronoi, int nDstVoronoiStep,
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
    if (pDstVoronoi) {
      Npp16s *row = (Npp16s *)((char *)pDstVoronoi + (size_t)y * nDstVoronoiStep);
      row[2 * x] = 0;
      row[2 * x + 1] = 0;
    }
    if (pDstVoronoiIndices) {
      Npp16s *row = (Npp16s *)((char *)pDstVoronoiIndices + (size_t)y * nDstVoronoiIndicesStep);
      row[2 * x] = 0;
      row[2 * x + 1] = 0;
    }
    if (pDstRelative) {
      Npp16u *row = (Npp16u *)((char *)pDstRelative + (size_t)y * nDstRelativeStep);
      row[2 * x] = 0;
      row[2 * x + 1] = 0;
    }
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
      const int2 s = outSites[i];
      const long long dx = (long long)x - s.x, dy = (long long)y - s.y;
      const long long d2 = dx * dx + dy * dy;
      if (outD2 < 0 || closerSite(d2, s.x, s.y, outD2, outX, outY)) {
        outD2 = d2;
        outX = s.x;
        outY = s.y;
      }
    }
  }

  int inX = 0, inY = 0;
  long long inD2 = -1;
  for (int i = 0; i < insideCount; ++i) {
    const int2 s = inSites[i];
    const long long dx = (long long)x - s.x, dy = (long long)y - s.y;
    const long long d2 = dx * dx + dy * dy;
    if (inD2 < 0 || closerSite(d2, s.x, s.y, inD2, inX, inY)) {
      inD2 = d2;
      inX = s.x;
      inY = s.y;
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

  if (pDstVoronoi) {
    Npp16s *row = (Npp16s *)((char *)pDstVoronoi + (size_t)y * nDstVoronoiStep);
    row[2 * x] = (Npp16s)vorX;
    row[2 * x + 1] = (Npp16s)vorY;
  }
  if (pDstVoronoiIndices) {
    Npp16s *row = (Npp16s *)((char *)pDstVoronoiIndices + (size_t)y * nDstVoronoiIndicesStep);
    row[2 * x] = (Npp16s)siteX;
    row[2 * x + 1] = (Npp16s)siteY;
  }
  if (pDstRelative) {
    Npp16u *row = (Npp16u *)((char *)pDstRelative + (size_t)y * nDstRelativeStep);
    row[2 * x] = (Npp16u)relX;
    row[2 * x + 1] = (Npp16u)relY;
  }
  if (pDstTransform) {
    *(OutT *)((char *)pDstTransform + (size_t)y * nDstTransformStep + (size_t)x * sizeof(OutT)) = value;
  }
}

// Best-effort antialiased path. The reference's AA workspace is input-sensitive
// (zero-filled works, arbitrary garbage can fail with NPP_NULL_POINTER_ERROR)
// and its per-pixel values are not a plain distance field, so this emits the
// true Euclidean distance to the nearest inside pixel (0 inside) plus the
// absolute nearest-inside site outputs. Only 0/1 masks with zero shifts are
// covered by the differential tests.
template <typename T, typename OutT>
__global__ void aaDistanceKernel(const T *pSrc, int nSrcStep, int width, int height, const int2 *inSites,
                                 const int *counters, Npp16s *pDstVoronoi, int nDstVoronoiStep,
                                 Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep, Npp16u *pDstRelative,
                                 int nDstRelativeStep, OutT *pDstTransform, int nDstTransformStep) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int insideCount = counters[1];
  int siteX = 0, siteY = 0;
  long long bestD2 = -1;
  for (int i = 0; i < insideCount; ++i) {
    const int2 s = inSites[i];
    const long long dx = (long long)x - s.x, dy = (long long)y - s.y;
    const long long d2 = dx * dx + dy * dy;
    if (bestD2 < 0 || closerSite(d2, s.x, s.y, bestD2, siteX, siteY)) {
      bestD2 = d2;
      siteX = s.x;
      siteY = s.y;
    }
  }

  const T v = *(const T *)((const char *)pSrc + (size_t)y * nSrcStep + (size_t)x * sizeof(T));
  const OutT value = insideClass(v) ? (OutT)0 : (OutT)sqrt((OutT)bestD2);

  if (pDstVoronoi) {
    Npp16s *row = (Npp16s *)((char *)pDstVoronoi + (size_t)y * nDstVoronoiStep);
    row[2 * x] = (Npp16s)siteX;
    row[2 * x + 1] = (Npp16s)siteY;
  }
  if (pDstVoronoiIndices) {
    Npp16s *row = (Npp16s *)((char *)pDstVoronoiIndices + (size_t)y * nDstVoronoiIndicesStep);
    row[2 * x] = (Npp16s)siteX;
    row[2 * x + 1] = (Npp16s)siteY;
  }
  if (pDstRelative) {
    Npp16u *row = (Npp16u *)((char *)pDstRelative + (size_t)y * nDstRelativeStep);
    row[2 * x] = (Npp16u)abs(siteX - x);
    row[2 * x + 1] = (Npp16u)abs(siteY - y);
  }
  if (pDstTransform) {
    *(OutT *)((char *)pDstTransform + (size_t)y * nDstTransformStep + (size_t)x * sizeof(OutT)) = value;
  }
}

template <typename T, typename OutT>
NppStatus validateAndRun(T *pSrc, int nSrcStep, T cutoff, OutT subX, OutT subY, Npp16s *pDstVoronoi,
                         int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
                         Npp16u *pDstRelative, int nDstRelativeStep, OutT *pDstTransform, int nDstTransformStep,
                         NppiSize oSizeROI, Npp8u *pDeviceBuffer, int mode, bool antialiasingRequired,
                         Npp8u *pAntialiasingDeviceBuffer, cudaStream_t stream) {
  if (pSrc == nullptr || pDeviceBuffer == nullptr) return NPP_NULL_POINTER_ERROR;
  const bool useAA = pAntialiasingDeviceBuffer != nullptr;
  if (antialiasingRequired && !useAA) return NPP_NULL_POINTER_ERROR;
  // The antialiasing path requires the transform output (checked before the
  // ROI, as measured on the 5090).
  if (useAA && pDstTransform == nullptr) return NPP_NULL_POINTER_ERROR;
  if (!pDstVoronoi && !pDstVoronoiIndices && !pDstRelative && !pDstTransform) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < kMinROI || oSizeROI.height < kMinROI) return NPP_SIZE_ERROR;

  const int width = oSizeROI.width, height = oSizeROI.height;
  if ((pDstVoronoi && nDstVoronoiStep < width * 4) ||
      (pDstVoronoiIndices && nDstVoronoiIndicesStep < width * 4) ||
      (pDstRelative && nDstRelativeStep < width * 4)) {
    return NPP_STEP_ERROR;
  }
  // Plain comparison: NaN passes, infinities do not.
  if (cutoff < (T)0 || cutoff >= (T)1) return NPP_RANGE_ERROR;

  const size_t numPixels = (size_t)width * height;
  int2 *outSites = (int2 *)pDeviceBuffer;
  int2 *inSites = outSites + numPixels;
  int *counters = (int *)(inSites + numPixels);
  cudaMemsetAsync(counters, 0, 2 * sizeof(int), stream);

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  classifySitesKernel<T><<<grid, block, 0, stream>>>(pSrc, nSrcStep, width, height, cutoff, outSites, inSites,
                                                      counters);

  int hostCounts[2] = {0, 0};
  cudaMemcpyAsync(hostCounts, counters, sizeof(hostCounts), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  if (hostCounts[1] == 0 && mode == 1) return NPP_NULL_POINTER_ERROR;

  if (mode == 1 && useAA) {
    aaDistanceKernel<T, OutT><<<grid, block, 0, stream>>>(pSrc, nSrcStep, width, height, inSites, counters,
                                                          pDstVoronoi, nDstVoronoiStep, pDstVoronoiIndices,
                                                          nDstVoronoiIndicesStep, pDstRelative, nDstRelativeStep,
                                                          pDstTransform, nDstTransformStep);
  } else if (mode == 1) {
    signedDistanceKernel<T, OutT, true, false><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, width, height, cutoff, subX, subY, outSites, inSites, counters, pDstVoronoi,
        nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstRelative, nDstRelativeStep, pDstTransform,
        nDstTransformStep);
  } else if (mode == 2) {
    signedDistanceKernel<T, OutT, false, true><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, width, height, cutoff, subX, subY, outSites, inSites, counters, pDstVoronoi,
        nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstRelative, nDstRelativeStep, pDstTransform,
        nDstTransformStep);
  } else {
    signedDistanceKernel<T, OutT, false, false><<<grid, block, 0, stream>>>(
        pSrc, nSrcStep, width, height, cutoff, subX, subY, outSites, inSites, counters, pDstVoronoi,
        nDstVoronoiStep, pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstRelative, nDstRelativeStep, pDstTransform,
        nDstTransformStep);
  }
  return cudaGetLastError() == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

}  // namespace

NppStatus nppiSignedDistanceTransformPBA_32f_C1R_Ctx(
    Npp32f *pSrc, int nSrcStep, Npp32f nCutoffValue, Npp32f nSubPixelXShift, Npp32f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16s *pDstVoronoiRelativeManhattanDistances, int nDstVoronoiRelativeManhattanDistancesStep,
    Npp32f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
    NppStreamContext nppStreamCtx) {
  return validateAndRun<Npp32f, Npp32f>(
      pSrc, nSrcStep, nCutoffValue, nSubPixelXShift, nSubPixelYShift, pDstVoronoi, nDstVoronoiStep,
      pDstVoronoiIndices, nDstVoronoiIndicesStep, (Npp16u *)pDstVoronoiRelativeManhattanDistances,
      nDstVoronoiRelativeManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, 0,
      false, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSignedDistanceTransformAbsPBA_32f_C1R_Ctx(
    Npp32f *pSrc, int nSrcStep, Npp32f nCutoffValue, Npp32f nSubPixelXShift, Npp32f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,
    Npp32f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
    NppStreamContext nppStreamCtx) {
  return validateAndRun<Npp32f, Npp32f>(
      pSrc, nSrcStep, nCutoffValue, nSubPixelXShift, nSubPixelYShift, pDstVoronoi, nDstVoronoiStep,
      pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,
      nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, 1,
      false, nullptr, nppStreamCtx.hStream);
}

NppStatus nppiSignedDistanceTransformAbsPBA_32f64f_C1R_Ctx(
    Npp32f *pSrc, int nSrcStep, Npp32f nCutoffValue, Npp64f nSubPixelXShift, Npp64f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,
    Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
    Npp8u *pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return validateAndRun<Npp32f, Npp64f>(
      pSrc, nSrcStep, nCutoffValue, nSubPixelXShift, nSubPixelYShift, pDstVoronoi, nDstVoronoiStep,
      pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,
      nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer, 1,
      true, pAntialiasingDeviceBuffer, nppStreamCtx.hStream);
}

NppStatus nppiSignedDistanceTransformAbsPBA_64f_C1R_Ctx(
    Npp64f *pSrc, int nSrcStep, Npp64f nCutoffValue, Npp64f nSubPixelXShift, Npp64f nSubPixelYShift,
    Npp16s *pDstVoronoi, int nDstVoronoiStep, Npp16s *pDstVoronoiIndices, int nDstVoronoiIndicesStep,
    Npp16u *pDstVoronoiAbsoluteManhattanDistances, int nDstVoronoiAbsoluteManhattanDistancesStep,
    Npp64f *pDstTransform, int nDstTransformStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
    Npp8u *pAntialiasingDeviceBuffer, NppStreamContext nppStreamCtx) {
  return validateAndRun<Npp64f, Npp64f>(
      pSrc, nSrcStep, nCutoffValue, nSubPixelXShift, nSubPixelYShift, pDstVoronoi, nDstVoronoiStep,
      pDstVoronoiIndices, nDstVoronoiIndicesStep, pDstVoronoiAbsoluteManhattanDistances,
      nDstVoronoiAbsoluteManhattanDistancesStep, pDstTransform, nDstTransformStep, oSizeROI, pDeviceBuffer,
      pAntialiasingDeviceBuffer ? 1 : 2, false, pAntialiasingDeviceBuffer, nppStreamCtx.hStream);
}
