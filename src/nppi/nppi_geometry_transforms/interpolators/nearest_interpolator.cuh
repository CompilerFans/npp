#ifndef NEAREST_INTERPOLATOR_CUH
#define NEAREST_INTERPOLATOR_CUH

#include "interpolator_common.cuh"

// Nearest neighbor interpolation
template <typename T, int CHANNELS> struct NearestInterpolator {
  __device__ static void interpolate(const T *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, T *pDst, int nDstStep, const NppiRect &oDstRectROI) {
    // Map destination coordinates to source coordinates
    int sx = (int)(dx * scaleX + 0.5f) + oSrcRectROI.x;
    int sy = (int)(dy * scaleY + 0.5f) + oSrcRectROI.y;

    // Clamp to source bounds
    sx = min(max(sx, oSrcRectROI.x), oSrcRectROI.x + oSrcRectROI.width - 1);
    sy = min(max(sy, oSrcRectROI.y), oSrcRectROI.y + oSrcRectROI.height - 1);

    // Copy pixels
    const T *src_row = (const T *)((const char *)pSrc + sy * nSrcStep);
    T *dst_row = (T *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

    int src_idx = sx * CHANNELS;
    int dst_idx = (dx + oDstRectROI.x) * CHANNELS;

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      dst_row[dst_idx + c] = src_row[src_idx + c];
    }
  }
};

#endif // NEAREST_INTERPOLATOR_CUH
