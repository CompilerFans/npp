#ifndef SUPER_INTERPOLATOR_CUH
#define SUPER_INTERPOLATOR_CUH

#include "interpolator_common.cuh"

// Super sampling interpolation (simple box filter)
template <typename T, int CHANNELS> struct SuperSamplingInterpolator {
  __device__ static void interpolate(const T *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, T *pDst, int nDstStep, const NppiRect &oDstRectROI) {
    // Calculate source region bounds
    float src_x_start = dx * scaleX + oSrcRectROI.x;
    float src_y_start = dy * scaleY + oSrcRectROI.y;
    float src_x_end = (dx + 1) * scaleX + oSrcRectROI.x;
    float src_y_end = (dy + 1) * scaleY + oSrcRectROI.y;

    int x_start = max((int)src_x_start, oSrcRectROI.x);
    int y_start = max((int)src_y_start, oSrcRectROI.y);
    int x_end = min((int)src_x_end, oSrcRectROI.x + oSrcRectROI.width);
    int y_end = min((int)src_y_end, oSrcRectROI.y + oSrcRectROI.height);

    float sum[CHANNELS];
#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      sum[c] = 0.0f;
    }
    int count = 0;

    // Average all source pixels that contribute to this destination pixel
    for (int y = y_start; y < y_end; y++) {
      const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);
      for (int x = x_start; x < x_end; x++) {
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sum[c] += src_row[x * CHANNELS + c];
        }
        count++;
      }
    }

    T *dst_row = (T *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

    if (count > 0) {
#pragma unroll
      for (int c = 0; c < CHANNELS; c++) {
        dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = (T)(sum[c] / count + 0.5f);
      }
    } else {
#pragma unroll
      for (int c = 0; c < CHANNELS; c++) {
        dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = 0;
      }
    }
  }
};

#endif // SUPER_INTERPOLATOR_CUH
