#ifndef CUBIC_INTERPOLATOR_CUH
#define CUBIC_INTERPOLATOR_CUH

#include "interpolator_common.cuh"

// Bicubic interpolation (generic for 8u and integer types)
template <typename T, int CHANNELS> struct CubicInterpolator {
  __device__ static void interpolate(const T *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, T *pDst, int nDstStep, const NppiRect &oDstRectROI) {
    // Map destination coordinates to source coordinates
    float fx = dx * scaleX + oSrcRectROI.x;
    float fy = dy * scaleY + oSrcRectROI.y;

    int x_center = (int)floorf(fx);
    int y_center = (int)floorf(fy);

    float tx = fx - x_center;
    float ty = fy - y_center;

    T *dst_row = (T *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      float sum = 0.0f;

      // Sample 4x4 neighborhood
      for (int j = -1; j <= 2; j++) {
        int sy = y_center + j;
        sy = max(sy, oSrcRectROI.y);
        sy = min(sy, oSrcRectROI.y + oSrcRectROI.height - 1);

        float wy = cubicWeight(ty - j);

        const T *src_row = (const T *)((const char *)pSrc + sy * nSrcStep);

        for (int i = -1; i <= 2; i++) {
          int sx = x_center + i;
          sx = max(sx, oSrcRectROI.x);
          sx = min(sx, oSrcRectROI.x + oSrcRectROI.width - 1);

          float wx = cubicWeight(tx - i);
          float weight = wx * wy;

          sum += src_row[sx * CHANNELS + c] * weight;
        }
      }

      // Clamp result to valid range
      sum = max(0.0f, min(sum, 255.0f));
      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = (T)(sum + 0.5f);
    }
  }
};

// Bicubic interpolation (specialized for float)
template <int CHANNELS> struct CubicInterpolator<float, CHANNELS> {
  __device__ static void interpolate(const float *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, float *pDst, int nDstStep,
                                     const NppiRect &oDstRectROI) {
    float fx = dx * scaleX + oSrcRectROI.x;
    float fy = dy * scaleY + oSrcRectROI.y;

    int x_center = (int)floorf(fx);
    int y_center = (int)floorf(fy);

    float tx = fx - x_center;
    float ty = fy - y_center;

    float *dst_row = (float *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      float sum = 0.0f;

      for (int j = -1; j <= 2; j++) {
        int sy = y_center + j;
        sy = max(sy, oSrcRectROI.y);
        sy = min(sy, oSrcRectROI.y + oSrcRectROI.height - 1);

        float wy = cubicWeight(ty - j);

        const float *src_row = (const float *)((const char *)pSrc + sy * nSrcStep);

        for (int i = -1; i <= 2; i++) {
          int sx = x_center + i;
          sx = max(sx, oSrcRectROI.x);
          sx = min(sx, oSrcRectROI.x + oSrcRectROI.width - 1);

          float wx = cubicWeight(tx - i);
          float weight = wx * wy;

          sum += src_row[sx * CHANNELS + c] * weight;
        }
      }

      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = sum;
    }
  }
};

#endif // CUBIC_INTERPOLATOR_CUH
