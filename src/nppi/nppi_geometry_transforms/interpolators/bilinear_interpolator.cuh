#ifndef BILINEAR_INTERPOLATOR_CUH
#define BILINEAR_INTERPOLATOR_CUH

#include "interpolator_common.cuh"

// Bilinear interpolation (generic for 8u and integer types)
template <typename T, int CHANNELS> struct BilinearInterpolator {
  __device__ static void interpolate(const T *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, T *pDst, int nDstStep, const NppiRect &oDstRectROI) {
    // Map destination coordinates to source coordinates
    float fx = dx * scaleX + oSrcRectROI.x;
    float fy = dy * scaleY + oSrcRectROI.y;

    int x1 = (int)fx;
    int y1 = (int)fy;
    int x2 = min(x1 + 1, oSrcRectROI.x + oSrcRectROI.width - 1);
    int y2 = min(y1 + 1, oSrcRectROI.y + oSrcRectROI.height - 1);

    x1 = max(x1, oSrcRectROI.x);
    y1 = max(y1, oSrcRectROI.y);

    float tx = fx - x1;
    float ty = fy - y1;

    const T *src_row1 = (const T *)((const char *)pSrc + y1 * nSrcStep);
    const T *src_row2 = (const T *)((const char *)pSrc + y2 * nSrcStep);
    T *dst_row = (T *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      float p11 = src_row1[x1 * CHANNELS + c];
      float p12 = src_row1[x2 * CHANNELS + c];
      float p21 = src_row2[x1 * CHANNELS + c];
      float p22 = src_row2[x2 * CHANNELS + c];

      float result = lerp(lerp(p11, p12, tx), lerp(p21, p22, tx), ty);
      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = (T)(result + 0.5f);
    }
  }
};

// Bilinear interpolation (specialized for float)
template <int CHANNELS> struct BilinearInterpolator<float, CHANNELS> {
  __device__ static void interpolate(const float *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, float *pDst, int nDstStep,
                                     const NppiRect &oDstRectROI) {
    float fx = dx * scaleX + oSrcRectROI.x;
    float fy = dy * scaleY + oSrcRectROI.y;

    int x1 = (int)fx;
    int y1 = (int)fy;
    int x2 = min(x1 + 1, oSrcRectROI.x + oSrcRectROI.width - 1);
    int y2 = min(y1 + 1, oSrcRectROI.y + oSrcRectROI.height - 1);

    x1 = max(x1, oSrcRectROI.x);
    y1 = max(y1, oSrcRectROI.y);

    float tx = fx - x1;
    float ty = fy - y1;

    const float *src_row1 = (const float *)((const char *)pSrc + y1 * nSrcStep);
    const float *src_row2 = (const float *)((const char *)pSrc + y2 * nSrcStep);
    float *dst_row = (float *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      float p11 = src_row1[x1 * CHANNELS + c];
      float p12 = src_row1[x2 * CHANNELS + c];
      float p21 = src_row2[x1 * CHANNELS + c];
      float p22 = src_row2[x2 * CHANNELS + c];

      float result = lerp(lerp(p11, p12, tx), lerp(p21, p22, tx), ty);
      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = result;
    }
  }
};

#endif // BILINEAR_INTERPOLATOR_CUH
