#ifndef BILINEAR_V2_INTERPOLATOR_CUH
#define BILINEAR_V2_INTERPOLATOR_CUH

#include "interpolator_common.cuh"

// Bilinear V2 interpolation - NPP-compatible linear resize algorithm
// This version implements the actual algorithm used by NPP, discovered through
// reverse engineering and validated against NPP outputs.
//
// Key differences from standard bilinear:
// 1. Three modes based on scale factor: upscale, fractional downscale, large downscale
// 2. Different coordinate mapping for large downscale
// 3. Fractional downscale uses threshold-based hybrid interpolation
// 4. Large downscale uses floor sampling (no interpolation)
//
// Reference: ref_code/linear_resize_cpu_reference/linear_resize_refactored.h
template <typename T, int CHANNELS> struct BilinearV2Interpolator {
  enum Mode { UPSCALE, FRACTIONAL_DOWN, LARGE_DOWN };

  __device__ static Mode getMode(float scale) {
    if (scale < 1.0f)
      return UPSCALE;
    if (scale < 2.0f)
      return FRACTIONAL_DOWN;
    return LARGE_DOWN;
  }

  __device__ static float interp1D(float v0, float v1, float f, float scale, Mode mode) {
    switch (mode) {
    case UPSCALE:
      // Standard linear interpolation
      return v0 * (1.0f - f) + v1 * f;

    case FRACTIONAL_DOWN: {
      // Hybrid: threshold-based floor or attenuated interpolation
      float threshold = 0.5f / scale;
      if (f <= threshold) {
        return v0; // Floor
      } else {
        float attenuation = 1.0f / scale;
        float f_att = f * attenuation;
        return v0 * (1.0f - f_att) + v1 * f_att;
      }
    }

    case LARGE_DOWN:
      // No interpolation, use floor
      return v0;
    }
    return v0; // Default
  }

  __device__ static void interpolate(const T *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, T *pDst, int nDstStep, const NppiRect &oDstRectROI) {
    // Determine mode for each dimension
    Mode modeX = getMode(scaleX);
    Mode modeY = getMode(scaleY);

    // Coordinate mapping depends on mode
    bool useLargeDownscale = (modeX == LARGE_DOWN && modeY == LARGE_DOWN);
    float srcX, srcY;

    if (useLargeDownscale) {
      // Large downscale: edge-to-edge mapping
      srcX = dx * scaleX + oSrcRectROI.x;
      srcY = dy * scaleY + oSrcRectROI.y;
    } else {
      // Upscale or fractional downscale: center-to-center mapping
      srcX = (dx + 0.5f) * scaleX - 0.5f + oSrcRectROI.x;
      srcY = (dy + 0.5f) * scaleY - 0.5f + oSrcRectROI.y;
    }

    // Clamp coordinates to ROI
    srcX = fmaxf(0.0f, fminf((float)(oSrcRectROI.x + oSrcRectROI.width - 1), srcX));
    srcY = fmaxf(0.0f, fminf((float)(oSrcRectROI.y + oSrcRectROI.height - 1), srcY));

    // Get neighbor coordinates
    int x0 = (int)floorf(srcX);
    int y0 = (int)floorf(srcY);
    int x1 = min(x0 + 1, oSrcRectROI.x + oSrcRectROI.width - 1);
    int y1 = min(y0 + 1, oSrcRectROI.y + oSrcRectROI.height - 1);

    x0 = max(x0, oSrcRectROI.x);
    y0 = max(y0, oSrcRectROI.y);

    // Fractional parts
    float fx = srcX - x0;
    float fy = srcY - y0;

    // Get source rows
    const T *src_row0 = (const T *)((const char *)pSrc + y0 * nSrcStep);
    const T *src_row1 = (const T *)((const char *)pSrc + y1 * nSrcStep);
    T *dst_row = (T *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      // Get 4 corner pixels
      float p00 = src_row0[x0 * CHANNELS + c];
      float p10 = src_row0[x1 * CHANNELS + c];
      float p01 = src_row1[x0 * CHANNELS + c];
      float p11 = src_row1[x1 * CHANNELS + c];

      // Separable 2D interpolation
      // Interpolate in X direction
      float v0 = interp1D(p00, p10, fx, scaleX, modeX);
      float v1 = interp1D(p01, p11, fx, scaleX, modeX);

      // Interpolate in Y direction
      float result = interp1D(v0, v1, fy, scaleY, modeY);

      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = (T)(result + 0.5f);
    }
  }
};

// Specialized version for float (no rounding)
template <int CHANNELS> struct BilinearV2Interpolator<float, CHANNELS> {
  enum Mode { UPSCALE, FRACTIONAL_DOWN, LARGE_DOWN };

  __device__ static Mode getMode(float scale) {
    if (scale < 1.0f)
      return UPSCALE;
    if (scale < 2.0f)
      return FRACTIONAL_DOWN;
    return LARGE_DOWN;
  }

  __device__ static float interp1D(float v0, float v1, float f, float scale, Mode mode) {
    switch (mode) {
    case UPSCALE:
      return v0 * (1.0f - f) + v1 * f;

    case FRACTIONAL_DOWN: {
      float threshold = 0.5f / scale;
      if (f <= threshold) {
        return v0;
      } else {
        float attenuation = 1.0f / scale;
        float f_att = f * attenuation;
        return v0 * (1.0f - f_att) + v1 * f_att;
      }
    }

    case LARGE_DOWN:
      return v0;
    }
    return v0;
  }

  __device__ static void interpolate(const float *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, float *pDst, int nDstStep,
                                     const NppiRect &oDstRectROI) {
    Mode modeX = getMode(scaleX);
    Mode modeY = getMode(scaleY);

    bool useLargeDownscale = (modeX == LARGE_DOWN && modeY == LARGE_DOWN);
    float srcX, srcY;

    if (useLargeDownscale) {
      srcX = dx * scaleX + oSrcRectROI.x;
      srcY = dy * scaleY + oSrcRectROI.y;
    } else {
      srcX = (dx + 0.5f) * scaleX - 0.5f + oSrcRectROI.x;
      srcY = (dy + 0.5f) * scaleY - 0.5f + oSrcRectROI.y;
    }

    srcX = fmaxf(0.0f, fminf((float)(oSrcRectROI.x + oSrcRectROI.width - 1), srcX));
    srcY = fmaxf(0.0f, fminf((float)(oSrcRectROI.y + oSrcRectROI.height - 1), srcY));

    int x0 = (int)floorf(srcX);
    int y0 = (int)floorf(srcY);
    int x1 = min(x0 + 1, oSrcRectROI.x + oSrcRectROI.width - 1);
    int y1 = min(y0 + 1, oSrcRectROI.y + oSrcRectROI.height - 1);

    x0 = max(x0, oSrcRectROI.x);
    y0 = max(y0, oSrcRectROI.y);

    float fx = srcX - x0;
    float fy = srcY - y0;

    const float *src_row0 = (const float *)((const char *)pSrc + y0 * nSrcStep);
    const float *src_row1 = (const float *)((const char *)pSrc + y1 * nSrcStep);
    float *dst_row = (float *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      float p00 = src_row0[x0 * CHANNELS + c];
      float p10 = src_row0[x1 * CHANNELS + c];
      float p01 = src_row1[x0 * CHANNELS + c];
      float p11 = src_row1[x1 * CHANNELS + c];

      float v0 = interp1D(p00, p10, fx, scaleX, modeX);
      float v1 = interp1D(p01, p11, fx, scaleX, modeX);
      float result = interp1D(v0, v1, fy, scaleY, modeY);

      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = result;
    }
  }
};

#endif // BILINEAR_V2_INTERPOLATOR_CUH
