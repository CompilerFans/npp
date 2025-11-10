#ifndef SUPER_V2_INTERPOLATOR_CUH
#define SUPER_V2_INTERPOLATOR_CUH

#include "interpolator_common.cuh"

// Super sampling V2 interpolation (weighted box filter, based on kunzmi/mpp)
// This version uses fractional edge pixel weights for accurate downsampling
//
// Boundary Handling Note:
// Unlike fixed-kernel neighborhood operations (e.g., 3×3 box filter) that require border
// expansion, super sampling for downsampling does NOT need nppiCopyConstBorder because:
// 1. Output size < input size (downsampling property)
// 2. Sampling regions naturally fit within input boundaries
// 3. Each output pixel samples from [center - scale/2, center + scale/2),
//    which is mathematically guaranteed to be within [ROI.x, ROI.x + ROI.width) for valid coordinates
// The boundary checks in this code are defensive programming for edge cases and floating-point rounding.
template <typename T, int CHANNELS> struct SuperSamplingV2Interpolator {
  __device__ static void interpolate(const T *pSrc, int nSrcStep, const NppiRect &oSrcRectROI, int dx, int dy,
                                     float scaleX, float scaleY, T *pDst, int nDstStep, const NppiRect &oDstRectROI) {
    // In kunzmi/mpp algorithm, the sampling region is calculated in a unified coordinate space
    // The destination pixel (dx, dy) maps to source space via: dx * scaleX
    // For downsampling (scaleX > 1), the sampling region has half-width = scaleX * 0.5

    // Calculate the center of sampling region in source space
    float srcCenterX = (dx + 0.5f) * scaleX + oSrcRectROI.x;
    float srcCenterY = (dy + 0.5f) * scaleY + oSrcRectROI.y;

    // Half-width of sampling region in source space
    float halfScaleX = scaleX * 0.5f;
    float halfScaleY = scaleY * 0.5f;

    // Calculate sampling region bounds
    float xMin = srcCenterX - halfScaleX;
    float xMax = srcCenterX + halfScaleX;
    float yMin = srcCenterY - halfScaleY;
    float yMax = srcCenterY + halfScaleY;

    // Integer bounds (pixels fully or partially inside sampling region)
    int xMinInt = (int)ceilf(xMin);
    int xMaxInt = (int)floorf(xMax);
    int yMinInt = (int)ceilf(yMin);
    int yMaxInt = (int)floorf(yMax);

    // Clamp to source ROI
    // Note: xMaxInt/yMaxInt are exclusive bounds (used with < in loops)
    xMinInt = max(xMinInt, oSrcRectROI.x);
    xMaxInt = min(xMaxInt, oSrcRectROI.x + oSrcRectROI.width);
    yMinInt = max(yMinInt, oSrcRectROI.y);
    yMaxInt = min(yMaxInt, oSrcRectROI.y + oSrcRectROI.height);

    // Edge weights (fractional contribution of edge pixels)
    float wxMin = ceilf(xMin) - xMin;
    float wxMax = xMax - floorf(xMax);
    float wyMin = ceilf(yMin) - yMin;
    float wyMax = yMax - floorf(yMax);

    // Handle edge cases where sampling region is within a single pixel
    if (wxMin > 1.0f)
      wxMin = 1.0f;
    if (wxMax > 1.0f)
      wxMax = 1.0f;
    if (wyMin > 1.0f)
      wyMin = 1.0f;
    if (wyMax > 1.0f)
      wyMax = 1.0f;

    float sum[CHANNELS];
#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      sum[c] = 0.0f;
    }

    // Process top edge pixels (y = yMinInt - 1)
    if (wyMin > 0.0f && yMinInt > oSrcRectROI.y) {
      int y = yMinInt - 1;
      const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);

      // Top-left corner
      if (wxMin > 0.0f && xMinInt > oSrcRectROI.x) {
        int x = xMinInt - 1;
        float weight = wxMin * wyMin;
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sum[c] += src_row[x * CHANNELS + c] * weight;
        }
      }

      // Top edge
      for (int x = xMinInt; x < xMaxInt; x++) { // Changed <= to <
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sum[c] += src_row[x * CHANNELS + c] * wyMin;
        }
      }

      // Top-right corner
      if (wxMax > 0.0f && xMaxInt < oSrcRectROI.x + oSrcRectROI.width) {
        int x = xMaxInt;
        float weight = wxMax * wyMin;
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sum[c] += src_row[x * CHANNELS + c] * weight;
        }
      }
    }

    // Process middle rows
    for (int y = yMinInt; y < yMaxInt; y++) { // Changed <= to <
      const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);

      // Left edge
      if (wxMin > 0.0f && xMinInt > oSrcRectROI.x) {
        int x = xMinInt - 1;
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sum[c] += src_row[x * CHANNELS + c] * wxMin;
        }
      }

      // Center pixels (fully contained)
      for (int x = xMinInt; x < xMaxInt; x++) { // Changed <= to <
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sum[c] += src_row[x * CHANNELS + c];
        }
      }

      // Right edge
      if (wxMax > 0.0f && xMaxInt < oSrcRectROI.x + oSrcRectROI.width) {
        int x = xMaxInt;
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sum[c] += src_row[x * CHANNELS + c] * wxMax;
        }
      }
    }

    // Process bottom edge pixels (y = yMaxInt)
    if (wyMax > 0.0f && yMaxInt < oSrcRectROI.y + oSrcRectROI.height) { // Changed bound check
      int y = yMaxInt;                                                  // Changed from yMaxInt + 1
      const T *src_row = (const T *)((const char *)pSrc + y * nSrcStep);

      // Bottom-left corner
      if (wxMin > 0.0f && xMinInt > oSrcRectROI.x) {
        int x = xMinInt - 1;
        float weight = wxMin * wyMax;
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sum[c] += src_row[x * CHANNELS + c] * weight;
        }
      }

      // Bottom edge
      for (int x = xMinInt; x < xMaxInt; x++) { // Changed <= to <
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sum[c] += src_row[x * CHANNELS + c] * wyMax;
        }
      }

      // Bottom-right corner
      if (wxMax > 0.0f && xMaxInt < oSrcRectROI.x + oSrcRectROI.width) { // Changed bound check
        int x = xMaxInt;                                                 // Changed from xMaxInt + 1
        float weight = wxMax * wyMax;
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sum[c] += src_row[x * CHANNELS + c] * weight;
        }
      }
    }

    // Normalize by total weight (scaleX * scaleY)
    float totalWeight = scaleX * scaleY;
    float invWeight = 1.0f / totalWeight;

    T *dst_row = (T *)((char *)pDst + (dy + oDstRectROI.y) * nDstStep);

#pragma unroll
    for (int c = 0; c < CHANNELS; c++) {
      dst_row[(dx + oDstRectROI.x) * CHANNELS + c] = (T)(sum[c] * invWeight + 0.5f);
    }
  }
};

#endif // SUPER_V2_INTERPOLATOR_CUH
