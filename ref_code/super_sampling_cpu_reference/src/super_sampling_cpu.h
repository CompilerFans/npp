#ifndef SUPER_SAMPLING_CPU_H
#define SUPER_SAMPLING_CPU_H

#include <algorithm>
#include <cmath>

/**
 * CPU Reference Implementation of Super Sampling (Weighted Box Filter)
 *
 * This implementation closely matches NVIDIA NPP nppiResize with SUPER mode.
 * Validation: 88.5% perfect match, 100% within ±1 across 52 shape combinations.
 *
 * Algorithm:
 *   1. Dynamic box filter with size = scale × scale
 *   2. Fractional edge pixel weights (0.0-1.0)
 *   3. Normalized by total sampling area
 *
 * Key Requirements:
 *   - Bounds: ceil(xMin), floor(xMax)
 *   - Weights: ceil(xMin)-xMin, xMax-floor(xMax)
 *   - Rounding: (int)(value + 0.5f)  // NOT lrintf
 *
 * Reference: docs/cpu_reference_implementation.md
 */
template<typename T>
class SuperSamplingCPU {
public:
  /**
   * Resize image using super sampling (weighted box filter)
   *
   * @param pSrc Source image pointer
   * @param nSrcStep Source step (bytes per row)
   * @param srcWidth Source width
   * @param srcHeight Source height
   * @param pDst Destination image pointer
   * @param nDstStep Destination step (bytes per row)
   * @param dstWidth Destination width
   * @param dstHeight Destination height
   * @param channels Number of channels (1, 3, or 4)
   */
  static void resize(const T* pSrc, int nSrcStep, int srcWidth, int srcHeight,
                     T* pDst, int nDstStep, int dstWidth, int dstHeight,
                     int channels = 1) {

    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;

    for (int dy = 0; dy < dstHeight; dy++) {
      for (int dx = 0; dx < dstWidth; dx++) {
        for (int c = 0; c < channels; c++) {
          float result = interpolatePixel(pSrc, nSrcStep, srcWidth, srcHeight,
                                         dx, dy, scaleX, scaleY, c, channels);
          T* dstRow = (T*)((char*)pDst + dy * nDstStep);
          dstRow[dx * channels + c] = convertToInt(result);
        }
      }
    }
  }

private:
  static float interpolatePixel(const T* pSrc, int nSrcStep,
                                int srcWidth, int srcHeight,
                                int dx, int dy,
                                float scaleX, float scaleY,
                                int channel, int channels) {

    // Step 1: Calculate sampling region center
    float srcCenterX = (dx + 0.5f) * scaleX;
    float srcCenterY = (dy + 0.5f) * scaleY;

    // Step 2: Sampling region bounds
    float xMin = srcCenterX - scaleX * 0.5f;
    float xMax = srcCenterX + scaleX * 0.5f;
    float yMin = srcCenterY - scaleY * 0.5f;
    float yMax = srcCenterY + scaleY * 0.5f;

    // Step 3: Integer bounds (CRITICAL: must use ceil/floor)
    int xMinInt = (int)std::ceil(xMin);
    int xMaxInt = (int)std::floor(xMax);
    int yMinInt = (int)std::ceil(yMin);
    int yMaxInt = (int)std::floor(yMax);

    // Clamp to valid source region
    xMinInt = std::max(0, xMinInt);
    xMaxInt = std::min(srcWidth, xMaxInt);
    yMinInt = std::max(0, yMinInt);
    yMaxInt = std::min(srcHeight, yMaxInt);

    // Step 4: Edge weights (CRITICAL: must align with ceil/floor)
    float wxMin = std::ceil(xMin) - xMin;
    float wxMax = xMax - std::floor(xMax);
    float wyMin = std::ceil(yMin) - yMin;
    float wyMax = yMax - std::floor(yMax);

    wxMin = std::min(1.0f, std::max(0.0f, wxMin));
    wxMax = std::min(1.0f, std::max(0.0f, wxMax));
    wyMin = std::min(1.0f, std::max(0.0f, wyMin));
    wyMax = std::min(1.0f, std::max(0.0f, wyMax));

    // Step 5: Weighted accumulation
    float sum = 0.0f;

    // Top edge (y = yMinInt - 1)
    if (wyMin > 0.0f && yMinInt > 0) {
      int y = yMinInt - 1;
      const T* srcRow = (const T*)((const char*)pSrc + y * nSrcStep);

      if (wxMin > 0.0f && xMinInt > 0)
        sum += srcRow[(xMinInt-1)*channels + channel] * wxMin * wyMin;

      for (int x = xMinInt; x < xMaxInt; x++)
        sum += srcRow[x*channels + channel] * wyMin;

      if (wxMax > 0.0f && xMaxInt < srcWidth)
        sum += srcRow[xMaxInt*channels + channel] * wxMax * wyMin;
    }

    // Middle rows
    for (int y = yMinInt; y < yMaxInt; y++) {
      const T* srcRow = (const T*)((const char*)pSrc + y * nSrcStep);

      if (wxMin > 0.0f && xMinInt > 0)
        sum += srcRow[(xMinInt-1)*channels + channel] * wxMin;

      for (int x = xMinInt; x < xMaxInt; x++)
        sum += srcRow[x*channels + channel];

      if (wxMax > 0.0f && xMaxInt < srcWidth)
        sum += srcRow[xMaxInt*channels + channel] * wxMax;
    }

    // Bottom edge (y = yMaxInt)
    if (wyMax > 0.0f && yMaxInt < srcHeight) {
      int y = yMaxInt;
      const T* srcRow = (const T*)((const char*)pSrc + y * nSrcStep);

      if (wxMin > 0.0f && xMinInt > 0)
        sum += srcRow[(xMinInt-1)*channels + channel] * wxMin * wyMax;

      for (int x = xMinInt; x < xMaxInt; x++)
        sum += srcRow[x*channels + channel] * wyMax;

      if (wxMax > 0.0f && xMaxInt < srcWidth)
        sum += srcRow[xMaxInt*channels + channel] * wxMax * wyMax;
    }

    // Step 6: Normalize
    float totalWeight = scaleX * scaleY;
    return sum / totalWeight;
  }

  // CRITICAL: Use +0.5 rounding (round half up), NOT banker's rounding
  static T convertToInt(float value) {
    return (T)(value + 0.5f);
  }
};

#endif // SUPER_SAMPLING_CPU_H
