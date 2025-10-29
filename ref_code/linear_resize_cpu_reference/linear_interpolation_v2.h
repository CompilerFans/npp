#ifndef LINEAR_INTERPOLATION_V2_H
#define LINEAR_INTERPOLATION_V2_H

#include <cmath>
#include <algorithm>
#include <cstdint>

// V2 Implementation: Based on kunzmi/mpp standard bilinear interpolation
// Uses separable filtering approach: horizontal then vertical
// Always uses bilinear interpolation for both upscale and downscale

namespace LinearInterpolationV2 {

// Border handling: replicate edge pixels (clamp coordinates)
template<typename T>
static inline T clampedPixel(const T* pSrc, int nSrcStep, int srcWidth, int srcHeight,
                             int x, int y, int channel, int channels) {
    x = std::max(0, std::min(srcWidth - 1, x));
    y = std::max(0, std::min(srcHeight - 1, y));

    const T* srcRow = (const T*)((const char*)pSrc + y * nSrcStep);
    return srcRow[x * channels + channel];
}

// Standard bilinear interpolation using separable filtering
// This follows the mpp approach: horizontal interpolation first, then vertical
template<typename T>
static float interpolatePixelSeparable(const T* pSrc, int nSrcStep, int srcWidth, int srcHeight,
                                       float srcX, float srcY, int channel, int channels) {
    // Compute integer coordinates using floor
    float x0_f = std::floor(srcX);
    float y0_f = std::floor(srcY);
    int x0 = static_cast<int>(x0_f);
    int y0 = static_cast<int>(y0_f);

    // Compute fractional parts
    float diffX = srcX - x0_f;
    float diffY = srcY - y0_f;

    // Get four corner pixels
    float pixel00 = clampedPixel(pSrc, nSrcStep, srcWidth, srcHeight, x0,     y0,     channel, channels);
    float pixel01 = clampedPixel(pSrc, nSrcStep, srcWidth, srcHeight, x0 + 1, y0,     channel, channels);
    float pixel10 = clampedPixel(pSrc, nSrcStep, srcWidth, srcHeight, x0,     y0 + 1, channel, channels);
    float pixel11 = clampedPixel(pSrc, nSrcStep, srcWidth, srcHeight, x0 + 1, y0 + 1, channel, channels);

    // Horizontal interpolation at y0
    float tmp0 = pixel00 + (pixel01 - pixel00) * diffX;

    // Horizontal interpolation at y0+1
    float tmp1 = pixel10 + (pixel11 - pixel10) * diffX;

    // Vertical interpolation
    float result = tmp0 + (tmp1 - tmp0) * diffY;

    return result;
}

// Float to integer conversion with rounding (mimics NearestTiesToEven)
template<typename T>
static inline T convertToInt(float value) {
    // Clamp to valid range
    if (value < 0.0f) value = 0.0f;
    if (value > 255.0f) value = 255.0f;

    // Round to nearest, ties to even
    float rounded = std::round(value);
    return static_cast<T>(rounded);
}

// Main resize function using standard bilinear interpolation
// Always uses bilinear for both upscale and downscale (mpp approach)
template<typename T>
static void resize(const T* pSrc, int nSrcStep, int srcWidth, int srcHeight,
                  T* pDst, int nDstStep, int dstWidth, int dstHeight,
                  int channels = 1) {
    // Compute scale factors
    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;

    // Process each destination pixel
    for (int dy = 0; dy < dstHeight; dy++) {
        for (int dx = 0; dx < dstWidth; dx++) {
            // Pixel-center coordinate mapping: (d + 0.5) * scale - 0.5
            float srcX = (dx + 0.5f) * scaleX - 0.5f;
            float srcY = (dy + 0.5f) * scaleY - 0.5f;

            T* dstRow = (T*)((char*)pDst + dy * nDstStep);

            // Always use bilinear interpolation (mpp approach)
            for (int c = 0; c < channels; c++) {
                float result = interpolatePixelSeparable(pSrc, nSrcStep, srcWidth, srcHeight,
                                                        srcX, srcY, c, channels);
                dstRow[dx * channels + c] = convertToInt<T>(result);
            }
        }
    }
}

// Grayscale resize (single channel)
template<typename T>
static void resizeGray(const T* pSrc, int nSrcStep, int srcWidth, int srcHeight,
                      T* pDst, int nDstStep, int dstWidth, int dstHeight) {
    resize(pSrc, nSrcStep, srcWidth, srcHeight,
           pDst, nDstStep, dstWidth, dstHeight, 1);
}

// RGB resize (3 channels)
template<typename T>
static void resizeRGB(const T* pSrc, int nSrcStep, int srcWidth, int srcHeight,
                     T* pDst, int nDstStep, int dstWidth, int dstHeight) {
    resize(pSrc, nSrcStep, srcWidth, srcHeight,
           pDst, nDstStep, dstWidth, dstHeight, 3);
}

} // namespace LinearInterpolationV2

#endif // LINEAR_INTERPOLATION_V2_H
