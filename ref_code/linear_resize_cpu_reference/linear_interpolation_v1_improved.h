#ifndef LINEAR_INTERPOLATION_V1_IMPROVED_H
#define LINEAR_INTERPOLATION_V1_IMPROVED_H

#include <algorithm>
#include <cmath>

// V1 Improved: Enhanced NPP-matching with fixes for problematic cases
//
// Key improvements:
// 1. Per-dimension algorithm selection (X and Y handled independently)
// 2. Edge-to-edge coordinate mapping for large downscales
// 3. Better handling of non-uniform scaling

template<typename T>
class LinearInterpolationV1Improved {
public:
    // Main resize function with improved algorithm selection
    static void resize(const T* pSrc, int nSrcStep, int srcWidth, int srcHeight,
                      T* pDst, int nDstStep, int dstWidth, int dstHeight,
                      int channels = 1) {

        float scaleX = (float)srcWidth / dstWidth;
        float scaleY = (float)srcHeight / dstHeight;

        // Determine coordinate mapping mode
        bool useLargeDownscaleMode = (scaleX >= 2.0f && scaleY >= 2.0f);

        // Per-dimension algorithm decision
        bool interpX = (scaleX < 1.0f);  // Upscale X: use interpolation
        bool interpY = (scaleY < 1.0f);  // Upscale Y: use interpolation

        for (int dy = 0; dy < dstHeight; dy++) {
            for (int dx = 0; dx < dstWidth; dx++) {
                T* dstRow = (T*)((char*)pDst + dy * nDstStep);

                // Coordinate mapping
                float srcX, srcY;
                if (useLargeDownscaleMode) {
                    // Large downscale (>= 2x): use edge-to-edge mapping
                    srcX = dx * scaleX;
                    srcY = dy * scaleY;
                } else {
                    // Standard: use pixel-center mapping
                    srcX = (dx + 0.5f) * scaleX - 0.5f;
                    srcY = (dy + 0.5f) * scaleY - 0.5f;
                }

                // Process each channel
                for (int c = 0; c < channels; c++) {
                    float result;

                    if (interpX && interpY) {
                        // Both dimensions upscale: full bilinear
                        result = bilinearInterpolate(pSrc, nSrcStep, srcWidth, srcHeight,
                                                    srcX, srcY, c, channels);
                    }
                    else if (interpX && !interpY) {
                        // X upscale, Y downscale: interpolate in X, floor in Y
                        result = interpolateXFloorY(pSrc, nSrcStep, srcWidth, srcHeight,
                                                    srcX, srcY, c, channels);
                    }
                    else if (!interpX && interpY) {
                        // X downscale, Y upscale: floor in X, interpolate in Y
                        result = floorXInterpolateY(pSrc, nSrcStep, srcWidth, srcHeight,
                                                    srcX, srcY, c, channels);
                    }
                    else {
                        // Both downscale: floor method
                        result = floorBoth(pSrc, nSrcStep, srcWidth, srcHeight,
                                          srcX, srcY, c, channels);
                    }

                    dstRow[dx * channels + c] = convertToInt(result);
                }
            }
        }
    }

private:
    // Full bilinear interpolation (both dimensions)
    static float bilinearInterpolate(const T* pSrc, int nSrcStep,
                                     int srcWidth, int srcHeight,
                                     float srcX, float srcY,
                                     int channel, int channels) {
        // Clamp coordinates
        srcX = std::max(0.0f, std::min((float)(srcWidth - 1), srcX));
        srcY = std::max(0.0f, std::min((float)(srcHeight - 1), srcY));

        int x0 = (int)std::floor(srcX);
        int y0 = (int)std::floor(srcY);
        int x1 = std::min(x0 + 1, srcWidth - 1);
        int y1 = std::min(y0 + 1, srcHeight - 1);

        float fx = srcX - x0;
        float fy = srcY - y0;

        const T* row0 = (const T*)((const char*)pSrc + y0 * nSrcStep);
        const T* row1 = (const T*)((const char*)pSrc + y1 * nSrcStep);

        float p00 = (float)row0[x0 * channels + channel];
        float p10 = (float)row0[x1 * channels + channel];
        float p01 = (float)row1[x0 * channels + channel];
        float p11 = (float)row1[x1 * channels + channel];

        float result = p00 * (1.0f - fx) * (1.0f - fy) +
                      p10 * fx * (1.0f - fy) +
                      p01 * (1.0f - fx) * fy +
                      p11 * fx * fy;

        return result;
    }

    // Interpolate in X, floor in Y
    static float interpolateXFloorY(const T* pSrc, int nSrcStep,
                                    int srcWidth, int srcHeight,
                                    float srcX, float srcY,
                                    int channel, int channels) {
        // Clamp coordinates
        srcX = std::max(0.0f, std::min((float)(srcWidth - 1), srcX));
        srcY = std::max(0.0f, std::min((float)(srcHeight - 1), srcY));

        int x0 = (int)std::floor(srcX);
        int y = (int)std::floor(srcY);
        int x1 = std::min(x0 + 1, srcWidth - 1);

        float fx = srcX - x0;

        const T* row = (const T*)((const char*)pSrc + y * nSrcStep);

        float p0 = (float)row[x0 * channels + channel];
        float p1 = (float)row[x1 * channels + channel];

        return p0 * (1.0f - fx) + p1 * fx;
    }

    // Floor in X, interpolate in Y
    static float floorXInterpolateY(const T* pSrc, int nSrcStep,
                                    int srcWidth, int srcHeight,
                                    float srcX, float srcY,
                                    int channel, int channels) {
        // Clamp coordinates
        srcX = std::max(0.0f, std::min((float)(srcWidth - 1), srcX));
        srcY = std::max(0.0f, std::min((float)(srcHeight - 1), srcY));

        int x = (int)std::floor(srcX);
        int y0 = (int)std::floor(srcY);
        int y1 = std::min(y0 + 1, srcHeight - 1);

        float fy = srcY - y0;

        const T* row0 = (const T*)((const char*)pSrc + y0 * nSrcStep);
        const T* row1 = (const T*)((const char*)pSrc + y1 * nSrcStep);

        float p0 = (float)row0[x * channels + channel];
        float p1 = (float)row1[x * channels + channel];

        return p0 * (1.0f - fy) + p1 * fy;
    }

    // Floor in both dimensions
    static float floorBoth(const T* pSrc, int nSrcStep,
                          int srcWidth, int srcHeight,
                          float srcX, float srcY,
                          int channel, int channels) {
        int x = (int)std::floor(srcX);
        int y = (int)std::floor(srcY);

        // Clamp to valid range
        x = std::max(0, std::min(srcWidth - 1, x));
        y = std::max(0, std::min(srcHeight - 1, y));

        const T* row = (const T*)((const char*)pSrc + y * nSrcStep);
        return (float)row[x * channels + channel];
    }

    // Convert float to integer with rounding
    static T convertToInt(float value) {
        // Clamp to valid range
        if (value < 0.0f) value = 0.0f;
        if (value > 255.0f) value = 255.0f;
        return (T)(value + 0.5f);
    }
};

#endif // LINEAR_INTERPOLATION_V1_IMPROVED_H
