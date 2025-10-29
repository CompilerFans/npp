#ifndef LINEAR_INTERPOLATION_V1_ULTIMATE_H
#define LINEAR_INTERPOLATION_V1_ULTIMATE_H

#include <algorithm>
#include <cmath>

// V1 Ultimate: Complete NPP-matching implementation
//
// Based on reverse engineering of NVIDIA NPP behavior:
// 1. Upscale (scale < 1.0): Standard bilinear interpolation
// 2. Large downscale (scale >= 2.0): Edge-to-edge mapping + floor
// 3. Fractional downscale (1.0 <= scale < 2.0): Threshold-based hybrid
//    - Uses floor when fx <= threshold (0.5/scale)
//    - Uses attenuated interpolation when fx > threshold
//    - Attenuation factor = 1.0/scale
// 4. Non-uniform scaling: Handle X and Y dimensions independently

template<typename T>
class LinearInterpolationV1Ultimate {
public:
    static void resize(const T* pSrc, int nSrcStep, int srcWidth, int srcHeight,
                      T* pDst, int nDstStep, int dstWidth, int dstHeight,
                      int channels = 1) {

        float scaleX = (float)srcWidth / dstWidth;
        float scaleY = (float)srcHeight / dstHeight;

        for (int dy = 0; dy < dstHeight; dy++) {
            for (int dx = 0; dx < dstWidth; dx++) {
                T* dstRow = (T*)((char*)pDst + dy * nDstStep);

                for (int c = 0; c < channels; c++) {
                    float result = resizePixel(pSrc, nSrcStep, srcWidth, srcHeight,
                                              dx, dy, c, channels,
                                              scaleX, scaleY);
                    dstRow[dx * channels + c] = convertToInt(result);
                }
            }
        }
    }

private:
    static float resizePixel(const T* pSrc, int nSrcStep,
                            int srcWidth, int srcHeight,
                            int dx, int dy,
                            int channel, int channels,
                            float scaleX, float scaleY) {

        // Determine algorithm mode for each dimension
        enum Mode { UPSCALE, FRACTIONAL_DOWN, LARGE_DOWN };

        Mode modeX = (scaleX < 1.0f) ? UPSCALE :
                    (scaleX >= 2.0f) ? LARGE_DOWN : FRACTIONAL_DOWN;
        Mode modeY = (scaleY < 1.0f) ? UPSCALE :
                    (scaleY >= 2.0f) ? LARGE_DOWN : FRACTIONAL_DOWN;

        // Calculate source coordinates based on mode
        float srcX, srcY;

        if (modeX == LARGE_DOWN && modeY == LARGE_DOWN) {
            // Both large downscale: edge-to-edge
            srcX = dx * scaleX;
            srcY = dy * scaleY;
        } else {
            // Any other combination: pixel-center
            srcX = (dx + 0.5f) * scaleX - 0.5f;
            srcY = (dy + 0.5f) * scaleY - 0.5f;
        }

        // Handle different mode combinations
        if (modeX == UPSCALE && modeY == UPSCALE) {
            // Both upscale: standard bilinear
            return bilinearInterpolate(pSrc, nSrcStep, srcWidth, srcHeight,
                                      srcX, srcY, channel, channels);
        }
        else if (modeX == UPSCALE && modeY == FRACTIONAL_DOWN) {
            // X upscale, Y fractional down
            return interpolateXFractionalY(pSrc, nSrcStep, srcWidth, srcHeight,
                                          srcX, srcY, channel, channels, scaleY);
        }
        else if (modeX == UPSCALE && modeY == LARGE_DOWN) {
            // X upscale, Y large down
            return interpolateXFloorY(pSrc, nSrcStep, srcWidth, srcHeight,
                                     srcX, srcY, channel, channels);
        }
        else if (modeX == FRACTIONAL_DOWN && modeY == UPSCALE) {
            // X fractional down, Y upscale
            return fractionalXInterpolateY(pSrc, nSrcStep, srcWidth, srcHeight,
                                          srcX, srcY, channel, channels, scaleX);
        }
        else if (modeX == FRACTIONAL_DOWN && modeY == FRACTIONAL_DOWN) {
            // Both fractional downscale
            return fractionalBoth(pSrc, nSrcStep, srcWidth, srcHeight,
                                 srcX, srcY, channel, channels, scaleX, scaleY);
        }
        else if (modeX == FRACTIONAL_DOWN && modeY == LARGE_DOWN) {
            // X fractional down, Y large down
            return fractionalXFloorY(pSrc, nSrcStep, srcWidth, srcHeight,
                                    srcX, srcY, channel, channels, scaleX);
        }
        else if (modeX == LARGE_DOWN && modeY == UPSCALE) {
            // X large down, Y upscale
            return floorXInterpolateY(pSrc, nSrcStep, srcWidth, srcHeight,
                                     srcX, srcY, channel, channels);
        }
        else if (modeX == LARGE_DOWN && modeY == FRACTIONAL_DOWN) {
            // X large down, Y fractional down
            return floorXFractionalY(pSrc, nSrcStep, srcWidth, srcHeight,
                                    srcX, srcY, channel, channels, scaleY);
        }
        else {
            // Both large downscale: floor both
            return floorBoth(pSrc, nSrcStep, srcWidth, srcHeight,
                           srcX, srcY, channel, channels);
        }
    }

    // Standard bilinear interpolation (both dimensions)
    static float bilinearInterpolate(const T* pSrc, int nSrcStep,
                                    int srcWidth, int srcHeight,
                                    float srcX, float srcY,
                                    int channel, int channels) {
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

        return p00 * (1.0f - fx) * (1.0f - fy) +
               p10 * fx * (1.0f - fy) +
               p01 * (1.0f - fx) * fy +
               p11 * fx * fy;
    }

    // Fractional downscale in one dimension
    static float fractionalInterp1D(float v0, float v1, float f, float scale) {
        float threshold = 0.5f / scale;
        if (f <= threshold) {
            return v0;  // Floor method
        } else {
            float attenuation = 1.0f / scale;
            float f_attenuated = f * attenuation;
            return v0 * (1.0f - f_attenuated) + v1 * f_attenuated;
        }
    }

    // Fractional downscale in both dimensions
    static float fractionalBoth(const T* pSrc, int nSrcStep,
                               int srcWidth, int srcHeight,
                               float srcX, float srcY,
                               int channel, int channels,
                               float scaleX, float scaleY) {
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

        // Apply fractional algorithm in X direction
        float v0 = fractionalInterp1D(p00, p10, fx, scaleX);  // Top
        float v1 = fractionalInterp1D(p01, p11, fx, scaleX);  // Bottom

        // Apply fractional algorithm in Y direction
        return fractionalInterp1D(v0, v1, fy, scaleY);
    }

    // Interpolate X, fractional Y
    static float interpolateXFractionalY(const T* pSrc, int nSrcStep,
                                        int srcWidth, int srcHeight,
                                        float srcX, float srcY,
                                        int channel, int channels,
                                        float scaleY) {
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

        // Standard interpolation in X
        float v0 = p00 * (1.0f - fx) + p10 * fx;
        float v1 = p01 * (1.0f - fx) + p11 * fx;

        // Fractional algorithm in Y
        return fractionalInterp1D(v0, v1, fy, scaleY);
    }

    // Fractional X, interpolate Y
    static float fractionalXInterpolateY(const T* pSrc, int nSrcStep,
                                        int srcWidth, int srcHeight,
                                        float srcX, float srcY,
                                        int channel, int channels,
                                        float scaleX) {
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

        // Fractional algorithm in X
        float v0 = fractionalInterp1D(p00, p10, fx, scaleX);
        float v1 = fractionalInterp1D(p01, p11, fx, scaleX);

        // Standard interpolation in Y
        return v0 * (1.0f - fy) + v1 * fy;
    }

    // Fractional X, floor Y
    static float fractionalXFloorY(const T* pSrc, int nSrcStep,
                                   int srcWidth, int srcHeight,
                                   float srcX, float srcY,
                                   int channel, int channels,
                                   float scaleX) {
        srcX = std::max(0.0f, std::min((float)(srcWidth - 1), srcX));
        srcY = std::max(0.0f, std::min((float)(srcHeight - 1), srcY));

        int x0 = (int)std::floor(srcX);
        int y = (int)std::floor(srcY);
        int x1 = std::min(x0 + 1, srcWidth - 1);

        float fx = srcX - x0;

        const T* row = (const T*)((const char*)pSrc + y * nSrcStep);

        float p0 = (float)row[x0 * channels + channel];
        float p1 = (float)row[x1 * channels + channel];

        return fractionalInterp1D(p0, p1, fx, scaleX);
    }

    // Floor X, fractional Y
    static float floorXFractionalY(const T* pSrc, int nSrcStep,
                                   int srcWidth, int srcHeight,
                                   float srcX, float srcY,
                                   int channel, int channels,
                                   float scaleY) {
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

        return fractionalInterp1D(p0, p1, fy, scaleY);
    }

    // Interpolate X, floor Y
    static float interpolateXFloorY(const T* pSrc, int nSrcStep,
                                   int srcWidth, int srcHeight,
                                   float srcX, float srcY,
                                   int channel, int channels) {
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

    // Floor X, interpolate Y
    static float floorXInterpolateY(const T* pSrc, int nSrcStep,
                                   int srcWidth, int srcHeight,
                                   float srcX, float srcY,
                                   int channel, int channels) {
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

    // Floor both dimensions
    static float floorBoth(const T* pSrc, int nSrcStep,
                          int srcWidth, int srcHeight,
                          float srcX, float srcY,
                          int channel, int channels) {
        int x = (int)std::floor(srcX);
        int y = (int)std::floor(srcY);

        x = std::max(0, std::min(srcWidth - 1, x));
        y = std::max(0, std::min(srcHeight - 1, y));

        const T* row = (const T*)((const char*)pSrc + y * nSrcStep);
        return (float)row[x * channels + channel];
    }

    static T convertToInt(float value) {
        if (value < 0.0f) value = 0.0f;
        if (value > 255.0f) value = 255.0f;
        return (T)(value + 0.5f);
    }
};

#endif // LINEAR_INTERPOLATION_V1_ULTIMATE_H
