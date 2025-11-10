#ifndef LINEAR_RESIZE_REFACTORED_H
#define LINEAR_RESIZE_REFACTORED_H

#include <algorithm>
#include <cmath>
#include <cstdint>

/**
 * Linear Interpolation Resize - Refactored Clean Implementation
 *
 * Key improvements:
 * - Unified sampling logic with mode enum
 * - Eliminated code duplication
 * - Clearer separation of concerns
 * - Better performance (mode computed once)
 */
template<typename T>
class LinearResizeRefactored {
public:
    static void resize(const T* pSrc, int nSrcStep, int srcWidth, int srcHeight,
                      T* pDst, int nDstStep, int dstWidth, int dstHeight,
                      int channels = 1) {

        float scaleX = (float)srcWidth / dstWidth;
        float scaleY = (float)srcHeight / dstHeight;

        // Determine mode once
        Mode modeX = getMode(scaleX);
        Mode modeY = getMode(scaleY);
        bool useLargeDownscale = (modeX == LARGE_DOWN && modeY == LARGE_DOWN);

        for (int dy = 0; dy < dstHeight; dy++) {
            for (int dx = 0; dx < dstWidth; dx++) {
                // Compute source coordinates
                float srcX = useLargeDownscale ? dx * scaleX : (dx + 0.5f) * scaleX - 0.5f;
                float srcY = useLargeDownscale ? dy * scaleY : (dy + 0.5f) * scaleY - 0.5f;

                T* dstRow = (T*)((char*)pDst + dy * nDstStep);

                for (int c = 0; c < channels; c++) {
                    float value = sample2D(pSrc, nSrcStep, srcWidth, srcHeight,
                                          srcX, srcY, c, channels,
                                          modeX, modeY, scaleX, scaleY);
                    dstRow[dx * channels + c] = clamp(value);
                }
            }
        }
    }

private:
    enum Mode { UPSCALE, FRACTIONAL_DOWN, LARGE_DOWN };

    // Determine interpolation mode from scale factor
    static Mode getMode(float scale) {
        if (scale < 1.0f) return UPSCALE;
        if (scale < 2.0f) return FRACTIONAL_DOWN;
        return LARGE_DOWN;
    }

    // Unified 2D sampling
    static float sample2D(const T* pSrc, int nSrcStep,
                         int srcWidth, int srcHeight,
                         float srcX, float srcY,
                         int channel, int channels,
                         Mode modeX, Mode modeY,
                         float scaleX, float scaleY) {

        // Clamp and get neighbors
        srcX = std::max(0.0f, std::min((float)(srcWidth - 1), srcX));
        srcY = std::max(0.0f, std::min((float)(srcHeight - 1), srcY));

        int x0 = (int)std::floor(srcX);
        int y0 = (int)std::floor(srcY);
        int x1 = std::min(x0 + 1, srcWidth - 1);
        int y1 = std::min(y0 + 1, srcHeight - 1);

        float fx = srcX - x0;
        float fy = srcY - y0;

        // Get 4 corner pixels
        const T* row0 = (const T*)((const char*)pSrc + y0 * nSrcStep);
        const T* row1 = (const T*)((const char*)pSrc + y1 * nSrcStep);

        float p00 = row0[x0 * channels + channel];
        float p10 = row0[x1 * channels + channel];
        float p01 = row1[x0 * channels + channel];
        float p11 = row1[x1 * channels + channel];

        // Interpolate in X direction
        float v0 = interp1D(p00, p10, fx, scaleX, modeX);
        float v1 = interp1D(p01, p11, fx, scaleX, modeX);

        // Interpolate in Y direction
        return interp1D(v0, v1, fy, scaleY, modeY);
    }

    // Unified 1D interpolation with mode selection
    static float interp1D(float v0, float v1, float f, float scale, Mode mode) {
        switch (mode) {
            case UPSCALE:
                // Standard linear interpolation
                return v0 * (1.0f - f) + v1 * f;

            case FRACTIONAL_DOWN: {
                // Hybrid: threshold-based floor or attenuated interpolation
                float threshold = 0.5f / scale;
                if (f <= threshold) {
                    return v0;  // Floor
                } else {
                    float attenuation = 1.0f / scale;
                    float f_att = f * attenuation;
                    return v0 * (1.0f - f_att) + v1 * f_att;
                }
            }

            case LARGE_DOWN:
                // No interpolation, use floor
                return v0;

            default:
                return v0;
        }
    }

    // Clamp and round
    static T clamp(float value) {
        value = std::max(0.0f, std::min(255.0f, value));
        return (T)(value + 0.5f);
    }
};

#endif // LINEAR_RESIZE_REFACTORED_H
