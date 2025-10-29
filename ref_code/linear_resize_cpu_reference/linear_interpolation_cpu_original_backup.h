/**
 * CPU Reference Implementation for Linear (Bilinear) Interpolation
 *
 * Purpose: High-accuracy CPU reference matching NVIDIA NPP nppiResize with LINEAR mode
 * Supports: Both upscaling and downscaling
 *
 * Algorithm:
 *   1. Coordinate mapping: srcCoord = (dstCoord + 0.5) * scale - 0.5
 *   2. UPSCALE (scale < 1.0): Bilinear interpolation using 4 neighboring pixels
 *   3. DOWNSCALE (scale >= 1.0): Floor method (no interpolation!)
 *   4. Boundary clamping to [0, width-1] and [0, height-1]
 *   5. Rounding: (int)(value + 0.5)
 *
 * Based on analysis: linear_interpolation_analysis.md
 */

#ifndef LINEAR_INTERPOLATION_CPU_H
#define LINEAR_INTERPOLATION_CPU_H

#include <algorithm>
#include <cmath>

template<typename T>
class LinearInterpolationCPU {
public:
    /**
     * Resize image using bilinear interpolation
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
                // Map destination pixel center to source space
                // Formula: srcCoord = (dstCoord + 0.5) * scale - 0.5
                float srcX = (dx + 0.5f) * scaleX - 0.5f;
                float srcY = (dy + 0.5f) * scaleY - 0.5f;

                T* dstRow = (T*)((char*)pDst + dy * nDstStep);

                // NPP uses bilinear for upscale, floor for downscale
                // For non-uniform scaling, each dimension is handled separately
                bool useInterpolation = (scaleX < 1.0f && scaleY < 1.0f);

                if (useInterpolation) {
                    // Both dimensions upscale: use bilinear interpolation
                    for (int c = 0; c < channels; c++) {
                        float result = interpolatePixel(pSrc, nSrcStep, srcWidth, srcHeight,
                                                       srcX, srcY, c, channels);
                        dstRow[dx * channels + c] = convertToInt(result);
                    }
                } else {
                    // At least one dimension downscale: use floor method
                    int x = (int)std::floor(srcX);
                    int y = (int)std::floor(srcY);

                    // Clamp to valid range
                    x = std::max(0, std::min(srcWidth - 1, x));
                    y = std::max(0, std::min(srcHeight - 1, y));

                    const T* srcRow = (const T*)((const char*)pSrc + y * nSrcStep);
                    for (int c = 0; c < channels; c++) {
                        dstRow[dx * channels + c] = srcRow[x * channels + c];
                    }
                }
            }
        }
    }

private:
    /**
     * Bilinear interpolation for a single pixel/channel
     */
    static float interpolatePixel(const T* pSrc, int nSrcStep,
                                  int srcWidth, int srcHeight,
                                  float srcX, float srcY,
                                  int channel, int channels) {

        // Clamp source coordinates to valid range [0, width-1] and [0, height-1]
        srcX = std::max(0.0f, std::min((float)(srcWidth - 1), srcX));
        srcY = std::max(0.0f, std::min((float)(srcHeight - 1), srcY));

        // Get integer coordinates and fractional parts
        int x0 = (int)std::floor(srcX);
        int y0 = (int)std::floor(srcY);

        // Ensure we don't go out of bounds for x1, y1
        int x1 = std::min(x0 + 1, srcWidth - 1);
        int y1 = std::min(y0 + 1, srcHeight - 1);

        // Fractional parts (0.0 to 1.0)
        float fx = srcX - x0;
        float fy = srcY - y0;

        // Get pointers to the two rows
        const T* row0 = (const T*)((const char*)pSrc + y0 * nSrcStep);
        const T* row1 = (const T*)((const char*)pSrc + y1 * nSrcStep);

        // Get the 4 neighboring pixel values
        float p00 = (float)row0[x0 * channels + channel];
        float p10 = (float)row0[x1 * channels + channel];
        float p01 = (float)row1[x0 * channels + channel];
        float p11 = (float)row1[x1 * channels + channel];

        // Bilinear interpolation formula:
        // result = p00 * (1-fx) * (1-fy) +
        //          p10 * fx     * (1-fy) +
        //          p01 * (1-fx) * fy     +
        //          p11 * fx     * fy
        float result = p00 * (1.0f - fx) * (1.0f - fy) +
                      p10 * fx * (1.0f - fy) +
                      p01 * (1.0f - fx) * fy +
                      p11 * fx * fy;

        return result;
    }

    /**
     * Convert float to integer with round half up
     */
    static T convertToInt(float value) {
        return (T)(value + 0.5f);
    }
};

#endif // LINEAR_INTERPOLATION_CPU_H
