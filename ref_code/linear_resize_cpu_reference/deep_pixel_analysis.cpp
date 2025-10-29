#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <npp.h>
#include "linear_interpolation_cpu.h"

// Deep analysis for a specific scale
void analyzeScale(const std::string& name, int srcW, int srcH, int dstW, int dstH) {
    std::cout << "\n========================================\n";
    std::cout << "DEEP ANALYSIS: " << name << "\n";
    std::cout << "Source: " << srcW << "x" << srcH << " -> Dst: " << dstW << "x" << dstH << "\n";
    std::cout << "Scale: " << std::fixed << std::setprecision(4)
             << (float)dstW/srcW << "x, " << (float)dstH/srcH << "x\n";
    std::cout << "========================================\n\n";

    // Create simple sequential pattern for easy analysis
    int srcSize = srcW * srcH;
    uint8_t* srcData = new uint8_t[srcSize];
    for (int i = 0; i < srcSize; i++) {
        srcData[i] = i % 256;
    }

    std::cout << "Source pattern (sequential 0-" << (srcSize-1) << "):\n";
    for (int y = 0; y < srcH; y++) {
        std::cout << "  Row " << y << ": ";
        for (int x = 0; x < srcW; x++) {
            std::cout << std::setw(3) << (int)srcData[y * srcW + x] << " ";
        }
        std::cout << "\n";
    }

    // NPP result
    int dstSize = dstW * dstH;
    uint8_t* nppResult = new uint8_t[dstSize];
    uint8_t* v1Result = new uint8_t[dstSize];

    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcSize);
    cudaMalloc(&d_dst, dstSize);
    cudaMemcpy(d_src, srcData, srcSize, cudaMemcpyHostToDevice);

    NppiSize srcSz = {srcW, srcH};
    NppiSize dstSz = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    nppiResize_8u_C1R(d_src, srcW, srcSz, srcROI,
                      d_dst, dstW, dstSz, dstROI,
                      NPPI_INTER_LINEAR);

    cudaMemcpy(nppResult, d_dst, dstSize, cudaMemcpyDeviceToHost);

    LinearInterpolationCPU<uint8_t>::resize(srcData, srcW, srcW, srcH,
                                            v1Result, dstW, dstW, dstH, 1);

    std::cout << "\nNPP Result:\n";
    for (int y = 0; y < dstH; y++) {
        std::cout << "  Row " << y << ": ";
        for (int x = 0; x < dstW; x++) {
            std::cout << std::setw(3) << (int)nppResult[y * dstW + x] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nV1 Result:\n";
    for (int y = 0; y < dstH; y++) {
        std::cout << "  Row " << y << ": ";
        for (int x = 0; x < dstW; x++) {
            std::cout << std::setw(3) << (int)v1Result[y * dstW + x] << " ";
        }
        std::cout << "\n";
    }

    // Detailed per-pixel analysis
    float scaleX = (float)srcW / dstW;
    float scaleY = (float)srcH / dstH;

    std::cout << "\nPer-pixel coordinate analysis:\n";
    std::cout << "Format: [NPP | V1 | diff] (coord, neighbors, calc)\n\n";

    for (int dy = 0; dy < dstH; dy++) {
        for (int dx = 0; dx < dstW; dx++) {
            // Standard pixel-center mapping
            float srcX = (dx + 0.5f) * scaleX - 0.5f;
            float srcY = (dy + 0.5f) * scaleY - 0.5f;

            // Alternative: edge-to-edge
            float srcX_edge = dx * scaleX;
            float srcY_edge = dy * scaleY;

            int x0 = (int)std::floor(srcX);
            int y0 = (int)std::floor(srcY);
            int x1 = std::min(x0 + 1, srcW - 1);
            int y1 = std::min(y0 + 1, srcH - 1);

            float fx = srcX - x0;
            float fy = srcY - y0;

            uint8_t npp = nppResult[dy * dstW + dx];
            uint8_t v1 = v1Result[dy * dstW + dx];
            int diff = std::abs((int)npp - (int)v1);

            std::cout << "dst(" << dx << "," << dy << "): "
                     << "[" << std::setw(3) << (int)npp << "|"
                     << std::setw(3) << (int)v1 << "|"
                     << std::setw(3) << diff << "]\n";

            std::cout << "  Pixel-center: srcX=" << std::fixed << std::setprecision(3)
                     << srcX << ", srcY=" << srcY << "\n";
            std::cout << "  Edge-to-edge: srcX=" << srcX_edge << ", srcY=" << srcY_edge << "\n";
            std::cout << "  Floor coords: (" << x0 << "," << y0 << ")\n";
            std::cout << "  Fractional: fx=" << fx << ", fy=" << fy << "\n";

            // Get 4 neighbors
            int p00 = srcData[y0 * srcW + x0];
            int p10 = srcData[y0 * srcW + x1];
            int p01 = srcData[y1 * srcW + x0];
            int p11 = srcData[y1 * srcW + x1];

            std::cout << "  4-neighbors: [" << p00 << " " << p10 << "]\n";
            std::cout << "               [" << p01 << " " << p11 << "]\n";

            // Try different interpolation methods
            float bilinear = p00 * (1-fx) * (1-fy) +
                           p10 * fx * (1-fy) +
                           p01 * (1-fx) * fy +
                           p11 * fx * fy;

            // Modified weights (from previous discovery)
            float fx_mod = (fx < 0.5f) ? (fx * 1.2f) : (0.6f + (fx-0.5f) * 0.8f);
            float fy_mod = (fy < 0.5f) ? (fy * 1.2f) : (0.6f + (fy-0.5f) * 0.8f);
            fx_mod = std::min(1.0f, fx_mod);
            fy_mod = std::min(1.0f, fy_mod);

            float bilinear_mod = p00 * (1-fx_mod) * (1-fy_mod) +
                               p10 * fx_mod * (1-fy_mod) +
                               p01 * (1-fx_mod) * fy_mod +
                               p11 * fx_mod * fy_mod;

            // Nearest neighbor
            int nearest = srcData[(int)std::round(srcY) * srcW + (int)std::round(srcX)];

            // Floor
            int floor_val = p00;

            std::cout << "  Standard bilinear: " << std::fixed << std::setprecision(2)
                     << bilinear << " -> " << (int)(bilinear + 0.5f) << "\n";
            std::cout << "  Modified weights:  " << bilinear_mod << " -> " << (int)(bilinear_mod + 0.5f) << "\n";
            std::cout << "  Nearest neighbor:  " << nearest << "\n";
            std::cout << "  Floor method:      " << floor_val << "\n";

            // Check which method matches NPP
            if (std::abs((int)(bilinear + 0.5f) - (int)npp) <= 1) {
                std::cout << "  >>> Standard bilinear matches! ✓\n";
            } else if (std::abs((int)(bilinear_mod + 0.5f) - (int)npp) <= 1) {
                std::cout << "  >>> Modified weights match! ✓\n";
            } else if (nearest == (int)npp) {
                std::cout << "  >>> Nearest neighbor matches! ✓\n";
            } else if (floor_val == (int)npp) {
                std::cout << "  >>> Floor method matches! ✓\n";
            } else {
                std::cout << "  >>> No match - unknown algorithm! ✗\n";

                // Try to reverse-engineer weights
                if (p00 != p10 && p00 != p01) {
                    // Try to solve for fx, fy assuming bilinear
                    // This is complex, let's just report
                    std::cout << "  >>> Attempting reverse calculation...\n";

                    // If p01 == p11 (same bottom row), we can isolate fx
                    if (p01 == p11 && p00 != p10) {
                        float actual_top = ((float)npp - p01 * fy) / (1 - fy);
                        float computed_fx = (actual_top - p00) / (p10 - p00);
                        std::cout << "  >>> Computed fx (assuming standard fy): " << computed_fx << "\n";
                    }
                }
            }

            std::cout << "\n";
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] srcData;
    delete[] nppResult;
    delete[] v1Result;
}

// Analyze fractional downscale pattern
void analyzeFractionalDownscale() {
    std::cout << "\n========================================\n";
    std::cout << "FRACTIONAL DOWNSCALE PATTERN ANALYSIS\n";
    std::cout << "========================================\n\n";

    struct TestCase {
        const char* name;
        int srcW, dstW;
        float scale;
    };

    TestCase tests[] = {
        {"0.875x (7/8)", 8, 7, 0.875f},
        {"0.8x (4/5)", 10, 8, 0.8f},
        {"0.75x (3/4)", 8, 6, 0.75f},
        {"0.67x (2/3)", 9, 6, 0.67f},
        {"0.6x (3/5)", 10, 6, 0.6f},
        {"0.5x (1/2)", 8, 4, 0.5f},
        {"0.4x (2/5)", 10, 4, 0.4f},
    };

    for (const auto& test : tests) {
        // Create gradient 0-255
        uint8_t srcData[32];
        for (int i = 0; i < test.srcW; i++) {
            srcData[i] = (i * 255) / (test.srcW - 1);
        }

        uint8_t nppResult[32];
        uint8_t *d_src, *d_dst;
        cudaMalloc(&d_src, test.srcW);
        cudaMalloc(&d_dst, test.dstW);
        cudaMemcpy(d_src, srcData, test.srcW, cudaMemcpyHostToDevice);

        NppiSize srcSz = {test.srcW, 1};
        NppiSize dstSz = {test.dstW, 1};
        NppiRect srcROI = {0, 0, test.srcW, 1};
        NppiRect dstROI = {0, 0, test.dstW, 1};

        nppiResize_8u_C1R(d_src, test.srcW, srcSz, srcROI,
                         d_dst, test.dstW, dstSz, dstROI,
                         NPPI_INTER_LINEAR);

        cudaMemcpy(nppResult, d_dst, test.dstW, cudaMemcpyDeviceToHost);

        std::cout << test.name << " [" << test.srcW << " -> " << test.dstW << "]:\n";
        std::cout << "  Src: ";
        for (int i = 0; i < test.srcW; i++) {
            std::cout << std::setw(3) << (int)srcData[i] << " ";
        }
        std::cout << "\n  NPP: ";
        for (int i = 0; i < test.dstW; i++) {
            std::cout << std::setw(3) << (int)nppResult[i] << " ";
        }
        std::cout << "\n";

        // Analyze coordinate mapping
        std::cout << "  Mapping analysis:\n";
        for (int dx = 0; dx < test.dstW; dx++) {
            float srcX_center = (dx + 0.5f) * test.scale - 0.5f;
            float srcX_edge = dx * test.scale;

            int x0 = (int)std::floor(srcX_center);
            float fx = srcX_center - x0;

            int npp = nppResult[dx];
            int floor_val = srcData[std::max(0, std::min(test.srcW-1, x0))];

            float bilinear = 0;
            if (x0 >= 0 && x0 < test.srcW - 1) {
                bilinear = srcData[x0] * (1 - fx) + srcData[x0+1] * fx;
            } else if (x0 < 0) {
                bilinear = srcData[0];
            } else {
                bilinear = srcData[test.srcW-1];
            }

            std::cout << "    dst[" << dx << "]: srcX=" << std::fixed << std::setprecision(3)
                     << srcX_center << " -> NPP=" << npp
                     << ", floor=" << floor_val
                     << ", bilinear=" << std::setprecision(1) << bilinear
                     << " (" << (int)(bilinear + 0.5f) << ")";

            if (npp == floor_val) {
                std::cout << " [FLOOR match]";
            } else if (std::abs(npp - (int)(bilinear + 0.5f)) <= 1) {
                std::cout << " [BILINEAR match]";
            } else {
                std::cout << " [NO match]";
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        cudaFree(d_src);
        cudaFree(d_dst);
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "DEEP PIXEL-LEVEL ANALYSIS\n";
    std::cout << "========================================\n";

    // Analyze problematic upscales
    analyzeScale("3x Upscale", 4, 4, 12, 12);
    analyzeScale("4x Upscale", 2, 2, 8, 8);

    // Analyze fractional downscales
    analyzeFractionalDownscale();

    return 0;
}
