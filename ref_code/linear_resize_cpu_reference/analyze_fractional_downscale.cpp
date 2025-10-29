#include <iostream>
#include <iomanip>
#include <cmath>
#include <npp.h>
#include "linear_interpolation_cpu.h"

// Analyze a specific fractional downscale with 2D input
void analyzeFractionalDownscale(const char* name, int srcW, int dstW, float expectedScale) {
    std::cout << "\n========================================\n";
    std::cout << "ANALYZING: " << name << " (" << srcW << " -> " << dstW << ")\n";
    std::cout << "Scale: " << std::fixed << std::setprecision(4) << expectedScale << "\n";
    std::cout << "========================================\n\n";

    // Create 2x2D pattern (use 2 rows for proper 2D, but analyze first row only)
    int srcH = 2, dstH = 2;
    uint8_t* srcData = new uint8_t[srcW * srcH];

    // First row: gradient 0-255
    for (int i = 0; i < srcW; i++) {
        srcData[i] = (i * 255) / (srcW - 1);
        srcData[srcW + i] = (i * 255) / (srcW - 1); // Same for row 2
    }

    std::cout << "Source (row 0): ";
    for (int i = 0; i < srcW; i++) {
        std::cout << std::setw(3) << (int)srcData[i] << " ";
    }
    std::cout << "\n\n";

    // NPP resize
    uint8_t* nppResult = new uint8_t[dstW * dstH];
    uint8_t* v1Result = new uint8_t[dstW * dstH];

    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcW * srcH);
    cudaMalloc(&d_dst, dstW * dstH);
    cudaMemcpy(d_src, srcData, srcW * srcH, cudaMemcpyHostToDevice);

    NppiSize srcSz = {srcW, srcH};
    NppiSize dstSz = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    nppiResize_8u_C1R(d_src, srcW, srcSz, srcROI,
                     d_dst, dstW, dstSz, dstROI,
                     NPPI_INTER_LINEAR);

    cudaMemcpy(nppResult, d_dst, dstW * dstH, cudaMemcpyDeviceToHost);

    // V1 resize
    LinearInterpolationCPU<uint8_t>::resize(srcData, srcW, srcW, srcH,
                                            v1Result, dstW, dstW, dstH, 1);

    std::cout << "NPP result (row 0): ";
    for (int i = 0; i < dstW; i++) {
        std::cout << std::setw(3) << (int)nppResult[i] << " ";
    }
    std::cout << "\n";

    std::cout << "V1 result (row 0):  ";
    for (int i = 0; i < dstW; i++) {
        std::cout << std::setw(3) << (int)v1Result[i] << " ";
    }
    std::cout << "\n\n";

    // Detailed per-pixel analysis
    float scale = (float)srcW / dstW;

    std::cout << "Per-pixel analysis:\n";
    std::cout << "Format: dst[i] -> NPP=X, V1=Y, diff=Z\n";
    std::cout << "        Coordinate mappings and interpolation tests\n\n";

    for (int dx = 0; dx < dstW; dx++) {
        // Standard pixel-center mapping
        float srcX_center = (dx + 0.5f) * scale - 0.5f;

        // Alternative: edge-to-edge (for large downscales >= 2.0)
        float srcX_edge = dx * scale;

        int x0 = (int)std::floor(srcX_center);
        int x1 = std::min(x0 + 1, srcW - 1);
        x0 = std::max(0, x0);

        float fx = srcX_center - (int)std::floor(srcX_center);

        uint8_t npp = nppResult[dx];
        uint8_t v1 = v1Result[dx];
        int diff = std::abs((int)npp - (int)v1);

        std::cout << "dst[" << dx << "]: NPP=" << std::setw(3) << (int)npp
                 << ", V1=" << std::setw(3) << (int)v1
                 << ", diff=" << std::setw(3) << diff;

        if (diff == 0) {
            std::cout << " ✓";
        } else {
            std::cout << " ✗";
        }
        std::cout << "\n";

        std::cout << "  srcX(center)=" << std::fixed << std::setprecision(3) << srcX_center;
        std::cout << ", srcX(edge)=" << srcX_edge << "\n";
        std::cout << "  floor(srcX)=" << x0 << ", neighbors=[" << (int)srcData[x0]
                 << ", " << (int)srcData[x1] << "], fx=" << fx << "\n";

        // Test different methods
        int floor_val = srcData[x0];
        float bilinear = srcData[x0] * (1 - fx) + srcData[x1] * fx;
        int bilinear_rounded = (int)(bilinear + 0.5f);

        std::cout << "  Methods: floor=" << floor_val
                 << ", bilinear=" << std::setprecision(2) << bilinear
                 << " (rounded=" << bilinear_rounded << ")\n";

        // Check which matches NPP
        if (npp == floor_val) {
            std::cout << "  >>> FLOOR method matches NPP! ✓\n";
        } else if (std::abs(npp - bilinear_rounded) <= 1) {
            std::cout << "  >>> BILINEAR matches NPP! ✓\n";
        } else {
            std::cout << "  >>> Neither matches - investigating...\n";

            // Try modified fractional values
            for (float test_fx = 0.0f; test_fx <= 1.0f; test_fx += 0.1f) {
                float test_val = srcData[x0] * (1 - test_fx) + srcData[x1] * test_fx;
                int test_rounded = (int)(test_val + 0.5f);
                if (test_rounded == npp) {
                    std::cout << "  >>> Found match with fx=" << std::fixed << std::setprecision(2)
                             << test_fx << " (original fx=" << fx << ")\n";
                    break;
                }
            }
        }
        std::cout << "\n";
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] srcData;
    delete[] nppResult;
    delete[] v1Result;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Fractional Downscale Deep Analysis\n";
    std::cout << "Using 2D input, analyzing first row\n";
    std::cout << "========================================\n";

    // Test all problematic fractional downscales
    analyzeFractionalDownscale("0.875x (7/8)", 8, 7, 0.875f);
    analyzeFractionalDownscale("0.8x (4/5)", 10, 8, 0.8f);
    analyzeFractionalDownscale("0.75x (3/4)", 8, 6, 0.75f);
    analyzeFractionalDownscale("0.67x (2/3)", 9, 6, 0.67f);
    analyzeFractionalDownscale("0.6x (3/5)", 10, 6, 0.6f);

    // Compare with working cases
    std::cout << "\n========================================\n";
    std::cout << "COMPARISON WITH WORKING CASES\n";
    std::cout << "========================================\n";

    analyzeFractionalDownscale("0.5x (1/2) [WORKING]", 8, 4, 0.5f);
    analyzeFractionalDownscale("0.4x (2/5)", 10, 4, 0.4f);
    analyzeFractionalDownscale("0.33x (1/3) [WORKING]", 12, 4, 0.33f);
    analyzeFractionalDownscale("0.25x (1/4) [WORKING]", 16, 4, 0.25f);

    return 0;
}
