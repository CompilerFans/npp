/**
 * Debug Test: Detailed coordinate mapping analysis
 *
 * Purpose: Trace exact coordinate mapping and interpolation values
 */

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_geometry_transforms.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

void printSrcImage(const std::vector<Npp8u>& src, int width, int height, int channels) {
    std::cout << "Source image (" << width << "x" << height << ", " << channels << " ch):\n";
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << "[";
            for (int c = 0; c < channels; c++) {
                int idx = (y * width + x) * channels + c;
                std::cout << std::setw(3) << (int)src[idx];
                if (c < channels - 1) std::cout << ",";
            }
            std::cout << "] ";
        }
        std::cout << "\n";
    }
}

void printDstImage(const std::vector<Npp8u>& dst, int width, int height, int channels) {
    std::cout << "Destination image (" << width << "x" << height << ", " << channels << " ch):\n";
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << "[";
            for (int c = 0; c < channels; c++) {
                int idx = (y * width + x) * channels + c;
                std::cout << std::setw(3) << (int)dst[idx];
                if (c < channels - 1) std::cout << ",";
            }
            std::cout << "] ";
        }
        std::cout << "\n";
    }
}

void testCase(int srcW, int srcH, int dstW, int dstH, const char* desc) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Test: " << desc << "\n";
    std::cout << srcW << "x" << srcH << " -> " << dstW << "x" << dstH << "\n";

    float scaleX = (float)srcW / dstW;
    float scaleY = (float)srcH / dstH;
    std::cout << "Scale: " << scaleX << " x " << scaleY << "\n\n";

    // Create simple pattern: sequential values
    int channels = 1;
    std::vector<Npp8u> srcData(srcW * srcH * channels);
    for (int i = 0; i < srcW * srcH * channels; i++) {
        srcData[i] = (Npp8u)(i * 10);  // 0, 10, 20, 30, ...
    }

    printSrcImage(srcData, srcW, srcH, channels);

    // Allocate device memory
    Npp8u* d_src;
    Npp8u* d_dst;
    size_t srcStep = srcW * channels * sizeof(Npp8u);
    size_t dstStep = dstW * channels * sizeof(Npp8u);

    cudaMalloc(&d_src, srcH * srcStep);
    cudaMalloc(&d_dst, dstH * dstStep);
    cudaMemcpy(d_src, srcData.data(), srcH * srcStep, cudaMemcpyHostToDevice);

    // NPP resize
    NppiSize srcSize = {srcW, srcH};
    NppiSize dstSize = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    NppStatus status = nppiResize_8u_C1R(d_src, srcStep, srcSize, srcROI,
                                         d_dst, dstStep, dstSize, dstROI,
                                         NPPI_INTER_LINEAR);

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP Error: " << status << "\n";
        cudaFree(d_src);
        cudaFree(d_dst);
        return;
    }

    // Get result
    std::vector<Npp8u> result(dstW * dstH * channels);
    cudaMemcpy(result.data(), d_dst, dstH * dstStep, cudaMemcpyDeviceToHost);

    std::cout << "\nNPP Result:\n";
    printDstImage(result, dstW, dstH, channels);

    // Show expected coordinates
    std::cout << "\nCoordinate mapping (pixel center formula):\n";
    for (int dy = 0; dy < dstH; dy++) {
        for (int dx = 0; dx < dstW; dx++) {
            float srcX = (dx + 0.5f) * scaleX - 0.5f;
            float srcY = (dy + 0.5f) * scaleY - 0.5f;

            int idx = dy * dstW + dx;
            std::cout << "dst(" << dx << "," << dy << ")=" << std::setw(3) << (int)result[idx]
                     << " <- src(" << std::setw(6) << std::fixed << std::setprecision(2)
                     << srcX << "," << srcY << ")\n";
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    std::cout << "Linear Interpolation Debug Analysis\n";
    std::cout << "Testing with simple sequential patterns\n";

    // Test 1: Simple 2x2 -> 4x4 upscale
    testCase(2, 2, 4, 4, "2x2 -> 4x4 (2x upscale)");

    // Test 2: Simple 3x3 -> 6x6 upscale
    testCase(3, 3, 6, 6, "3x3 -> 6x6 (2x upscale)");

    // Test 3: Simple 4x4 -> 2x2 downscale
    testCase(4, 4, 2, 2, "4x4 -> 2x2 (0.5x downscale)");

    // Test 4: Simple 2x2 -> 3x3 fractional upscale
    testCase(2, 2, 3, 3, "2x2 -> 3x3 (1.5x upscale)");

    return 0;
}
