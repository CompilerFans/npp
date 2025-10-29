#include <iostream>
#include <iomanip>
#include <cmath>
#include <npp.h>
#include "linear_interpolation_cpu.h"

// Debug 0.25x downscale issue
void debug_025x_downscale() {
    std::cout << "\n========================================\n";
    std::cout << "DEBUG: 0.25x Downscale (16x16 -> 4x4)\n";
    std::cout << "========================================\n\n";

    const int srcW = 16, srcH = 16;
    const int dstW = 4, dstH = 4;
    const int channels = 1;

    // Create simple pattern: sequential values 0, 1, 2, ..., 255
    uint8_t* srcData = new uint8_t[srcW * srcH];
    for (int i = 0; i < srcW * srcH; i++) {
        srcData[i] = i % 256;
    }

    std::cout << "Source 16x16 (showing first 4x4 and last 4x4):\n";
    for (int y = 0; y < 4; y++) {
        std::cout << "  Row " << y << ": ";
        for (int x = 0; x < 4; x++) {
            std::cout << std::setw(3) << (int)srcData[y * srcW + x] << " ";
        }
        std::cout << "... ";
        for (int x = srcW-4; x < srcW; x++) {
            std::cout << std::setw(3) << (int)srcData[y * srcW + x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "  ...\n";
    for (int y = srcH-4; y < srcH; y++) {
        std::cout << "  Row " << y << ": ";
        for (int x = 0; x < 4; x++) {
            std::cout << std::setw(3) << (int)srcData[y * srcW + x] << " ";
        }
        std::cout << "... ";
        for (int x = srcW-4; x < srcW; x++) {
            std::cout << std::setw(3) << (int)srcData[y * srcW + x] << " ";
        }
        std::cout << "\n";
    }

    // NPP result
    int srcStep = srcW * channels;
    int dstStep = dstW * channels;
    uint8_t* nppResult = new uint8_t[dstW * dstH];
    uint8_t* v1Result = new uint8_t[dstW * dstH];

    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcH * srcStep);
    cudaMalloc(&d_dst, dstH * dstStep);
    cudaMemcpy(d_src, srcData, srcH * srcStep, cudaMemcpyHostToDevice);

    NppiSize srcSize = {srcW, srcH};
    NppiSize dstSize = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    nppiResize_8u_C1R(d_src, srcStep, srcSize, srcROI,
                      d_dst, dstStep, dstSize, dstROI,
                      NPPI_INTER_LINEAR);

    cudaMemcpy(nppResult, d_dst, dstH * dstStep, cudaMemcpyDeviceToHost);

    // V1 result
    LinearInterpolationCPU<uint8_t>::resize(srcData, srcStep, srcW, srcH,
                                            v1Result, dstStep, dstW, dstH, channels);

    // Analyze each destination pixel
    std::cout << "\nDestination 4x4 analysis:\n\n";

    float scaleX = (float)srcW / dstW;  // 4.0
    float scaleY = (float)srcH / dstH;  // 4.0

    for (int dy = 0; dy < dstH; dy++) {
        for (int dx = 0; dx < dstW; dx++) {
            float srcX = (dx + 0.5f) * scaleX - 0.5f;
            float srcY = (dy + 0.5f) * scaleY - 0.5f;

            int floorX = (int)std::floor(srcX);
            int floorY = (int)std::floor(srcY);

            // Get V1's sample point
            int clampX = std::max(0, std::min(srcW - 1, floorX));
            int clampY = std::max(0, std::min(srcH - 1, floorY));
            uint8_t v1Sample = srcData[clampY * srcW + clampX];

            uint8_t npp = nppResult[dy * dstW + dx];
            uint8_t v1 = v1Result[dy * dstW + dx];

            std::cout << "dst(" << dx << "," << dy << "):\n";
            std::cout << "  srcX=" << std::fixed << std::setprecision(2) << srcX
                     << ", srcY=" << srcY << "\n";
            std::cout << "  floor: x=" << floorX << ", y=" << floorY << "\n";
            std::cout << "  V1 samples src[" << clampY << "][" << clampX << "] = "
                     << (int)v1Sample << "\n";
            std::cout << "  NPP=" << (int)npp << ", V1=" << (int)v1
                     << ", diff=" << std::abs((int)npp - (int)v1) << "\n\n";
        }
    }

    // Check if NPP might be using a different coordinate mapping
    std::cout << "Alternative coordinate mappings:\n\n";

    for (int dy = 0; dy < dstH; dy++) {
        for (int dx = 0; dx < dstW; dx++) {
            uint8_t npp = nppResult[dy * dstW + dx];

            // Try different mappings
            float mapping1 = dx * scaleX;  // Edge-to-edge
            float mapping2 = (dx + 0.5f) * scaleX;  // Pixel-center to edge
            float mapping3 = (dx + 0.5f) * scaleX - 0.5f;  // Standard (our V1)

            std::cout << "dst(" << dx << "," << dy << ") NPP=" << (int)npp << ":\n";
            std::cout << "  mapping1 (edge): " << mapping1 << " -> src["
                     << (int)mapping1 << "] = "
                     << (int)srcData[dy * 4 * srcW + (int)mapping1] << "\n";
            std::cout << "  mapping2 (center->edge): " << mapping2 << " -> src["
                     << (int)mapping2 << "] = "
                     << (int)srcData[dy * 4 * srcW + (int)mapping2] << "\n";
            std::cout << "  mapping3 (standard): " << mapping3 << " -> src["
                     << (int)mapping3 << "] = "
                     << (int)srcData[dy * 4 * srcW + (int)mapping3] << "\n\n";
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] srcData;
    delete[] nppResult;
    delete[] v1Result;
}

// Debug non-uniform scaling
void debug_nonuniform_scaling() {
    std::cout << "\n========================================\n";
    std::cout << "DEBUG: Non-uniform Scaling (4x8 -> 8x4)\n";
    std::cout << "Width: 4->8 (2x upscale), Height: 8->4 (0.5x downscale)\n";
    std::cout << "========================================\n\n";

    const int srcW = 4, srcH = 8;
    const int dstW = 8, dstH = 4;
    const int channels = 1;

    // Simple gradient pattern
    uint8_t* srcData = new uint8_t[srcW * srcH];
    for (int y = 0; y < srcH; y++) {
        for (int x = 0; x < srcW; x++) {
            srcData[y * srcW + x] = (x * 255 / (srcW - 1));
        }
    }

    std::cout << "Source 4x8:\n";
    for (int y = 0; y < srcH; y++) {
        std::cout << "  Row " << y << ": ";
        for (int x = 0; x < srcW; x++) {
            std::cout << std::setw(3) << (int)srcData[y * srcW + x] << " ";
        }
        std::cout << "\n";
    }

    // NPP and V1 results
    int srcStep = srcW * channels;
    int dstStep = dstW * channels;
    uint8_t* nppResult = new uint8_t[dstW * dstH];
    uint8_t* v1Result = new uint8_t[dstW * dstH];

    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcH * srcStep);
    cudaMalloc(&d_dst, dstH * dstStep);
    cudaMemcpy(d_src, srcData, srcH * srcStep, cudaMemcpyHostToDevice);

    NppiSize srcSize = {srcW, srcH};
    NppiSize dstSize = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    nppiResize_8u_C1R(d_src, srcStep, srcSize, srcROI,
                      d_dst, dstStep, dstSize, dstROI,
                      NPPI_INTER_LINEAR);

    cudaMemcpy(nppResult, d_dst, dstH * dstStep, cudaMemcpyDeviceToHost);

    LinearInterpolationCPU<uint8_t>::resize(srcData, srcStep, srcW, srcH,
                                            v1Result, dstStep, dstW, dstH, channels);

    std::cout << "\nDestination 8x4:\n";
    std::cout << "Format: [NPP | V1 | diff]\n\n";
    for (int y = 0; y < dstH; y++) {
        std::cout << "Row " << y << ": ";
        for (int x = 0; x < dstW; x++) {
            uint8_t npp = nppResult[y * dstW + x];
            uint8_t v1 = v1Result[y * dstW + x];
            int diff = std::abs((int)npp - (int)v1);
            std::cout << "[" << std::setw(3) << (int)npp << "|"
                     << std::setw(3) << (int)v1 << "|"
                     << std::setw(2) << diff << "] ";
        }
        std::cout << "\n";
    }

    // Analyze V1's decision
    float scaleX = (float)srcW / dstW;  // 0.5 (upscale)
    float scaleY = (float)srcH / dstH;  // 2.0 (downscale)
    bool v1UsesInterpolation = (scaleX < 1.0f && scaleY < 1.0f);

    std::cout << "\nV1 Analysis:\n";
    std::cout << "  scaleX = " << scaleX << " (upscale)\n";
    std::cout << "  scaleY = " << scaleY << " (downscale)\n";
    std::cout << "  useInterpolation = (" << scaleX << " < 1.0 && " << scaleY
             << " < 1.0) = " << (v1UsesInterpolation ? "true" : "false") << "\n";
    std::cout << "  V1 decision: " << (v1UsesInterpolation ? "Bilinear" : "Floor") << "\n";

    std::cout << "\nNPP likely behavior:\n";
    std::cout << "  X dimension (upscale): Bilinear interpolation\n";
    std::cout << "  Y dimension (downscale): Floor method\n";
    std::cout << "  -> Hybrid approach per dimension\n";

    // Sample detailed analysis for center pixel
    std::cout << "\nDetailed analysis for dst(4,2):\n";
    int dx = 4, dy = 2;
    float srcX = (dx + 0.5f) * scaleX - 0.5f;
    float srcY = (dy + 0.5f) * scaleY - 0.5f;

    std::cout << "  srcX = " << srcX << ", srcY = " << srcY << "\n";

    if (v1UsesInterpolation) {
        std::cout << "  V1 uses bilinear (wrong for this case)\n";
    } else {
        int x = (int)std::floor(srcX);
        int y = (int)std::floor(srcY);
        std::cout << "  V1 uses floor: src[" << y << "][" << x << "]\n";
    }

    // What NPP might do
    int x0 = (int)std::floor(srcX);
    int y0 = (int)std::floor(srcY);
    float fx = srcX - x0;

    std::cout << "  NPP likely does:\n";
    std::cout << "    X: bilinear between src[y][" << x0 << "] and src[y][" << (x0+1) << "]\n";
    std::cout << "    Y: floor to row " << y0 << "\n";
    std::cout << "    Result: " << (int)srcData[y0 * srcW + x0] << " * "
             << (1-fx) << " + " << (int)srcData[y0 * srcW + (x0+1)] << " * " << fx << "\n";

    float nppEstimate = srcData[y0 * srcW + x0] * (1-fx) + srcData[y0 * srcW + (x0+1)] * fx;
    std::cout << "    Estimated: " << nppEstimate << "\n";
    std::cout << "    Actual NPP: " << (int)nppResult[dy * dstW + dx] << "\n";
    std::cout << "    V1: " << (int)v1Result[dy * dstW + dx] << "\n";

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] srcData;
    delete[] nppResult;
    delete[] v1Result;
}

// Debug 0.75x downscale
void debug_075x_downscale() {
    std::cout << "\n========================================\n";
    std::cout << "DEBUG: 0.75x Downscale (8x8 -> 6x6)\n";
    std::cout << "========================================\n\n";

    const int srcW = 8, srcH = 8;
    const int dstW = 6, dstH = 6;
    const int channels = 1;

    // Simple pattern
    uint8_t* srcData = new uint8_t[srcW * srcH];
    for (int y = 0; y < srcH; y++) {
        for (int x = 0; x < srcW; x++) {
            srcData[y * srcW + x] = (x * 255 / (srcW - 1));
        }
    }

    std::cout << "Source 8x8:\n";
    for (int y = 0; y < srcH; y++) {
        std::cout << "  Row " << y << ": ";
        for (int x = 0; x < srcW; x++) {
            std::cout << std::setw(3) << (int)srcData[y * srcW + x] << " ";
        }
        std::cout << "\n";
    }

    // Results
    int srcStep = srcW * channels;
    int dstStep = dstW * channels;
    uint8_t* nppResult = new uint8_t[dstW * dstH];
    uint8_t* v1Result = new uint8_t[dstW * dstH];

    uint8_t *d_src, *d_dst;
    cudaMalloc(&d_src, srcH * srcStep);
    cudaMalloc(&d_dst, dstH * dstStep);
    cudaMemcpy(d_src, srcData, srcH * srcStep, cudaMemcpyHostToDevice);

    NppiSize srcSize = {srcW, srcH};
    NppiSize dstSize = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    nppiResize_8u_C1R(d_src, srcStep, srcSize, srcROI,
                      d_dst, dstStep, dstSize, dstROI,
                      NPPI_INTER_LINEAR);

    cudaMemcpy(nppResult, d_dst, dstH * dstStep, cudaMemcpyDeviceToHost);

    LinearInterpolationCPU<uint8_t>::resize(srcData, srcStep, srcW, srcH,
                                            v1Result, dstStep, dstW, dstH, channels);

    std::cout << "\nDestination 6x6:\n";
    std::cout << "Format: [NPP | V1 | diff]\n\n";
    for (int y = 0; y < dstH; y++) {
        std::cout << "Row " << y << ": ";
        for (int x = 0; x < dstW; x++) {
            uint8_t npp = nppResult[y * dstW + x];
            uint8_t v1 = v1Result[y * dstW + x];
            int diff = std::abs((int)npp - (int)v1);
            std::cout << "[" << std::setw(3) << (int)npp << "|"
                     << std::setw(3) << (int)v1 << "|"
                     << std::setw(3) << diff << "] ";
        }
        std::cout << "\n";
    }

    // Analyze coordinate mapping
    float scaleX = (float)srcW / dstW;  // 1.333...
    float scaleY = (float)srcH / dstH;  // 1.333...

    std::cout << "\nScale analysis:\n";
    std::cout << "  scaleX = " << scaleX << " (downscale)\n";
    std::cout << "  scaleY = " << scaleY << " (downscale)\n";
    std::cout << "  V1 uses: Floor method\n";

    std::cout << "\nCoordinate mapping for each dst pixel:\n";
    for (int dx = 0; dx < dstW; dx++) {
        float srcX = (dx + 0.5f) * scaleX - 0.5f;
        int floorX = (int)std::floor(srcX);
        uint8_t npp = nppResult[0 * dstW + dx];
        uint8_t v1 = v1Result[0 * dstW + dx];

        std::cout << "  dst[" << dx << "]: srcX=" << std::fixed << std::setprecision(3)
                 << srcX << " -> floor=" << floorX
                 << " (src=" << (int)srcData[floorX] << ")"
                 << " | NPP=" << (int)npp << ", V1=" << (int)v1 << "\n";
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] srcData;
    delete[] nppResult;
    delete[] v1Result;
}

int main() {
    std::cout << "==============================================\n";
    std::cout << "V1 Problem Case Debugging\n";
    std::cout << "==============================================\n";

    debug_025x_downscale();
    debug_nonuniform_scaling();
    debug_075x_downscale();

    return 0;
}
