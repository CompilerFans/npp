/**
 * Detailed pixel-level analysis of 2x2 -> 3x3 upscale
 *
 * This test manually calculates every interpolation step
 * to understand the exact difference between CPU and NPP
 */

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_geometry_transforms.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

void analyzePixel(int dx, int dy, const std::vector<Npp8u>& nppResult,
                  int dstW, int srcData[2][2]) {
    int srcW = 2, srcH = 2;
    int dstH = 3;

    float scaleX = (float)srcW / dstW;  // 0.6667
    float scaleY = (float)srcH / dstH;  // 0.6667

    // Pixel center mapping
    float srcX = (dx + 0.5f) * scaleX - 0.5f;
    float srcY = (dy + 0.5f) * scaleY - 0.5f;

    // Clamp
    float srcX_clamped = std::max(0.0f, std::min(1.0f, srcX));
    float srcY_clamped = std::max(0.0f, std::min(1.0f, srcY));

    // Get integer coordinates
    int x0 = (int)std::floor(srcX_clamped);
    int y0 = (int)std::floor(srcY_clamped);
    int x1 = std::min(x0 + 1, 1);
    int y1 = std::min(y0 + 1, 1);

    // Get fractional parts
    float fx = srcX_clamped - x0;
    float fy = srcY_clamped - y0;

    // Get 4 neighbors
    float p00 = srcData[y0][x0];
    float p10 = srcData[y0][x1];
    float p01 = srcData[y1][x0];
    float p11 = srcData[y1][x1];

    std::cout << "\n=== dst(" << dx << "," << dy << ") ===\n";
    std::cout << "Source coordinate (before clamp): (" << srcX << ", " << srcY << ")\n";
    std::cout << "Source coordinate (after clamp):  (" << srcX_clamped << ", " << srcY_clamped << ")\n";
    std::cout << "Integer coordinates: x0=" << x0 << ", y0=" << y0
              << ", x1=" << x1 << ", y1=" << y1 << "\n";
    std::cout << "Fractional parts: fx=" << std::fixed << std::setprecision(6) << fx
              << ", fy=" << fy << "\n";
    std::cout << "Four neighbors:\n";
    std::cout << "  p00 = src[" << y0 << "][" << x0 << "] = " << (int)p00 << "\n";
    std::cout << "  p10 = src[" << y0 << "][" << x1 << "] = " << (int)p10 << "\n";
    std::cout << "  p01 = src[" << y1 << "][" << x0 << "] = " << (int)p01 << "\n";
    std::cout << "  p11 = src[" << y1 << "][" << x1 << "] = " << (int)p11 << "\n";

    // Standard bilinear interpolation
    float w00 = (1 - fx) * (1 - fy);
    float w10 = fx * (1 - fy);
    float w01 = (1 - fx) * fy;
    float w11 = fx * fy;

    std::cout << "Weights:\n";
    std::cout << "  w00 = (1-fx)*(1-fy) = " << w00 << "\n";
    std::cout << "  w10 = fx*(1-fy)     = " << w10 << "\n";
    std::cout << "  w01 = (1-fx)*fy     = " << w01 << "\n";
    std::cout << "  w11 = fx*fy         = " << w11 << "\n";
    std::cout << "  Sum of weights = " << (w00 + w10 + w01 + w11) << "\n";

    float bilinear = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11;

    std::cout << "Calculation:\n";
    std::cout << "  " << (int)p00 << "*" << w00 << " + "
              << (int)p10 << "*" << w10 << " + "
              << (int)p01 << "*" << w01 << " + "
              << (int)p11 << "*" << w11 << "\n";
    std::cout << "  = " << (p00*w00) << " + " << (p10*w10) << " + "
              << (p01*w01) << " + " << (p11*w11) << "\n";
    std::cout << "  = " << bilinear << "\n";

    int cpu_round_half_up = (int)(bilinear + 0.5f);
    int cpu_truncate = (int)bilinear;
    int cpu_floor = (int)std::floor(bilinear);
    int cpu_ceil = (int)std::ceil(bilinear);

    int npp_value = (int)nppResult[dy * dstW + dx];

    std::cout << "Rounding options:\n";
    std::cout << "  Round half up: " << cpu_round_half_up;
    if (cpu_round_half_up == npp_value) std::cout << " ✓ MATCH";
    std::cout << "\n";
    std::cout << "  Truncate:      " << cpu_truncate;
    if (cpu_truncate == npp_value) std::cout << " ✓ MATCH";
    std::cout << "\n";
    std::cout << "  Floor:         " << cpu_floor;
    if (cpu_floor == npp_value) std::cout << " ✓ MATCH";
    std::cout << "\n";
    std::cout << "  Ceil:          " << cpu_ceil;
    if (cpu_ceil == npp_value) std::cout << " ✓ MATCH";
    std::cout << "\n";

    std::cout << "NPP result: " << npp_value << "\n";
    std::cout << "Difference: " << (cpu_round_half_up - npp_value) << "\n";

    // Try alternative calculations
    // Maybe NPP uses different order of operations?
    float alt1 = p00 + (p10 - p00) * fx;  // Horizontal interpolation at y0
    float alt2 = p01 + (p11 - p01) * fx;  // Horizontal interpolation at y1
    float alt_result = alt1 + (alt2 - alt1) * fy;  // Vertical interpolation

    std::cout << "\nAlternative calculation (horizontal then vertical):\n";
    std::cout << "  Horizontal at y0: " << p00 << " + (" << p10 << " - " << p00 << ") * " << fx << " = " << alt1 << "\n";
    std::cout << "  Horizontal at y1: " << p01 << " + (" << p11 << " - " << p01 << ") * " << fx << " = " << alt2 << "\n";
    std::cout << "  Vertical: " << alt1 << " + (" << alt2 << " - " << alt1 << ") * " << fy << " = " << alt_result << "\n";
    std::cout << "  Rounded: " << (int)(alt_result + 0.5f);
    if ((int)(alt_result + 0.5f) == npp_value) std::cout << " ✓ MATCH";
    std::cout << "\n";
}

int main() {
    std::cout << "Detailed Analysis: 2x2 -> 3x3 Upscale\n";
    std::cout << "=====================================\n\n";

    // Source data: simple 2x2
    int srcData[2][2] = {
        {0, 10},
        {20, 30}
    };

    std::cout << "Source image:\n";
    std::cout << "[  0] [ 10]\n";
    std::cout << "[ 20] [ 30]\n\n";

    // Prepare for NPP
    int srcW = 2, srcH = 2;
    int dstW = 3, dstH = 3;
    int channels = 1;

    std::vector<Npp8u> srcVec(srcW * srcH);
    for (int y = 0; y < srcH; y++) {
        for (int x = 0; x < srcW; x++) {
            srcVec[y * srcW + x] = srcData[y][x];
        }
    }

    // Allocate device memory
    Npp8u* d_src;
    Npp8u* d_dst;
    size_t srcStep = srcW * sizeof(Npp8u);
    size_t dstStep = dstW * sizeof(Npp8u);

    cudaMalloc(&d_src, srcH * srcStep);
    cudaMalloc(&d_dst, dstH * dstStep);
    cudaMemcpy(d_src, srcVec.data(), srcH * srcStep, cudaMemcpyHostToDevice);

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
        return 1;
    }

    // Get NPP result
    std::vector<Npp8u> nppResult(dstW * dstH);
    cudaMemcpy(nppResult.data(), d_dst, dstH * dstStep, cudaMemcpyDeviceToHost);

    std::cout << "NPP Result:\n";
    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            std::cout << "[" << std::setw(3) << (int)nppResult[y * dstW + x] << "] ";
        }
        std::cout << "\n";
    }

    // Analyze each pixel that doesn't match
    std::cout << "\n\nDETAILED PIXEL ANALYSIS\n";
    std::cout << "=======================\n";

    // Focus on pixels that showed differences
    analyzePixel(1, 0, nppResult, dstW, srcData);  // Expected 4
    analyzePixel(0, 1, nppResult, dstW, srcData);  // Expected 8
    analyzePixel(1, 1, nppResult, dstW, srcData);  // Expected 13
    analyzePixel(2, 1, nppResult, dstW, srcData);  // Expected 18
    analyzePixel(1, 2, nppResult, dstW, srcData);  // Expected 24

    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
