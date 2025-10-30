#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <npp.h>

void analyze1DInterpolation(int p0, int p1, float fx, float scale) {
    // Create 1D test case
    std::vector<unsigned char> src(2);
    src[0] = p0;
    src[1] = p1;

    // Calculate destination size based on scale
    int dstW = (int)std::round(2.0f / scale);
    std::vector<unsigned char> dst(dstW);

    // Run NVIDIA NPP (need to make 2D image)
    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, 2 * 1);
    cudaMalloc(&d_dst, dstW * 1);
    cudaMemcpy(d_src, src.data(), 2, cudaMemcpyHostToDevice);

    NppiSize srcSize = {2, 1};
    NppiSize dstSize = {dstW, 1};
    NppiRect srcROI = {0, 0, 2, 1};
    NppiRect dstROI = {0, 0, dstW, 1};

    nppiResize_8u_C1R(d_src, 2, srcSize, srcROI,
                      d_dst, dstW, dstSize, dstROI,
                      NPPI_INTER_LINEAR);

    cudaMemcpy(dst.data(), d_dst, dstW, cudaMemcpyDeviceToHost);

    // Find the pixel that has the target fx
    float actual_scale = 2.0f / dstW;
    for (int dx = 0; dx < dstW; dx++) {
        float srcX = (dx + 0.5f) * actual_scale - 0.5f;
        srcX = std::max(0.0f, std::min(1.0f, srcX));

        int x0 = (int)std::floor(srcX);
        float calc_fx = srcX - x0;

        if (std::abs(calc_fx - fx) < 0.05f) {
            int npp_result = dst[dx];
            float std_bilinear = p0 * (1 - calc_fx) + p1 * calc_fx;

            std::cout << "p0=" << std::setw(3) << p0
                      << " p1=" << std::setw(3) << p1
                      << " scale=" << std::fixed << std::setprecision(3) << std::setw(5) << actual_scale
                      << " fx=" << std::setw(5) << std::setprecision(3) << calc_fx
                      << " | StdBilin=" << std::setw(6) << std::setprecision(2) << std_bilinear
                      << " NPP=" << std::setw(3) << npp_result
                      << " Diff=" << std::setw(5) << std::setprecision(2) << (std_bilinear - npp_result);

            if (std::abs(p1 - p0) > 1) {
                float fx_eff = (npp_result - p0) / (float)(p1 - p0);
                float modifier = fx_eff / calc_fx;
                std::cout << " | fx_eff=" << std::setw(6) << std::setprecision(4) << fx_eff
                          << " mod=" << std::setw(6) << std::setprecision(4) << modifier;
            }

            std::cout << "\n";
            break;
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    std::cout << "Final Analysis: Testing Simple 1D Cases\n";
    std::cout << "==========================================\n\n";

    // Test different patterns with fx=0.5 (2→3 scaling, scale=0.667)
    std::cout << "Testing fx=0.5 with scale=0.667 (2→3):\n";
    std::cout << std::string(100, '-') << "\n";
    analyze1DInterpolation(0, 100, 0.5, 0.667);
    analyze1DInterpolation(0, 127, 0.5, 0.667);
    analyze1DInterpolation(0, 255, 0.5, 0.667);
    analyze1DInterpolation(50, 150, 0.5, 0.667);
    analyze1DInterpolation(100, 200, 0.5, 0.667);
    analyze1DInterpolation(127, 255, 0.5, 0.667);

    // Test different fx values with same pattern (0→127)
    std::cout << "\nTesting pattern [0,127] with different scales/fx:\n";
    std::cout << std::string(100, '-') << "\n";
    analyze1DInterpolation(0, 127, 0.5, 0.667);   // 2→3
    analyze1DInterpolation(0, 127, 0.625, 0.75);  // 3→4
    analyze1DInterpolation(0, 127, 0.375, 0.75);  // 3→4
    analyze1DInterpolation(0, 127, 0.7, 0.8);     // 4→5
    analyze1DInterpolation(0, 127, 0.5, 0.8);     // 4→5
    analyze1DInterpolation(0, 127, 0.3, 0.8);     // 4→5

    // Test different patterns with fx=0.5
    std::cout << "\nTesting fx=0.5 with different patterns and scale=0.667:\n";
    std::cout << std::string(100, '-') << "\n";
    analyze1DInterpolation(0, 50, 0.5, 0.667);
    analyze1DInterpolation(0, 100, 0.5, 0.667);
    analyze1DInterpolation(0, 150, 0.5, 0.667);
    analyze1DInterpolation(0, 200, 0.5, 0.667);
    analyze1DInterpolation(0, 255, 0.5, 0.667);

    return 0;
}
