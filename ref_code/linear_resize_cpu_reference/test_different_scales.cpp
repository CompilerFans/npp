#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <npp.h>

void testScale(int srcW, int srcH, int dstW, int dstH) {
    // Use simple pattern [0, 255] in both dimensions
    std::vector<unsigned char> src(srcW * srcH, 0);

    // Create gradient
    for (int y = 0; y < srcH; y++) {
        for (int x = 0; x < srcW; x++) {
            src[y * srcW + x] = (unsigned char)(
                (x * 255 / std::max(1, srcW - 1)) * 0.5f +
                (y * 255 / std::max(1, srcH - 1)) * 0.5f
            );
        }
    }

    // Get NVIDIA NPP result
    int channels = 1;
    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, srcW * srcH * channels);
    cudaMalloc(&d_dst, dstW * dstH * channels);
    cudaMemcpy(d_src, src.data(), srcW * srcH * channels, cudaMemcpyHostToDevice);

    NppiSize srcSize = {srcW, srcH};
    NppiSize dstSize = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    nppiResize_8u_C1R(d_src, srcW * channels, srcSize, srcROI,
                      d_dst, dstW * channels, dstSize, dstROI,
                      NPPI_INTER_LINEAR);

    std::vector<unsigned char> nppResult(dstW * dstH * channels);
    cudaMemcpy(nppResult.data(), d_dst, dstW * dstH * channels, cudaMemcpyDeviceToHost);

    // Analyze a few interesting pixels
    float scaleX = (float)srcW / dstW;
    float scaleY = (float)srcH / dstH;

    std::cout << "\n" << srcW << "x" << srcH << " → " << dstW << "x" << dstH
              << " (scale: " << std::fixed << std::setprecision(3) << scaleX << "x" << scaleY << ")\n";
    std::cout << std::string(100, '-') << "\n";

    int sampleCount = 0;
    for (int dy = 0; dy < dstH && sampleCount < 5; dy++) {
        for (int dx = 0; dx < dstW && sampleCount < 5; dx++) {
            float srcX = (dx + 0.5f) * scaleX - 0.5f;
            float srcY = (dy + 0.5f) * scaleY - 0.5f;
            srcX = std::max(0.0f, std::min((float)(srcW - 1), srcX));
            srcY = std::max(0.0f, std::min((float)(srcH - 1), srcY));

            int x0 = (int)std::floor(srcX);
            int y0 = (int)std::floor(srcY);
            float fx = srcX - x0;
            float fy = srcY - y0;

            if (fx < 0.01f && fy < 0.01f) continue;  // Skip corners

            int x1 = std::min(x0 + 1, srcW - 1);
            int y1 = std::min(y0 + 1, srcH - 1);

            int p00 = src[y0 * srcW + x0];
            int p10 = src[y0 * srcW + x1];
            int p01 = src[y1 * srcW + x0];
            int p11 = src[y1 * srcW + x1];

            int npp = nppResult[dy * dstW + dx];

            std::cout << "Dst[" << dx << "," << dy << "] fx=" << std::setprecision(3) << fx
                      << " fy=" << fy << " | p00=" << std::setw(3) << p00
                      << " p10=" << std::setw(3) << p10 << " p01=" << std::setw(3) << p01
                      << " p11=" << std::setw(3) << p11 << " → NPP=" << std::setw(3) << npp;

            // Try to find modifier for 1D cases
            if (std::abs(fy) < 0.01f && std::abs(p10 - p00) > 1) {
                float expected_std = p00 * (1 - fx) + p10 * fx;
                float effective_fx = (npp - p00) / (float)(p10 - p00);
                float modifier = effective_fx / fx;
                std::cout << " | 1D-X: mod=" << std::setprecision(4) << modifier;
            } else if (std::abs(fx) < 0.01f && std::abs(p01 - p00) > 1) {
                float expected_std = p00 * (1 - fy) + p01 * fy;
                float effective_fy = (npp - p00) / (float)(p01 - p00);
                float modifier = effective_fy / fy;
                std::cout << " | 1D-Y: mod=" << std::setprecision(4) << modifier;
            }

            std::cout << "\n";
            sampleCount++;
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    std::cout << "Testing Different Scales to Find Modifier Pattern\n";
    std::cout << "==================================================\n";

    // Test different upscale factors to get different fx/fy values
    testScale(2, 2, 3, 3);   // fx=fy=0.5
    testScale(3, 3, 4, 4);   // fx=fy=0.25
    testScale(4, 4, 5, 5);   // fx=fy=0.2
    testScale(2, 2, 5, 5);   // fx=fy=0.125
    testScale(5, 5, 7, 7);   // fx=fy≈0.4286

    return 0;
}
