#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <npp.h>

struct InterpolationData {
    float fx, fy;
    int p00, p10, p01, p11;
    int npp_result;
    float std_bilinear;

    // Derived values
    float v0_std, v1_std;
    int delta_x0, delta_x1;  // p10-p00, p11-p01
    int delta_y0, delta_y1;  // p01-p00, p11-p10

    // Try to reverse-engineer effective weights
    float effective_fx, effective_fy;
};

void analyzePattern(const std::vector<unsigned char>& src, int srcW, int srcH,
                   int dstW, int dstH, const std::string& name) {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "Pattern: " << name << "\n";
    std::cout << "Source: ";
    for (auto v : src) std::cout << (int)v << " ";
    std::cout << "\n" << std::string(100, '=') << "\n\n";

    int channels = 1;

    // Get NVIDIA NPP result
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

    // Analyze
    float scaleX = (float)srcW / dstW;
    float scaleY = (float)srcH / dstH;

    std::vector<InterpolationData> data;

    for (int dy = 0; dy < dstH; dy++) {
        for (int dx = 0; dx < dstW; dx++) {
            int idx = dy * dstW + dx;

            float srcX = (dx + 0.5f) * scaleX - 0.5f;
            float srcY = (dy + 0.5f) * scaleY - 0.5f;
            srcX = std::max(0.0f, std::min((float)(srcW - 1), srcX));
            srcY = std::max(0.0f, std::min((float)(srcH - 1), srcY));

            int x0 = (int)std::floor(srcX);
            int y0 = (int)std::floor(srcY);
            int x1 = std::min(x0 + 1, srcW - 1);
            int y1 = std::min(y0 + 1, srcH - 1);

            float fx = srcX - x0;
            float fy = srcY - y0;

            if (fx < 0.001f && fy < 0.001f) continue;  // Skip trivial cases

            InterpolationData d;
            d.fx = fx;
            d.fy = fy;
            d.p00 = src[y0 * srcW + x0];
            d.p10 = src[y0 * srcW + x1];
            d.p01 = src[y1 * srcW + x0];
            d.p11 = src[y1 * srcW + x1];
            d.npp_result = nppResult[idx];

            d.v0_std = d.p00 * (1 - fx) + d.p10 * fx;
            d.v1_std = d.p01 * (1 - fx) + d.p11 * fx;
            d.std_bilinear = d.v0_std * (1 - fy) + d.v1_std * fy;

            d.delta_x0 = d.p10 - d.p00;
            d.delta_x1 = d.p11 - d.p01;
            d.delta_y0 = d.p01 - d.p00;
            d.delta_y1 = d.p11 - d.p10;

            // Try to reverse-engineer effective fx/fy
            // NPP = p00*(1-fx_eff) + p10*fx_eff when fy=0
            if (std::abs(fy) < 0.001f && std::abs(d.delta_x0) > 0.1f) {
                d.effective_fx = (d.npp_result - d.p00) / (float)d.delta_x0;
            } else if (std::abs(fx) < 0.001f && std::abs(d.delta_y0) > 0.1f) {
                d.effective_fy = (d.npp_result - d.p00) / (float)d.delta_y0;
            } else {
                d.effective_fx = -1;
                d.effective_fy = -1;
            }

            data.push_back(d);
        }
    }

    // Print analysis
    std::cout << "Detailed Analysis:\n";
    std::cout << std::string(120, '-') << "\n";

    for (auto& d : data) {
        std::cout << "fx=" << std::fixed << std::setprecision(3) << d.fx
                  << " fy=" << d.fy << " | ";
        std::cout << "p00=" << std::setw(3) << d.p00
                  << " p10=" << std::setw(3) << d.p10
                  << " p01=" << std::setw(3) << d.p01
                  << " p11=" << std::setw(3) << d.p11 << " | ";
        std::cout << "NPP=" << std::setw(3) << d.npp_result
                  << " Std=" << std::setw(6) << std::setprecision(2) << d.std_bilinear
                  << " Diff=" << std::setw(5) << std::setprecision(2)
                  << (d.std_bilinear - d.npp_result) << "\n";

        if (std::abs(d.fy) < 0.001f && d.effective_fx >= 0) {
            float modifier = d.effective_fx / d.fx;
            std::cout << "  → 1D X-interp: effective_fx=" << std::setprecision(4) << d.effective_fx
                      << " (nominal=" << d.fx << ") modifier=" << modifier << "\n";
        } else if (std::abs(d.fx) < 0.001f && d.effective_fy >= 0) {
            float modifier = d.effective_fy / d.fy;
            std::cout << "  → 1D Y-interp: effective_fy=" << std::setprecision(4) << d.effective_fy
                      << " (nominal=" << d.fy << ") modifier=" << modifier << "\n";
        } else if (d.fx > 0.001f && d.fy > 0.001f) {
            // Try to solve for effective weights in 2D case
            // This is more complex, need to try different hypotheses
            std::cout << "  → 2D interp: delta_x0=" << d.delta_x0
                      << " delta_x1=" << d.delta_x1
                      << " delta_y0=" << d.delta_y0
                      << " delta_y1=" << d.delta_y1 << "\n";

            // Test hypothesis: separate X and Y modifiers
            for (float mod_x = 0.7f; mod_x <= 1.0f; mod_x += 0.01f) {
                for (float mod_y = 0.7f; mod_y <= 1.0f; mod_y += 0.01f) {
                    float fx_eff = d.fx * mod_x;
                    float fy_eff = d.fy * mod_y;
                    float v0 = d.p00 * (1 - fx_eff) + d.p10 * fx_eff;
                    float v1 = d.p01 * (1 - fx_eff) + d.p11 * fx_eff;
                    float result = v0 * (1 - fy_eff) + v1 * fy_eff;
                    int rounded = (int)std::floor(result);

                    if (rounded == d.npp_result) {
                        std::cout << "  ✓ Found match: mod_x=" << std::setprecision(3) << mod_x
                                  << " mod_y=" << mod_y
                                  << " (fx_eff=" << fx_eff << " fy_eff=" << fy_eff << ")\n";
                        break;
                    }
                }
            }
        }
        std::cout << "\n";
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    std::cout << "Analyzing Weight Modification Patterns in NVIDIA NPP\n";
    std::cout << "====================================================\n";

    analyzePattern({0, 100, 50, 150}, 2, 2, 3, 3, "[0,100,50,150]");
    analyzePattern({0, 127, 127, 255}, 2, 2, 3, 3, "[0,127,127,255]");
    analyzePattern({0, 255, 0, 255}, 2, 2, 3, 3, "[0,255,0,255]");

    return 0;
}
