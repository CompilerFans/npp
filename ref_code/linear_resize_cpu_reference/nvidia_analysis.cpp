#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <npp.h>

// Helper to copy data
void copyToDevice(const std::vector<unsigned char>& host, unsigned char*& dev, int size) {
    cudaMalloc(&dev, size);
    cudaMemcpy(dev, host.data(), size, cudaMemcpyHostToDevice);
}

void copyFromDevice(unsigned char* dev, std::vector<unsigned char>& host, int size) {
    cudaMemcpy(host.data(), dev, size, cudaMemcpyDeviceToHost);
}

int main() {
    // Test case: 2x2 -> 3x3 (same as failing test)
    int srcW = 2, srcH = 2;
    int dstW = 3, dstH = 3;
    int channels = 3;
    
    // Source data (same as test)
    std::vector<unsigned char> src = {
        0, 0, 0,      100, 100, 100,
        50, 50, 50,   150, 150, 150
    };
    
    // Allocate device memory
    unsigned char *d_src, *d_dst;
    int srcStep = srcW * channels;
    int dstStep = dstW * channels;
    
    copyToDevice(src, d_src, srcW * srcH * channels);
    cudaMalloc(&d_dst, dstW * dstH * channels);
    
    // Setup NPP structures
    NppiSize srcSize = {srcW, srcH};
    NppiSize dstSize = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};
    
    // Call NVIDIA NPP
    NppStatus status = nppiResize_8u_C3R(d_src, srcStep, srcSize, srcROI,
                                          d_dst, dstStep, dstSize, dstROI,
                                          NPPI_INTER_LINEAR);
    
    if (status != NPP_SUCCESS) {
        std::cerr << "NPP error: " << status << std::endl;
        return 1;
    }
    
    // Copy result back
    std::vector<unsigned char> nppResult(dstW * dstH * channels);
    copyFromDevice(d_dst, nppResult, dstW * dstH * channels);
    
    // Print NVIDIA NPP results
    std::cout << "NVIDIA NPP Results (2x2 -> 3x3):\n";
    std::cout << "Scale: " << (float)srcW/dstW << " x " << (float)srcH/dstH << "\n\n";
    
    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            int idx = (y * dstW + x) * channels;
            std::cout << "Pixel[" << x << "," << y << "]: ("
                      << (int)nppResult[idx] << ", "
                      << (int)nppResult[idx+1] << ", "
                      << (int)nppResult[idx+2] << ")\n";
        }
    }
    
    std::cout << "\n\nDetailed analysis for reverse engineering:\n";
    
    // Analyze each pixel to find pattern
    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            int idx = (y * dstW + x) * channels;
            int value = nppResult[idx]; // All channels same
            
            // Try different coordinate mappings
            float scale = (float)srcW / dstW;
            
            // Method 1: (dx+0.5)*scale - 0.5 (current CPU ref)
            float sx1 = (x + 0.5f) * scale - 0.5f;
            float sy1 = (y + 0.5f) * scale - 0.5f;
            
            // Method 2: dx*scale (edge-to-edge)
            float sx2 = x * scale;
            float sy2 = y * scale;
            
            // Method 3: (dx+0.5)*scale (center, no offset)
            float sx3 = (x + 0.5f) * scale;
            float sy3 = (y + 0.5f) * scale;
            
            std::cout << "Pixel[" << x << "," << y << "] = " << value << "\n";
            std::cout << "  Method1 (ref): sx=" << sx1 << ", sy=" << sy1 << "\n";
            std::cout << "  Method2 (e2e): sx=" << sx2 << ", sy=" << sy2 << "\n";
            std::cout << "  Method3 (ctr): sx=" << sx3 << ", sy=" << sy3 << "\n";
            
            // Calculate what each method would produce with standard bilinear
            auto bilinear = [&](float sx, float sy) -> float {
                sx = std::max(0.0f, std::min((float)(srcW-1), sx));
                sy = std::max(0.0f, std::min((float)(srcH-1), sy));
                
                int x0 = (int)std::floor(sx);
                int y0 = (int)std::floor(sy);
                int x1 = std::min(x0 + 1, srcW - 1);
                int y1 = std::min(y0 + 1, srcH - 1);
                
                float fx = sx - x0;
                float fy = sy - y0;
                
                float p00 = src[(y0 * srcW + x0) * channels];
                float p10 = src[(y0 * srcW + x1) * channels];
                float p01 = src[(y1 * srcW + x0) * channels];
                float p11 = src[(y1 * srcW + x1) * channels];
                
                float v0 = p00 * (1-fx) + p10 * fx;
                float v1 = p01 * (1-fx) + p11 * fx;
                return v0 * (1-fy) + v1 * fy;
            };
            
            float r1 = bilinear(sx1, sy1);
            float r2 = bilinear(sx2, sy2);
            float r3 = bilinear(sx3, sy3);
            
            std::cout << "  Result1: " << (int)(r1+0.5f) << " (diff=" << std::abs(value - (int)(r1+0.5f)) << ")\n";
            std::cout << "  Result2: " << (int)(r2+0.5f) << " (diff=" << std::abs(value - (int)(r2+0.5f)) << ")\n";
            std::cout << "  Result3: " << (int)(r3+0.5f) << " (diff=" << std::abs(value - (int)(r3+0.5f)) << ")\n";
            std::cout << "\n";
        }
    }
    
    // Cleanup
    cudaFree(d_src);
    cudaFree(d_dst);
    
    return 0;
}
