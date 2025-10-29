/**
 * Linear Interpolation Analysis Test
 *
 * Purpose: Analyze NVIDIA NPP linear interpolation behavior to create
 *          a high-accuracy CPU reference implementation.
 *
 * Test scenarios:
 * 1. Integer scale (2x, 4x upscale)
 * 2. Fractional scale (1.5x, 2.5x)
 * 3. Downscale (0.5x, 0.75x)
 * 4. Non-uniform scale (different X/Y ratios)
 * 5. Edge pixel behavior
 * 6. Coordinate mapping verification
 */

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_geometry_transforms.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Helper function to print image data
template<typename T>
void printImage(const std::vector<T>& data, int width, int height, int channels, const char* name) {
    std::cout << "\n=== " << name << " (" << width << "x" << height << ", " << channels << " channels) ===" << std::endl;
    for (int y = 0; y < std::min(8, height); y++) {
        for (int x = 0; x < std::min(8, width); x++) {
            std::cout << "[";
            for (int c = 0; c < channels; c++) {
                int idx = (y * width + x) * channels + c;
                std::cout << std::setw(3) << (int)data[idx];
                if (c < channels - 1) std::cout << ",";
            }
            std::cout << "] ";
        }
        if (width > 8) std::cout << "...";
        std::cout << std::endl;
    }
    if (height > 8) std::cout << "..." << std::endl;
}

// Test case structure
struct LinearTestCase {
    int srcWidth, srcHeight;
    int dstWidth, dstHeight;
    int channels;
    const char* description;
};

// Run a single test case
void runLinearTest(const LinearTestCase& test) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Test: " << test.description << std::endl;
    std::cout << "Scale: " << test.srcWidth << "x" << test.srcHeight
              << " -> " << test.dstWidth << "x" << test.dstHeight << std::endl;
    std::cout << "Scale factor: " << (float)test.dstWidth/test.srcWidth << "x (X), "
              << (float)test.dstHeight/test.srcHeight << "x (Y)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Create source image with simple pattern
    std::vector<Npp8u> srcData(test.srcWidth * test.srcHeight * test.channels);

    // Fill with gradient pattern
    for (int y = 0; y < test.srcHeight; y++) {
        for (int x = 0; x < test.srcWidth; x++) {
            for (int c = 0; c < test.channels; c++) {
                int idx = (y * test.srcWidth + x) * test.channels + c;
                // Create distinct pattern for each channel
                if (test.channels == 1) {
                    srcData[idx] = (Npp8u)((x + y * test.srcWidth) * 16 % 256);
                } else {
                    srcData[idx] = (Npp8u)((x * 40 + y * 20 + c * 60) % 256);
                }
            }
        }
    }

    printImage(srcData, test.srcWidth, test.srcHeight, test.channels, "Source Image");

    // Allocate device memory
    Npp8u* d_src;
    Npp8u* d_dst;
    size_t srcStep = test.srcWidth * test.channels * sizeof(Npp8u);
    size_t dstStep = test.dstWidth * test.channels * sizeof(Npp8u);

    cudaMalloc(&d_src, test.srcHeight * srcStep);
    cudaMalloc(&d_dst, test.dstHeight * dstStep);

    cudaMemcpy(d_src, srcData.data(), test.srcHeight * srcStep, cudaMemcpyHostToDevice);

    // Setup NPP parameters
    NppiSize srcSize = {test.srcWidth, test.srcHeight};
    NppiSize dstSize = {test.dstWidth, test.dstHeight};
    NppiRect srcROI = {0, 0, test.srcWidth, test.srcHeight};
    NppiRect dstROI = {0, 0, test.dstWidth, test.dstHeight};

    // Call NPP linear interpolation
    NppStatus status;
    if (test.channels == 1) {
        status = nppiResize_8u_C1R(d_src, srcStep, srcSize, srcROI,
                                   d_dst, dstStep, dstSize, dstROI,
                                   NPPI_INTER_LINEAR);
    } else if (test.channels == 3) {
        status = nppiResize_8u_C3R(d_src, srcStep, srcSize, srcROI,
                                   d_dst, dstStep, dstSize, dstROI,
                                   NPPI_INTER_LINEAR);
    } else {
        status = nppiResize_8u_C4R(d_src, srcStep, srcSize, srcROI,
                                   d_dst, dstStep, dstSize, dstROI,
                                   NPPI_INTER_LINEAR);
    }

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP Error: " << status << std::endl;
        cudaFree(d_src);
        cudaFree(d_dst);
        return;
    }

    // Copy result back
    std::vector<Npp8u> dstData(test.dstWidth * test.dstHeight * test.channels);
    cudaMemcpy(dstData.data(), d_dst, test.dstHeight * dstStep, cudaMemcpyDeviceToHost);

    printImage(dstData, test.dstWidth, test.dstHeight, test.channels, "NPP Linear Result");

    // Analyze specific pixels for coordinate mapping
    std::cout << "\n=== Coordinate Mapping Analysis ===" << std::endl;
    std::cout << "Analyzing key pixels to understand NPP's coordinate system..." << std::endl;

    float scaleX = (float)test.srcWidth / test.dstWidth;
    float scaleY = (float)test.srcHeight / test.dstHeight;

    // Analyze corner and center pixels
    int testPoints[][2] = {{0, 0}, {test.dstWidth-1, 0}, {0, test.dstHeight-1},
                           {test.dstWidth-1, test.dstHeight-1},
                           {test.dstWidth/2, test.dstHeight/2}};
    const char* pointNames[] = {"Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"};

    for (int i = 0; i < 5; i++) {
        int dx = testPoints[i][0];
        int dy = testPoints[i][1];

        if (dx >= test.dstWidth || dy >= test.dstHeight) continue;

        std::cout << "\n" << pointNames[i] << " pixel (" << dx << "," << dy << "):" << std::endl;
        std::cout << "  NPP value: ";
        for (int c = 0; c < test.channels; c++) {
            int idx = (dy * test.dstWidth + dx) * test.channels + c;
            std::cout << (int)dstData[idx];
            if (c < test.channels - 1) std::cout << ",";
        }
        std::cout << std::endl;

        // Try different coordinate mapping formulas
        float srcX1 = dx * scaleX;  // Simple mapping
        float srcY1 = dy * scaleY;
        float srcX2 = (dx + 0.5f) * scaleX - 0.5f;  // Pixel center mapping
        float srcY2 = (dy + 0.5f) * scaleY - 0.5f;

        std::cout << "  Source coord (simple): (" << srcX1 << ", " << srcY1 << ")" << std::endl;
        std::cout << "  Source coord (center): (" << srcX2 << ", " << srcY2 << ")" << std::endl;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    std::cout << "Linear Interpolation Analysis for NVIDIA NPP" << std::endl;
    std::cout << "Purpose: Understand coordinate mapping and interpolation behavior" << std::endl;

    // Test cases covering various scenarios
    std::vector<LinearTestCase> testCases = {
        // Upscaling - integer
        {4, 4, 8, 8, 1, "Upscale 2x - Single Channel (Grayscale)"},
        {4, 4, 8, 8, 3, "Upscale 2x - 3 Channels (RGB)"},
        {2, 2, 8, 8, 1, "Upscale 4x - Single Channel"},

        // Upscaling - fractional
        {4, 4, 6, 6, 1, "Upscale 1.5x - Single Channel"},
        {4, 4, 10, 10, 3, "Upscale 2.5x - 3 Channels"},

        // Downscaling
        {8, 8, 4, 4, 1, "Downscale 0.5x - Single Channel"},
        {8, 8, 6, 6, 3, "Downscale to 0.75x - 3 Channels"},

        // Non-uniform scaling
        {4, 8, 8, 4, 1, "Non-uniform (2x width, 0.5x height)"},
        {8, 4, 4, 8, 3, "Non-uniform (0.5x width, 2x height)"},

        // Small to large
        {2, 2, 10, 10, 1, "Extreme upscale 5x"},
    };

    for (const auto& test : testCases) {
        runLinearTest(test);
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Analysis Complete" << std::endl;
    std::cout << "Next steps:" << std::endl;
    std::cout << "1. Identify coordinate mapping formula" << std::endl;
    std::cout << "2. Determine interpolation method (bilinear)" << std::endl;
    std::cout << "3. Understand rounding/clamping behavior" << std::endl;
    std::cout << "4. Implement CPU reference based on findings" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
