#include <gtest/gtest.h>
#include <npp.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <algorithm>

// Type alias for buffer size to handle API differences
#ifdef USE_NVIDIA_NPP_TESTS
using BufferSizeType = size_t;
#else
using BufferSizeType = int;
#endif

class HistogramRangeTest : public ::testing::Test {
protected:
    void SetUp() override {
        width = 64;
        height = 64;
        totalPixels = width * height;
        nLevels = 10; // Test with custom levels
        
        oSizeROI = {width, height};
    }

    void TearDown() override {
        cleanupDeviceMemory();
    }

    void cleanupDeviceMemory() {
        if (d_src_8u) { cudaFree(d_src_8u); d_src_8u = nullptr; }
        if (d_src_32f) { cudaFree(d_src_32f); d_src_32f = nullptr; }
        if (d_hist) { cudaFree(d_hist); d_hist = nullptr; }
        if (d_buffer) { cudaFree(d_buffer); d_buffer = nullptr; }
        if (d_levels_8u) { cudaFree(d_levels_8u); d_levels_8u = nullptr; }
        if (d_levels_32f) { cudaFree(d_levels_32f); d_levels_32f = nullptr; }
    }

    void allocateTestData8u() {
        // Create test data for 8u
        h_src_8u.resize(totalPixels);
        for (int i = 0; i < totalPixels; i++) {
            h_src_8u[i] = static_cast<Npp8u>(i % 256);
        }

        // Allocate device memory
        nSrcStep_8u = width * sizeof(Npp8u);
        ASSERT_EQ(cudaMalloc(&d_src_8u, totalPixels * sizeof(Npp8u)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_src_8u, h_src_8u.data(), totalPixels * sizeof(Npp8u), 
                           cudaMemcpyHostToDevice), cudaSuccess);
    }

    void allocateTestData32f() {
        // Create test data for 32f
        h_src_32f.resize(totalPixels);
        for (int i = 0; i < totalPixels; i++) {
            h_src_32f[i] = static_cast<Npp32f>(i % 256) / 256.0f; // Values [0.0, 1.0)
        }

        nSrcStep_32f = width * sizeof(Npp32f);
        ASSERT_EQ(cudaMalloc(&d_src_32f, totalPixels * sizeof(Npp32f)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_src_32f, h_src_32f.data(), totalPixels * sizeof(Npp32f),
                           cudaMemcpyHostToDevice), cudaSuccess);
    }

    void setupLevels8u() {
        // Create custom levels for 8u: [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255]
        h_levels_8u.resize(nLevels);
        for (int i = 0; i < nLevels; i++) {
            h_levels_8u[i] = static_cast<Npp32s>(i * 255 / (nLevels - 1));
        }

        ASSERT_EQ(cudaMalloc(&d_levels_8u, nLevels * sizeof(Npp32s)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_levels_8u, h_levels_8u.data(), nLevels * sizeof(Npp32s),
                           cudaMemcpyHostToDevice), cudaSuccess);
    }

    void setupLevels32f() {
        // Create custom levels for 32f: [0.0, 0.1, 0.2, ..., 0.9, 1.0] 
        h_levels_32f.resize(nLevels);
        for (int i = 0; i < nLevels; i++) {
            h_levels_32f[i] = static_cast<Npp32f>(i) / static_cast<Npp32f>(nLevels - 1);
        }

        ASSERT_EQ(cudaMalloc(&d_levels_32f, nLevels * sizeof(Npp32f)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_levels_32f, h_levels_32f.data(), nLevels * sizeof(Npp32f),
                           cudaMemcpyHostToDevice), cudaSuccess);
    }

    void allocateHistogramMemory() {
        h_hist.resize(nLevels - 1, 0);
        ASSERT_EQ(cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);

        // Get buffer size and allocate
        BufferSizeType bufferSize;
        ASSERT_EQ(nppiHistogramRangeGetBufferSize_8u_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
        ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);
    }

protected:
    int width, height, totalPixels;
    int nLevels;
    int nSrcStep_8u, nSrcStep_32f;
    NppiSize oSizeROI;
    
    std::vector<Npp8u> h_src_8u;
    std::vector<Npp32f> h_src_32f;
    std::vector<Npp32s> h_hist;
    std::vector<Npp32s> h_levels_8u;
    std::vector<Npp32f> h_levels_32f;
    
    Npp8u* d_src_8u = nullptr;
    Npp32f* d_src_32f = nullptr;
    Npp32s* d_hist = nullptr;
    Npp8u* d_buffer = nullptr;
    Npp32s* d_levels_8u = nullptr;
    Npp32f* d_levels_32f = nullptr;
};

TEST_F(HistogramRangeTest, BufferSize_8u_C1R_Basic) {
    BufferSizeType bufferSize;
    
    EXPECT_EQ(nppiHistogramRangeGetBufferSize_8u_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0);

    // Test with stream context
    EXPECT_EQ(nppiHistogramRangeGetBufferSize_8u_C1R_Ctx(oSizeROI, nLevels, &bufferSize, NppStreamContext{}), 
             NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramRangeTest, BufferSize_32f_C1R_Basic) {
    BufferSizeType bufferSize;
    
    EXPECT_EQ(nppiHistogramRangeGetBufferSize_32f_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0);

    EXPECT_EQ(nppiHistogramRangeGetBufferSize_32f_C1R_Ctx(oSizeROI, nLevels, &bufferSize, NppStreamContext{}),
             NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0);
}

TEST_F(HistogramRangeTest, HistogramRange_8u_C1R_Basic) {
    allocateTestData8u();
    setupLevels8u();
    allocateHistogramMemory();

    // Print debug info
    printf("Debug: width=%d, height=%d, nLevels=%d\n", width, height, nLevels);
    printf("Debug: h_levels_8u size=%zu, first few levels: ", h_levels_8u.size());
    for (int i = 0; i < std::min(5, (int)h_levels_8u.size()); i++) {
        printf("%d ", h_levels_8u[i]);
    }
    printf("\n");

    // Compute histogram
    NppStatus status = nppiHistogramRange_8u_C1R(d_src_8u, nSrcStep_8u, oSizeROI, d_hist, d_levels_8u, nLevels, d_buffer);
    printf("Debug: nppiHistogramRange_8u_C1R returned status=%d\n", status);
    EXPECT_EQ(status, NPP_SUCCESS);

    if (status == NPP_SUCCESS) {
        // Copy result back to host
        ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), 
                            cudaMemcpyDeviceToHost), cudaSuccess);

        // Verify total count matches expected pixels in range
        int totalCount = 0;
        for (int i = 0; i < nLevels - 1; i++) {
            totalCount += h_hist[i];
        }
        
        // Count expected pixels in range manually
        int expectedCount = 0;
        for (int i = 0; i < totalPixels; i++) {
            int pixel_value = h_src_8u[i];
            for (int j = 0; j < nLevels - 1; j++) {
                if (pixel_value >= h_levels_8u[j] && pixel_value < h_levels_8u[j + 1]) {
                    expectedCount++;
                    break;
                }
            }
        }
        
        printf("Debug: totalCount=%d, expectedCount=%d\n", totalCount, expectedCount);
        EXPECT_EQ(totalCount, expectedCount);
    }
}

TEST_F(HistogramRangeTest, HistogramRange_32f_C1R_Basic) {
    allocateTestData32f();
    setupLevels32f();
    
    // Allocate histogram memory for 32f test
    h_hist.resize(nLevels - 1, 0);
    ASSERT_EQ(cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);

    BufferSizeType bufferSize;
    ASSERT_EQ(nppiHistogramRangeGetBufferSize_32f_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
    ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);

    EXPECT_EQ(nppiHistogramRange_32f_C1R(d_src_32f, nSrcStep_32f, oSizeROI, d_hist, d_levels_32f, nLevels, d_buffer),
             NPP_SUCCESS);

    ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s),
                        cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify total count
    int totalCount = 0;
    for (int i = 0; i < nLevels - 1; i++) {
        totalCount += h_hist[i];
    }
    
    int expectedCount = 0;
    for (int i = 0; i < totalPixels; i++) {
        float pixel_value = h_src_32f[i];
        for (int j = 0; j < nLevels - 1; j++) {
            if (pixel_value >= h_levels_32f[j] && pixel_value < h_levels_32f[j + 1]) {
                expectedCount++;
                break;
            }
        }
    }
    
    EXPECT_EQ(totalCount, expectedCount);
}

TEST_F(HistogramRangeTest, HistogramRange_8u_C1R_WithContext) {
    allocateTestData8u();
    setupLevels8u();
    allocateHistogramMemory();

    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);

    EXPECT_EQ(nppiHistogramRange_8u_C1R_Ctx(d_src_8u, nSrcStep_8u, oSizeROI, d_hist, d_levels_8u, nLevels, 
                                           d_buffer, nppStreamCtx), NPP_SUCCESS);

    cudaStreamSynchronize(nppStreamCtx.hStream);

    ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s),
                        cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify result consistency
    int totalCount = 0;
    for (int i = 0; i < nLevels - 1; i++) {
        totalCount += h_hist[i];
    }
    EXPECT_GT(totalCount, 0);
}

TEST_F(HistogramRangeTest, CustomLevels_NonUniform) {
    allocateTestData8u();
    allocateHistogramMemory();

    // Create non-uniform levels: [0, 10, 50, 100, 200, 255]
    nLevels = 6;
    h_levels_8u = {0, 10, 50, 100, 200, 255};
    
    ASSERT_EQ(cudaMalloc(&d_levels_8u, nLevels * sizeof(Npp32s)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_levels_8u, h_levels_8u.data(), nLevels * sizeof(Npp32s),
                        cudaMemcpyHostToDevice), cudaSuccess);

    // Reallocate histogram for new levels
    if (d_hist) cudaFree(d_hist);
    h_hist.resize(nLevels - 1, 0);
    ASSERT_EQ(cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);

    EXPECT_EQ(nppiHistogramRange_8u_C1R(d_src_8u, nSrcStep_8u, oSizeROI, d_hist, d_levels_8u, nLevels, d_buffer),
             NPP_SUCCESS);

    ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s),
                        cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify histogram distribution follows the custom ranges
    for (int i = 0; i < nLevels - 1; i++) {
        EXPECT_GE(h_hist[i], 0) << "Bin " << i << " should be non-negative";
    }

    int totalCount = 0;
    for (int i = 0; i < nLevels - 1; i++) {
        totalCount += h_hist[i];
    }
    EXPECT_GT(totalCount, 0) << "Total histogram count should be positive";
}

TEST_F(HistogramRangeTest, SharedMemoryOptimization) {
    allocateTestData8u();
    setupLevels8u();

    // Use very small levels to trigger shared memory optimization
    nLevels = 4; // Small number of levels (3 bins)
    h_levels_8u = {0, 85, 170, 255};
    
    ASSERT_EQ(cudaMalloc(&d_levels_8u, nLevels * sizeof(Npp32s)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_levels_8u, h_levels_8u.data(), nLevels * sizeof(Npp32s),
                        cudaMemcpyHostToDevice), cudaSuccess);

    h_hist.resize(nLevels - 1, 0);
    ASSERT_EQ(cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s)), cudaSuccess);

    BufferSizeType bufferSize;
    ASSERT_EQ(nppiHistogramRangeGetBufferSize_8u_C1R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
    ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);

    // This should use shared memory kernel due to small bins and image size
    EXPECT_EQ(nppiHistogramRange_8u_C1R(d_src_8u, nSrcStep_8u, oSizeROI, d_hist, d_levels_8u, nLevels, d_buffer),
             NPP_SUCCESS);

    ASSERT_EQ(cudaMemcpy(h_hist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s),
                        cudaMemcpyDeviceToHost), cudaSuccess);

    int totalCount = 0;
    for (int i = 0; i < nLevels - 1; i++) {
        totalCount += h_hist[i];
    }
    EXPECT_GT(totalCount, 0) << "Shared memory optimization should produce valid results";
}