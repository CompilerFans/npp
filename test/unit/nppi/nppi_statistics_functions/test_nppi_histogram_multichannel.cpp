#include <gtest/gtest.h>
#include <npp.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <algorithm>

class HistogramEvenMultiChannelTest : public ::testing::Test {
protected:
    void SetUp() override {
        width = 64;
        height = 64;
        totalPixels = width * height;
        
        // Set up for 4-channel test
        for (int c = 0; c < 4; c++) {
            nLevels[c] = 256;
            nLowerLevel[c] = 0;
            nUpperLevel[c] = 256;
        }
        
        oSizeROI = {width, height};
    }

    void TearDown() override {
        cleanupDeviceMemory();
    }

    void cleanupDeviceMemory() {
        if (d_src_8u_c4) { cudaFree(d_src_8u_c4); d_src_8u_c4 = nullptr; }
        for (int c = 0; c < 4; c++) {
            if (d_hist[c]) { cudaFree(d_hist[c]); d_hist[c] = nullptr; }
        }
        if (d_buffer) { cudaFree(d_buffer); d_buffer = nullptr; }
    }

    void allocateTestData8uC4R() {
        // Create test data for 8u C4R (4 channels)
        h_src_8u_c4.resize(totalPixels * 4);
        for (int i = 0; i < totalPixels; i++) {
            for (int c = 0; c < 4; c++) {
                // Create varied data per channel
                h_src_8u_c4[i * 4 + c] = static_cast<Npp8u>((i + c * 64) % 256);
            }
        }
        
        // Allocate device memory
        nSrcStep_8u_c4 = width * 4 * sizeof(Npp8u);
        ASSERT_EQ(cudaMalloc(&d_src_8u_c4, totalPixels * 4 * sizeof(Npp8u)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_src_8u_c4, h_src_8u_c4.data(), totalPixels * 4 * sizeof(Npp8u), 
                           cudaMemcpyHostToDevice), cudaSuccess);
    }

    void allocateHistogramMemory() {
        // Allocate histogram memory for each channel
        for (int c = 0; c < 4; c++) {
            h_hist[c].resize(nLevels[c] - 1, 0);
            ASSERT_EQ(cudaMalloc(&d_hist[c], (nLevels[c] - 1) * sizeof(Npp32s)), cudaSuccess);
        }
        
        // Get buffer size and allocate
        int bufferSize;
        ASSERT_EQ(nppiHistogramEvenGetBufferSize_8u_C4R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
        ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);
    }

protected:
    int width, height, totalPixels;
    int nLevels[4], nLowerLevel[4], nUpperLevel[4];
    int nSrcStep_8u_c4;
    NppiSize oSizeROI;
    
    std::vector<Npp8u> h_src_8u_c4;
    std::vector<Npp32s> h_hist[4];
    
    Npp8u* d_src_8u_c4 = nullptr;
    Npp32s* d_hist[4] = {nullptr, nullptr, nullptr, nullptr};
    Npp8u* d_buffer = nullptr;
};

TEST_F(HistogramEvenMultiChannelTest, HistogramEven_8u_C4R_Basic) {
    allocateTestData8uC4R();
    allocateHistogramMemory();
    
    // Compute multi-channel histogram
    EXPECT_EQ(nppiHistogramEven_8u_C4R(d_src_8u_c4, nSrcStep_8u_c4, oSizeROI, d_hist, nLevels,
                                      nLowerLevel, nUpperLevel, d_buffer), NPP_SUCCESS);
    
    // Copy results back to host for each channel
    for (int c = 0; c < 4; c++) {
        ASSERT_EQ(cudaMemcpy(h_hist[c].data(), d_hist[c], (nLevels[c] - 1) * sizeof(Npp32s), 
                            cudaMemcpyDeviceToHost), cudaSuccess);
    }
    
    // Verify total count for each channel
    for (int c = 0; c < 4; c++) {
        int totalCount = 0;
        for (int i = 0; i < nLevels[c] - 1; i++) {
            totalCount += h_hist[c][i];
        }
        EXPECT_EQ(totalCount, totalPixels) << "Channel " << c << " total count mismatch";
        
        // Verify histogram has reasonable distribution
        int nonZeroBins = 0;
        for (int i = 0; i < nLevels[c] - 1; i++) {
            if (h_hist[c][i] > 0) nonZeroBins++;
        }
        EXPECT_GT(nonZeroBins, 0) << "Channel " << c << " should have non-zero bins";
    }
}

TEST_F(HistogramEvenMultiChannelTest, HistogramEven_8u_C4R_WithContext) {
    allocateTestData8uC4R();
    allocateHistogramMemory();
    
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    EXPECT_EQ(nppiHistogramEven_8u_C4R_Ctx(d_src_8u_c4, nSrcStep_8u_c4, oSizeROI, d_hist, nLevels,
                                          nLowerLevel, nUpperLevel, d_buffer, nppStreamCtx), NPP_SUCCESS);
    
    cudaStreamSynchronize(nppStreamCtx.hStream);
    
    // Verify results
    for (int c = 0; c < 4; c++) {
        ASSERT_EQ(cudaMemcpy(h_hist[c].data(), d_hist[c], (nLevels[c] - 1) * sizeof(Npp32s), 
                            cudaMemcpyDeviceToHost), cudaSuccess);
        
        int totalCount = 0;
        for (int i = 0; i < nLevels[c] - 1; i++) {
            totalCount += h_hist[c][i];
        }
        EXPECT_EQ(totalCount, totalPixels) << "Channel " << c << " total count mismatch with context";
    }
}

TEST_F(HistogramEvenMultiChannelTest, ParameterValidation_8u_C4R) {
    allocateTestData8uC4R();
    allocateHistogramMemory();
    
    // Test null pointer errors - source
    EXPECT_EQ(nppiHistogramEven_8u_C4R(nullptr, nSrcStep_8u_c4, oSizeROI, d_hist, nLevels,
                                      nLowerLevel, nUpperLevel, d_buffer), NPP_NULL_POINTER_ERROR);
    
    // Test null pointer errors - histogram array
    EXPECT_EQ(nppiHistogramEven_8u_C4R(d_src_8u_c4, nSrcStep_8u_c4, oSizeROI, nullptr, nLevels,
                                      nLowerLevel, nUpperLevel, d_buffer), NPP_NULL_POINTER_ERROR);
    
    // Test null pointer errors - individual histogram channel
    Npp32s* backup = d_hist[1];
    d_hist[1] = nullptr;
    EXPECT_EQ(nppiHistogramEven_8u_C4R(d_src_8u_c4, nSrcStep_8u_c4, oSizeROI, d_hist, nLevels,
                                      nLowerLevel, nUpperLevel, d_buffer), NPP_NULL_POINTER_ERROR);
    d_hist[1] = backup;
    
    // Test size error
    NppiSize invalidSize = {0, height};
    EXPECT_EQ(nppiHistogramEven_8u_C4R(d_src_8u_c4, nSrcStep_8u_c4, invalidSize, d_hist, nLevels,
                                      nLowerLevel, nUpperLevel, d_buffer), NPP_SIZE_ERROR);
    
    // Test range error - one channel has invalid range
    int backupUpper = nUpperLevel[2];
    nUpperLevel[2] = nLowerLevel[2] - 1;
    EXPECT_EQ(nppiHistogramEven_8u_C4R(d_src_8u_c4, nSrcStep_8u_c4, oSizeROI, d_hist, nLevels,
                                      nLowerLevel, nUpperLevel, d_buffer), NPP_RANGE_ERROR);
    nUpperLevel[2] = backupUpper;
}

TEST_F(HistogramEvenMultiChannelTest, DifferentLevelsPerChannel) {
    allocateTestData8uC4R();
    
    // Set different levels for each channel
    nLevels[0] = 64;   // Channel 0: 64 levels
    nLevels[1] = 128;  // Channel 1: 128 levels  
    nLevels[2] = 256;  // Channel 2: 256 levels
    nLevels[3] = 32;   // Channel 3: 32 levels
    
    // Reallocate with new levels
    for (int c = 0; c < 4; c++) {
        h_hist[c].resize(nLevels[c] - 1, 0);
        ASSERT_EQ(cudaMalloc(&d_hist[c], (nLevels[c] - 1) * sizeof(Npp32s)), cudaSuccess);
    }
    
    int bufferSize;
    ASSERT_EQ(nppiHistogramEvenGetBufferSize_8u_C4R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
    ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);
    
    // Compute histogram
    EXPECT_EQ(nppiHistogramEven_8u_C4R(d_src_8u_c4, nSrcStep_8u_c4, oSizeROI, d_hist, nLevels,
                                      nLowerLevel, nUpperLevel, d_buffer), NPP_SUCCESS);
    
    // Verify each channel
    for (int c = 0; c < 4; c++) {
        ASSERT_EQ(cudaMemcpy(h_hist[c].data(), d_hist[c], (nLevels[c] - 1) * sizeof(Npp32s), 
                            cudaMemcpyDeviceToHost), cudaSuccess);
        
        int totalCount = 0;
        for (int i = 0; i < nLevels[c] - 1; i++) {
            totalCount += h_hist[c][i];
        }
        EXPECT_EQ(totalCount, totalPixels) << "Channel " << c << " with " << nLevels[c] << " levels";
    }
}

TEST_F(HistogramEvenMultiChannelTest, SharedMemoryOptimization) {
    allocateTestData8uC4R();
    
    // Use small levels to trigger shared memory optimization
    for (int c = 0; c < 4; c++) {
        nLevels[c] = 16;  // Small number of levels
    }
    
    // Allocate with small levels
    for (int c = 0; c < 4; c++) {
        h_hist[c].resize(nLevels[c] - 1, 0);
        ASSERT_EQ(cudaMalloc(&d_hist[c], (nLevels[c] - 1) * sizeof(Npp32s)), cudaSuccess);
    }
    
    int bufferSize;
    ASSERT_EQ(nppiHistogramEvenGetBufferSize_8u_C4R(oSizeROI, nLevels, &bufferSize), NPP_SUCCESS);
    ASSERT_EQ(cudaMalloc(&d_buffer, bufferSize), cudaSuccess);
    
    // This should use shared memory kernel due to small total bins (4 * 15 = 60 bins)
    EXPECT_EQ(nppiHistogramEven_8u_C4R(d_src_8u_c4, nSrcStep_8u_c4, oSizeROI, d_hist, nLevels,
                                      nLowerLevel, nUpperLevel, d_buffer), NPP_SUCCESS);
    
    // Verify results
    for (int c = 0; c < 4; c++) {
        ASSERT_EQ(cudaMemcpy(h_hist[c].data(), d_hist[c], (nLevels[c] - 1) * sizeof(Npp32s), 
                            cudaMemcpyDeviceToHost), cudaSuccess);
        
        int totalCount = 0;
        for (int i = 0; i < nLevels[c] - 1; i++) {
            totalCount += h_hist[c][i];
        }
        EXPECT_EQ(totalCount, totalPixels) << "Channel " << c << " shared memory optimization";
    }
}

TEST_F(HistogramEvenMultiChannelTest, ChannelIndependence) {
    allocateTestData8uC4R();
    allocateHistogramMemory();
    
    // Set different ranges for each channel to test independence
    nLowerLevel[0] = 0;   nUpperLevel[0] = 64;   // Channel 0: [0, 64)
    nLowerLevel[1] = 64;  nUpperLevel[1] = 128;  // Channel 1: [64, 128)
    nLowerLevel[2] = 128; nUpperLevel[2] = 192;  // Channel 2: [128, 192)
    nLowerLevel[3] = 192; nUpperLevel[3] = 256;  // Channel 3: [192, 256)
    
    EXPECT_EQ(nppiHistogramEven_8u_C4R(d_src_8u_c4, nSrcStep_8u_c4, oSizeROI, d_hist, nLevels,
                                      nLowerLevel, nUpperLevel, d_buffer), NPP_SUCCESS);
    
    // Copy and verify each channel independently
    for (int c = 0; c < 4; c++) {
        ASSERT_EQ(cudaMemcpy(h_hist[c].data(), d_hist[c], (nLevels[c] - 1) * sizeof(Npp32s), 
                            cudaMemcpyDeviceToHost), cudaSuccess);
        
        // Count expected pixels in range for this channel
        int expectedCount = 0;
        for (int i = 0; i < totalPixels; i++) {
            int pixelValue = h_src_8u_c4[i * 4 + c];
            if (pixelValue >= nLowerLevel[c] && pixelValue < nUpperLevel[c]) {
                expectedCount++;
            }
        }
        
        int actualCount = 0;
        for (int i = 0; i < nLevels[c] - 1; i++) {
            actualCount += h_hist[c][i];
        }
        
        EXPECT_EQ(actualCount, expectedCount) << "Channel " << c << " range independence test";
    }
}