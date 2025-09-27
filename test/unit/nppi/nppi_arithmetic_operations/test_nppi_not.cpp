#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

class NppiNotTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }

    void TearDown() override {
        cudaError_t err = cudaDeviceSynchronize();
        EXPECT_EQ(err, cudaSuccess);
    }
};

TEST_F(NppiNotTest, Not_8u_C1R_BasicOperation) {
    const int width = 4;
    const int height = 4;
    const int totalPixels = width * height;

    // Create test data
    std::vector<Npp8u> hostSrc = {0x00, 0xFF, 0xAA, 0x55, 
                                  0x33, 0xCC, 0x0F, 0xF0,
                                  0x3C, 0xC3, 0x69, 0x96,
                                  0x81, 0x7E, 0x42, 0xBD};

    std::vector<Npp8u> expected = {0xFF, 0x00, 0x55, 0xAA,
                                   0xCC, 0x33, 0xF0, 0x0F,
                                   0xC3, 0x3C, 0x96, 0x69,
                                   0x7E, 0x81, 0xBD, 0x42};

    // Allocate GPU memory
    int step;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &step);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &step);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiNot_8u_C1R(d_src, step, d_dst, step, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp8u> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
}

TEST_F(NppiNotTest, Not_8u_C1IR_InPlace) {
    const int width = 3;
    const int height = 3;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostData = {0x00, 0xFF, 0x80, 0x7F, 0x55, 0xAA, 0x33, 0xCC, 0x0F};
    std::vector<Npp8u> expected = {0xFF, 0x00, 0x7F, 0x80, 0xAA, 0x55, 0xCC, 0x33, 0xF0};

    // Allocate GPU memory
    int step;
    Npp8u *d_data = nppiMalloc_8u_C1(width, height, &step);
    ASSERT_NE(d_data, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_data, step, hostData.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute in-place operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiNot_8u_C1IR(d_data, step, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp8u> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_data, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_data);
}

TEST_F(NppiNotTest, Not_8u_C3R_MultiChannel) {
    const int width = 2;
    const int height = 2;
    const int channels = 3;
    const int totalElements = width * height * channels;

    std::vector<Npp8u> hostSrc = {0x00, 0xFF, 0x80,  // pixel (0,0)
                                  0x7F, 0x55, 0xAA,  // pixel (1,0)
                                  0x33, 0xCC, 0x0F,  // pixel (0,1)
                                  0xF0, 0x69, 0x96}; // pixel (1,1)

    std::vector<Npp8u> expected = {0xFF, 0x00, 0x7F,  // ~pixel (0,0)
                                   0x80, 0xAA, 0x55,  // ~pixel (1,0)
                                   0xCC, 0x33, 0xF0,  // ~pixel (0,1)
                                   0x0F, 0x96, 0x69}; // ~pixel (1,1)

    // Allocate GPU memory
    int step;
    Npp8u *d_src = nppiMalloc_8u_C3(width, height, &step);
    Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &step);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Copy data to GPU
    int hostStep = width * channels * sizeof(Npp8u);
    cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiNot_8u_C3R(d_src, step, d_dst, step, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp8u> hostResult(totalElements);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalElements; ++i) {
        EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
}

TEST_F(NppiNotTest, Not_8u_C1R_ZeroSizeROI) {
    int step;
    Npp8u *d_src = nppiMalloc_8u_C1(1, 1, &step);
    Npp8u *d_dst = nppiMalloc_8u_C1(1, 1, &step);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Test with zero width
    NppiSize oSizeROI = {0, 1};
    NppStatus status = nppiNot_8u_C1R(d_src, step, d_dst, step, oSizeROI);
    EXPECT_EQ(status, NPP_SUCCESS);

    // Test with zero height
    oSizeROI = {1, 0};
    status = nppiNot_8u_C1R(d_src, step, d_dst, step, oSizeROI);
    EXPECT_EQ(status, NPP_SUCCESS);

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
}

TEST_F(NppiNotTest, Not_8u_C1R_ErrorHandling) {
    int step;
    Npp8u *d_valid = nppiMalloc_8u_C1(2, 2, &step);
    ASSERT_NE(d_valid, nullptr);

    NppiSize oSizeROI = {2, 2};

    // Test null pointer errors
    EXPECT_EQ(nppiNot_8u_C1R(nullptr, step, d_valid, step, oSizeROI), NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiNot_8u_C1R(d_valid, step, nullptr, step, oSizeROI), NPP_NULL_POINTER_ERROR);

    // Test negative size errors
    NppiSize invalidSize = {-1, 2};
    EXPECT_EQ(nppiNot_8u_C1R(d_valid, step, d_valid, step, invalidSize), NPP_SIZE_ERROR);

    invalidSize = {2, -1};
    EXPECT_EQ(nppiNot_8u_C1R(d_valid, step, d_valid, step, invalidSize), NPP_SIZE_ERROR);

    // Cleanup
    nppiFree(d_valid);
}