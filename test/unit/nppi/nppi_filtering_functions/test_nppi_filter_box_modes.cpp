#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <npp.h>
#include <vector>

// Test for filter box boundary handling modes
class NppiFilterBoxModesTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 10;
    height = 10;
    srcStep = width * sizeof(Npp8u);
    dstStep = width * sizeof(Npp8u);
    roiSize = {width, height};
    
    // Allocate host memory
    h_src.resize(width * height);
    h_dst.resize(width * height);
    
    // Initialize source data with a pattern
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        h_src[y * width + x] = (y * 10 + x) % 256;
      }
    }
    
    // Allocate device memory
    ASSERT_EQ(cudaMalloc(&d_src, width * height * sizeof(Npp8u)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, width * height * sizeof(Npp8u)), cudaSuccess);
    
    // Copy source data to device
    ASSERT_EQ(cudaMemcpy(d_src, h_src.data(), width * height * sizeof(Npp8u), cudaMemcpyHostToDevice), cudaSuccess);
  }
  
  void TearDown() override {
    if (d_src) cudaFree(d_src);
    if (d_dst) cudaFree(d_dst);
  }
  
  void TestFilterBox(NppiSize maskSize, NppiPoint anchor, bool expectSuccess) {
    NppStatus status = nppiFilterBox_8u_C1R(d_src, srcStep, d_dst, dstStep, roiSize, maskSize, anchor);
    
    if (expectSuccess) {
      EXPECT_EQ(status, NPP_SUCCESS) << "Filter should succeed with mask " 
                                     << maskSize.width << "x" << maskSize.height 
                                     << " anchor (" << anchor.x << "," << anchor.y << ")";
      
      // Copy result back to host
      ASSERT_EQ(cudaMemcpy(h_dst.data(), d_dst, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost), cudaSuccess);
      
      // Basic validation - check that some filtering occurred
      // For 1x1 mask, the output should be identical to input
      bool hasChanges = false;
      for (int i = 0; i < width * height; i++) {
        if (h_src[i] != h_dst[i]) {
          hasChanges = true;
          break;
        }
      }
      
      // Only expect changes if mask is larger than 1x1
      if (maskSize.width > 1 || maskSize.height > 1) {
        EXPECT_TRUE(hasChanges) << "Filter should modify the image";
      } else {
        EXPECT_FALSE(hasChanges) << "1x1 filter should not modify the image";
      }
    } else {
      EXPECT_NE(status, NPP_SUCCESS) << "Filter should fail with mask " 
                                     << maskSize.width << "x" << maskSize.height 
                                     << " anchor (" << anchor.x << "," << anchor.y << ")";
    }
  }
  
  int width, height;
  int srcStep, dstStep;
  NppiSize roiSize;
  std::vector<Npp8u> h_src, h_dst;
  Npp8u *d_src = nullptr;
  Npp8u *d_dst = nullptr;
};

// Test small masks that should work in both modes
TEST_F(NppiFilterBoxModesTest, SmallMaskTest) {
  // 3x3 mask with center anchor
  TestFilterBox({3, 3}, {1, 1}, true);
  
  // 5x5 mask with center anchor  
  TestFilterBox({5, 5}, {2, 2}, true);
  
  // 3x3 mask with corner anchor
  TestFilterBox({3, 3}, {0, 0}, true);
  TestFilterBox({3, 3}, {2, 2}, true);
}

// Test edge cases
TEST_F(NppiFilterBoxModesTest, EdgeCasesTest) {
  // 1x1 mask (should just copy)
  TestFilterBox({1, 1}, {0, 0}, true);
  
  // Non-square masks
  TestFilterBox({5, 3}, {2, 1}, true);
  TestFilterBox({3, 5}, {1, 2}, true);
  
  // Asymmetric anchors
  TestFilterBox({4, 4}, {0, 0}, true);
  TestFilterBox({4, 4}, {3, 3}, true);
}

// Test masks larger than ROI 
// In zero padding mode, these should succeed
// In direct access mode, these should fail (unless we have sufficient buffer)
TEST_F(NppiFilterBoxModesTest, LargeMaskTest) {
  // Mask equal to ROI size
  TestFilterBox({10, 10}, {5, 5}, true);
  
  // Mask larger than ROI - behavior depends on mode
  // Note: In direct access mode this would fail with current validation
  // In zero padding mode this would succeed
#ifdef MPP_FILTERBOX_ZERO_PADDING
  TestFilterBox({15, 15}, {7, 7}, true);
  TestFilterBox({20, 20}, {10, 10}, true);
#else
  TestFilterBox({15, 15}, {7, 7}, false);
  TestFilterBox({20, 20}, {10, 10}, false);
#endif
}

// Test with larger buffer to verify direct access mode
TEST_F(NppiFilterBoxModesTest, LargerBufferTest) {
  // Create a larger buffer with padding
  int paddedWidth = width + 10;
  int paddedHeight = height + 10;
  int paddedSrcStep = paddedWidth * sizeof(Npp8u);
  
  std::vector<Npp8u> h_padded_src(paddedWidth * paddedHeight, 0);
  
  // Copy original data to center of padded buffer
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      h_padded_src[(y + 5) * paddedWidth + (x + 5)] = h_src[y * width + x];
    }
  }
  
  Npp8u *d_padded_src = nullptr;
  ASSERT_EQ(cudaMalloc(&d_padded_src, paddedWidth * paddedHeight * sizeof(Npp8u)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_padded_src, h_padded_src.data(), paddedWidth * paddedHeight * sizeof(Npp8u), 
                       cudaMemcpyHostToDevice), cudaSuccess);
  
  // Point to the ROI within the padded buffer
  Npp8u *d_src_roi = d_padded_src + 5 * paddedWidth + 5;
  
  // Now larger masks should work even in direct access mode
  NppStatus status = nppiFilterBox_8u_C1R(d_src_roi, paddedSrcStep, d_dst, dstStep, roiSize, {7, 7}, {3, 3});
  
#ifdef MPP_FILTERBOX_ZERO_PADDING
  // In zero padding mode, this should always work
  EXPECT_EQ(status, NPP_SUCCESS);
#else
  // In direct access mode, this works because we have sufficient buffer
  EXPECT_EQ(status, NPP_SUCCESS);
#endif
  
  cudaFree(d_padded_src);
}

// Test 32-bit float version
TEST_F(NppiFilterBoxModesTest, Float32Test) {
  std::vector<Npp32f> h_src_32f(width * height);
  std::vector<Npp32f> h_dst_32f(width * height);
  
  // Initialize with float data
  for (int i = 0; i < width * height; i++) {
    h_src_32f[i] = static_cast<float>(i) / 100.0f;
  }
  
  Npp32f *d_src_32f = nullptr;
  Npp32f *d_dst_32f = nullptr;
  
  int srcStep32f = width * sizeof(Npp32f);
  int dstStep32f = width * sizeof(Npp32f);
  
  ASSERT_EQ(cudaMalloc(&d_src_32f, width * height * sizeof(Npp32f)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_dst_32f, width * height * sizeof(Npp32f)), cudaSuccess);
  
  ASSERT_EQ(cudaMemcpy(d_src_32f, h_src_32f.data(), width * height * sizeof(Npp32f), cudaMemcpyHostToDevice), cudaSuccess);
  
  // Test 3x3 filter
  NppStatus status = nppiFilterBox_32f_C1R(d_src_32f, srcStep32f, d_dst_32f, dstStep32f, roiSize, {3, 3}, {1, 1});
  EXPECT_EQ(status, NPP_SUCCESS);
  
  // Copy result back
  ASSERT_EQ(cudaMemcpy(h_dst_32f.data(), d_dst_32f, width * height * sizeof(Npp32f), cudaMemcpyDeviceToHost), cudaSuccess);
  
  // Verify averaging occurred
  bool hasAveraging = false;
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      float expected = 0.0f;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          expected += h_src_32f[(y + dy) * width + (x + dx)];
        }
      }
      expected /= 9.0f;
      
      float actual = h_dst_32f[y * width + x];
      if (std::abs(actual - expected) < 0.001f) {
        hasAveraging = true;
        break;
      }
    }
    if (hasAveraging) break;
  }
  EXPECT_TRUE(hasAveraging) << "Box filter should perform averaging";
  
  cudaFree(d_src_32f);
  cudaFree(d_dst_32f);
}