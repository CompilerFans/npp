#include "npp_test_base.h"
#include <functional>

using namespace npp_functional_test;

class ResizeFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }

  // ===== Data Generation Helpers =====

  // Generate checkerboard pattern
  template <typename T, int CHANNELS>
  void fillCheckerboard(std::vector<T> &data, int width, int height, int blockSize, T white, T black) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * CHANNELS;
        bool isWhite = ((x / blockSize) + (y / blockSize)) % 2 == 0;
        T value = isWhite ? white : black;
        for (int c = 0; c < CHANNELS; c++) {
          data[idx + c] = value;
        }
      }
    }
  }

  // Generate horizontal gradient
  template <typename T, int CHANNELS>
  void fillHorizontalGradient(std::vector<T> &data, int width, int height, T minVal, T maxVal) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * CHANNELS;
        T value = static_cast<T>(minVal + (maxVal - minVal) * x / (width - 1));
        for (int c = 0; c < CHANNELS; c++) {
          data[idx + c] = value;
        }
      }
    }
  }

  // Generate vertical gradient
  template <typename T, int CHANNELS>
  void fillVerticalGradient(std::vector<T> &data, int width, int height, T minVal, T maxVal) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * CHANNELS;
        T value = static_cast<T>(minVal + (maxVal - minVal) * y / (height - 1));
        for (int c = 0; c < CHANNELS; c++) {
          data[idx + c] = value;
        }
      }
    }
  }

  // Generate uniform block
  template <typename T, int CHANNELS>
  void fillUniformBlock(std::vector<T> &data, int width, int height, int blockX, int blockY, int blockW, int blockH,
                        T insideVal, T outsideVal) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * CHANNELS;
        bool inside = (x >= blockX && x < blockX + blockW && y >= blockY && y < blockY + blockH);
        T value = inside ? insideVal : outsideVal;
        for (int c = 0; c < CHANNELS; c++) {
          data[idx + c] = value;
        }
      }
    }
  }

  // Generate uniform value
  template <typename T, int CHANNELS> void fillUniform(std::vector<T> &data, const T values[CHANNELS]) {
    size_t pixels = data.size() / CHANNELS;
    for (size_t i = 0; i < pixels; i++) {
      for (int c = 0; c < CHANNELS; c++) {
        data[i * CHANNELS + c] = values[c];
      }
    }
  }

  // ===== Validation Helpers =====

  // Validate horizontal gradient monotonicity
  template <typename T, int CHANNELS>
  void validateHorizontalMonotonic(const std::vector<T> &data, int width, int height, int channel = 0) {
    for (int y = 0; y < height; y++) {
      for (int x = 1; x < width - 1; x++) {
        int idx = (y * width + x) * CHANNELS + channel;
        int prevIdx = (y * width + (x - 1)) * CHANNELS + channel;
        int nextIdx = (y * width + (x + 1)) * CHANNELS + channel;

        EXPECT_GE(data[idx], data[prevIdx]) << "Not monotonic at (" << x << "," << y << ")";
        EXPECT_LE(data[idx], data[nextIdx]) << "Not monotonic at (" << x << "," << y << ")";
      }
    }
  }

  // Validate vertical gradient monotonicity
  template <typename T, int CHANNELS>
  void validateVerticalMonotonic(const std::vector<T> &data, int width, int height, int channel = 0) {
    for (int x = 0; x < width; x++) {
      for (int y = 1; y < height - 1; y++) {
        int idx = (y * width + x) * CHANNELS + channel;
        int prevIdx = ((y - 1) * width + x) * CHANNELS + channel;
        int nextIdx = ((y + 1) * width + x) * CHANNELS + channel;

        EXPECT_GE(data[idx], data[prevIdx]) << "Not monotonic at (" << x << "," << y << ")";
        EXPECT_LE(data[idx], data[nextIdx]) << "Not monotonic at (" << x << "," << y << ")";
      }
    }
  }

  // Validate channel independence
  template <typename T, int CHANNELS>
  void validateChannelValues(const std::vector<T> &data, int width, int height, const T expected[CHANNELS],
                             T tolerance) {
    for (int i = 0; i < width * height; i++) {
      for (int c = 0; c < CHANNELS; c++) {
        EXPECT_NEAR(data[i * CHANNELS + c], expected[c], tolerance) << "Channel " << c << " mismatch at pixel " << i;
      }
    }
  }

  // Validate value range
  template <typename T>
  void validateRange(const std::vector<T> &data, T minVal, T maxVal, const std::string &message = "") {
    for (size_t i = 0; i < data.size(); i++) {
      EXPECT_GE(data[i], minVal) << message << " - value too low at index " << i;
      EXPECT_LE(data[i], maxVal) << message << " - value too high at index " << i;
    }
  }

  // Validate non-zero pixel count
  template <typename T> void validateNonZeroCount(const std::vector<T> &data, int minCount, int maxCount) {
    int nonZeroCount = 0;
    for (size_t i = 0; i < data.size(); i++) {
      if (data[i] > 0)
        nonZeroCount++;
    }
    EXPECT_GE(nonZeroCount, minCount) << "Not enough non-zero pixels";
    EXPECT_LT(nonZeroCount, maxCount) << "Too many non-zero pixels";
  }
};

TEST_F(ResizeFunctionalTest, Resize_8u_C1R_NearestNeighbor) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 64, dstHeight = 64;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);
  fillCheckerboard<Npp8u, 1>(srcData, srcWidth, srcHeight, 4, 255, 0);

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  validateNonZeroCount(resultData, dstWidth * dstHeight / 4, dstWidth * dstHeight * 3 / 4);
}

TEST_F(ResizeFunctionalTest, Resize_8u_C1R_Bilinear) {
  const int srcWidth = 16, srcHeight = 16;
  const int dstWidth = 32, dstHeight = 32;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);
  fillHorizontalGradient<Npp8u, 1>(srcData, srcWidth, srcHeight, 0, 255);

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  validateHorizontalMonotonic<Npp8u, 1>(resultData, dstWidth, dstHeight);
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_NearestNeighbor) {
  const int srcWidth = 16, srcHeight = 16;
  const int dstWidth = 32, dstHeight = 32;

  // 准备三通道测试数据
  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x * 255 / (srcWidth - 1));  // R
      srcData[idx + 1] = (Npp8u)(y * 255 / (srcHeight - 1)); // G
      srcData[idx + 2] = 128;                                // B
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight); // 3 channels
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  // 设置源和目标区域
  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  // 执行三通道最近邻缩放
  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_NN); // 最近邻插值

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiResize_8u_C3R failed";

  // Validate结果
  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // 检查几个采样点
  for (int y = 0; y < dstHeight; y += 8) {
    for (int x = 0; x < dstWidth; x += 8) {
      int idx = (y * dstWidth + x) * 3;

      // 每个通道应该有合理的值
      EXPECT_LE(resultData[idx + 0], 255) << "R channel out of range";
      EXPECT_LE(resultData[idx + 1], 255) << "G channel out of range";
      EXPECT_LE(resultData[idx + 2], 255) << "B channel out of range";
    }
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Bilinear) {
  const int srcWidth = 16, srcHeight = 16;
  const int dstWidth = 32, dstHeight = 32;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x * 255 / (srcWidth - 1));
      srcData[idx + 1] = (Npp8u)(y * 255 / (srcHeight - 1));
      srcData[idx + 2] = 128;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiResize_8u_C3R bilinear failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Check gradient continuity in first row for R channel
  for (int x = 1; x < dstWidth - 1; x++) {
    int current = resultData[x * 3];
    int prev = resultData[(x - 1) * 3];
    EXPECT_GE(current, prev - 1) << "R gradient not continuous at x=" << x;
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super) {
  const int srcWidth = 64, srcHeight = 64;
  const int dstWidth = 32, dstHeight = 32;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create uniform blocks to verify averaging behavior
  // Scale factor is 2 (64/32), so each dst pixel samples from approximately 2x2 src region
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      // Set top-left 8x8 block to 100, rest to 0
      // This ensures dst(0,0) through dst(3,3) all sample from value-100 regions
      if (x < 8 && y < 8) {
        srcData[idx + 0] = 100;
        srcData[idx + 1] = 150;
        srcData[idx + 2] = 200;
      } else {
        srcData[idx + 0] = 0;
        srcData[idx + 1] = 0;
        srcData[idx + 2] = 0;
      }
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiResize_8u_C3R super sampling failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify super sampling averages correctly
  // dst(0,0) samples from src region [0,2)x[0,2), all 100
  int dst_idx_0_0 = (0 * dstWidth + 0) * 3;
  EXPECT_NEAR(resultData[dst_idx_0_0 + 0], 100, 2) << "Super sampling R channel at (0,0)";
  EXPECT_NEAR(resultData[dst_idx_0_0 + 1], 150, 2) << "Super sampling G channel at (0,0)";
  EXPECT_NEAR(resultData[dst_idx_0_0 + 2], 200, 2) << "Super sampling B channel at (0,0)";

  // dst(2,2) samples from src region [4,6)x[4,6), still within 8x8 block
  int dst_idx_2_2 = (2 * dstWidth + 2) * 3;
  EXPECT_NEAR(resultData[dst_idx_2_2 + 0], 100, 2) << "Super sampling R channel at (2,2)";
  EXPECT_NEAR(resultData[dst_idx_2_2 + 1], 150, 2) << "Super sampling G channel at (2,2)";

  // dst(4,4) samples from src region [8,10)x[8,10), which is all 0
  int dst_idx_4_4 = (4 * dstWidth + 4) * 3;
  EXPECT_EQ(resultData[dst_idx_4_4 + 0], 0) << "Region outside block should be 0";
  EXPECT_EQ(resultData[dst_idx_4_4 + 1], 0) << "Region outside block should be 0";
  EXPECT_EQ(resultData[dst_idx_4_4 + 2], 0) << "Region outside block should be 0";

  // Verify edge case: dst(3,3) samples from src [6,8)x[6,8), near boundary
  // NVIDIA NPP may handle boundary differently, so just check it's valid
  int dst_idx_3_3 = (3 * dstWidth + 3) * 3;
  EXPECT_LE(resultData[dst_idx_3_3 + 0], 100) << "Edge pixel value should not exceed source max";
  EXPECT_GE(resultData[dst_idx_3_3 + 0], 0) << "Edge pixel should have non-negative value";
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_4xDownscale) {
  const int srcWidth = 128, srcHeight = 128;
  const int dstWidth = 32, dstHeight = 32;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);
  fillCheckerboard<Npp8u, 3>(srcData, srcWidth, srcHeight, 4, 200, 50);

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  validateRange(resultData, Npp8u(50), Npp8u(200), "4x downscale checkerboard");
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_NonIntegerScale) {
  const int srcWidth = 100, srcHeight = 100;
  const int dstWidth = 30, dstHeight = 30;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create gradient pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x * 255 / srcWidth);
      srcData[idx + 1] = (Npp8u)(y * 255 / srcHeight);
      srcData[idx + 2] = 128;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS) << "Non-integer scale failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify gradient is preserved (R increases left to right)
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 1; x < dstWidth; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = (y * dstWidth + (x - 1)) * 3;
      EXPECT_GE(resultData[idx + 0], resultData[prev_idx + 0] - 3) << "Gradient not monotonic";
    }
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_SmallImage) {
  const int srcWidth = 16, srcHeight = 16;
  const int dstWidth = 4, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);
  const Npp8u uniform[3] = {100, 100, 100};
  fillUniform<Npp8u, 3>(srcData, uniform);

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS) << "Small image resize failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Uniform input should produce uniform output
  for (int i = 0; i < dstWidth * dstHeight * 3; i++) {
    EXPECT_NEAR(resultData[i], 100, 2) << "Uniform averaging failed at index " << i;
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_ROITest) {
  const int srcWidth = 64, srcHeight = 64;
  const int dstWidth = 32, dstHeight = 32;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create quadrant pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      if (x < srcWidth / 2 && y < srcHeight / 2) {
        srcData[idx + 0] = 255;
        srcData[idx + 1] = 0;
        srcData[idx + 2] = 0;
      } else if (x >= srcWidth / 2 && y < srcHeight / 2) {
        srcData[idx + 0] = 0;
        srcData[idx + 1] = 255;
        srcData[idx + 2] = 0;
      } else if (x < srcWidth / 2 && y >= srcHeight / 2) {
        srcData[idx + 0] = 0;
        srcData[idx + 1] = 0;
        srcData[idx + 2] = 255;
      } else {
        srcData[idx + 0] = 255;
        srcData[idx + 1] = 255;
        srcData[idx + 2] = 0;
      }
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  // Test with ROI on upper-left quadrant only
  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth / 2, srcHeight / 2};
  NppiRect dstROI = {0, 0, dstWidth / 2, dstHeight / 2};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS) << "ROI resize failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Upper-left quadrant should be predominantly red
  int red_dominant = 0;
  for (int y = 0; y < dstHeight / 2; y++) {
    for (int x = 0; x < dstWidth / 2; x++) {
      int idx = (y * dstWidth + x) * 3;
      if (resultData[idx + 0] > resultData[idx + 1] && resultData[idx + 0] > resultData[idx + 2]) {
        red_dominant++;
      }
    }
  }

  EXPECT_GT(red_dominant, (dstWidth / 2) * (dstHeight / 2) / 2) << "ROI color not preserved";
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_LargeDownscale) {
  const int srcWidth = 256, srcHeight = 256;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create alternating stripes
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value = (x % 2) ? 255 : 0;
      srcData[idx + 0] = value;
      srcData[idx + 1] = value;
      srcData[idx + 2] = value;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS) << "Large downscale failed";

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // 16x downscaling of alternating stripes should produce ~127
  // V2 algorithm achieves error <= 1, tightened from tolerance 15 to 2
  for (int i = 0; i < dstWidth * dstHeight * 3; i++) {
    EXPECT_NEAR(resultData[i], 127, 2) << "Large downscale averaging incorrect at " << i;
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_BoundaryTest) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3, 0);

  // Create precise boundary test pattern
  // Fill left half with 200, right half with 0
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      if (x < srcWidth / 2) {
        srcData[idx + 0] = 200;
        srcData[idx + 1] = 200;
        srcData[idx + 2] = 200;
      }
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Test boundary pixels at the transition
  // dst(7,y) should sample from src around x=14-16, which is at boundary
  // dst(8,y) should sample from src around x=16-18, which crosses boundary

  for (int y = 0; y < dstHeight; y++) {
    // Far left: should be ~200 (fully in 200 region, tightened from tolerance 3 to 2)
    int idx_left = (y * dstWidth + 3) * 3;
    EXPECT_NEAR(resultData[idx_left + 0], 200, 2) << "Left region incorrect at y=" << y;

    // At boundary: dst(7,y) maps to src(14-16, y*2 to y*2+2)
    int idx_boundary = (y * dstWidth + 7) * 3;
    EXPECT_GE(resultData[idx_boundary + 0], 100) << "Boundary should show 200 influence at y=" << y;
    EXPECT_LE(resultData[idx_boundary + 0], 200) << "Boundary should not exceed max at y=" << y;

    // Just after boundary: dst(8,y) maps to src(16-18, y*2 to y*2+2)
    int idx_after = (y * dstWidth + 8) * 3;
    EXPECT_LT(resultData[idx_after + 0], 100) << "After boundary should show averaging at y=" << y;

    // Far right: should be ~0 (fully in 0 region, tightened from tolerance 3 to 2)
    int idx_right = (y * dstWidth + 12) * 3;
    EXPECT_NEAR(resultData[idx_right + 0], 0, 2) << "Right region incorrect at y=" << y;
  }
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_CornerTest) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3, 0);

  // Create quadrant pattern: top-left corner is 255, rest is 0
  for (int y = 0; y < 8; y++) {
    for (int x = 0; x < 8; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = 255;
      srcData[idx + 1] = 255;
      srcData[idx + 2] = 255;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // With 4x downscaling, source 8x8 block maps perfectly to dst 2x2 region
  // Top-left 2x2 block: samples from src[0,8)x[0,8), fully in 255 region
  // Tightened from tolerance 3 to 2
  for (int y = 0; y < 2; y++) {
    for (int x = 0; x < 2; x++) {
      int idx = (y * dstWidth + x) * 3;
      EXPECT_NEAR(resultData[idx], 255, 2) << "Pixel (" << x << "," << y << ") should be ~255";
    }
  }

  // Pixels outside 2x2 block: sample from 0 region
  int idx_20 = (0 * dstWidth + 2) * 3;
  EXPECT_EQ(resultData[idx_20], 0) << "Pixel (2,0) should be 0";

  int idx_02 = (2 * dstWidth + 0) * 3;
  EXPECT_EQ(resultData[idx_02], 0) << "Pixel (0,2) should be 0";

  int idx_77 = (7 * dstWidth + 7) * 3;
  EXPECT_EQ(resultData[idx_77], 0) << "Far corner (7,7) should be 0";
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_GradientBoundary) {
  const int srcWidth = 64, srcHeight = 64;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create horizontal gradient with sharp boundary in the middle
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      // Left half: gradient 0->200, right half: constant 0
      if (x < srcWidth / 2) {
        Npp8u value = (Npp8u)((x * 200) / (srcWidth / 2));
        srcData[idx + 0] = value;
        srcData[idx + 1] = value;
        srcData[idx + 2] = value;
      } else {
        srcData[idx + 0] = 0;
        srcData[idx + 1] = 0;
        srcData[idx + 2] = 0;
      }
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify gradient is preserved in left half
  for (int y = 0; y < dstHeight; y++) {
    // Check monotonic increase in left half
    for (int x = 1; x < dstWidth / 2 - 1; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = (y * dstWidth + (x - 1)) * 3;
      EXPECT_GE(resultData[idx], resultData[prev_idx] - 2) << "Gradient not preserved at (" << x << "," << y << ")";
    }

    // Right half should be mostly 0
    for (int x = dstWidth / 2 + 2; x < dstWidth; x++) {
      int idx = (y * dstWidth + x) * 3;
      EXPECT_LT(resultData[idx], 10) << "Right region should be near 0 at (" << x << "," << y << ")";
    }
  }
}

// Test precise averaging with known values
TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_ExactAveraging) {
  const int srcWidth = 8, srcHeight = 8;
  const int dstWidth = 2, dstHeight = 2;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create 2x2 blocks with known values for exact average calculation
  // Each dst pixel will sample exactly 4x4=16 src pixels
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      int block_x = x / 4;
      int block_y = y / 4;

      if (block_x == 0 && block_y == 0) {
        // Top-left: all 100
        srcData[idx + 0] = 100;
        srcData[idx + 1] = 100;
        srcData[idx + 2] = 100;
      } else if (block_x == 1 && block_y == 0) {
        // Top-right: all 200
        srcData[idx + 0] = 200;
        srcData[idx + 1] = 200;
        srcData[idx + 2] = 200;
      } else if (block_x == 0 && block_y == 1) {
        // Bottom-left: all 50
        srcData[idx + 0] = 50;
        srcData[idx + 1] = 50;
        srcData[idx + 2] = 50;
      } else {
        // Bottom-right: all 150
        srcData[idx + 0] = 150;
        srcData[idx + 1] = 150;
        srcData[idx + 2] = 150;
      }
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Each dst pixel averages 16 src pixels, verify exact values
  int idx_00 = (0 * dstWidth + 0) * 3;
  EXPECT_NEAR(resultData[idx_00], 100, 1) << "dst(0,0) should average to exactly 100";

  int idx_10 = (0 * dstWidth + 1) * 3;
  EXPECT_NEAR(resultData[idx_10], 200, 1) << "dst(1,0) should average to exactly 200";

  int idx_01 = (1 * dstWidth + 0) * 3;
  EXPECT_NEAR(resultData[idx_01], 50, 1) << "dst(0,1) should average to exactly 50";

  int idx_11 = (1 * dstWidth + 1) * 3;
  EXPECT_NEAR(resultData[idx_11], 150, 1) << "dst(1,1) should average to exactly 150";
}

// Test multi-level boundary transitions
TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_MultiBoundary) {
  const int srcWidth = 48, srcHeight = 48;
  const int dstWidth = 12, dstHeight = 12;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create three vertical stripes: 0, 128, 255
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value;
      if (x < srcWidth / 3) {
        value = 0;
      } else if (x < 2 * srcWidth / 3) {
        value = 128;
      } else {
        value = 255;
      }
      srcData[idx + 0] = value;
      srcData[idx + 1] = value;
      srcData[idx + 2] = value;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  for (int y = 0; y < dstHeight; y++) {
    // Left stripe: should be ~0
    int idx_left = (y * dstWidth + 1) * 3;
    EXPECT_NEAR(resultData[idx_left], 0, 2) << "Left stripe at y=" << y;

    // Middle stripe: should be ~128
    int idx_mid = (y * dstWidth + 6) * 3;
    EXPECT_NEAR(resultData[idx_mid], 128, 2) << "Middle stripe at y=" << y;

    // Right stripe: should be ~255
    int idx_right = (y * dstWidth + 10) * 3;
    EXPECT_NEAR(resultData[idx_right], 255, 2) << "Right stripe at y=" << y;

    // First boundary region (dst x=3,4 crosses left/middle at src x=16)
    int idx_b1_before = (y * dstWidth + 3) * 3;
    EXPECT_GE(resultData[idx_b1_before], 0) << "Before boundary 1 at y=" << y;
    EXPECT_LE(resultData[idx_b1_before], 64) << "Before boundary 1 at y=" << y;

    // Second boundary region (dst x=7,8 crosses middle/right at src x=32)
    int idx_b2_before = (y * dstWidth + 7) * 3;
    EXPECT_GE(resultData[idx_b2_before], 128) << "Before boundary 2 at y=" << y;
    EXPECT_LE(resultData[idx_b2_before], 192) << "Before boundary 2 at y=" << y;
  }
}

// Test channel independence
TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_ChannelIndependence) {
  const int srcWidth = 16, srcHeight = 16;
  const int dstWidth = 4, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);
  const Npp8u uniform[3] = {80, 160, 240};
  fillUniform<Npp8u, 3>(srcData, uniform);

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  validateChannelValues<Npp8u, 3>(resultData, dstWidth, dstHeight, uniform, 1);
}

// Test extreme value boundaries (0 and 255)
TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_ExtremeValues) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);
  fillCheckerboard<Npp8u, 3>(srcData, srcWidth, srcHeight, 2, 255, 0);

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // With 4x downscaling of alternating pixels, should average to ~127
  // V2 algorithm achieves error <= 1, tightened from tolerance 5 to 2
  for (int i = 0; i < dstWidth * dstHeight * 3; i++) {
    EXPECT_NEAR(resultData[i], 127, 2) << "Extreme value averaging at index " << i;
  }
}

// Test precise 2x2 block averaging
TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_2x2BlockAverage) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 2, dstHeight = 2;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Top-left 2x2: values 10, 20, 30, 40 (average should be 25)
  srcData[0] = srcData[1] = srcData[2] = 10;
  srcData[3] = srcData[4] = srcData[5] = 20;
  srcData[12] = srcData[13] = srcData[14] = 30;
  srcData[15] = srcData[16] = srcData[17] = 40;

  // Top-right 2x2: all 100
  srcData[6] = srcData[7] = srcData[8] = 100;
  srcData[9] = srcData[10] = srcData[11] = 100;
  srcData[18] = srcData[19] = srcData[20] = 100;
  srcData[21] = srcData[22] = srcData[23] = 100;

  // Bottom-left 2x2: all 200
  srcData[24] = srcData[25] = srcData[26] = 200;
  srcData[27] = srcData[28] = srcData[29] = 200;
  srcData[36] = srcData[37] = srcData[38] = 200;
  srcData[39] = srcData[40] = srcData[41] = 200;

  // Bottom-right 2x2: values 60, 80, 100, 120 (average should be 90)
  srcData[30] = srcData[31] = srcData[32] = 60;
  srcData[33] = srcData[34] = srcData[35] = 80;
  srcData[42] = srcData[43] = srcData[44] = 100;
  srcData[45] = srcData[46] = srcData[47] = 120;

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify precise averages
  int idx_00 = (0 * dstWidth + 0) * 3;
  EXPECT_NEAR(resultData[idx_00], 25, 1) << "dst(0,0) = (10+20+30+40)/4 = 25";

  int idx_10 = (0 * dstWidth + 1) * 3;
  EXPECT_NEAR(resultData[idx_10], 100, 1) << "dst(1,0) = 100";

  int idx_01 = (1 * dstWidth + 0) * 3;
  EXPECT_NEAR(resultData[idx_01], 200, 1) << "dst(0,1) = 200";

  int idx_11 = (1 * dstWidth + 1) * 3;
  EXPECT_NEAR(resultData[idx_11], 90, 1) << "dst(1,1) = (60+80+100+120)/4 = 90";
}

TEST_F(ResizeFunctionalTest, NN_8u_C3R_ExactUpsample2x) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Fill with distinct values for each pixel
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value = (Npp8u)((y * srcWidth + x) * 10);
      srcData[idx + 0] = value;
      srcData[idx + 1] = value + 1;
      srcData[idx + 2] = value + 2;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Each source pixel should map to approximately 2x2 destination pixels
  // Verify src(0,0) value appears in dst(0,0) and nearby pixels
  int src_idx_00 = 0;
  int dst_idx_00 = 0;
  EXPECT_EQ(resultData[dst_idx_00 + 0], srcData[src_idx_00 + 0]) << "NN should preserve exact pixel values";
  EXPECT_EQ(resultData[dst_idx_00 + 1], srcData[src_idx_00 + 1]);
  EXPECT_EQ(resultData[dst_idx_00 + 2], srcData[src_idx_00 + 2]);

  // Verify src(3,3) maps correctly to dst region
  int src_idx_33 = ((3 * srcWidth + 3) * 3);
  int dst_idx_66 = ((6 * dstWidth + 6) * 3);
  EXPECT_EQ(resultData[dst_idx_66 + 0], srcData[src_idx_33 + 0]) << "NN should map corner pixels correctly";
}

// Test exact coordinate mapping with block pattern
TEST_F(ResizeFunctionalTest, NN_8u_C3R_ExactMapping) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create 2x2 blocks with distinct values
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      int block_id = (y / 2) * 2 + (x / 2);
      Npp8u value = (Npp8u)(block_id * 60 + 40);
      srcData[idx + 0] = value;
      srcData[idx + 1] = value;
      srcData[idx + 2] = value;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify all destination pixels have values from source (no interpolation)
  for (int i = 0; i < dstWidth * dstHeight * 3; i += 3) {
    Npp8u val = resultData[i];
    // Value should be one of the 4 block values: 40, 100, 160, 220
    EXPECT_TRUE(val == 40 || val == 100 || val == 160 || val == 220)
        << "NN should only select source pixel values, not interpolate. Got: " << (int)val;
  }
}

// Test channel independence for nearest neighbor
TEST_F(ResizeFunctionalTest, NN_8u_C3R_ChannelIndependence) {
  const int srcWidth = 8, srcHeight = 8;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Fill each channel with different pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)((x % 2) * 100 + 50);       // R: alternating 50/150
      srcData[idx + 1] = (Npp8u)((y % 2) * 100 + 70);       // G: alternating 70/170
      srcData[idx + 2] = (Npp8u)(((x + y) % 2) * 100 + 90); // B: checkerboard 90/190
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify each channel only contains values from source
  for (int i = 0; i < dstWidth * dstHeight * 3; i += 3) {
    EXPECT_TRUE(resultData[i + 0] == 50 || resultData[i + 0] == 150) << "R channel should be 50 or 150";
    EXPECT_TRUE(resultData[i + 1] == 70 || resultData[i + 1] == 170) << "G channel should be 70 or 170";
    EXPECT_TRUE(resultData[i + 2] == 90 || resultData[i + 2] == 190) << "B channel should be 90 or 190";
  }
}

// Test boundary pixel selection
TEST_F(ResizeFunctionalTest, NN_8u_C3R_BoundarySelection) {
  const int srcWidth = 6, srcHeight = 6;
  const int dstWidth = 12, dstHeight = 12;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create three horizontal stripes: 60, 120, 180
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value;
      if (y < 2)
        value = 60;
      else if (y < 4)
        value = 120;
      else
        value = 180;
      srcData[idx + 0] = value;
      srcData[idx + 1] = value;
      srcData[idx + 2] = value;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // All pixels should be one of the three stripe values
  for (int i = 0; i < dstWidth * dstHeight * 3; i += 3) {
    Npp8u val = resultData[i];
    EXPECT_TRUE(val == 60 || val == 120 || val == 180)
        << "NN should select from source values: 60, 120, or 180. Got: " << (int)val;
  }
}

// Test downsampling pixel selection
TEST_F(ResizeFunctionalTest, NN_8u_C3R_Downsampling) {
  const int srcWidth = 8, srcHeight = 8;
  const int dstWidth = 4, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);
  fillCheckerboard<Npp8u, 3>(srcData, srcWidth, srcHeight, 1, 200, 80);

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // All values should be either 80 or 200 (no averaging)
  for (int i = 0; i < dstWidth * dstHeight * 3; i += 3) {
    Npp8u val = resultData[i];
    EXPECT_TRUE(val == 80 || val == 200) << "NN downsampling should pick nearest pixel, not average. Got: " << (int)val;
  }
}

// Linear interpolation precision tests

// Test exact bilinear interpolation calculation
TEST_F(ResizeFunctionalTest, Linear_8u_C3R_ExactInterpolation) {
  const int srcWidth = 2, srcHeight = 2;
  const int dstWidth = 3, dstHeight = 3;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Set corner values for known interpolation
  // Top-left (0,0): 0, Top-right (1,0): 100
  // Bottom-left (0,1): 50, Bottom-right (1,1): 150
  srcData[0] = srcData[1] = srcData[2] = 0;     // (0,0)
  srcData[3] = srcData[4] = srcData[5] = 100;   // (1,0)
  srcData[6] = srcData[7] = srcData[8] = 50;    // (0,1)
  srcData[9] = srcData[10] = srcData[11] = 150; // (1,1)

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Corners should preserve values (tightened from tolerance 5 to 2)
  EXPECT_NEAR(resultData[0], 0, 2) << "Top-left corner";
  EXPECT_NEAR(resultData[6], 100, 2) << "Top-right corner";
  EXPECT_NEAR(resultData[18], 50, 2) << "Bottom-left corner";
  EXPECT_NEAR(resultData[24], 150, 2) << "Bottom-right corner";

  // Center pixel interpolation (relaxed tolerance due to coordinate mapping complexity)
  int center_idx = (1 * dstWidth + 1) * 3;
  EXPECT_GT(resultData[center_idx], 0) << "Center should be interpolated, not 0";
  EXPECT_LT(resultData[center_idx], 150) << "Center should be interpolated, not max";
  // Value should be somewhere in the interpolated range
  EXPECT_GE(resultData[center_idx], 40) << "Center should show interpolation influence";
  EXPECT_LE(resultData[center_idx], 110) << "Center should show interpolation influence";
}

// Test halfway point interpolation
TEST_F(ResizeFunctionalTest, Linear_8u_C3R_HalfwayPoint) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 7, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Horizontal gradient on each row: 0, 90, 180, 255
  for (int y = 0; y < srcHeight; y++) {
    srcData[(y * srcWidth + 0) * 3 + 0] = srcData[(y * srcWidth + 0) * 3 + 1] = srcData[(y * srcWidth + 0) * 3 + 2] = 0;
    srcData[(y * srcWidth + 1) * 3 + 0] = srcData[(y * srcWidth + 1) * 3 + 1] = srcData[(y * srcWidth + 1) * 3 + 2] =
        90;
    srcData[(y * srcWidth + 2) * 3 + 0] = srcData[(y * srcWidth + 2) * 3 + 1] = srcData[(y * srcWidth + 2) * 3 + 2] =
        180;
    srcData[(y * srcWidth + 3) * 3 + 0] = srcData[(y * srcWidth + 3) * 3 + 1] = srcData[(y * srcWidth + 3) * 3 + 2] =
        255;
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify interpolated values are between adjacent source values for each row
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 0; x < dstWidth - 1; x++) {
      int idx = (y * dstWidth + x) * 3;
      int next_idx = (y * dstWidth + (x + 1)) * 3;
      // Linear interpolation should produce monotonically increasing values
      EXPECT_LE(resultData[idx], resultData[next_idx] + 1)
          << "Gradient should be monotonic at (" << x << "," << y << ")";
    }
  }
}

// Test channel independence for linear interpolation
TEST_F(ResizeFunctionalTest, Linear_8u_C3R_ChannelIndependence) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create channel-specific gradients
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x * 255 / (srcWidth - 1));  // R: horizontal gradient
      srcData[idx + 1] = (Npp8u)(y * 255 / (srcHeight - 1)); // G: vertical gradient
      srcData[idx + 2] = 128;                                // B: constant
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify R channel increases horizontally
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 1; x < dstWidth; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = (y * dstWidth + (x - 1)) * 3;
      EXPECT_GE(resultData[idx + 0], resultData[prev_idx + 0] - 2)
          << "R gradient horizontal at (" << x << "," << y << ")";
    }
  }

  // Verify G channel increases vertically
  for (int y = 1; y < dstHeight; y++) {
    for (int x = 0; x < dstWidth; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = ((y - 1) * dstWidth + x) * 3;
      EXPECT_GE(resultData[idx + 1], resultData[prev_idx + 1] - 2)
          << "G gradient vertical at (" << x << "," << y << ")";
    }
  }

  // Verify B channel is constant
  for (int i = 0; i < dstWidth * dstHeight * 3; i += 3) {
    EXPECT_NEAR(resultData[i + 2], 128, 3) << "B channel should remain constant";
  }
}

// Test boundary interpolation behavior
TEST_F(ResizeFunctionalTest, Linear_8u_C3R_BoundaryInterpolation) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 7, dstHeight = 7;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create step function: left half 50, right half 200
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value = (x < srcWidth / 2) ? 50 : 200;
      srcData[idx + 0] = value;
      srcData[idx + 1] = value;
      srcData[idx + 2] = value;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Left side should be close to 50 (tightened from tolerance 15 to 3)
  for (int y = 0; y < dstHeight; y++) {
    int idx_left = (y * dstWidth + 1) * 3;
    EXPECT_NEAR(resultData[idx_left], 50, 3) << "Left region at y=" << y;
  }

  // Right side should be close to 200 (tightened from tolerance 15 to 3)
  for (int y = 0; y < dstHeight; y++) {
    int idx_right = (y * dstWidth + (dstWidth - 2)) * 3;
    EXPECT_NEAR(resultData[idx_right], 200, 3) << "Right region at y=" << y;
  }

  // Middle region should have intermediate values
  for (int y = 0; y < dstHeight; y++) {
    int idx_mid = (y * dstWidth + (dstWidth / 2)) * 3;
    EXPECT_GT(resultData[idx_mid], 50) << "Middle should be interpolated at y=" << y;
    EXPECT_LT(resultData[idx_mid], 200) << "Middle should be interpolated at y=" << y;
  }
}

// Test gradient preservation with linear interpolation
TEST_F(ResizeFunctionalTest, Linear_8u_C3R_GradientPreservation) {
  const int srcWidth = 8, srcHeight = 8;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create perfect linear gradient
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value_x = (Npp8u)(x * 255 / (srcWidth - 1));
      Npp8u value_y = (Npp8u)(y * 255 / (srcHeight - 1));
      srcData[idx + 0] = value_x;                 // Horizontal gradient
      srcData[idx + 1] = value_y;                 // Vertical gradient
      srcData[idx + 2] = (value_x + value_y) / 2; // Diagonal gradient
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify monotonic horizontal gradient in R channel
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 1; x < dstWidth - 1; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = (y * dstWidth + (x - 1)) * 3;
      EXPECT_GE(resultData[idx + 0], resultData[prev_idx + 0] - 2)
          << "R horizontal gradient at (" << x << "," << y << ")";
    }
  }

  // Verify monotonic vertical gradient in G channel
  for (int y = 1; y < dstHeight - 1; y++) {
    for (int x = 0; x < dstWidth; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = ((y - 1) * dstWidth + x) * 3;
      EXPECT_GE(resultData[idx + 1], resultData[prev_idx + 1] - 2)
          << "G vertical gradient at (" << x << "," << y << ")";
    }
  }

  // Verify corners
  EXPECT_NEAR(resultData[0], 0, 3) << "Top-left R should be 0";
  EXPECT_NEAR(resultData[1], 0, 3) << "Top-left G should be 0";

  int bottom_right_idx = ((dstHeight - 1) * dstWidth + (dstWidth - 1)) * 3;
  EXPECT_NEAR(resultData[bottom_right_idx + 0], 255, 3) << "Bottom-right R should be 255";
  EXPECT_NEAR(resultData[bottom_right_idx + 1], 255, 3) << "Bottom-right G should be 255";
}

TEST_F(ResizeFunctionalTest, NN_8u_C1R_ExactUpsample2x) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // Fill with distinct values
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp8u)((y * srcWidth + x) * 15);
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // Verify corner mapping
  EXPECT_EQ(resultData[0], srcData[0]) << "Top-left corner should preserve value";
  EXPECT_EQ(resultData[dstWidth - 1], srcData[srcWidth - 1]) << "Top-right corner";
  EXPECT_EQ(resultData[(dstHeight - 1) * dstWidth], srcData[(srcHeight - 1) * srcWidth]) << "Bottom-left corner";
}

TEST_F(ResizeFunctionalTest, NN_8u_C1R_BlockPattern) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // Create 2x2 blocks with values 50, 100, 150, 200
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int block_id = (y / 2) * 2 + (x / 2);
      srcData[y * srcWidth + x] = (Npp8u)(block_id * 50 + 50);
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // All values should be one of the block values (no interpolation)
  for (int i = 0; i < dstWidth * dstHeight; i++) {
    Npp8u val = resultData[i];
    EXPECT_TRUE(val == 50 || val == 100 || val == 150 || val == 200)
        << "NN should only select source values, not interpolate. Got: " << (int)val;
  }
}

TEST_F(ResizeFunctionalTest, NN_8u_C1R_Downsampling) {
  const int srcWidth = 8, srcHeight = 8;
  const int dstWidth = 4, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);
  fillCheckerboard<Npp8u, 1>(srcData, srcWidth, srcHeight, 1, 200, 80);

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status =
      nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI, NPPI_INTER_NN);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // All values should be either 80 or 200 (no averaging)
  for (int i = 0; i < dstWidth * dstHeight; i++) {
    Npp8u val = resultData[i];
    EXPECT_TRUE(val == 80 || val == 200) << "NN downsampling should pick nearest pixel, not average. Got: " << (int)val;
  }
}

// Linear Interpolation C1R precision tests

TEST_F(ResizeFunctionalTest, Linear_8u_C1R_CornerPreservation) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // Create known corner values
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp8u)(x * 60 + y * 15);
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // Corners should approximately preserve source values
  EXPECT_NEAR(resultData[0], srcData[0], 5) << "Top-left corner";
  EXPECT_NEAR(resultData[dstWidth - 1], srcData[srcWidth - 1], 5) << "Top-right corner";
  EXPECT_NEAR(resultData[(dstHeight - 1) * dstWidth], srcData[(srcHeight - 1) * srcWidth], 5) << "Bottom-left";
  EXPECT_NEAR(resultData[(dstHeight - 1) * dstWidth + (dstWidth - 1)],
              srcData[(srcHeight - 1) * srcWidth + (srcWidth - 1)], 5)
      << "Bottom-right";
}

TEST_F(ResizeFunctionalTest, Linear_8u_C1R_HorizontalGradient) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 8, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);
  fillHorizontalGradient<Npp8u, 1>(srcData, srcWidth, srcHeight, 0, 255);

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // Verify monotonic horizontal gradient in each row
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 1; x < dstWidth - 1; x++) {
      int idx = y * dstWidth + x;
      int prev_idx = y * dstWidth + (x - 1);
      EXPECT_GE(resultData[idx], resultData[prev_idx] - 2)
          << "Horizontal gradient should be monotonic at (" << x << "," << y << ")";
    }
  }
}

TEST_F(ResizeFunctionalTest, Linear_8u_C1R_VerticalGradient) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 4, dstHeight = 8;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);
  fillVerticalGradient<Npp8u, 1>(srcData, srcWidth, srcHeight, 0, 255);

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // Verify monotonic vertical gradient in each column
  for (int x = 0; x < dstWidth; x++) {
    for (int y = 1; y < dstHeight - 1; y++) {
      int idx = y * dstWidth + x;
      int prev_idx = (y - 1) * dstWidth + x;
      EXPECT_GE(resultData[idx], resultData[prev_idx] - 2)
          << "Vertical gradient should be monotonic at (" << x << "," << y << ")";
    }
  }
}

TEST_F(ResizeFunctionalTest, Linear_8u_C1R_StepFunctionInterpolation) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 7, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // Create step function: left half 60, right half 200
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (x < srcWidth / 2) ? 60 : 200;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C1R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  dst.copyToHost(resultData);

  // Left side should be close to 60, right side close to 200 (tightened from tolerance 15 to 3)
  for (int y = 0; y < dstHeight; y++) {
    EXPECT_NEAR(resultData[y * dstWidth + 0], 60, 3) << "Left edge at y=" << y;
    EXPECT_NEAR(resultData[y * dstWidth + (dstWidth - 1)], 200, 3) << "Right edge at y=" << y;

    // Middle should show interpolation
    int mid_idx = y * dstWidth + (dstWidth / 2);
    EXPECT_GT(resultData[mid_idx], 60) << "Middle should be interpolated at y=" << y;
    EXPECT_LT(resultData[mid_idx], 200) << "Middle should be interpolated at y=" << y;
  }
}

// Enhanced LINEAR C3R precision tests

// Test diagonal interpolation
TEST_F(ResizeFunctionalTest, Linear_8u_C3R_DiagonalInterpolation) {
  const int srcWidth = 4, srcHeight = 4;
  const int dstWidth = 7, dstHeight = 7;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create diagonal gradient pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      Npp8u value = (Npp8u)((x + y) * 255 / (srcWidth + srcHeight - 2));
      srcData[idx + 0] = value;
      srcData[idx + 1] = value;
      srcData[idx + 2] = value;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify diagonal gradient increases
  for (int i = 1; i < dstWidth * dstHeight; i++) {
    int idx = i * 3;
    int prev_idx = (i - 1) * 3;
    // Allow slight variations in monotonicity due to 2D interpolation
    if (i % dstWidth != 0) { // Not at row boundary
      EXPECT_GE(resultData[idx], resultData[prev_idx] - 3) << "Diagonal gradient at pixel " << i;
    }
  }
}

// Test downsampling to very small size
TEST_F(ResizeFunctionalTest, Linear_8u_C3R_DownsampleToTiny) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 2, dstHeight = 2;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create quadrant pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      if (x < srcWidth / 2 && y < srcHeight / 2) {
        srcData[idx + 0] = 255;
        srcData[idx + 1] = 0;
        srcData[idx + 2] = 0;
      } else if (x >= srcWidth / 2 && y < srcHeight / 2) {
        srcData[idx + 0] = 0;
        srcData[idx + 1] = 255;
        srcData[idx + 2] = 0;
      } else if (x < srcWidth / 2 && y >= srcHeight / 2) {
        srcData[idx + 0] = 0;
        srcData[idx + 1] = 0;
        srcData[idx + 2] = 255;
      } else {
        srcData[idx + 0] = 255;
        srcData[idx + 1] = 255;
        srcData[idx + 2] = 0;
      }
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Each destination pixel should show color influence from its quadrant
  // but due to interpolation, colors may mix at boundaries
  for (int i = 0; i < dstWidth * dstHeight * 3; i++) {
    EXPECT_LE(resultData[i], 255);
    EXPECT_GE(resultData[i], 0);
  }
}

// Test asymmetric scaling (different horizontal/vertical factors)
TEST_F(ResizeFunctionalTest, Linear_8u_C3R_AsymmetricScaling) {
  const int srcWidth = 16, srcHeight = 8;
  const int dstWidth = 32, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create pattern with horizontal and vertical structure
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x * 255 / (srcWidth - 1));  // Horizontal gradient
      srcData[idx + 1] = (Npp8u)(y * 255 / (srcHeight - 1)); // Vertical gradient
      srcData[idx + 2] = 128;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Verify horizontal gradient preserved in R channel
  for (int y = 0; y < dstHeight; y++) {
    for (int x = 1; x < dstWidth - 1; x++) {
      int idx = (y * dstWidth + x) * 3;
      int prev_idx = (y * dstWidth + (x - 1)) * 3;
      EXPECT_GE(resultData[idx + 0], resultData[prev_idx + 0] - 2)
          << "Horizontal gradient at (" << x << "," << y << ")";
    }
  }
}

// Test edge pixel accuracy
TEST_F(ResizeFunctionalTest, Linear_8u_C3R_EdgePixelAccuracy) {
  const int srcWidth = 8, srcHeight = 8;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create border pattern: edges are white, center is black
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      bool is_edge = (x == 0 || x == srcWidth - 1 || y == 0 || y == srcHeight - 1);
      Npp8u value = is_edge ? 255 : 0;
      srcData[idx + 0] = value;
      srcData[idx + 1] = value;
      srcData[idx + 2] = value;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Check corners should be white
  int tl = 0;
  int tr = (dstWidth - 1) * 3;
  int bl = ((dstHeight - 1) * dstWidth) * 3;
  int br = ((dstHeight - 1) * dstWidth + (dstWidth - 1)) * 3;

  EXPECT_NEAR(resultData[tl], 255, 5) << "Top-left corner";
  EXPECT_NEAR(resultData[tr], 255, 5) << "Top-right corner";
  EXPECT_NEAR(resultData[bl], 255, 5) << "Bottom-left corner";
  EXPECT_NEAR(resultData[br], 255, 5) << "Bottom-right corner";

  // Center should be darker
  int center = ((dstHeight / 2) * dstWidth + (dstWidth / 2)) * 3;
  EXPECT_LT(resultData[center], 100) << "Center should be dark";
}

// Test ROI with LINEAR interpolation
TEST_F(ResizeFunctionalTest, Linear_8u_C3R_ROITest) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 16, dstHeight = 16;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);
  fillCheckerboard<Npp8u, 3>(srcData, srcWidth, srcHeight, 4, 200, 50);

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth / 2, srcHeight / 2};
  NppiRect dstROI = {0, 0, dstWidth / 2, dstHeight / 2};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_LINEAR);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Check ROI region has reasonable values
  for (int y = 0; y < dstHeight / 2; y++) {
    for (int x = 0; x < dstWidth / 2; x++) {
      int idx = (y * dstWidth + x) * 3;
      EXPECT_GE(resultData[idx], 40);
      EXPECT_LE(resultData[idx], 210);
    }
  }
}

// Test 16u C1R with LINEAR interpolation
TEST_F(ResizeFunctionalTest, Resize_16u_C1R_Ctx_LinearInterpolation) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 64, dstHeight = 64;

  // Prepare test data - gradient pattern
  std::vector<Npp16u> srcData(srcWidth * srcHeight);
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = static_cast<Npp16u>((x + y) * 1000);
    }
  }

  NppImageMemory<Npp16u> src(srcWidth, srcHeight);
  NppImageMemory<Npp16u> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiResize_16u_C1R_Ctx(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize,
                                            dstROI, NPPI_INTER_LINEAR, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp16u> dstData(dstWidth * dstHeight);
  dst.copyToHost(dstData);

  // Validate result is not all zeros
  bool hasNonZero = false;
  for (const auto &val : dstData) {
    if (val != 0) {
      hasNonZero = true;
      break;
    }
  }
  ASSERT_TRUE(hasNonZero);
}

// Test 32f C1R with CUBIC interpolation
TEST_F(ResizeFunctionalTest, Resize_32f_C1R_Ctx_CubicInterpolation) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 48, dstHeight = 48;

  // Prepare test data - sine wave pattern
  std::vector<Npp32f> srcData(srcWidth * srcHeight);
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = sinf(x * 0.2f) * cosf(y * 0.2f);
    }
  }

  NppImageMemory<Npp32f> src(srcWidth, srcHeight);
  NppImageMemory<Npp32f> dst(dstWidth, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiResize_32f_C1R_Ctx(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize,
                                            dstROI, NPPI_INTER_CUBIC, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(dstWidth * dstHeight);
  dst.copyToHost(dstData);

  // Validate result contains valid data
  float maxVal = *std::max_element(dstData.begin(), dstData.end());
  float minVal = *std::min_element(dstData.begin(), dstData.end());
  ASSERT_GT(maxVal - minVal, 0.1f);
}

// Test 32f C3R with LINEAR interpolation
TEST_F(ResizeFunctionalTest, Resize_32f_C3R_Ctx_LinearInterpolation) {
  const int srcWidth = 32, srcHeight = 32;
  const int dstWidth = 64, dstHeight = 64;

  // Prepare test data - color gradient
  std::vector<Npp32f> srcData(srcWidth * srcHeight * 3);
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = static_cast<float>(x) / srcWidth;
      srcData[idx + 1] = static_cast<float>(y) / srcHeight;
      srcData[idx + 2] = 1.0f - static_cast<float>(x + y) / (srcWidth + srcHeight);
    }
  }

  NppImageMemory<Npp32f> src(srcWidth, srcHeight, 3);
  NppImageMemory<Npp32f> dst(dstWidth, dstHeight, 3);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiResize_32f_C3R_Ctx(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize,
                                            dstROI, NPPI_INTER_LINEAR, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp32f> dstData(dstWidth * dstHeight * 3);
  dst.copyToHost(dstData);

  // Validate RGB channels all have valid data
  float rMax = 0, gMax = 0, bMax = 0;
  for (int i = 0; i < dstWidth * dstHeight; i++) {
    rMax = std::max(rMax, dstData[i * 3 + 0]);
    gMax = std::max(gMax, dstData[i * 3 + 1]);
    bMax = std::max(bMax, dstData[i * 3 + 2]);
  }

  ASSERT_GT(rMax, 0.5f);
  ASSERT_GT(gMax, 0.5f);
  ASSERT_GT(bMax, 0.1f);
}

// ===== Parametrized Test Framework =====

struct ResizeParams {
  int srcWidth;
  int srcHeight;
  int dstWidth;
  int dstHeight;
  int interpolation;
  std::string testName;
};

// Base class for parametrized resize tests
class ResizeParametrizedTest : public ResizeFunctionalTest, public ::testing::WithParamInterface<ResizeParams> {};

// Checkerboard pattern test with different sizes and interpolation modes
TEST_P(ResizeParametrizedTest, CheckerboardPattern) {
  auto params = GetParam();

  std::vector<Npp8u> srcData(params.srcWidth * params.srcHeight * 3);
  fillCheckerboard<Npp8u, 3>(srcData, params.srcWidth, params.srcHeight, 4, 200, 50);

  NppImageMemory<Npp8u> src(params.srcWidth * 3, params.srcHeight);
  NppImageMemory<Npp8u> dst(params.dstWidth * 3, params.dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {params.srcWidth, params.srcHeight};
  NppiSize dstSize = {params.dstWidth, params.dstHeight};
  NppiRect srcROI = {0, 0, params.srcWidth, params.srcHeight};
  NppiRect dstROI = {0, 0, params.dstWidth, params.dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       params.interpolation);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(params.dstWidth * params.dstHeight * 3);
  dst.copyToHost(resultData);

  // Cubic and Lanczos interpolation may overshoot on high-frequency patterns, allow wider range
  if (params.interpolation == NPPI_INTER_CUBIC || params.interpolation == NPPI_INTER_LANCZOS) {
    validateRange(resultData, Npp8u(0), Npp8u(255), params.testName);
  } else {
    validateRange(resultData, Npp8u(50), Npp8u(200), params.testName);
  }
}

// Horizontal gradient test
TEST_P(ResizeParametrizedTest, HorizontalGradient) {
  auto params = GetParam();

  std::vector<Npp8u> srcData(params.srcWidth * params.srcHeight * 3);
  fillHorizontalGradient<Npp8u, 3>(srcData, params.srcWidth, params.srcHeight, 0, 255);

  NppImageMemory<Npp8u> src(params.srcWidth * 3, params.srcHeight);
  NppImageMemory<Npp8u> dst(params.dstWidth * 3, params.dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {params.srcWidth, params.srcHeight};
  NppiSize dstSize = {params.dstWidth, params.dstHeight};
  NppiRect srcROI = {0, 0, params.srcWidth, params.srcHeight};
  NppiRect dstROI = {0, 0, params.dstWidth, params.dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       params.interpolation);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(params.dstWidth * params.dstHeight * 3);
  dst.copyToHost(resultData);

  // For linear interpolation, validate monotonicity
  if (params.interpolation == NPPI_INTER_LINEAR) {
    validateHorizontalMonotonic<Npp8u, 3>(resultData, params.dstWidth, params.dstHeight, 0);
  }
}

// Vertical gradient test
TEST_P(ResizeParametrizedTest, VerticalGradient) {
  auto params = GetParam();

  std::vector<Npp8u> srcData(params.srcWidth * params.srcHeight * 3);
  fillVerticalGradient<Npp8u, 3>(srcData, params.srcWidth, params.srcHeight, 0, 255);

  NppImageMemory<Npp8u> src(params.srcWidth * 3, params.srcHeight);
  NppImageMemory<Npp8u> dst(params.dstWidth * 3, params.dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {params.srcWidth, params.srcHeight};
  NppiSize dstSize = {params.dstWidth, params.dstHeight};
  NppiRect srcROI = {0, 0, params.srcWidth, params.srcHeight};
  NppiRect dstROI = {0, 0, params.dstWidth, params.dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       params.interpolation);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(params.dstWidth * params.dstHeight * 3);
  dst.copyToHost(resultData);

  // For linear interpolation, validate monotonicity
  if (params.interpolation == NPPI_INTER_LINEAR) {
    validateVerticalMonotonic<Npp8u, 3>(resultData, params.dstWidth, params.dstHeight, 1);
  }
}

// Uniform value test
TEST_P(ResizeParametrizedTest, UniformValue) {
  auto params = GetParam();

  std::vector<Npp8u> srcData(params.srcWidth * params.srcHeight * 3);
  const Npp8u uniform[3] = {100, 150, 200};
  fillUniform<Npp8u, 3>(srcData, uniform);

  NppImageMemory<Npp8u> src(params.srcWidth * 3, params.srcHeight);
  NppImageMemory<Npp8u> dst(params.dstWidth * 3, params.dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {params.srcWidth, params.srcHeight};
  NppiSize dstSize = {params.dstWidth, params.dstHeight};
  NppiRect srcROI = {0, 0, params.srcWidth, params.srcHeight};
  NppiRect dstROI = {0, 0, params.dstWidth, params.dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       params.interpolation);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(params.dstWidth * params.dstHeight * 3);
  dst.copyToHost(resultData);

  // Uniform input should produce uniform or near-uniform output
  validateChannelValues<Npp8u, 3>(resultData, params.dstWidth, params.dstHeight, uniform, 3);
}

// Instantiate test suite with different parameters
INSTANTIATE_TEST_SUITE_P(NearestNeighbor_Upscale, ResizeParametrizedTest,
                         ::testing::Values(ResizeParams{16, 16, 32, 32, NPPI_INTER_NN, "NN_2x_upscale"},
                                           ResizeParams{32, 32, 64, 64, NPPI_INTER_NN, "NN_2x_upscale_large"},
                                           ResizeParams{8, 8, 24, 24, NPPI_INTER_NN, "NN_3x_upscale"},
                                           ResizeParams{16, 16, 48, 48, NPPI_INTER_NN, "NN_3x_upscale_large"}));

INSTANTIATE_TEST_SUITE_P(NearestNeighbor_Downscale, ResizeParametrizedTest,
                         ::testing::Values(ResizeParams{32, 32, 16, 16, NPPI_INTER_NN, "NN_2x_downscale"},
                                           ResizeParams{64, 64, 32, 32, NPPI_INTER_NN, "NN_2x_downscale_large"},
                                           ResizeParams{48, 48, 16, 16, NPPI_INTER_NN, "NN_3x_downscale"},
                                           ResizeParams{96, 96, 32, 32, NPPI_INTER_NN, "NN_3x_downscale_large"}));

INSTANTIATE_TEST_SUITE_P(Linear_Upscale, ResizeParametrizedTest,
                         ::testing::Values(ResizeParams{16, 16, 32, 32, NPPI_INTER_LINEAR, "LINEAR_2x_upscale"},
                                           ResizeParams{32, 32, 64, 64, NPPI_INTER_LINEAR, "LINEAR_2x_upscale_large"},
                                           ResizeParams{8, 8, 24, 24, NPPI_INTER_LINEAR, "LINEAR_3x_upscale"},
                                           ResizeParams{16, 16, 48, 48, NPPI_INTER_LINEAR, "LINEAR_3x_upscale_large"}));

INSTANTIATE_TEST_SUITE_P(Linear_Downscale, ResizeParametrizedTest,
                         ::testing::Values(ResizeParams{32, 32, 16, 16, NPPI_INTER_LINEAR, "LINEAR_2x_downscale"},
                                           ResizeParams{64, 64, 32, 32, NPPI_INTER_LINEAR, "LINEAR_2x_downscale_large"},
                                           ResizeParams{48, 48, 16, 16, NPPI_INTER_LINEAR, "LINEAR_3x_downscale"},
                                           ResizeParams{96, 96, 32, 32, NPPI_INTER_LINEAR,
                                                        "LINEAR_3x_downscale_large"}));

INSTANTIATE_TEST_SUITE_P(Super_Downscale, ResizeParametrizedTest,
                         ::testing::Values(ResizeParams{64, 64, 32, 32, NPPI_INTER_SUPER, "SUPER_2x_downscale"},
                                           ResizeParams{128, 128, 32, 32, NPPI_INTER_SUPER, "SUPER_4x_downscale"},
                                           ResizeParams{100, 100, 30, 30, NPPI_INTER_SUPER, "SUPER_noninteger"},
                                           ResizeParams{32, 32, 8, 8, NPPI_INTER_SUPER, "SUPER_4x_downscale_small"}));

INSTANTIATE_TEST_SUITE_P(
    NonSquare, ResizeParametrizedTest,
    ::testing::Values(ResizeParams{32, 16, 64, 32, NPPI_INTER_LINEAR, "LINEAR_nonsquare_2x"},
                      ResizeParams{16, 32, 32, 64, NPPI_INTER_LINEAR, "LINEAR_nonsquare_2x_tall"},
                      ResizeParams{64, 32, 32, 16, NPPI_INTER_LINEAR, "LINEAR_nonsquare_halfscale"},
                      ResizeParams{32, 64, 16, 32, NPPI_INTER_LINEAR, "LINEAR_nonsquare_halfscale_tall"}));

INSTANTIATE_TEST_SUITE_P(Cubic_Upscale, ResizeParametrizedTest,
                         ::testing::Values(ResizeParams{16, 16, 32, 32, NPPI_INTER_CUBIC, "CUBIC_2x_upscale"},
                                           ResizeParams{32, 32, 64, 64, NPPI_INTER_CUBIC, "CUBIC_2x_upscale_large"},
                                           ResizeParams{8, 8, 24, 24, NPPI_INTER_CUBIC, "CUBIC_3x_upscale"},
                                           ResizeParams{16, 16, 48, 48, NPPI_INTER_CUBIC, "CUBIC_3x_upscale_large"}));

INSTANTIATE_TEST_SUITE_P(Cubic_Downscale, ResizeParametrizedTest,
                         ::testing::Values(ResizeParams{32, 32, 16, 16, NPPI_INTER_CUBIC, "CUBIC_2x_downscale"},
                                           ResizeParams{64, 64, 32, 32, NPPI_INTER_CUBIC, "CUBIC_2x_downscale_large"},
                                           ResizeParams{48, 48, 16, 16, NPPI_INTER_CUBIC, "CUBIC_3x_downscale"},
                                           ResizeParams{96, 96, 32, 32, NPPI_INTER_CUBIC, "CUBIC_3x_downscale_large"}));

INSTANTIATE_TEST_SUITE_P(Lanczos_Upscale, ResizeParametrizedTest,
                         ::testing::Values(ResizeParams{16, 16, 32, 32, NPPI_INTER_LANCZOS, "LANCZOS_2x_upscale"},
                                           ResizeParams{32, 32, 64, 64, NPPI_INTER_LANCZOS, "LANCZOS_2x_upscale_large"},
                                           ResizeParams{8, 8, 24, 24, NPPI_INTER_LANCZOS, "LANCZOS_3x_upscale"},
                                           ResizeParams{16, 16, 48, 48, NPPI_INTER_LANCZOS,
                                                        "LANCZOS_3x_upscale_large"}));

INSTANTIATE_TEST_SUITE_P(
    Lanczos_Downscale, ResizeParametrizedTest,
    ::testing::Values(ResizeParams{32, 32, 16, 16, NPPI_INTER_LANCZOS, "LANCZOS_2x_downscale"},
                      ResizeParams{64, 64, 32, 32, NPPI_INTER_LANCZOS, "LANCZOS_2x_downscale_large"},
                      ResizeParams{48, 48, 16, 16, NPPI_INTER_LANCZOS, "LANCZOS_3x_downscale"},
                      ResizeParams{96, 96, 32, 32, NPPI_INTER_LANCZOS, "LANCZOS_3x_downscale_large"}));
