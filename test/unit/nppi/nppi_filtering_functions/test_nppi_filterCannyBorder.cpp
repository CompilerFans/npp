#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

class NPPIFilterCannyTest : public ::testing::Test {
protected:
  void SetUp() override {
    srcWidth = 32;
    srcHeight = 24;
    oSrcSizeROI.width = srcWidth;
    oSrcSizeROI.height = srcHeight;

    dstWidth = 30;
    dstHeight = 22;
    oDstSizeROI.width = dstWidth;
    oDstSizeROI.height = dstHeight;

    oSrcOffset.x = 1;
    oSrcOffset.y = 1;
  }

  int srcWidth, srcHeight, dstWidth, dstHeight;
  NppiSize oSrcSizeROI, oDstSizeROI;
  NppiPoint oSrcOffset;
};

// 测试缓冲区大小获取
TEST_F(NPPIFilterCannyTest, CannyBorderGetBufferSize_Basic) {
  int bufferSize = 0;

  NppStatus status = nppiFilterCannyBorderGetBufferSize(oSrcSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

// 测试Canny边缘检测基础功能
TEST_F(NPPIFilterCannyTest, FilterCannyBorder_8u_C1R_Basic) {
  size_t srcDataSize = srcWidth * srcHeight;
  size_t dstDataSize = dstWidth * dstHeight;
  std::vector<Npp8u> srcData(srcDataSize), dstData(dstDataSize);

  // 生成测试图像：中央有一个矩形
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      if (x >= 8 && x < 24 && y >= 6 && y < 18) {
        srcData[y * srcWidth + x] = 255; // 矩形内部
      } else {
        srcData[y * srcWidth + x] = 0; // 背景
      }
    }
  }

  // 获取缓冲区大小
  int bufferSize = 0;
  NppStatus status = nppiFilterCannyBorderGetBufferSize(oSrcSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 分配GPU内存
  Npp8u *d_src, *d_dst, *d_buffer;
  int srcStep = srcWidth * sizeof(Npp8u);
  int dstStep = dstWidth * sizeof(Npp8u);

  cudaMalloc(&d_src, srcDataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dstDataSize * sizeof(Npp8u));
  cudaMalloc(&d_buffer, bufferSize);

  cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // CallCanny边缘检测
  status = nppiFilterCannyBorder_8u_C1R(d_src, srcStep, oSrcSizeROI, oSrcOffset, d_dst, dstStep, oDstSizeROI,
                                        NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, 50.0f, 150.0f, nppiNormL2,
                                        NPP_BORDER_REPLICATE, d_buffer);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // 基本Validate：应该检测到边缘
  int edgePixelCount = 0;
  for (size_t i = 0; i < dstDataSize; i++) {
    if (dstData[i] > 0) {
      edgePixelCount++;
    }
  }
  EXPECT_GT(edgePixelCount, 0); // 应该有边缘像素

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_buffer);
}
