#include "../../framework/npp_test_base.h"
#include <algorithm>

using namespace npp_functional_test;

class NV12ToBGRFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

// 测试NV12到BGR三通道转换
TEST_F(NV12ToBGRFunctionalTest, NV12ToBGR_8u_P2C3R_Ctx_Basic) {
  const int width = 64, height = 64;
  const int chromaHeight = height / 2;

  // 准备Y平面数据 (亮度)
  std::vector<Npp8u> yData(width * height);
  for (int i = 0; i < width * height; i++) {
    yData[i] = static_cast<Npp8u>(128 + (i % 128)); // 中等亮度
  }

  // 准备UV平面数据 (色度)
  std::vector<Npp8u> uvData(width * chromaHeight);
  for (int i = 0; i < width * chromaHeight; i += 2) {
    uvData[i] = 128;     // U分量 (中性)
    uvData[i + 1] = 128; // V分量 (中性)
  }

  // 分配设备内存
  NppImageMemory<Npp8u> yPlane(width, height);
  NppImageMemory<Npp8u> uvPlane(width, chromaHeight);
  NppImageMemory<Npp8u> bgrImage(width, height, 3);

  yPlane.copyFromHost(yData);
  uvPlane.copyFromHost(uvData);

  // 设置NV12源指针数组
  const Npp8u *pSrc[2] = {yPlane.get(), uvPlane.get()};
  int srcStep = yPlane.step();

  // 设置参数
  NppiSize oSizeROI = {width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行转换
  NppStatus status = nppiNV12ToBGR_8u_P2C3R_Ctx(pSrc, srcStep, bgrImage.get(), bgrImage.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  gpuStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> bgrData(width * height * 3);
  bgrImage.copyToHost(bgrData);

  // Validate结果 - 应该产生灰色图像（因为UV是中性的）
  bool hasValidData = false;
  for (int i = 0; i < width * height; i++) {
    Npp8u b = bgrData[i * 3 + 0];
    Npp8u g = bgrData[i * 3 + 1];
    Npp8u r = bgrData[i * 3 + 2];

    if (b > 0 || g > 0 || r > 0) {
      hasValidData = true;
    }

    // 由于UV是中性的，RGB值应该接近
    int diff = std::abs(r - g) + std::abs(g - b) + std::abs(r - b);
    ASSERT_LT(diff, 30); // 允许一定的误差
  }

  ASSERT_TRUE(hasValidData);
}
