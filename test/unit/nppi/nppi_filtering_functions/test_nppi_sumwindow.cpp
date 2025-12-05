#include "npp_test_base.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <vector>

using namespace npp_functional_test;
// CPU参考实现 - 零边界
static void cpuSumWindowRow_8u32f_zero(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                       int nMaskSize, int nAnchor) {
  for (int y = 0; y < oROI.height; y++) {
    const Npp8u *pSrcRow = (const Npp8u *)((char *)pSrc + y * nSrcStep);
    Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);

    for (int x = 0; x < oROI.width; x++) {
      Npp32f sum = 0.0f;
      int startX = x - nAnchor;
      int endX = startX + nMaskSize;

      for (int kx = startX; kx < endX; kx++) {
        if (kx >= 0 && kx < oROI.width) {
          sum += static_cast<Npp32f>(pSrcRow[kx]);
        }
        // 边界外贡献为0
      }
      pDstRow[x] = sum;
    }
  }
}

// CPU参考实现 - 零边界
static void cpuSumWindowColumn_8u32f_zero(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                          int nMaskSize, int nAnchor) {
  for (int y = 0; y < oROI.height; y++) {
    Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);

    for (int x = 0; x < oROI.width; x++) {
      Npp32f sum = 0.0f;
      int startY = y - nAnchor;
      int endY = startY + nMaskSize;

      for (int ky = startY; ky < endY; ky++) {
        if (ky >= 0 && ky < oROI.height) {
          const Npp8u *pSrcRow = (const Npp8u *)((char *)pSrc + ky * nSrcStep);
          sum += static_cast<Npp32f>(pSrcRow[x]);
        }
        // 边界外贡献为0
      }
      pDstRow[x] = sum;
    }
  }
}

class NppiSumWindowTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }

  // 创建常量图像
  std::vector<Npp8u> createConstantImage(int width, int height, Npp8u value) {
    return std::vector<Npp8u>(width * height, value);
  }

  // 创建递增图像
  std::vector<Npp8u> createIncrementImage(int width, int height) {
    std::vector<Npp8u> image(width * height);
    for (int i = 0; i < width * height; i++) {
      image[i] = static_cast<Npp8u>(i % 256);
    }
    return image;
  }

  // 创建垂直条纹
  std::vector<Npp8u> createVerticalStripes(int width, int height) {
    std::vector<Npp8u> image(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        image[y * width + x] = (x % 4 == 0) ? 100 : 200;
      }
    }
    return image;
  }

  // 创建水平条纹
  std::vector<Npp8u> createHorizontalStripes(int width, int height) {
    std::vector<Npp8u> image(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        image[y * width + x] = (y % 4 == 0) ? 50 : 150;
      }
    }
    return image;
  }

  // 创建中心脉冲
  std::vector<Npp8u> createCenterImpulse(int width, int height) {
    std::vector<Npp8u> image(width * height, 10);
    int centerX = width / 2;
    int centerY = height / 2;
    image[centerY * width + centerX] = 255;
    return image;
  }

  // 验证结果（使用相对误差）
  bool validateResults(const Npp32f *cpu, const Npp32f *gpu, int width, int height, int step, float absEpsilon = 1e-3f,
                       float relEpsilon = 1e-4f) {
    for (int y = 0; y < height; y++) {
      const Npp32f *pCpuRow = (const Npp32f *)((char *)cpu + y * step);
      const Npp32f *pGpuRow = (const Npp32f *)((char *)gpu + y * step);

      for (int x = 0; x < width; x++) {
        float cpuVal = pCpuRow[x];
        float gpuVal = pGpuRow[x];
        float absDiff = std::fabs(cpuVal - gpuVal);
        float relDiff = (cpuVal != 0.0f) ? absDiff / std::fabs(cpuVal) : absDiff;

        if (absDiff > absEpsilon && relDiff > relEpsilon) {
          std::cout << "Mismatch at (" << x << ", " << y << "): "
                    << "CPU=" << cpuVal << ", GPU=" << gpuVal << ", absDiff=" << absDiff << ", relDiff=" << relDiff
                    << std::endl;
          return false;
        }
      }
    }
    return true;
  }
};

// ==================== 基本功能测试 ====================

TEST_F(NppiSumWindowTest, SumWindowRow_ConstantImage) {
  const int width = 32;
  const int height = 16;
  const int maskSize = 5;
  const int anchor = 2;

  NppiSize roi = {width, height};

  // 创建常量图像
  std::vector<Npp8u> hostSrc = createConstantImage(width, height, 20);
  std::vector<Npp32f> hostDstCpu(width * height);
  std::vector<Npp32f> hostDstGpu(width * height);

  // 分配GPU内存
  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(hostSrc);

  // 调用GPU函数
  NppStatus status = nppiSumWindowRow_8u32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, maskSize, anchor);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiSumWindowRow_8u32f_C1R failed";

  // 复制结果
  dst.copyToHost(hostDstGpu);

  // 计算CPU参考结果
  cpuSumWindowRow_8u32f_zero(hostSrc.data(), width * sizeof(Npp8u), hostDstCpu.data(), width * sizeof(Npp32f), roi,
                             maskSize, anchor);

  // 验证结果
  bool matches = validateResults(hostDstCpu.data(), hostDstGpu.data(), width, height, width * sizeof(Npp32f));

  EXPECT_TRUE(matches) << "GPU results do not match CPU reference";

  // 验证数值正确性（常量图像）
  float expectedValue = maskSize * 20.0f;
  float actualValue = hostDstGpu[(height / 2) * width + (width / 2)];
  EXPECT_NEAR(actualValue, expectedValue, 1e-3f) << "Center pixel value incorrect for constant image";
}

TEST_F(NppiSumWindowTest, SumWindowColumn_ConstantImage) {
  const int width = 32;
  const int height = 16;
  const int maskSize = 7;
  const int anchor = 3;

  NppiSize roi = {width, height};

  // 创建常量图像
  std::vector<Npp8u> hostSrc = createConstantImage(width, height, 30);
  std::vector<Npp32f> hostDstCpu(width * height);
  std::vector<Npp32f> hostDstGpu(width * height);

  // 分配GPU内存
  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(hostSrc);

  // 调用GPU函数
  NppStatus status = nppiSumWindowColumn_8u32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, maskSize, anchor);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiSumWindowColumn_8u32f_C1R failed";

  // 复制结果
  dst.copyToHost(hostDstGpu);

  // 计算CPU参考结果
  cpuSumWindowColumn_8u32f_zero(hostSrc.data(), width * sizeof(Npp8u), hostDstCpu.data(), width * sizeof(Npp32f), roi,
                                maskSize, anchor);

  // 验证结果
  bool matches = validateResults(hostDstCpu.data(), hostDstGpu.data(), width, height, width * sizeof(Npp32f));

  EXPECT_TRUE(matches) << "GPU results do not match CPU reference";

  // 验证数值正确性
  float expectedValue = maskSize * 30.0f;
  float actualValue = hostDstGpu[(height / 2) * width + (width / 2)];
  EXPECT_NEAR(actualValue, expectedValue, 1e-3f) << "Center pixel value incorrect for constant image";
}

// ==================== 递增图像测试 ====================

TEST_F(NppiSumWindowTest, SumWindowRow_IncrementImage) {
  const int width = 16;
  const int height = 8;
  const int maskSize = 3;
  const int anchor = 1;

  NppiSize roi = {width, height};

  // 创建递增图像
  std::vector<Npp8u> hostSrc = createIncrementImage(width, height);
  std::vector<Npp32f> hostDstCpu(width * height);
  std::vector<Npp32f> hostDstGpu(width * height);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(hostSrc);

  // 调用GPU函数
  NppStatus status = nppiSumWindowRow_8u32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, maskSize, anchor);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiSumWindowRow_8u32f_C1R failed for increment image";

  dst.copyToHost(hostDstGpu);
  cpuSumWindowRow_8u32f_zero(hostSrc.data(), width * sizeof(Npp8u), hostDstCpu.data(), width * sizeof(Npp32f), roi,
                             maskSize, anchor);

  bool matches = validateResults(hostDstCpu.data(), hostDstGpu.data(), width, height, width * sizeof(Npp32f));

  EXPECT_TRUE(matches) << "Failed for increment image";
}

// ==================== 不同掩码大小测试 ====================

TEST_F(NppiSumWindowTest, SumWindowRow_VariousMaskSizes) {
  const int width = 16;
  const int height = 8;

  std::vector<std::pair<int, int>> testCases = {
      {1, 0}, // 最小掩码
      {3, 1}, // 小掩码
      {5, 2}, // 中等掩码
      {7, 3}, // 大掩码
      {9, 4}, // 更大掩码
  };

  std::vector<Npp8u> hostSrc = createIncrementImage(width, height);
  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(hostSrc);

  NppiSize roi = {width, height};

  for (const auto &testCase : testCases) {
    int maskSize = testCase.first;
    int anchor = testCase.second;

    SCOPED_TRACE("Testing maskSize=" + std::to_string(maskSize) + ", anchor=" + std::to_string(anchor));

    std::vector<Npp32f> hostDstCpu(width * height);
    std::vector<Npp32f> hostDstGpu(width * height);
    NppImageMemory<Npp32f> dst(width, height);

    // 调用GPU函数
    NppStatus status = nppiSumWindowRow_8u32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, maskSize, anchor);

    ASSERT_EQ(status, NPP_SUCCESS) << "Failed with maskSize=" << maskSize << ", anchor=" << anchor;

    dst.copyToHost(hostDstGpu);
    cpuSumWindowRow_8u32f_zero(hostSrc.data(), width * sizeof(Npp8u), hostDstCpu.data(), width * sizeof(Npp32f), roi,
                               maskSize, anchor);

    bool matches = validateResults(hostDstCpu.data(), hostDstGpu.data(), width, height, width * sizeof(Npp32f));

    EXPECT_TRUE(matches) << "Failed with maskSize=" << maskSize << ", anchor=" << anchor;
  }
}

// ==================== 不同锚点位置测试 ====================

TEST_F(NppiSumWindowTest, SumWindowColumn_VariousAnchors) {
  const int width = 12;
  const int height = 12;
  const int maskSize = 5;

  std::vector<int> anchors = {0, 1, 2, 3, 4}; // 所有可能的锚点

  std::vector<Npp8u> hostSrc = createVerticalStripes(width, height);
  NppImageMemory<Npp8u> src(width, height);
  src.copyFromHost(hostSrc);

  NppiSize roi = {width, height};

  for (int anchor : anchors) {
    SCOPED_TRACE("Testing anchor=" + std::to_string(anchor));

    std::vector<Npp32f> hostDstCpu(width * height);
    std::vector<Npp32f> hostDstGpu(width * height);
    NppImageMemory<Npp32f> dst(width, height);

    // 调用GPU函数
    NppStatus status =
        nppiSumWindowColumn_8u32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, maskSize, anchor);

    ASSERT_EQ(status, NPP_SUCCESS) << "Failed with anchor=" << anchor;

    dst.copyToHost(hostDstGpu);
    cpuSumWindowColumn_8u32f_zero(hostSrc.data(), width * sizeof(Npp8u), hostDstCpu.data(), width * sizeof(Npp32f), roi,
                                  maskSize, anchor);

    bool matches = validateResults(hostDstCpu.data(), hostDstGpu.data(), width, height, width * sizeof(Npp32f));

    EXPECT_TRUE(matches) << "Failed with anchor=" << anchor;
  }
}

TEST_F(NppiSumWindowTest, SumWindowRow_NullPointers) {
  const int width = 8;
  const int height = 8;

  NppiSize roi = {width, height};
  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  // 测试空源指针
  NppStatus status = nppiSumWindowRow_8u32f_C1R(nullptr, src.step(), dst.get(), dst.step(), roi, 3, 1);

  EXPECT_EQ(status, NPP_NULL_POINTER_ERROR) << "Should return NPP_NULL_POINTER_ERROR for null source";

  // 测试空目标指针
  status = nppiSumWindowRow_8u32f_C1R(src.get(), src.step(), nullptr, dst.step(), roi, 3, 1);

  EXPECT_EQ(status, NPP_NULL_POINTER_ERROR) << "Should return NPP_NULL_POINTER_ERROR for null destination";
}

// ==================== Ctx版本测试 ====================

TEST_F(NppiSumWindowTest, SumWindowRow_CtxVersion) {
  const int width = 16;
  const int height = 8;
  const int maskSize = 3;
  const int anchor = 1;

  NppiSize roi = {width, height};

  // 创建测试数据
  std::vector<Npp8u> hostSrc = createIncrementImage(width, height);
  std::vector<Npp32f> hostDstRegular(width * height);
  std::vector<Npp32f> hostDstCtx(width * height);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst1(width, height);
  NppImageMemory<Npp32f> dst2(width, height);

  src.copyFromHost(hostSrc);

  // 创建流上下文（使用默认流）
  NppStreamContext ctx;
  ctx.hStream = 0;

  // 测试常规版本
  NppStatus status1 = nppiSumWindowRow_8u32f_C1R(src.get(), src.step(), dst1.get(), dst1.step(), roi, maskSize, anchor);

  ASSERT_EQ(status1, NPP_SUCCESS) << "Regular version failed";

  // 测试_Ctx版本
  NppStatus status2 =
      nppiSumWindowRow_8u32f_C1R_Ctx(src.get(), src.step(), dst2.get(), dst2.step(), roi, maskSize, anchor, ctx);

  ASSERT_EQ(status2, NPP_SUCCESS) << "Ctx version failed";

  // 复制结果
  dst1.copyToHost(hostDstRegular);
  dst2.copyToHost(hostDstCtx);

  // 验证两个版本的结果一致
  bool matches = validateResults(hostDstRegular.data(), hostDstCtx.data(), width, height, width * sizeof(Npp32f));

  EXPECT_TRUE(matches) << "Ctx version results differ from regular version";
}

TEST_F(NppiSumWindowTest, SumWindowColumn_CtxVersion) {
  const int width = 16;
  const int height = 8;
  const int maskSize = 5;
  const int anchor = 2;

  NppiSize roi = {width, height};

  // 创建测试数据
  std::vector<Npp8u> hostSrc = createVerticalStripes(width, height);
  std::vector<Npp32f> hostDstRegular(width * height);
  std::vector<Npp32f> hostDstCtx(width * height);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst1(width, height);
  NppImageMemory<Npp32f> dst2(width, height);

  src.copyFromHost(hostSrc);

  // 创建流上下文
  NppStreamContext ctx;
  ctx.hStream = 0;

  // 测试常规版本
  NppStatus status1 =
      nppiSumWindowColumn_8u32f_C1R(src.get(), src.step(), dst1.get(), dst1.step(), roi, maskSize, anchor);

  ASSERT_EQ(status1, NPP_SUCCESS) << "Regular version failed";

  // 测试_Ctx版本
  NppStatus status2 =
      nppiSumWindowColumn_8u32f_C1R_Ctx(src.get(), src.step(), dst2.get(), dst2.step(), roi, maskSize, anchor, ctx);

  ASSERT_EQ(status2, NPP_SUCCESS) << "Ctx version failed";

  // 复制结果
  dst1.copyToHost(hostDstRegular);
  dst2.copyToHost(hostDstCtx);

  // 验证两个版本的结果一致
  bool matches = validateResults(hostDstRegular.data(), hostDstCtx.data(), width, height, width * sizeof(Npp32f));

  EXPECT_TRUE(matches) << "Ctx version results differ from regular version";
}

TEST_F(NppiSumWindowTest, SumWindowRow_BoundaryHandling) {
  const int width = 8;
  const int height = 6;
  const int maskSize = 5;
  const int anchor = 2;

  NppiSize roi = {width, height};

  // 创建中心脉冲图像
  std::vector<Npp8u> hostSrc = createCenterImpulse(width, height);

  std::vector<Npp32f> hostDstCpu(width * height);
  std::vector<Npp32f> hostDstGpu(width * height);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(hostSrc);

  // 调用GPU函数
  NppStatus status = nppiSumWindowRow_8u32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, maskSize, anchor);

  ASSERT_EQ(status, NPP_SUCCESS) << "Boundary test failed";

  dst.copyToHost(hostDstGpu);
  cpuSumWindowRow_8u32f_zero(hostSrc.data(), width * sizeof(Npp8u), hostDstCpu.data(), width * sizeof(Npp32f), roi,
                             maskSize, anchor);

  bool matches = validateResults(hostDstCpu.data(), hostDstGpu.data(), width, height, width * sizeof(Npp32f));

  EXPECT_TRUE(matches) << "Boundary test failed";

  int y = 0;
  // 检查中心像素 (4,3)
  y = height / 2;           // 3

  // 检查边界像素的实际计算
  float cornerValue = hostDstGpu[0];

  float actualSum = 0.0f;
  for (int kx = 0; kx < 3; kx++) {
    actualSum += hostSrc[y * width + kx];
  }

  EXPECT_NEAR(cornerValue, actualSum, 1e-3f) << "左上角像素值应该等于实际窗口内像素的和";
}
