#include "npp_test_base.h"
#include <algorithm>
#include <random>

using namespace npp_functional_test;

class CFAToRGBFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

namespace {
struct CfaCase {
  int width;
  int height;
  NppiBayerGridPosition pattern;
  unsigned int seed;
};

template <typename T>
void fill_bayer(std::vector<T> &src, int width, int height, NppiBayerGridPosition pattern, unsigned int seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(50, 200);
  src.resize(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = y * width + x;
      int v = dist(rng);
      // Add slight variation per channel location.
      if (pattern == NPPI_BAYER_BGGR) {
        v += ((y & 1) == 0 && (x & 1) == 0) ? 10 : 0;
      } else if (pattern == NPPI_BAYER_RGGB) {
        v += ((y & 1) == 0 && (x & 1) == 0) ? 20 : 0;
      } else if (pattern == NPPI_BAYER_GBRG) {
        v += ((y & 1) == 0 && (x & 1) == 1) ? 15 : 0;
      } else {
        v += ((y & 1) == 1 && (x & 1) == 0) ? 12 : 0;
      }
      src[idx] = static_cast<T>(v);
    }
  }
}
} // namespace

class CFAToRGBParamTest : public NppTestBase, public ::testing::WithParamInterface<CfaCase> {};

// 测试8位CFA到RGB转换 - BGGR模式
TEST_F(CFAToRGBFunctionalTest, CFAToRGB_8u_C1C3R_Ctx_BGGR) {
  const int width = 16, height = 16; // 使用最小尺寸

  // prepare test data - BGGR Bayer模式
  std::vector<Npp8u> srcData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      if (y % 2 == 0) {
        if (x % 2 == 0) {
          srcData[idx] = 100; // B
        } else {
          srcData[idx] = 150; // G
        }
      } else {
        if (x % 2 == 0) {
          srcData[idx] = 150; // G
        } else {
          srcData[idx] = 200; // R
        }
      }
    }
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height, 3);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize oSrcSize = {width, height};
  NppiRect oSrcROI = {0, 0, width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行CFA到RGB转换
  NppStatus status = nppiCFAToRGB_8u_C1C3R_Ctx(src.get(), src.step(), oSrcSize, oSrcROI, dst.get(), dst.step(),
                                               NPPI_BAYER_BGGR, NPPI_INTER_UNDEFINED, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  // Validate结果 - 检查RGB通道都有数据
  int rCount = 0, gCount = 0, bCount = 0;
  for (int i = 0; i < width * height; i++) {
    if (dstData[i * 3 + 0] > 50)
      rCount++;
    if (dstData[i * 3 + 1] > 50)
      gCount++;
    if (dstData[i * 3 + 2] > 50)
      bCount++;
  }

  // 每个通道都应该有大量有效数据
  ASSERT_GT(rCount, width * height / 2);
  ASSERT_GT(gCount, width * height / 2);
  ASSERT_GT(bCount, width * height / 2);
}

// 测试16位CFA到RGB转换 - RGGB模式
TEST_F(CFAToRGBFunctionalTest, CFAToRGB_16u_C1C3R_Ctx_RGGB) {
  const int width = 32, height = 32;

  // prepare test data - RGGB Bayer模式，使用16位值
  std::vector<Npp16u> srcData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      if (y % 2 == 0) {
        if (x % 2 == 0) {
          srcData[idx] = 40000; // R
        } else {
          srcData[idx] = 30000; // G
        }
      } else {
        if (x % 2 == 0) {
          srcData[idx] = 30000; // G
        } else {
          srcData[idx] = 20000; // B
        }
      }
    }
  }

  NppImageMemory<Npp16u> src(width, height);
  NppImageMemory<Npp16u> dst(width, height, 3);

  src.copyFromHost(srcData);

  // 设置参数
  NppiSize oSrcSize = {width, height};
  NppiRect oSrcROI = {0, 0, width, height};

  // 获取流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行CFA到RGB转换
  NppStatus status = nppiCFAToRGB_16u_C1C3R_Ctx(src.get(), src.step(), oSrcSize, oSrcROI, dst.get(), dst.step(),
                                                NPPI_BAYER_RGGB, NPPI_INTER_UNDEFINED, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 同步并获取结果
  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp16u> dstData(width * height * 3);
  dst.copyToHost(dstData);

  // Validate结果
  bool hasValidData = false;
  for (int i = 0; i < width * height; i++) {
    Npp16u r = dstData[i * 3 + 0];
    Npp16u g = dstData[i * 3 + 1];
    Npp16u b = dstData[i * 3 + 2];

    if (r > 10000 || g > 10000 || b > 10000) {
      hasValidData = true;
    }
  }

  ASSERT_TRUE(hasValidData);

  // Validate第一个像素（应该是红色为主）
  ASSERT_GT(dstData[0], 30000);
}

// 测试不同Bayer模式的基本功能
TEST_F(CFAToRGBFunctionalTest, CFAToRGB_8u_C1C3R_Ctx_AllPatterns) {
  const int width = 16, height = 16;

  NppiBayerGridPosition patterns[] = {NPPI_BAYER_BGGR, NPPI_BAYER_RGGB, NPPI_BAYER_GBRG, NPPI_BAYER_GRBG};

  for (auto pattern : patterns) {
    // 准备简单的测试数据
    std::vector<Npp8u> srcData(width * height, 128);

    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(width, height, 3);

    src.copyFromHost(srcData);

    // 设置参数
    NppiSize oSrcSize = {width, height};
    NppiRect oSrcROI = {0, 0, width, height};

    // 获取流上下文
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);

    // 执行CFA到RGB转换
    NppStatus status = nppiCFAToRGB_8u_C1C3R_Ctx(src.get(), src.step(), oSrcSize, oSrcROI, dst.get(), dst.step(),
                                                 pattern, NPPI_INTER_UNDEFINED, nppStreamCtx);

    ASSERT_EQ(status, NPP_SUCCESS);

    // Validate基本功能正常
    cudaStreamSynchronize(nppStreamCtx.hStream);
    std::vector<Npp8u> dstData(width * height * 3);
    dst.copyToHost(dstData);

    // 确保有输出
    bool hasNonZero = false;
    for (const auto &val : dstData) {
      if (val != 0) {
        hasNonZero = true;
        break;
      }
    }
    ASSERT_TRUE(hasNonZero);
  }
}

TEST_P(CFAToRGBParamTest, CFAToRGB_8u_C1C3R_MatchesCtx) {
  const auto param = GetParam();
  std::vector<Npp8u> srcData;
  fill_bayer(srcData, param.width, param.height, param.pattern, param.seed);

  NppImageMemory<Npp8u> src(param.width, param.height);
  NppImageMemory<Npp8u> dst(param.width, param.height, 3);
  NppImageMemory<Npp8u> dst_ctx(param.width, param.height, 3);
  src.copyFromHost(srcData);

  NppiSize oSrcSize = {param.width, param.height};
  NppiRect oSrcROI = {0, 0, param.width, param.height};

  NppStatus status = nppiCFAToRGB_8u_C1C3R(src.get(), src.step(), oSrcSize, oSrcROI, dst.get(), dst.step(),
                                           param.pattern, NPPI_INTER_UNDEFINED);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  status = nppiCFAToRGB_8u_C1C3R_Ctx(src.get(), src.step(), oSrcSize, oSrcROI, dst_ctx.get(), dst_ctx.step(),
                                     param.pattern, NPPI_INTER_UNDEFINED, ctx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> out, out_ctx;
  dst.copyToHost(out);
  dst_ctx.copyToHost(out_ctx);
  ASSERT_EQ(out.size(), out_ctx.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out[i], out_ctx[i]) << "Mismatch at " << i;
  }
}

TEST_P(CFAToRGBParamTest, CFAToRGB_16u_C1C3R_MatchesCtx) {
  const auto param = GetParam();
  std::vector<Npp16u> srcData;
  fill_bayer(srcData, param.width, param.height, param.pattern, param.seed + 7);

  NppImageMemory<Npp16u> src(param.width, param.height);
  NppImageMemory<Npp16u> dst(param.width, param.height, 3);
  NppImageMemory<Npp16u> dst_ctx(param.width, param.height, 3);
  src.copyFromHost(srcData);

  NppiSize oSrcSize = {param.width, param.height};
  NppiRect oSrcROI = {0, 0, param.width, param.height};

  NppStatus status = nppiCFAToRGB_16u_C1C3R(src.get(), src.step(), oSrcSize, oSrcROI, dst.get(), dst.step(),
                                            param.pattern, NPPI_INTER_UNDEFINED);
  ASSERT_EQ(status, NPP_SUCCESS);

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  status = nppiCFAToRGB_16u_C1C3R_Ctx(src.get(), src.step(), oSrcSize, oSrcROI, dst_ctx.get(), dst_ctx.step(),
                                      param.pattern, NPPI_INTER_UNDEFINED, ctx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> out, out_ctx;
  dst.copyToHost(out);
  dst_ctx.copyToHost(out_ctx);
  ASSERT_EQ(out.size(), out_ctx.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out[i], out_ctx[i]) << "Mismatch at " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(FunctionalCases, CFAToRGBParamTest,
                         ::testing::Values(CfaCase{16, 16, NPPI_BAYER_BGGR, 11},
                                           CfaCase{16, 16, NPPI_BAYER_RGGB, 22}));
INSTANTIATE_TEST_SUITE_P(PrecisionCases, CFAToRGBParamTest,
                         ::testing::Values(CfaCase{64, 64, NPPI_BAYER_GBRG, 33},
                                           CfaCase{64, 64, NPPI_BAYER_GRBG, 44}));
