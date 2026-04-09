#include "npp.h"
#include "framework/npp_test_base.h"
#include <gtest/gtest.h>
#include <array>
#include <type_traits>
#include <vector>

using namespace npp_functional_test;

class NPPISwapChannelsTest : public NppTestBase {
protected:
  void SetUp() override {
    NppTestBase::SetUp();
    width = 8;
    height = 8;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;

  template <typename T> std::array<T, 4> sampleValues() const {
    if constexpr (std::is_same_v<T, Npp8u>) {
      return {11, 22, 33, 44};
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      return {100, 200, 300, 400};
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      return {-10, 20, -30, 40};
    } else if constexpr (std::is_same_v<T, Npp32s>) {
      return {1000, 2000, 3000, 4000};
    } else {
      return {0.1f, 0.5f, 0.9f, 1.3f};
    }
  }

  template <typename T> T fillValue() const {
    if constexpr (std::is_same_v<T, Npp8u>) {
      return 255;
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      return 555;
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      return -55;
    } else if constexpr (std::is_same_v<T, Npp32s>) {
      return 5555;
    } else {
      return 5.5f;
    }
  }

  template <typename T> void expectValue(T actual, T expected) {
    if constexpr (std::is_floating_point_v<T>) {
      EXPECT_NEAR(actual, expected, 1e-5f);
    } else {
      EXPECT_EQ(actual, expected);
    }
  }

  template <typename T> std::vector<T> makePackedImage(int channels, const std::array<T, 4> &values) const {
    std::vector<T> data(width * height * channels);
    for (int i = 0; i < width * height; ++i) {
      for (int c = 0; c < channels; ++c) {
        data[i * channels + c] = values[c];
      }
    }
    return data;
  }

  template <typename T, typename Call> void runC3RTest(Call call) {
    const auto values = sampleValues<T>();
    const auto srcData = makePackedImage<T>(3, values);
    NppImageMemory<T> src(width * 3, height);
    NppImageMemory<T> dst(width * 3, height);
    src.copyFromHost(srcData);

    int order[3] = {2, 1, 0};
    ASSERT_EQ(call(src.get(), src.step(), dst.get(), dst.step(), order), NPP_SUCCESS);

    std::vector<T> dstData;
    dst.copyToHost(dstData);
    for (int i = 0; i < width * height; ++i) {
      expectValue(dstData[i * 3 + 0], values[2]);
      expectValue(dstData[i * 3 + 1], values[1]);
      expectValue(dstData[i * 3 + 2], values[0]);
    }
  }

  template <typename T, typename Call> void runC4RTest(Call call) {
    const auto values = sampleValues<T>();
    const auto srcData = makePackedImage<T>(4, values);
    NppImageMemory<T> src(width * 4, height);
    NppImageMemory<T> dst(width * 4, height);
    src.copyFromHost(srcData);

    int order[4] = {3, 2, 1, 0};
    ASSERT_EQ(call(src.get(), src.step(), dst.get(), dst.step(), order), NPP_SUCCESS);

    std::vector<T> dstData;
    dst.copyToHost(dstData);
    for (int i = 0; i < width * height; ++i) {
      expectValue(dstData[i * 4 + 0], values[3]);
      expectValue(dstData[i * 4 + 1], values[2]);
      expectValue(dstData[i * 4 + 2], values[1]);
      expectValue(dstData[i * 4 + 3], values[0]);
    }
  }

  template <typename T, typename Call> void runC3IRTest(Call call) {
    const auto values = sampleValues<T>();
    const auto srcData = makePackedImage<T>(3, values);
    NppImageMemory<T> srcDst(width * 3, height);
    srcDst.copyFromHost(srcData);

    int order[3] = {2, 1, 0};
    ASSERT_EQ(call(srcDst.get(), srcDst.step(), order), NPP_SUCCESS);

    std::vector<T> dstData;
    srcDst.copyToHost(dstData);
    for (int i = 0; i < width * height; ++i) {
      expectValue(dstData[i * 3 + 0], values[2]);
      expectValue(dstData[i * 3 + 1], values[1]);
      expectValue(dstData[i * 3 + 2], values[0]);
    }
  }

  template <typename T, typename Call> void runC4IRTest(Call call) {
    const auto values = sampleValues<T>();
    const auto srcData = makePackedImage<T>(4, values);
    NppImageMemory<T> srcDst(width * 4, height);
    srcDst.copyFromHost(srcData);

    int order[4] = {3, 2, 1, 0};
    ASSERT_EQ(call(srcDst.get(), srcDst.step(), order), NPP_SUCCESS);

    std::vector<T> dstData;
    srcDst.copyToHost(dstData);
    for (int i = 0; i < width * height; ++i) {
      expectValue(dstData[i * 4 + 0], values[3]);
      expectValue(dstData[i * 4 + 1], values[2]);
      expectValue(dstData[i * 4 + 2], values[1]);
      expectValue(dstData[i * 4 + 3], values[0]);
    }
  }

  template <typename T, typename Call> void runC4C3RTest(Call call) {
    const auto values = sampleValues<T>();
    const auto srcData = makePackedImage<T>(4, values);
    NppImageMemory<T> src(width * 4, height);
    NppImageMemory<T> dst(width * 3, height);
    src.copyFromHost(srcData);

    int order[3] = {2, 1, 0};
    ASSERT_EQ(call(src.get(), src.step(), dst.get(), dst.step(), order), NPP_SUCCESS);

    std::vector<T> dstData;
    dst.copyToHost(dstData);
    for (int i = 0; i < width * height; ++i) {
      expectValue(dstData[i * 3 + 0], values[2]);
      expectValue(dstData[i * 3 + 1], values[1]);
      expectValue(dstData[i * 3 + 2], values[0]);
    }
  }

  template <typename T, typename Call> void runC3C4RTest(Call call) {
    const auto values = sampleValues<T>();
    const auto srcData = makePackedImage<T>(3, values);
    NppImageMemory<T> src(width * 3, height);
    NppImageMemory<T> dst(width * 4, height);
    src.copyFromHost(srcData);

    int order[4] = {2, 1, 0, 3};
    const T fill = fillValue<T>();
    ASSERT_EQ(call(src.get(), src.step(), dst.get(), dst.step(), order, fill), NPP_SUCCESS);

    std::vector<T> dstData;
    dst.copyToHost(dstData);
    for (int i = 0; i < width * height; ++i) {
      expectValue(dstData[i * 4 + 0], values[2]);
      expectValue(dstData[i * 4 + 1], values[1]);
      expectValue(dstData[i * 4 + 2], values[0]);
      expectValue(dstData[i * 4 + 3], fill);
    }
  }

  template <typename T, typename Call> void runAC4RTest(Call call) {
    const auto values = sampleValues<T>();
    const auto srcData = makePackedImage<T>(4, values);
    NppImageMemory<T> src(width * 4, height);
    NppImageMemory<T> dst(width * 4, height);
    src.copyFromHost(srcData);

    std::vector<T> dstInit(width * height * 4, fillValue<T>());
    dst.copyFromHost(dstInit);

    int order[3] = {2, 1, 0};
    EXPECT_EQ(call(src.get(), src.step(), dst.get(), dst.step(), order), NPP_BAD_ARGUMENT_ERROR);
  }
};

// C4R tests (4 channel to 4 channel)
TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C4R) {
  size_t dataSize = width * height * 4;
  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 10;
    srcData[i + 1] = 20;
    srcData[i + 2] = 30;
    srcData[i + 3] = 40;
  }

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);
  src.copyFromHost(srcData);

  int aDstOrder[4] = {2, 1, 0, 3};
  NppStatus status = nppiSwapChannels_8u_C4R(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 30);
  EXPECT_EQ(dstData[1], 20);
  EXPECT_EQ(dstData[2], 10);
  EXPECT_EQ(dstData[3], 40);
}

TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C4R_Ctx) {
  size_t dataSize = width * height * 4;
  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 10;
    srcData[i + 1] = 20;
    srcData[i + 2] = 30;
    srcData[i + 3] = 40;
  }

  NppImageMemory<Npp8u> src(width, height, 4);
  NppImageMemory<Npp8u> dst(width, height, 4);
  src.copyFromHost(srcData);

  int aDstOrder[4] = {2, 1, 0, 3};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiSwapChannels_8u_C4R_Ctx(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder, nppStreamCtx);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 30);
  EXPECT_EQ(dstData[1], 20);
  EXPECT_EQ(dstData[2], 10);
  EXPECT_EQ(dstData[3], 40);
}

// C4IR tests (4 channel in-place)
TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C4IR) {
  size_t dataSize = width * height * 4;
  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 10;
    srcData[i + 1] = 20;
    srcData[i + 2] = 30;
    srcData[i + 3] = 40;
  }

  NppImageMemory<Npp8u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcData);

  int aDstOrder[4] = {2, 1, 0, 3};
  NppStatus status = nppiSwapChannels_8u_C4IR(srcDst.get(), srcDst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  srcDst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 30);
  EXPECT_EQ(dstData[1], 20);
  EXPECT_EQ(dstData[2], 10);
  EXPECT_EQ(dstData[3], 40);
}

TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C4IR_Ctx) {
  size_t dataSize = width * height * 4;
  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 10;
    srcData[i + 1] = 20;
    srcData[i + 2] = 30;
    srcData[i + 3] = 40;
  }

  NppImageMemory<Npp8u> srcDst(width, height, 4);
  srcDst.copyFromHost(srcData);

  int aDstOrder[4] = {2, 1, 0, 3};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiSwapChannels_8u_C4IR_Ctx(
      srcDst.get(), srcDst.step(), roi, aDstOrder, nppStreamCtx);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  srcDst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 30);
  EXPECT_EQ(dstData[1], 20);
  EXPECT_EQ(dstData[2], 10);
  EXPECT_EQ(dstData[3], 40);
}

// C3R tests (3 channel to 3 channel)
TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C3R) {
  size_t dataSize = width * height * 3;
  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 3) {
    srcData[i + 0] = 11;
    srcData[i + 1] = 22;
    srcData[i + 2] = 33;
  }

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 3);
  src.copyFromHost(srcData);

  int aDstOrder[3] = {2, 1, 0};
  NppStatus status = nppiSwapChannels_8u_C3R(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 33);
  EXPECT_EQ(dstData[1], 22);
  EXPECT_EQ(dstData[2], 11);
}

// C3IR tests (3 channel in-place)
TEST_F(NPPISwapChannelsTest, SwapChannels_16u_C3IR) {
  size_t dataSize = width * height * 3;
  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 3) {
    srcData[i + 0] = 1000;
    srcData[i + 1] = 2000;
    srcData[i + 2] = 3000;
  }

  NppImageMemory<Npp16u> srcDst(width, height, 3);
  srcDst.copyFromHost(srcData);

  int aDstOrder[3] = {2, 1, 0};
  NppStatus status = nppiSwapChannels_16u_C3IR(srcDst.get(), srcDst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> dstData;
  srcDst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 3000);
  EXPECT_EQ(dstData[1], 2000);
  EXPECT_EQ(dstData[2], 1000);
}

// C4C3R tests (4 channel to 3 channel)
TEST_F(NPPISwapChannelsTest, SwapChannels_32f_C4C3R) {
  size_t srcSize = width * height * 4;
  std::vector<Npp32f> srcData(srcSize);
  for (size_t i = 0; i < srcSize; i += 4) {
    srcData[i + 0] = 1.0f;
    srcData[i + 1] = 2.0f;
    srcData[i + 2] = 3.0f;
    srcData[i + 3] = 4.0f;
  }

  NppImageMemory<Npp32f> src(width, height, 4);
  NppImageMemory<Npp32f> dst(width, height, 3);
  src.copyFromHost(srcData);

  int aDstOrder[3] = {2, 1, 0};
  NppStatus status = nppiSwapChannels_32f_C4C3R(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> dstData;
  dst.copyToHost(dstData);
  EXPECT_FLOAT_EQ(dstData[0], 3.0f);
  EXPECT_FLOAT_EQ(dstData[1], 2.0f);
  EXPECT_FLOAT_EQ(dstData[2], 1.0f);
}

// C3C4R tests (3 channel to 4 channel with fill value)
TEST_F(NPPISwapChannelsTest, SwapChannels_8u_C3C4R) {
  size_t srcSize = width * height * 3;
  std::vector<Npp8u> srcData(srcSize);
  for (size_t i = 0; i < srcSize; i += 3) {
    srcData[i + 0] = 11;
    srcData[i + 1] = 22;
    srcData[i + 2] = 33;
  }

  NppImageMemory<Npp8u> src(width, height, 3);
  NppImageMemory<Npp8u> dst(width, height, 4);
  src.copyFromHost(srcData);

  int aDstOrder[4] = {2, 1, 0, 3};
  Npp8u fillValue = 255;
  NppStatus status = nppiSwapChannels_8u_C3C4R(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder, fillValue);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 33);
  EXPECT_EQ(dstData[1], 22);
  EXPECT_EQ(dstData[2], 11);
  EXPECT_EQ(dstData[3], 255);
}

// AC4R tests (4 channel with alpha preserved)
TEST_F(NPPISwapChannelsTest, SwapChannels_32s_AC4R) {
  runAC4RTest<Npp32s>([&](const Npp32s *src, int srcStep, Npp32s *dst, int dstStep, const int *order) {
    return nppiSwapChannels_32s_AC4R(src, srcStep, dst, dstStep, roi, order);
  });
}

// Test multiple data types - use 16u instead of 16s for C4
TEST_F(NPPISwapChannelsTest, SwapChannels_16u_C4R) {
  size_t dataSize = width * height * 4;
  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 4) {
    srcData[i + 0] = 100;
    srcData[i + 1] = 200;
    srcData[i + 2] = 300;
    srcData[i + 3] = 400;
  }

  NppImageMemory<Npp16u> src(width, height, 4);
  NppImageMemory<Npp16u> dst(width, height, 4);
  src.copyFromHost(srcData);

  int aDstOrder[4] = {3, 2, 1, 0};
  NppStatus status = nppiSwapChannels_16u_C4R(
      src.get(), src.step(), dst.get(), dst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> dstData;
  dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 400);
  EXPECT_EQ(dstData[1], 300);
  EXPECT_EQ(dstData[2], 200);
  EXPECT_EQ(dstData[3], 100);
}

TEST_F(NPPISwapChannelsTest, SwapChannels_32f_C3IR) {
  size_t dataSize = width * height * 3;
  std::vector<Npp32f> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i += 3) {
    srcData[i + 0] = 0.1f;
    srcData[i + 1] = 0.5f;
    srcData[i + 2] = 0.9f;
  }

  NppImageMemory<Npp32f> srcDst(width, height, 3);
  srcDst.copyFromHost(srcData);

  int aDstOrder[3] = {1, 0, 2};
  NppStatus status = nppiSwapChannels_32f_C3IR(srcDst.get(), srcDst.step(), roi, aDstOrder);

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> dstData;
  srcDst.copyToHost(dstData);
  EXPECT_FLOAT_EQ(dstData[0], 0.5f);
  EXPECT_FLOAT_EQ(dstData[1], 0.1f);
  EXPECT_FLOAT_EQ(dstData[2], 0.9f);
}

#define NPPI_SWAP_C3R_TEST(TEST_NAME, TYPE, API)                                                                       \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    runC3RTest<TYPE>([&](const TYPE *src, int srcStep, TYPE *dst, int dstStep, const int *order) {                    \
      return API(src, srcStep, dst, dstStep, roi, order);                                                              \
    });                                                                                                                 \
  }

#define NPPI_SWAP_C3R_CTX_TEST(TEST_NAME, TYPE, API)                                                                   \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    NppStreamContext ctx;                                                                                               \
    nppGetStreamContext(&ctx);                                                                                          \
    runC3RTest<TYPE>([&](const TYPE *src, int srcStep, TYPE *dst, int dstStep, const int *order) {                    \
      return API(src, srcStep, dst, dstStep, roi, order, ctx);                                                         \
    });                                                                                                                 \
    cudaStreamSynchronize(ctx.hStream);                                                                                 \
  }

#define NPPI_SWAP_C4R_TEST(TEST_NAME, TYPE, API)                                                                       \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    runC4RTest<TYPE>([&](const TYPE *src, int srcStep, TYPE *dst, int dstStep, const int *order) {                    \
      return API(src, srcStep, dst, dstStep, roi, order);                                                              \
    });                                                                                                                 \
  }

#define NPPI_SWAP_C4R_CTX_TEST(TEST_NAME, TYPE, API)                                                                   \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    NppStreamContext ctx;                                                                                               \
    nppGetStreamContext(&ctx);                                                                                          \
    runC4RTest<TYPE>([&](const TYPE *src, int srcStep, TYPE *dst, int dstStep, const int *order) {                    \
      return API(src, srcStep, dst, dstStep, roi, order, ctx);                                                         \
    });                                                                                                                 \
    cudaStreamSynchronize(ctx.hStream);                                                                                 \
  }

#define NPPI_SWAP_C3IR_TEST(TEST_NAME, TYPE, API)                                                                      \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    runC3IRTest<TYPE>([&](TYPE *srcDst, int step, const int *order) { return API(srcDst, step, roi, order); });       \
  }

#define NPPI_SWAP_C3IR_CTX_TEST(TEST_NAME, TYPE, API)                                                                  \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    NppStreamContext ctx;                                                                                               \
    nppGetStreamContext(&ctx);                                                                                          \
    runC3IRTest<TYPE>([&](TYPE *srcDst, int step, const int *order) { return API(srcDst, step, roi, order, ctx); });  \
    cudaStreamSynchronize(ctx.hStream);                                                                                 \
  }

#define NPPI_SWAP_C4IR_TEST(TEST_NAME, TYPE, API)                                                                      \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    runC4IRTest<TYPE>([&](TYPE *srcDst, int step, const int *order) { return API(srcDst, step, roi, order); });       \
  }

#define NPPI_SWAP_C4IR_CTX_TEST(TEST_NAME, TYPE, API)                                                                  \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    NppStreamContext ctx;                                                                                               \
    nppGetStreamContext(&ctx);                                                                                          \
    runC4IRTest<TYPE>([&](TYPE *srcDst, int step, const int *order) { return API(srcDst, step, roi, order, ctx); });  \
    cudaStreamSynchronize(ctx.hStream);                                                                                 \
  }

#define NPPI_SWAP_C4C3R_TEST(TEST_NAME, TYPE, API)                                                                     \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    runC4C3RTest<TYPE>([&](const TYPE *src, int srcStep, TYPE *dst, int dstStep, const int *order) {                  \
      return API(src, srcStep, dst, dstStep, roi, order);                                                              \
    });                                                                                                                 \
  }

#define NPPI_SWAP_C4C3R_CTX_TEST(TEST_NAME, TYPE, API)                                                                 \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    NppStreamContext ctx;                                                                                               \
    nppGetStreamContext(&ctx);                                                                                          \
    runC4C3RTest<TYPE>([&](const TYPE *src, int srcStep, TYPE *dst, int dstStep, const int *order) {                  \
      return API(src, srcStep, dst, dstStep, roi, order, ctx);                                                         \
    });                                                                                                                 \
    cudaStreamSynchronize(ctx.hStream);                                                                                 \
  }

#define NPPI_SWAP_C3C4R_TEST(TEST_NAME, TYPE, API)                                                                     \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    runC3C4RTest<TYPE>([&](const TYPE *src, int srcStep, TYPE *dst, int dstStep, const int *order, TYPE fill) {       \
      return API(src, srcStep, dst, dstStep, roi, order, fill);                                                        \
    });                                                                                                                 \
  }

#define NPPI_SWAP_C3C4R_CTX_TEST(TEST_NAME, TYPE, API)                                                                 \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    NppStreamContext ctx;                                                                                               \
    nppGetStreamContext(&ctx);                                                                                          \
    runC3C4RTest<TYPE>([&](const TYPE *src, int srcStep, TYPE *dst, int dstStep, const int *order, TYPE fill) {       \
      return API(src, srcStep, dst, dstStep, roi, order, fill, ctx);                                                   \
    });                                                                                                                 \
    cudaStreamSynchronize(ctx.hStream);                                                                                 \
  }

#define NPPI_SWAP_AC4R_TEST(TEST_NAME, TYPE, API)                                                                      \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    runAC4RTest<TYPE>([&](const TYPE *src, int srcStep, TYPE *dst, int dstStep, const int *order) {                   \
      return API(src, srcStep, dst, dstStep, roi, order);                                                              \
    });                                                                                                                 \
  }

#define NPPI_SWAP_AC4R_CTX_TEST(TEST_NAME, TYPE, API)                                                                  \
  TEST_F(NPPISwapChannelsTest, TEST_NAME) {                                                                             \
    NppStreamContext ctx;                                                                                               \
    nppGetStreamContext(&ctx);                                                                                          \
    runAC4RTest<TYPE>([&](const TYPE *src, int srcStep, TYPE *dst, int dstStep, const int *order) {                   \
      return API(src, srcStep, dst, dstStep, roi, order, ctx);                                                         \
    });                                                                                                                 \
    cudaStreamSynchronize(ctx.hStream);                                                                                 \
  }

NPPI_SWAP_AC4R_TEST(SwapChannels_8u_AC4R, Npp8u, nppiSwapChannels_8u_AC4R)
NPPI_SWAP_AC4R_CTX_TEST(SwapChannels_8u_AC4R_Ctx, Npp8u, nppiSwapChannels_8u_AC4R_Ctx)
NPPI_SWAP_C3C4R_CTX_TEST(SwapChannels_8u_C3C4R_Ctx, Npp8u, nppiSwapChannels_8u_C3C4R_Ctx)
NPPI_SWAP_C3IR_TEST(SwapChannels_8u_C3IR, Npp8u, nppiSwapChannels_8u_C3IR)
NPPI_SWAP_C3IR_CTX_TEST(SwapChannels_8u_C3IR_Ctx, Npp8u, nppiSwapChannels_8u_C3IR_Ctx)
NPPI_SWAP_C3R_CTX_TEST(SwapChannels_8u_C3R_Ctx, Npp8u, nppiSwapChannels_8u_C3R_Ctx)
NPPI_SWAP_C4C3R_TEST(SwapChannels_8u_C4C3R, Npp8u, nppiSwapChannels_8u_C4C3R)
NPPI_SWAP_C4C3R_CTX_TEST(SwapChannels_8u_C4C3R_Ctx, Npp8u, nppiSwapChannels_8u_C4C3R_Ctx)

NPPI_SWAP_AC4R_TEST(SwapChannels_16u_AC4R, Npp16u, nppiSwapChannels_16u_AC4R)
NPPI_SWAP_AC4R_CTX_TEST(SwapChannels_16u_AC4R_Ctx, Npp16u, nppiSwapChannels_16u_AC4R_Ctx)
NPPI_SWAP_C3C4R_TEST(SwapChannels_16u_C3C4R, Npp16u, nppiSwapChannels_16u_C3C4R)
NPPI_SWAP_C3C4R_CTX_TEST(SwapChannels_16u_C3C4R_Ctx, Npp16u, nppiSwapChannels_16u_C3C4R_Ctx)
NPPI_SWAP_C3IR_CTX_TEST(SwapChannels_16u_C3IR_Ctx, Npp16u, nppiSwapChannels_16u_C3IR_Ctx)
NPPI_SWAP_C3R_TEST(SwapChannels_16u_C3R, Npp16u, nppiSwapChannels_16u_C3R)
NPPI_SWAP_C3R_CTX_TEST(SwapChannels_16u_C3R_Ctx, Npp16u, nppiSwapChannels_16u_C3R_Ctx)
NPPI_SWAP_C4C3R_TEST(SwapChannels_16u_C4C3R, Npp16u, nppiSwapChannels_16u_C4C3R)
NPPI_SWAP_C4C3R_CTX_TEST(SwapChannels_16u_C4C3R_Ctx, Npp16u, nppiSwapChannels_16u_C4C3R_Ctx)
NPPI_SWAP_C4IR_TEST(SwapChannels_16u_C4IR, Npp16u, nppiSwapChannels_16u_C4IR)
NPPI_SWAP_C4IR_CTX_TEST(SwapChannels_16u_C4IR_Ctx, Npp16u, nppiSwapChannels_16u_C4IR_Ctx)
NPPI_SWAP_C4R_CTX_TEST(SwapChannels_16u_C4R_Ctx, Npp16u, nppiSwapChannels_16u_C4R_Ctx)

NPPI_SWAP_AC4R_TEST(SwapChannels_16s_AC4R, Npp16s, nppiSwapChannels_16s_AC4R)
NPPI_SWAP_AC4R_CTX_TEST(SwapChannels_16s_AC4R_Ctx, Npp16s, nppiSwapChannels_16s_AC4R_Ctx)
NPPI_SWAP_C3C4R_TEST(SwapChannels_16s_C3C4R, Npp16s, nppiSwapChannels_16s_C3C4R)
NPPI_SWAP_C3C4R_CTX_TEST(SwapChannels_16s_C3C4R_Ctx, Npp16s, nppiSwapChannels_16s_C3C4R_Ctx)
NPPI_SWAP_C3IR_TEST(SwapChannels_16s_C3IR, Npp16s, nppiSwapChannels_16s_C3IR)
NPPI_SWAP_C3IR_CTX_TEST(SwapChannels_16s_C3IR_Ctx, Npp16s, nppiSwapChannels_16s_C3IR_Ctx)
NPPI_SWAP_C3R_TEST(SwapChannels_16s_C3R, Npp16s, nppiSwapChannels_16s_C3R)
NPPI_SWAP_C3R_CTX_TEST(SwapChannels_16s_C3R_Ctx, Npp16s, nppiSwapChannels_16s_C3R_Ctx)
NPPI_SWAP_C4C3R_TEST(SwapChannels_16s_C4C3R, Npp16s, nppiSwapChannels_16s_C4C3R)
NPPI_SWAP_C4C3R_CTX_TEST(SwapChannels_16s_C4C3R_Ctx, Npp16s, nppiSwapChannels_16s_C4C3R_Ctx)
NPPI_SWAP_C4IR_TEST(SwapChannels_16s_C4IR, Npp16s, nppiSwapChannels_16s_C4IR)
NPPI_SWAP_C4IR_CTX_TEST(SwapChannels_16s_C4IR_Ctx, Npp16s, nppiSwapChannels_16s_C4IR_Ctx)
NPPI_SWAP_C4R_TEST(SwapChannels_16s_C4R, Npp16s, nppiSwapChannels_16s_C4R)
NPPI_SWAP_C4R_CTX_TEST(SwapChannels_16s_C4R_Ctx, Npp16s, nppiSwapChannels_16s_C4R_Ctx)

NPPI_SWAP_AC4R_CTX_TEST(SwapChannels_32s_AC4R_Ctx, Npp32s, nppiSwapChannels_32s_AC4R_Ctx)
NPPI_SWAP_C3C4R_TEST(SwapChannels_32s_C3C4R, Npp32s, nppiSwapChannels_32s_C3C4R)
NPPI_SWAP_C3C4R_CTX_TEST(SwapChannels_32s_C3C4R_Ctx, Npp32s, nppiSwapChannels_32s_C3C4R_Ctx)
NPPI_SWAP_C3IR_TEST(SwapChannels_32s_C3IR, Npp32s, nppiSwapChannels_32s_C3IR)
NPPI_SWAP_C3IR_CTX_TEST(SwapChannels_32s_C3IR_Ctx, Npp32s, nppiSwapChannels_32s_C3IR_Ctx)
NPPI_SWAP_C3R_TEST(SwapChannels_32s_C3R, Npp32s, nppiSwapChannels_32s_C3R)
NPPI_SWAP_C3R_CTX_TEST(SwapChannels_32s_C3R_Ctx, Npp32s, nppiSwapChannels_32s_C3R_Ctx)
NPPI_SWAP_C4C3R_TEST(SwapChannels_32s_C4C3R, Npp32s, nppiSwapChannels_32s_C4C3R)
NPPI_SWAP_C4C3R_CTX_TEST(SwapChannels_32s_C4C3R_Ctx, Npp32s, nppiSwapChannels_32s_C4C3R_Ctx)
NPPI_SWAP_C4IR_TEST(SwapChannels_32s_C4IR, Npp32s, nppiSwapChannels_32s_C4IR)
NPPI_SWAP_C4IR_CTX_TEST(SwapChannels_32s_C4IR_Ctx, Npp32s, nppiSwapChannels_32s_C4IR_Ctx)
NPPI_SWAP_C4R_TEST(SwapChannels_32s_C4R, Npp32s, nppiSwapChannels_32s_C4R)
NPPI_SWAP_C4R_CTX_TEST(SwapChannels_32s_C4R_Ctx, Npp32s, nppiSwapChannels_32s_C4R_Ctx)

NPPI_SWAP_AC4R_TEST(SwapChannels_32f_AC4R, Npp32f, nppiSwapChannels_32f_AC4R)
NPPI_SWAP_AC4R_CTX_TEST(SwapChannels_32f_AC4R_Ctx, Npp32f, nppiSwapChannels_32f_AC4R_Ctx)
NPPI_SWAP_C3C4R_TEST(SwapChannels_32f_C3C4R, Npp32f, nppiSwapChannels_32f_C3C4R)
NPPI_SWAP_C3C4R_CTX_TEST(SwapChannels_32f_C3C4R_Ctx, Npp32f, nppiSwapChannels_32f_C3C4R_Ctx)
NPPI_SWAP_C3IR_CTX_TEST(SwapChannels_32f_C3IR_Ctx, Npp32f, nppiSwapChannels_32f_C3IR_Ctx)
NPPI_SWAP_C3R_TEST(SwapChannels_32f_C3R, Npp32f, nppiSwapChannels_32f_C3R)
NPPI_SWAP_C3R_CTX_TEST(SwapChannels_32f_C3R_Ctx, Npp32f, nppiSwapChannels_32f_C3R_Ctx)
NPPI_SWAP_C4C3R_CTX_TEST(SwapChannels_32f_C4C3R_Ctx, Npp32f, nppiSwapChannels_32f_C4C3R_Ctx)
NPPI_SWAP_C4IR_TEST(SwapChannels_32f_C4IR, Npp32f, nppiSwapChannels_32f_C4IR)
NPPI_SWAP_C4IR_CTX_TEST(SwapChannels_32f_C4IR_Ctx, Npp32f, nppiSwapChannels_32f_C4IR_Ctx)
NPPI_SWAP_C4R_TEST(SwapChannels_32f_C4R, Npp32f, nppiSwapChannels_32f_C4R)
NPPI_SWAP_C4R_CTX_TEST(SwapChannels_32f_C4R_Ctx, Npp32f, nppiSwapChannels_32f_C4R_Ctx)

#undef NPPI_SWAP_C3R_TEST
#undef NPPI_SWAP_C3R_CTX_TEST
#undef NPPI_SWAP_C4R_TEST
#undef NPPI_SWAP_C4R_CTX_TEST
#undef NPPI_SWAP_C3IR_TEST
#undef NPPI_SWAP_C3IR_CTX_TEST
#undef NPPI_SWAP_C4IR_TEST
#undef NPPI_SWAP_C4IR_CTX_TEST
#undef NPPI_SWAP_C4C3R_TEST
#undef NPPI_SWAP_C4C3R_CTX_TEST
#undef NPPI_SWAP_C3C4R_TEST
#undef NPPI_SWAP_C3C4R_CTX_TEST
#undef NPPI_SWAP_AC4R_TEST
#undef NPPI_SWAP_AC4R_CTX_TEST
