#include "npp_test_base.h"
#include <iomanip>
#include <type_traits>

using namespace npp_functional_test;

class TransposeFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }

  // 辅助函数：生成测试矩阵
  template <typename T> void generateTestMatrix(std::vector<T> &data, int width, int height) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        data[y * width + x] = static_cast<T>(y * width + x + 1);
      }
    }
  }

  // 辅助函数：计算转置后的期望结果
  template <typename T>
  void computeExpectedTranspose(const std::vector<T> &src, std::vector<T> &dst, int src_width, int src_height) {
    dst.resize(src_width * src_height);
    for (int y = 0; y < src_height; y++) {
      for (int x = 0; x < src_width; x++) {
        dst[x * src_height + y] = src[y * src_width + x];
      }
    }
  }

  template <typename T, int CHANNELS>
  void runInterleavedTransposeTest(NppStatus (*fn)(const T *, int, T *, int, NppiSize), int width, int height) {
    std::vector<T> src(width * height * CHANNELS);
    for (size_t i = 0; i < src.size(); ++i) {
      src[i] = static_cast<T>((i * 7 + 3) % 251);
    }

    std::vector<T> expected(width * height * CHANNELS);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < CHANNELS; ++c) {
          expected[(x * height + y) * CHANNELS + c] = src[(y * width + x) * CHANNELS + c];
        }
      }
    }

    std::vector<T> dst(width * height * CHANNELS);
    NppiSize roi = {width, height};

    if constexpr (std::is_same_v<T, Npp16s> && CHANNELS > 1) {
      T *srcDev = nullptr;
      T *dstDev = nullptr;
      size_t srcPitch = 0;
      size_t dstPitch = 0;
      ASSERT_EQ(cudaMallocPitch(&srcDev, &srcPitch, width * CHANNELS * sizeof(T), height), cudaSuccess);
      ASSERT_EQ(cudaMallocPitch(&dstDev, &dstPitch, height * CHANNELS * sizeof(T), width), cudaSuccess);
      ASSERT_EQ(cudaMemcpy2D(srcDev, srcPitch, src.data(), width * CHANNELS * sizeof(T), width * CHANNELS * sizeof(T),
                             height, cudaMemcpyHostToDevice),
                cudaSuccess);

      ASSERT_EQ(fn(srcDev, static_cast<int>(srcPitch), dstDev, static_cast<int>(dstPitch), roi), NPP_NO_ERROR);
      ASSERT_EQ(cudaMemcpy2D(dst.data(), height * CHANNELS * sizeof(T), dstDev, dstPitch, height * CHANNELS * sizeof(T),
                             width, cudaMemcpyDeviceToHost),
                cudaSuccess);

      cudaFree(srcDev);
      cudaFree(dstDev);
    } else {
      NppImageMemory<T> srcMem(width, height, CHANNELS);
      NppImageMemory<T> dstMem(height, width, CHANNELS);
      srcMem.copyFromHost(src);

      ASSERT_EQ(fn(srcMem.get(), srcMem.step(), dstMem.get(), dstMem.step(), roi), NPP_NO_ERROR);
      dstMem.copyToHost(dst);
    }

    if constexpr (std::is_floating_point_v<T>) {
      EXPECT_TRUE(ResultValidator::arraysEqual(dst, expected, 1e-5f));
    } else {
      EXPECT_TRUE(ResultValidator::arraysEqual(dst, expected));
    }
  }

  template <typename T, int CHANNELS>
  void runInterleavedTransposeCtxTest(NppStatus (*fn)(const T *, int, T *, int, NppiSize, NppStreamContext), int width,
                                      int height) {
    std::vector<T> src(width * height * CHANNELS);
    for (size_t i = 0; i < src.size(); ++i) {
      src[i] = static_cast<T>((i * 5 + 11) % 241);
    }

    std::vector<T> expected(width * height * CHANNELS);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        for (int c = 0; c < CHANNELS; ++c) {
          expected[(x * height + y) * CHANNELS + c] = src[(y * width + x) * CHANNELS + c];
        }
      }
    }

    std::vector<T> dst(width * height * CHANNELS);
    NppiSize roi = {width, height};
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);

    if constexpr (std::is_same_v<T, Npp16s> && CHANNELS > 1) {
      T *srcDev = nullptr;
      T *dstDev = nullptr;
      size_t srcPitch = 0;
      size_t dstPitch = 0;
      ASSERT_EQ(cudaMallocPitch(&srcDev, &srcPitch, width * CHANNELS * sizeof(T), height), cudaSuccess);
      ASSERT_EQ(cudaMallocPitch(&dstDev, &dstPitch, height * CHANNELS * sizeof(T), width), cudaSuccess);
      ASSERT_EQ(cudaMemcpy2D(srcDev, srcPitch, src.data(), width * CHANNELS * sizeof(T), width * CHANNELS * sizeof(T),
                             height, cudaMemcpyHostToDevice),
                cudaSuccess);

      ASSERT_EQ(fn(srcDev, static_cast<int>(srcPitch), dstDev, static_cast<int>(dstPitch), roi, ctx), NPP_NO_ERROR);
      cudaStreamSynchronize(ctx.hStream);
      ASSERT_EQ(cudaMemcpy2D(dst.data(), height * CHANNELS * sizeof(T), dstDev, dstPitch, height * CHANNELS * sizeof(T),
                             width, cudaMemcpyDeviceToHost),
                cudaSuccess);

      cudaFree(srcDev);
      cudaFree(dstDev);
    } else {
      NppImageMemory<T> srcMem(width, height, CHANNELS);
      NppImageMemory<T> dstMem(height, width, CHANNELS);
      srcMem.copyFromHost(src);

      ASSERT_EQ(fn(srcMem.get(), srcMem.step(), dstMem.get(), dstMem.step(), roi, ctx), NPP_NO_ERROR);
      cudaStreamSynchronize(ctx.hStream);
      dstMem.copyToHost(dst);
    }

    if constexpr (std::is_floating_point_v<T>) {
      EXPECT_TRUE(ResultValidator::arraysEqual(dst, expected, 1e-5f));
    } else {
      EXPECT_TRUE(ResultValidator::arraysEqual(dst, expected));
    }
  }
};

// ==================== 基本转置测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_8u_C1R_BasicSquareMatrix) {
  const int size = 4; // 4x4矩阵

  // prepare test data
  std::vector<Npp8u> src_data(size * size);
  generateTestMatrix(src_data, size, size);

  std::vector<Npp8u> expected_data;
  computeExpectedTranspose(src_data, expected_data, size, size);

  // 分配GPU内存
  NppImageMemory<Npp8u> src(size, size);
  NppImageMemory<Npp8u> dst(size, size);

  src.copyFromHost(src_data);

  // 执行转置操作
  NppiSize roi = {size, size};
  NppStatus status = nppiTranspose_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose failed";

  // Validate结果
  std::vector<Npp8u> result(size * size);
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data)) << "Transpose result incorrect for 4x4 matrix";

  // 打印矩阵用于调试
  std::cout << "Original 4x4 matrix:" << std::endl;
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      std::cout << std::setw(3) << static_cast<int>(src_data[y * size + x]) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Transposed 4x4 matrix:" << std::endl;
  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      std::cout << std::setw(3) << static_cast<int>(result[y * size + x]) << " ";
    }
    std::cout << std::endl;
  }
}

TEST_F(TransposeFunctionalTest, Transpose_32f_C1R_BasicSquareMatrix) {
  const int size = 3; // 3x3矩阵

  // prepare test data
  std::vector<Npp32f> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

  std::vector<Npp32f> expected_data = {1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f};

  NppImageMemory<Npp32f> src(size, size);
  NppImageMemory<Npp32f> dst(size, size);

  src.copyFromHost(src_data);

  NppiSize roi = {size, size};
  NppStatus status = nppiTranspose_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose 32f failed";

  std::vector<Npp32f> result(size * size);
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f)) << "Transpose 32f result incorrect";
}

// ==================== 矩形矩阵转置测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_8u_C1R_RectangularMatrix) {
  const int width = 6;
  const int height = 4;

  // prepare test data
  std::vector<Npp8u> src_data(width * height);
  generateTestMatrix(src_data, width, height);

  std::vector<Npp8u> expected_data;
  computeExpectedTranspose(src_data, expected_data, width, height);

  // 注意：转置后尺寸变为height x width
  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(height, width); // 转置后的尺寸

  src.copyFromHost(src_data);

  NppiSize roi = {width, height};
  NppStatus status = nppiTranspose_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose rectangular failed";

  std::vector<Npp8u> result(width * height);
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data)) << "Transpose rectangular result incorrect";

  // 打印矩阵Validate
  std::cout << "Original " << width << "x" << height << " matrix:" << std::endl;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      std::cout << std::setw(3) << static_cast<int>(src_data[y * width + x]) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Transposed " << height << "x" << width << " matrix:" << std::endl;
  for (int y = 0; y < width; y++) { // 注意尺寸交换
    for (int x = 0; x < height; x++) {
      std::cout << std::setw(3) << static_cast<int>(result[y * height + x]) << " ";
    }
    std::cout << std::endl;
  }
}

// ==================== 大矩阵转置测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_32f_C1R_LargeMatrix) {
  const int width = 128;
  const int height = 64;

  // 生成大矩阵测试数据
  std::vector<Npp32f> src_data(width * height);
  TestDataGenerator::generateRandom(src_data, 0.0f, 100.0f, 12345);

  std::vector<Npp32f> expected_data;
  computeExpectedTranspose(src_data, expected_data, width, height);

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(height, width);

  src.copyFromHost(src_data);

  NppiSize roi = {width, height};
  NppStatus status = nppiTranspose_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose large matrix failed";

  std::vector<Npp32f> result(width * height);
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f)) << "Transpose large matrix result incorrect";

  std::cout << "Large matrix transpose (" << width << "x" << height << ") completed successfully" << std::endl;
}

// ==================== 边界条件测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_8u_C1R_SinglePixel) {
  const int size = 1;

  std::vector<Npp8u> src_data = {42};
  std::vector<Npp8u> expected_data = {42};

  NppImageMemory<Npp8u> src(size, size);
  NppImageMemory<Npp8u> dst(size, size);

  src.copyFromHost(src_data);

  NppiSize roi = {size, size};
  NppStatus status = nppiTranspose_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose single pixel failed";

  std::vector<Npp8u> result(1);
  dst.copyToHost(result);

  EXPECT_EQ(result[0], 42) << "Single pixel transpose failed";
}

TEST_F(TransposeFunctionalTest, Transpose_32f_C1R_SingleRow) {
  const int width = 8;
  const int height = 1;

  std::vector<Npp32f> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<Npp32f> expected_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(height, width); // 转置后变成8x1

  src.copyFromHost(src_data);

  NppiSize roi = {width, height};
  NppStatus status = nppiTranspose_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose single row failed";

  std::vector<Npp32f> result(width * height);
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f)) << "Single row transpose result incorrect";
}

TEST_F(TransposeFunctionalTest, Transpose_32f_C1R_SingleColumn) {
  const int width = 1;
  const int height = 6;

  std::vector<Npp32f> src_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<Npp32f> expected_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(height, width); // 转置后变成6x1

  src.copyFromHost(src_data);

  NppiSize roi = {width, height};
  NppStatus status = nppiTranspose_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose single column failed";

  std::vector<Npp32f> result(width * height);
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data, 1e-5f)) << "Single column transpose result incorrect";
}

// ==================== 双重转置测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_8u_C1R_DoubleTranspose) {
  const int width = 5;
  const int height = 7;

  // 准备原始数据
  std::vector<Npp8u> original_data(width * height);
  generateTestMatrix(original_data, width, height);

  // 第一次转置
  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> temp(height, width);
  NppImageMemory<Npp8u> result(width, height);

  src.copyFromHost(original_data);

  // 转置1: width x height -> height x width
  NppiSize roi1 = {width, height};
  NppStatus status1 = nppiTranspose_8u_C1R(src.get(), src.step(), temp.get(), temp.step(), roi1);
  ASSERT_EQ(status1, NPP_NO_ERROR) << "First transpose failed";

  // 转置2: height x width -> width x height (应该恢复原样)
  NppiSize roi2 = {height, width};
  NppStatus status2 = nppiTranspose_8u_C1R(temp.get(), temp.step(), result.get(), result.step(), roi2);
  ASSERT_EQ(status2, NPP_NO_ERROR) << "Second transpose failed";

  // Validate双重转置后的结果与原始数据相同
  std::vector<Npp8u> final_result(width * height);
  result.copyToHost(final_result);

  EXPECT_TRUE(ResultValidator::arraysEqual(final_result, original_data))
      << "Double transpose should restore original data";
}

#define NPPI_TRANSPOSE_TEST(TEST_NAME, TYPE, CHANNELS, API)                                                            \
  TEST_F(TransposeFunctionalTest, TEST_NAME) { runInterleavedTransposeTest<TYPE, CHANNELS>(API, 5, 3); }

#define NPPI_TRANSPOSE_CTX_TEST(TEST_NAME, TYPE, CHANNELS, API)                                                        \
  TEST_F(TransposeFunctionalTest, TEST_NAME) { runInterleavedTransposeCtxTest<TYPE, CHANNELS>(API, 4, 6); }

NPPI_TRANSPOSE_TEST(Transpose_8u_C3R_Interleaved, Npp8u, 3, nppiTranspose_8u_C3R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_8u_C3R_Ctx_Interleaved, Npp8u, 3, nppiTranspose_8u_C3R_Ctx)
NPPI_TRANSPOSE_TEST(Transpose_8u_C4R_Interleaved, Npp8u, 4, nppiTranspose_8u_C4R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_8u_C4R_Ctx_Interleaved, Npp8u, 4, nppiTranspose_8u_C4R_Ctx)

NPPI_TRANSPOSE_TEST(Transpose_16u_C3R_Interleaved, Npp16u, 3, nppiTranspose_16u_C3R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_16u_C3R_Ctx_Interleaved, Npp16u, 3, nppiTranspose_16u_C3R_Ctx)
NPPI_TRANSPOSE_TEST(Transpose_16u_C4R_Interleaved, Npp16u, 4, nppiTranspose_16u_C4R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_16u_C4R_Ctx_Interleaved, Npp16u, 4, nppiTranspose_16u_C4R_Ctx)

NPPI_TRANSPOSE_TEST(Transpose_16s_C1R_Interleaved, Npp16s, 1, nppiTranspose_16s_C1R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_16s_C1R_Ctx_Interleaved, Npp16s, 1, nppiTranspose_16s_C1R_Ctx)
NPPI_TRANSPOSE_TEST(Transpose_16s_C3R_Interleaved, Npp16s, 3, nppiTranspose_16s_C3R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_16s_C3R_Ctx_Interleaved, Npp16s, 3, nppiTranspose_16s_C3R_Ctx)
NPPI_TRANSPOSE_TEST(Transpose_16s_C4R_Interleaved, Npp16s, 4, nppiTranspose_16s_C4R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_16s_C4R_Ctx_Interleaved, Npp16s, 4, nppiTranspose_16s_C4R_Ctx)

NPPI_TRANSPOSE_TEST(Transpose_32s_C1R_Interleaved, Npp32s, 1, nppiTranspose_32s_C1R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_32s_C1R_Ctx_Interleaved, Npp32s, 1, nppiTranspose_32s_C1R_Ctx)
NPPI_TRANSPOSE_TEST(Transpose_32s_C3R_Interleaved, Npp32s, 3, nppiTranspose_32s_C3R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_32s_C3R_Ctx_Interleaved, Npp32s, 3, nppiTranspose_32s_C3R_Ctx)
NPPI_TRANSPOSE_TEST(Transpose_32s_C4R_Interleaved, Npp32s, 4, nppiTranspose_32s_C4R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_32s_C4R_Ctx_Interleaved, Npp32s, 4, nppiTranspose_32s_C4R_Ctx)

NPPI_TRANSPOSE_TEST(Transpose_32f_C3R_Interleaved, Npp32f, 3, nppiTranspose_32f_C3R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_32f_C3R_Ctx_Interleaved, Npp32f, 3, nppiTranspose_32f_C3R_Ctx)
NPPI_TRANSPOSE_TEST(Transpose_32f_C4R_Interleaved, Npp32f, 4, nppiTranspose_32f_C4R)
NPPI_TRANSPOSE_CTX_TEST(Transpose_32f_C4R_Ctx_Interleaved, Npp32f, 4, nppiTranspose_32f_C4R_Ctx)

#undef NPPI_TRANSPOSE_TEST
#undef NPPI_TRANSPOSE_CTX_TEST

// ==================== 不同数据类型测试 ====================

TEST_F(TransposeFunctionalTest, Transpose_16u_C1R_DataTypeTest) {
  const int size = 4;

  std::vector<Npp16u> src_data = {1000, 2000,  3000,  4000,  5000,  6000,  7000,  8000,
                                  9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000};

  std::vector<Npp16u> expected_data = {1000, 5000, 9000,  13000, 2000, 6000, 10000, 14000,
                                       3000, 7000, 11000, 15000, 4000, 8000, 12000, 16000};

  NppImageMemory<Npp16u> src(size, size);
  NppImageMemory<Npp16u> dst(size, size);

  src.copyFromHost(src_data);

  NppiSize roi = {size, size};
  NppStatus status = nppiTranspose_16u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_NO_ERROR) << "Transpose 16u failed";

  std::vector<Npp16u> result(size * size);
  dst.copyToHost(result);

  EXPECT_TRUE(ResultValidator::arraysEqual(result, expected_data)) << "Transpose 16u result incorrect";
}
