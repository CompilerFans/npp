#include "npp.h"
#include "npp_test_base.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace npp_functional_test;

namespace {

Npp8u round_and_clamp_to_byte(Npp32f value, NppRoundMode mode) {
  Npp32f rounded = value;
  if (mode == NPP_RND_NEAR) {
    Npp32f floor_val = floorf(value);
    Npp32f diff = value - floor_val;
    if (diff > 0.5f) {
      rounded = floor_val + 1.0f;
    } else if (diff < 0.5f) {
      rounded = floor_val;
    } else {
      int int_floor = static_cast<int>(floor_val);
      rounded = (int_floor % 2 == 0) ? floor_val : floor_val + 1.0f;
    }
  } else if (mode == NPP_RND_FINANCIAL) {
    rounded = roundf(value);
  }
  if (rounded < 0.0f) {
    rounded = 0.0f;
  }
  if (rounded > 255.0f) {
    rounded = 255.0f;
  }
  return static_cast<Npp8u>(rounded);
}

} // namespace

// ============================================================================
// Common Test Utilities and Data Generators
// ============================================================================

template <typename SrcType, typename DstType> class ConvertTestHelper {
public:
  struct TestData {
    std::vector<SrcType> src;
    std::vector<DstType> expected;
    int width, height, channels;

    TestData(int w, int h, int c) : width(w), height(h), channels(c) {
      int size = w * h * c;
      src.resize(size);
      expected.resize(size);
    }
  };

  // Data generation patterns
  enum class DataPattern {
    SEQUENTIAL,    // 0, 1, 2, 3, ...
    GRADIENT,      // Gradient across image
    RANDOM,        // Random values
    BOUNDARY,      // Min/max boundary values
    CHECKERBOARD,  // Alternating pattern
    COLOR_GRADIENT // RGB gradient for color images
  };

  static TestData generateTestData(int width, int height, int channels, DataPattern pattern,
                                   NppRoundMode rndMode = NPP_RND_NEAR, int seed = 42) {
    TestData data(width, height, channels);
    std::mt19937 gen(seed);
    (void)rndMode;

    switch (pattern) {
    case DataPattern::SEQUENTIAL:
      generateSequential(data);
      break;
    case DataPattern::GRADIENT:
      generateGradient(data);
      break;
    case DataPattern::RANDOM:
      generateRandom(data, gen);
      break;
    case DataPattern::BOUNDARY:
      generateBoundary(data);
      break;
    case DataPattern::CHECKERBOARD:
      generateCheckerboard(data);
      break;
    case DataPattern::COLOR_GRADIENT:
      generateColorGradient(data);
      break;
    }

    // Generate expected results based on conversion
    for (size_t i = 0; i < data.src.size(); i++) {
      if constexpr (std::is_same_v<SrcType, Npp32f> && std::is_same_v<DstType, Npp8u>) {
        data.expected[i] = round_and_clamp_to_byte(data.src[i], rndMode);
      } else {
        data.expected[i] = static_cast<DstType>(data.src[i]);
      }
    }

    return data;
  }

  static void validateResults(const std::vector<DstType> &actual, const std::vector<DstType> &expected,
                              float tolerance = 1e-6f) {
    ASSERT_EQ(actual.size(), expected.size());
    (void)tolerance;

    for (size_t i = 0; i < actual.size(); i++) {
      if constexpr (std::is_floating_point_v<DstType>) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "Mismatch at index " << i;
      } else {
        EXPECT_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
      }
    }
  }

private:
  static void generateSequential(TestData &data) {
    for (size_t i = 0; i < data.src.size(); i++) {
      data.src[i] = static_cast<SrcType>(i % 256);
    }
  }

  static void generateGradient(TestData &data) {
    int totalPixels = data.width * data.height;
    for (int i = 0; i < totalPixels; i++) {
      for (int c = 0; c < data.channels; c++) {
        int idx = i * data.channels + c;
        data.src[idx] = static_cast<SrcType>((i * 255) / totalPixels);
      }
    }
  }

  static void generateRandom(TestData &data, std::mt19937 &gen) {
    std::uniform_int_distribution<int> dis(0, 255);
    for (auto &val : data.src) {
      val = static_cast<SrcType>(dis(gen));
    }
  }

  static void generateBoundary(TestData &data) {
    // Fill with boundary values: min, max, and some intermediate values
    std::vector<SrcType> boundaryVals = {0, 1, 127, 128, 254, 255};
    for (size_t i = 0; i < data.src.size(); i++) {
      data.src[i] = boundaryVals[i % boundaryVals.size()];
    }
  }

  static void generateCheckerboard(TestData &data) {
    for (int y = 0; y < data.height; y++) {
      for (int x = 0; x < data.width; x++) {
        SrcType val = ((x + y) % 2) ? 255 : 0;
        for (int c = 0; c < data.channels; c++) {
          int idx = (y * data.width + x) * data.channels + c;
          data.src[idx] = val;
        }
      }
    }
  }

  static void generateColorGradient(TestData &data) {
    if (data.channels < 3) {
      generateGradient(data);
      return;
    }

    for (int y = 0; y < data.height; y++) {
      for (int x = 0; x < data.width; x++) {
        int idx = (y * data.width + x) * data.channels;
        data.src[idx + 0] = static_cast<SrcType>((x * 255) / std::max(1, data.width - 1));  // R
        data.src[idx + 1] = static_cast<SrcType>((y * 255) / std::max(1, data.height - 1)); // G
        data.src[idx + 2] = static_cast<SrcType>(128);                                      // B

        // Additional channels
        for (int c = 3; c < data.channels; c++) {
          data.src[idx + c] = static_cast<SrcType>((x + y + c) % 256);
        }
      }
    }
  }
};

// ============================================================================
// Memory Management Helper
// ============================================================================

template <typename T> class GPUMemoryManager {
private:
  T *ptr_;
  int step_;
  bool allocated_;

public:
  GPUMemoryManager() : ptr_(nullptr), step_(0), allocated_(false) {}

  GPUMemoryManager(int width, int height, int channels = 1) : allocated_(false) { allocate(width, height, channels); }

  ~GPUMemoryManager() { free(); }

  // Non-copyable but movable
  GPUMemoryManager(const GPUMemoryManager &) = delete;
  GPUMemoryManager &operator=(const GPUMemoryManager &) = delete;

  GPUMemoryManager(GPUMemoryManager &&other) noexcept
      : ptr_(other.ptr_), step_(other.step_), allocated_(other.allocated_) {
    other.ptr_ = nullptr;
    other.allocated_ = false;
  }

  GPUMemoryManager &operator=(GPUMemoryManager &&other) noexcept {
    if (this != &other) {
      free();
      ptr_ = other.ptr_;
      step_ = other.step_;
      allocated_ = other.allocated_;
      other.ptr_ = nullptr;
      other.allocated_ = false;
    }
    return *this;
  }

  bool allocate(int width, int height, int channels = 1) {
    free();

    if constexpr (std::is_same_v<T, Npp8u>) {
      if (channels == 1)
        ptr_ = nppiMalloc_8u_C1(width, height, &step_);
      else if (channels == 3)
        ptr_ = nppiMalloc_8u_C3(width, height, &step_);
      else if (channels == 4)
        ptr_ = nppiMalloc_8u_C4(width, height, &step_);
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      if (channels == 1)
        ptr_ = nppiMalloc_32f_C1(width, height, &step_);
      else if (channels == 3)
        ptr_ = nppiMalloc_32f_C3(width, height, &step_);
      else if (channels == 4)
        ptr_ = nppiMalloc_32f_C4(width, height, &step_);
    }

    allocated_ = (ptr_ != nullptr);
    return allocated_;
  }

  void free() {
    if (allocated_ && ptr_) {
      nppiFree(ptr_);
      ptr_ = nullptr;
      allocated_ = false;
    }
  }

  T *get() const { return ptr_; }
  int step() const { return step_; }
  bool isValid() const { return allocated_ && ptr_ != nullptr; }

  bool copyFromHost(const std::vector<T> &data, int width, int height, int channels = 1) {
    if (!isValid())
      return false;

    for (int y = 0; y < height; y++) {
      const T *srcRow = data.data() + y * width * channels;
      T *dstRow = reinterpret_cast<T *>(reinterpret_cast<char *>(ptr_) + y * step_);
      cudaError_t err = cudaMemcpy(dstRow, srcRow, width * channels * sizeof(T), cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
        return false;
    }
    return true;
  }

  bool copyToHost(std::vector<T> &data, int width, int height, int channels = 1) const {
    if (!isValid())
      return false;

    data.resize(width * height * channels);
    for (int y = 0; y < height; y++) {
      const T *srcRow = reinterpret_cast<const T *>(reinterpret_cast<const char *>(ptr_) + y * step_);
      T *dstRow = data.data() + y * width * channels;
      cudaError_t err = cudaMemcpy(dstRow, srcRow, width * channels * sizeof(T), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess)
        return false;
    }
    return true;
  }
};

// ============================================================================
// Unified Convert Test Class
// ============================================================================

class ConvertTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }

  // Generic conversion test template
  template <typename SrcType, typename DstType, int Channels>
  void testConversion(int width, int height, typename ConvertTestHelper<SrcType, DstType>::DataPattern pattern,
                      bool useContext = false, NppRoundMode rndMode = NPP_RND_NEAR) {
    using Helper = ConvertTestHelper<SrcType, DstType>;

    // Generate test data
    auto testData = Helper::generateTestData(width, height, Channels, pattern, rndMode);

    // Allocate GPU memory
    GPUMemoryManager<SrcType> srcGPU(width, height, Channels);
    GPUMemoryManager<DstType> dstGPU(width, height, Channels);

    ASSERT_TRUE(srcGPU.isValid()) << "Failed to allocate source GPU memory";
    ASSERT_TRUE(dstGPU.isValid()) << "Failed to allocate destination GPU memory";

    // Copy test data to GPU
    ASSERT_TRUE(srcGPU.copyFromHost(testData.src, width, height, Channels));

    // Perform conversion
    NppiSize roi = {width, height};
    NppStatus status;

    if (useContext) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = performConversionWithContext<SrcType, DstType, Channels>(srcGPU.get(), srcGPU.step(), dstGPU.get(),
                                                                        dstGPU.step(), roi, rndMode, ctx);
      cudaStreamSynchronize(ctx.hStream);
    } else {
      status = performConversion<SrcType, DstType, Channels>(srcGPU.get(), srcGPU.step(), dstGPU.get(), dstGPU.step(), roi,
                                                            rndMode);
    }

    ASSERT_EQ(status, NPP_SUCCESS) << "Conversion failed";

    // Copy result back and validate
    std::vector<DstType> result;
    ASSERT_TRUE(dstGPU.copyToHost(result, width, height, Channels));

    Helper::validateResults(result, testData.expected);
  }

private:
  // Specialized conversion function dispatch
  template <typename SrcType, typename DstType, int Channels>
  NppStatus performConversion(const SrcType *src, int srcStep, DstType *dst, int dstStep, NppiSize roi,
                              NppRoundMode rndMode) {
    (void)rndMode;
    if constexpr (std::is_same_v<SrcType, Npp8u> && std::is_same_v<DstType, Npp32f>) {
      if constexpr (Channels == 1) {
        return nppiConvert_8u32f_C1R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 3) {
        return nppiConvert_8u32f_C3R(src, srcStep, dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<SrcType, Npp32f> && std::is_same_v<DstType, Npp8u>) {
      if constexpr (Channels == 1) {
        return nppiConvert_32f8u_C1R(src, srcStep, dst, dstStep, roi, rndMode);
      } else if constexpr (Channels == 3) {
        return nppiConvert_32f8u_C3R(src, srcStep, dst, dstStep, roi, rndMode);
      }
    }
    return NPP_NOT_IMPLEMENTED_ERROR;
  }

  template <typename SrcType, typename DstType, int Channels>
  NppStatus performConversionWithContext(const SrcType *src, int srcStep, DstType *dst, int dstStep, NppiSize roi,
                                         NppRoundMode rndMode, NppStreamContext ctx) {
    (void)rndMode;
    if constexpr (std::is_same_v<SrcType, Npp8u> && std::is_same_v<DstType, Npp32f>) {
      if constexpr (Channels == 1) {
        return nppiConvert_8u32f_C1R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 3) {
        return nppiConvert_8u32f_C3R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<SrcType, Npp32f> && std::is_same_v<DstType, Npp8u>) {
      if constexpr (Channels == 1) {
        return nppiConvert_32f8u_C1R_Ctx(src, srcStep, dst, dstStep, roi, rndMode, ctx);
      } else if constexpr (Channels == 3) {
        return nppiConvert_32f8u_C3R_Ctx(src, srcStep, dst, dstStep, roi, rndMode, ctx);
      }
    }
    return NPP_NOT_IMPLEMENTED_ERROR;
  }
};

// ============================================================================
// Simplified Test Cases
// ============================================================================

// Basic functionality tests
TEST_F(ConvertTest, Convert_8u32f_C1R_Sequential) {
  testConversion<Npp8u, Npp32f, 1>(64, 64, ConvertTestHelper<Npp8u, Npp32f>::DataPattern::SEQUENTIAL);
}

TEST_F(ConvertTest, Convert_8u32f_C1R_Boundary) {
  testConversion<Npp8u, Npp32f, 1>(16, 16, ConvertTestHelper<Npp8u, Npp32f>::DataPattern::BOUNDARY);
}

TEST_F(ConvertTest, Convert_8u32f_C3R_ColorGradient) {
  testConversion<Npp8u, Npp32f, 3>(32, 32, ConvertTestHelper<Npp8u, Npp32f>::DataPattern::COLOR_GRADIENT);
}

TEST_F(ConvertTest, Convert_8u32f_C3R_Random) {
  testConversion<Npp8u, Npp32f, 3>(128, 128, ConvertTestHelper<Npp8u, Npp32f>::DataPattern::RANDOM);
}

TEST_F(ConvertTest, Convert_8u32f_C1R_WithContext) {
  testConversion<Npp8u, Npp32f, 1>(64, 64, ConvertTestHelper<Npp8u, Npp32f>::DataPattern::GRADIENT, true);
}

TEST_F(ConvertTest, Convert_32f8u_C3R_Near) {
  testConversion<Npp32f, Npp8u, 3>(64, 64, ConvertTestHelper<Npp32f, Npp8u>::DataPattern::COLOR_GRADIENT, false,
                                    NPP_RND_NEAR);
}

TEST_F(ConvertTest, Convert_32f8u_C3R_FinancialWithContext) {
  testConversion<Npp32f, Npp8u, 3>(32, 32, ConvertTestHelper<Npp32f, Npp8u>::DataPattern::COLOR_GRADIENT, true,
                                    NPP_RND_FINANCIAL);
}

TEST_F(ConvertTest, Convert_32f8u_C3R_RoundingAndClampBehavior) {
  constexpr int kWidth = 3;
  constexpr int kHeight = 2;
  std::vector<Npp32f> srcData = {
      -1.0f, 0.49f, 0.50f,
      1.50f, 2.50f, 3.50f,
      127.49f, 127.50f, 128.50f,
      254.50f, 255.49f, 300.0f,
      -12.5f, -0.5f, 42.5f,
      43.5f, 254.6f, 255.6f,
  };

  auto runCase = [&](NppRoundMode rndMode, const std::vector<Npp8u> &expected) {
    GPUMemoryManager<Npp32f> src(kWidth, kHeight, 3);
    GPUMemoryManager<Npp8u> dst(kWidth, kHeight, 3);
    ASSERT_TRUE(src.isValid());
    ASSERT_TRUE(dst.isValid());
    ASSERT_TRUE(src.copyFromHost(srcData, kWidth, kHeight, 3));

    NppiSize roi{kWidth, kHeight};
    ASSERT_EQ(nppiConvert_32f8u_C3R(src.get(), src.step(), dst.get(), dst.step(), roi, rndMode), NPP_SUCCESS);

    std::vector<Npp8u> result;
    ASSERT_TRUE(dst.copyToHost(result, kWidth, kHeight, 3));
    EXPECT_EQ(result, expected);
  };

  std::vector<Npp8u> expectedNear(srcData.size());
  std::vector<Npp8u> expectedFinancial(srcData.size());
  std::transform(srcData.begin(), srcData.end(), expectedNear.begin(),
                 [](Npp32f value) { return round_and_clamp_to_byte(value, NPP_RND_NEAR); });
  std::transform(srcData.begin(), srcData.end(), expectedFinancial.begin(),
                 [](Npp32f value) { return round_and_clamp_to_byte(value, NPP_RND_FINANCIAL); });

  runCase(NPP_RND_NEAR, expectedNear);
  runCase(NPP_RND_FINANCIAL, expectedFinancial);
}

TEST_F(ConvertTest, Convert_32f8u_C3R_InvalidArguments) {
  GPUMemoryManager<Npp32f> src(4, 3, 3);
  GPUMemoryManager<Npp8u> dst(4, 3, 3);
  ASSERT_TRUE(src.isValid());
  ASSERT_TRUE(dst.isValid());

  const NppiSize validRoi{4, 3};
  const NppiSize invalidRoi{-1, 3};

  EXPECT_EQ(nppiConvert_32f8u_C3R(nullptr, src.step(), dst.get(), dst.step(), validRoi, NPP_RND_NEAR),
            NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiConvert_32f8u_C3R(src.get(), src.step(), dst.get(), 0, validRoi, NPP_RND_NEAR), NPP_STEP_ERROR);
  EXPECT_EQ(nppiConvert_32f8u_C3R(src.get(), 0, dst.get(), dst.step(), validRoi, NPP_RND_NEAR), NPP_SUCCESS);
  EXPECT_EQ(nppiConvert_32f8u_C3R(src.get(), src.step(), dst.get(), dst.step(), invalidRoi, NPP_RND_NEAR),
            NPP_SIZE_ERROR);
}

// Size variation tests
class ConvertSizeTest : public ConvertTest, public ::testing::WithParamInterface<std::tuple<int, int, int>> {};

TEST_P(ConvertSizeTest, Convert_8u32f_VariousSizes) {
  auto [width, height, channels] = GetParam();

  if (channels == 1) {
    testConversion<Npp8u, Npp32f, 1>(width, height, ConvertTestHelper<Npp8u, Npp32f>::DataPattern::GRADIENT);
  } else if (channels == 3) {
    testConversion<Npp8u, Npp32f, 3>(width, height, ConvertTestHelper<Npp8u, Npp32f>::DataPattern::COLOR_GRADIENT);
  }
}

INSTANTIATE_TEST_SUITE_P(SizeVariations, ConvertSizeTest,
                         ::testing::Values(std::make_tuple(1, 1, 1),      // Minimum
                                           std::make_tuple(15, 15, 1),    // Odd size
                                           std::make_tuple(16, 16, 1),    // Power of 2
                                           std::make_tuple(256, 256, 3),  // Medium
                                           std::make_tuple(1920, 1080, 3) // HD
                                           ));
