#include "../../framework/npp_test_base.h"
#include "npp.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace npp_functional_test;

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

  static TestData generateTestData(int width, int height, int channels, DataPattern pattern, int seed = 42) {
    TestData data(width, height, channels);
    std::mt19937 gen(seed);

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
      data.expected[i] = static_cast<DstType>(data.src[i]);
    }

    return data;
  }

  static void validateResults(const std::vector<DstType> &actual, const std::vector<DstType> &expected,
                              float tolerance = 1e-6f) {
    ASSERT_EQ(actual.size(), expected.size());

    for (size_t i = 0; i < actual.size(); i++) {
      if constexpr (std::is_floating_point_v<DstType>) {
        (void)tolerance;
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
                      bool useContext = false) {
    using Helper = ConvertTestHelper<SrcType, DstType>;

    // Generate test data
    auto testData = Helper::generateTestData(width, height, Channels, pattern);

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
                                                                        dstGPU.step(), roi, ctx);
      cudaStreamSynchronize(ctx.hStream);
    } else {
      status =
          performConversion<SrcType, DstType, Channels>(srcGPU.get(), srcGPU.step(), dstGPU.get(), dstGPU.step(), roi);
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
  NppStatus performConversion(const SrcType *src, int srcStep, DstType *dst, int dstStep, NppiSize roi) {
    if constexpr (std::is_same_v<SrcType, Npp8u> && std::is_same_v<DstType, Npp32f>) {
      if constexpr (Channels == 1) {
        return nppiConvert_8u32f_C1R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 3) {
        return nppiConvert_8u32f_C3R(src, srcStep, dst, dstStep, roi);
      }
    }
    return NPP_NOT_IMPLEMENTED_ERROR;
  }

  template <typename SrcType, typename DstType, int Channels>
  NppStatus performConversionWithContext(const SrcType *src, int srcStep, DstType *dst, int dstStep, NppiSize roi,
                                         NppStreamContext ctx) {
    if constexpr (std::is_same_v<SrcType, Npp8u> && std::is_same_v<DstType, Npp32f>) {
      if constexpr (Channels == 1) {
        return nppiConvert_8u32f_C1R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 3) {
        return nppiConvert_8u32f_C3R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
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

// Error handling tests
TEST_F(ConvertTest, Convert_ErrorHandling) {
  GPUMemoryManager<Npp8u> src(16, 16, 1);
  GPUMemoryManager<Npp32f> dst(16, 16, 1);
  NppiSize roi = {16, 16};

  // Test null pointer
  EXPECT_EQ(nppiConvert_8u32f_C1R(nullptr, src.step(), dst.get(), dst.step(), roi), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiConvert_8u32f_C1R(src.get(), src.step(), nullptr, dst.step(), roi), NPP_NULL_POINTER_ERROR);

  // Test invalid size
  NppiSize invalidROI = {-1, -1};
  EXPECT_EQ(nppiConvert_8u32f_C1R(src.get(), src.step(), dst.get(), dst.step(), invalidROI), NPP_SIZE_ERROR);
}

// Performance comparison test
TEST_F(ConvertTest, Convert_Performance) {
  const int width = 1920, height = 1080;
  const int iterations = 10;

  using Helper = ConvertTestHelper<Npp8u, Npp32f>;
  auto testData = Helper::generateTestData(width, height, 3, Helper::DataPattern::RANDOM);

  GPUMemoryManager<Npp8u> srcGPU(width, height, 3);
  GPUMemoryManager<Npp32f> dstGPU(width, height, 3);

  ASSERT_TRUE(srcGPU.copyFromHost(testData.src, width, height, 3));

  NppiSize roi = {width, height};

  // Time regular version
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    nppiConvert_8u32f_C3R(srcGPU.get(), srcGPU.step(), dstGPU.get(), dstGPU.step(), roi);
  }
  cudaDeviceSynchronize();
  auto regularTime =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

  // Time context version
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    nppiConvert_8u32f_C3R_Ctx(srcGPU.get(), srcGPU.step(), dstGPU.get(), dstGPU.step(), roi, ctx);
  }
  cudaStreamSynchronize(ctx.hStream);
  auto contextTime =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

  std::cout << "\nConversion Performance Results (" << iterations << " iterations, " << width << "x" << height
            << "):\n";
  std::cout << "  Regular version: " << regularTime << " μs (avg: " << regularTime / iterations << " μs)\n";
  std::cout << "  Context version: " << contextTime << " μs (avg: " << contextTime / iterations << " μs)\n";
}