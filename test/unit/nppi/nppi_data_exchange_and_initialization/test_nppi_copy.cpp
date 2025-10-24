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

// ============================================================================
// Common Test Utilities and Data Generators
// ============================================================================

template <typename T> class CopyTestHelper {
public:
  struct TestData {
    std::vector<T> src;
    std::vector<T> expected;
    int width, height, channels;

    TestData(int w, int h, int c) : width(w), height(h), channels(c) {
      int size = w * h * c;
      src.resize(size);
      expected.resize(size);
    }
  };

  enum class DataPattern {
    SEQUENTIAL,     // 0, 1, 2, 3, ...
    GRADIENT,       // Gradient across image
    RANDOM,         // Random values
    BOUNDARY,       // Min/max boundary values
    CHECKERBOARD,   // Alternating pattern
    COLOR_GRADIENT, // RGB gradient for color images
    COORDINATE,     // Position-encoded values
    CONSTANT,       // Constant value
    SINE_WAVE       // Sine wave pattern
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
    case DataPattern::COORDINATE:
      generateCoordinate(data);
      break;
    case DataPattern::CONSTANT:
      generateConstant(data, static_cast<T>(42));
      break;
    case DataPattern::SINE_WAVE:
      generateSineWave(data);
      break;
    }

    // For copy operations, expected result is the same as source
    data.expected = data.src;

    return data;
  }

  static void validateResults(const std::vector<T> &actual, const std::vector<T> &expected, float tolerance = 1e-6f) {
    ASSERT_EQ(actual.size(), expected.size());
    (void)tolerance;

    for (size_t i = 0; i < actual.size(); i++) {
      if constexpr (std::is_floating_point_v<T>) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "Mismatch at index " << i;
      } else {
        EXPECT_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
      }
    }
  }

  static void validateROIResults(const std::vector<T> &actual, const std::vector<T> &expected, int srcWidth,
                                 int /* srcHeight */, int roiX, int roiY, int roiWidth, int roiHeight,
                                 int channels = 1) {
    for (int y = 0; y < roiHeight; y++) {
      for (int x = 0; x < roiWidth; x++) {
        for (int c = 0; c < channels; c++) {
          int srcIdx = ((roiY + y) * srcWidth + (roiX + x)) * channels + c;
          int dstIdx = (y * roiWidth + x) * channels + c;

          if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(actual[dstIdx], expected[srcIdx], 1e-6f)
                << "ROI mismatch at (" << x << "," << y << ") channel " << c;
          } else {
            EXPECT_EQ(actual[dstIdx], expected[srcIdx]) << "ROI mismatch at (" << x << "," << y << ") channel " << c;
          }
        }
      }
    }
  }

private:
  static void generateSequential(TestData &data) {
    for (size_t i = 0; i < data.src.size(); i++) {
      if constexpr (std::is_floating_point_v<T>) {
        data.src[i] = static_cast<T>(i % 256);
      } else {
        data.src[i] = static_cast<T>(i % 256);
      }
    }
  }

  static void generateGradient(TestData &data) {
    int totalPixels = data.width * data.height;
    for (int i = 0; i < totalPixels; i++) {
      for (int c = 0; c < data.channels; c++) {
        int idx = i * data.channels + c;
        if constexpr (std::is_floating_point_v<T>) {
          data.src[idx] = static_cast<T>((i * 255.0f) / (totalPixels - 1));
        } else {
          data.src[idx] = static_cast<T>((i * 255) / std::max(1, totalPixels - 1));
        }
      }
    }
  }

  static void generateRandom(TestData &data, std::mt19937 &gen) {
    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<float> dis(0.0f, 255.0f);
      for (auto &val : data.src) {
        val = static_cast<T>(dis(gen));
      }
    } else {
      std::uniform_int_distribution<int> dis(0, 255);
      for (auto &val : data.src) {
        val = static_cast<T>(dis(gen));
      }
    }
  }

  static void generateBoundary(TestData &data) {
    std::vector<T> boundaryVals;
    if constexpr (std::is_floating_point_v<T>) {
      boundaryVals = {T(0), T(1), T(127.5), T(128), T(254), T(255)};
    } else {
      boundaryVals = {T(0), T(1), T(127), T(128), T(254), T(255)};
    }

    for (size_t i = 0; i < data.src.size(); i++) {
      data.src[i] = boundaryVals[i % boundaryVals.size()];
    }
  }

  static void generateCheckerboard(TestData &data) {
    for (int y = 0; y < data.height; y++) {
      for (int x = 0; x < data.width; x++) {
        T val = ((x + y) % 2) ? T(255) : T(0);
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

        // RGB gradient
        if constexpr (std::is_floating_point_v<T>) {
          data.src[idx + 0] = static_cast<T>((x * 255.0f) / std::max(1.0f, float(data.width - 1)));
          data.src[idx + 1] = static_cast<T>((y * 255.0f) / std::max(1.0f, float(data.height - 1)));
          data.src[idx + 2] = static_cast<T>(128.0f);
        } else {
          data.src[idx + 0] = static_cast<T>((x * 255) / std::max(1, data.width - 1));
          data.src[idx + 1] = static_cast<T>((y * 255) / std::max(1, data.height - 1));
          data.src[idx + 2] = static_cast<T>(128);
        }

        // Additional channels
        for (int c = 3; c < data.channels; c++) {
          data.src[idx + c] = static_cast<T>((x + y + c) % 256);
        }
      }
    }
  }

  static void generateCoordinate(TestData &data) {
    for (int y = 0; y < data.height; y++) {
      for (int x = 0; x < data.width; x++) {
        int idx = (y * data.width + x) * data.channels;

        if (data.channels >= 3) {
          if constexpr (std::is_floating_point_v<T>) {
            data.src[idx + 0] = static_cast<T>(x + y * 100);  // Position encoding
            data.src[idx + 1] = static_cast<T>(x * y + 1000); // Product encoding
            data.src[idx + 2] = static_cast<T>(x - y + 2000); // Difference encoding
          } else {
            data.src[idx + 0] = static_cast<T>((x + y * 100) % 256);
            data.src[idx + 1] = static_cast<T>((x * y + 1000) % 256);
            data.src[idx + 2] = static_cast<T>((x - y + 2000) % 256);
          }

          for (int c = 3; c < data.channels; c++) {
            data.src[idx + c] = static_cast<T>((x + y * c) % 256);
          }
        } else {
          for (int c = 0; c < data.channels; c++) {
            data.src[idx + c] = static_cast<T>((x + y * 100 + c) % 256);
          }
        }
      }
    }
  }

  static void generateConstant(TestData &data, T value) { std::fill(data.src.begin(), data.src.end(), value); }

  static void generateSineWave(TestData &data) {
    for (int y = 0; y < data.height; y++) {
      for (int x = 0; x < data.width; x++) {
        int idx = (y * data.width + x) * data.channels;
        float t = static_cast<float>(x + y) / (data.width + data.height);

        for (int c = 0; c < data.channels; c++) {
          float val = 127.5f * (1.0f + std::sin(2.0f * M_PI * (2 + c) * t));
          data.src[idx + c] = static_cast<T>(val);
        }
      }
    }
  }
};

// ============================================================================
// Memory Management Helper (reuse from convert test)
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
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      if (channels == 1)
        ptr_ = nppiMalloc_16u_C1(width, height, &step_);
      else if (channels == 3)
        ptr_ = nppiMalloc_16u_C3(width, height, &step_);
      else if (channels == 4)
        ptr_ = nppiMalloc_16u_C4(width, height, &step_);
    } else if constexpr (std::is_same_v<T, Npp16s> || std::is_same_v<T, Npp32s>) {
      size_t pitch;
      cudaError_t err = cudaMallocPitch(&ptr_, &pitch, width * channels * sizeof(T), height);
      if (err == cudaSuccess) {
        step_ = static_cast<int>(pitch);
      }
    }

    allocated_ = (ptr_ != nullptr);
    return allocated_;
  }

  void free() {
    if (allocated_ && ptr_) {
      if constexpr (std::is_same_v<T, Npp16s> || std::is_same_v<T, Npp32s>) {
        cudaFree(ptr_);
      } else {
        nppiFree(ptr_);
      }
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

  bool copyROIToHost(std::vector<T> &data, int srcWidth, int srcHeight, int roiX, int roiY, int roiWidth, int roiHeight,
                     int channels = 1) const {
    if (!isValid())
      return false;

    data.resize(roiWidth * roiHeight * channels);
    for (int y = 0; y < roiHeight; y++) {
      const T *srcRow = reinterpret_cast<const T *>(reinterpret_cast<const char *>(ptr_) + (roiY + y) * step_);
      T *dstRow = data.data() + y * roiWidth * channels;
      cudaError_t err =
          cudaMemcpy(dstRow, srcRow + roiX * channels, roiWidth * channels * sizeof(T), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess)
        return false;
    }
    return true;
  }
};

// ============================================================================
// Planar Memory Management Helper
// ============================================================================

template <typename T, int NumPlanes> class PlanarGPUMemoryManager {
private:
  T *planes_[NumPlanes];
  int step_;
  bool allocated_;

public:
  PlanarGPUMemoryManager() : step_(0), allocated_(false) {
    for (int i = 0; i < NumPlanes; i++) {
      planes_[i] = nullptr;
    }
  }

  PlanarGPUMemoryManager(int width, int height) : allocated_(false) {
    for (int i = 0; i < NumPlanes; i++) {
      planes_[i] = nullptr;
    }
    allocate(width, height);
  }

  ~PlanarGPUMemoryManager() { free(); }

  PlanarGPUMemoryManager(const PlanarGPUMemoryManager &) = delete;
  PlanarGPUMemoryManager &operator=(const PlanarGPUMemoryManager &) = delete;

  bool allocate(int width, int height) {
    free();

    if constexpr (std::is_same_v<T, Npp8u>) {
      for (int i = 0; i < NumPlanes; i++) {
        planes_[i] = nppiMalloc_8u_C1(width, height, &step_);
        if (!planes_[i])
          return false;
      }
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      for (int i = 0; i < NumPlanes; i++) {
        planes_[i] = nppiMalloc_32f_C1(width, height, &step_);
        if (!planes_[i])
          return false;
      }
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      for (int i = 0; i < NumPlanes; i++) {
        planes_[i] = nppiMalloc_16u_C1(width, height, &step_);
        if (!planes_[i])
          return false;
      }
    } else if constexpr (std::is_same_v<T, Npp16s> || std::is_same_v<T, Npp32s>) {
      size_t pitch;
      for (int i = 0; i < NumPlanes; i++) {
        cudaError_t err = cudaMallocPitch(&planes_[i], &pitch, width * sizeof(T), height);
        if (err != cudaSuccess)
          return false;
        step_ = static_cast<int>(pitch);
      }
    }

    allocated_ = true;
    return allocated_;
  }

  void free() {
    if (allocated_) {
      for (int i = 0; i < NumPlanes; i++) {
        if (planes_[i]) {
          if constexpr (std::is_same_v<T, Npp16s> || std::is_same_v<T, Npp32s>) {
            cudaFree(planes_[i]);
          } else {
            nppiFree(planes_[i]);
          }
          planes_[i] = nullptr;
        }
      }
      allocated_ = false;
    }
  }

  T *getPlane(int index) const {
    if (index >= 0 && index < NumPlanes)
      return planes_[index];
    return nullptr;
  }

  T *const *getPlanes() const { return const_cast<T *const *>(planes_); }

  int step() const { return step_; }
  bool isValid() const { return allocated_; }

  bool copyFromHostPacked(const std::vector<T> &packedData, int width, int height) {
    if (!isValid())
      return false;

    std::vector<T> planeData(width * height);

    for (int plane = 0; plane < NumPlanes; plane++) {
      for (int i = 0; i < width * height; i++) {
        planeData[i] = packedData[i * NumPlanes + plane];
      }

      for (int y = 0; y < height; y++) {
        const T *srcRow = planeData.data() + y * width;
        T *dstRow = reinterpret_cast<T *>(reinterpret_cast<char *>(planes_[plane]) + y * step_);
        cudaError_t err = cudaMemcpy(dstRow, srcRow, width * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
          return false;
      }
    }
    return true;
  }

  bool copyToHostPacked(std::vector<T> &packedData, int width, int height) const {
    if (!isValid())
      return false;

    packedData.resize(width * height * NumPlanes);
    std::vector<T> planeData(width * height);

    for (int plane = 0; plane < NumPlanes; plane++) {
      for (int y = 0; y < height; y++) {
        const T *srcRow = reinterpret_cast<const T *>(reinterpret_cast<const char *>(planes_[plane]) + y * step_);
        T *dstRow = planeData.data() + y * width;
        cudaError_t err = cudaMemcpy(dstRow, srcRow, width * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
          return false;
      }

      for (int i = 0; i < width * height; i++) {
        packedData[i * NumPlanes + plane] = planeData[i];
      }
    }
    return true;
  }
};

// ============================================================================
// Unified Copy Test Class
// ============================================================================

class CopyTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }

  // Generic copy test template
  template <typename T, int Channels>
  void testCopy(int width, int height, typename CopyTestHelper<T>::DataPattern pattern, bool useContext = false) {
    using Helper = CopyTestHelper<T>;

    // Generate test data
    auto testData = Helper::generateTestData(width, height, Channels, pattern);

    // Allocate GPU memory
    GPUMemoryManager<T> srcGPU(width, height, Channels);
    GPUMemoryManager<T> dstGPU(width, height, Channels);

    ASSERT_TRUE(srcGPU.isValid()) << "Failed to allocate source GPU memory";
    ASSERT_TRUE(dstGPU.isValid()) << "Failed to allocate destination GPU memory";

    // Copy test data to GPU
    ASSERT_TRUE(srcGPU.copyFromHost(testData.src, width, height, Channels));

    // Perform copy
    NppiSize roi = {width, height};
    NppStatus status;

    if (useContext) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = performCopyWithContext<T, Channels>(srcGPU.get(), srcGPU.step(), dstGPU.get(), dstGPU.step(), roi, ctx);
      cudaStreamSynchronize(ctx.hStream);
    } else {
      status = performCopy<T, Channels>(srcGPU.get(), srcGPU.step(), dstGPU.get(), dstGPU.step(), roi);
    }

    ASSERT_EQ(status, NPP_SUCCESS) << "Copy failed";

    // Copy result back and validate
    std::vector<T> result;
    ASSERT_TRUE(dstGPU.copyToHost(result, width, height, Channels));

    Helper::validateResults(result, testData.expected);
  }

  // ROI copy test template
  template <typename T, int Channels>
  void testCopyROI(int srcWidth, int srcHeight, int roiX, int roiY, int roiWidth, int roiHeight,
                   typename CopyTestHelper<T>::DataPattern pattern) {
    using Helper = CopyTestHelper<T>;

    // Generate source data
    auto testData = Helper::generateTestData(srcWidth, srcHeight, Channels, pattern);

    // Allocate GPU memory
    GPUMemoryManager<T> srcGPU(srcWidth, srcHeight, Channels);
    GPUMemoryManager<T> dstGPU(roiWidth, roiHeight, Channels);

    ASSERT_TRUE(srcGPU.isValid()) << "Failed to allocate source GPU memory";
    ASSERT_TRUE(dstGPU.isValid()) << "Failed to allocate destination GPU memory";

    // Copy test data to GPU
    ASSERT_TRUE(srcGPU.copyFromHost(testData.src, srcWidth, srcHeight, Channels));

    // Perform ROI copy
    NppiSize roiSize = {roiWidth, roiHeight};
    T *srcPtr = reinterpret_cast<T *>(reinterpret_cast<char *>(srcGPU.get()) + roiY * srcGPU.step()) + roiX * Channels;

    NppStatus status = performCopy<T, Channels>(srcPtr, srcGPU.step(), dstGPU.get(), dstGPU.step(), roiSize);

    ASSERT_EQ(status, NPP_SUCCESS) << "ROI copy failed";

    // Copy result back and validate
    std::vector<T> result;
    ASSERT_TRUE(dstGPU.copyToHost(result, roiWidth, roiHeight, Channels));

    Helper::validateROIResults(result, testData.src, srcWidth, srcHeight, roiX, roiY, roiWidth, roiHeight, Channels);
  }

  // Planar to Packed copy test (P3C3R/P4C4R)
  template <typename T, int NumPlanes>
  void testCopyPlanarToPacked(int width, int height, typename CopyTestHelper<T>::DataPattern pattern,
                              bool useContext = false) {
    using Helper = CopyTestHelper<T>;

    // Generate packed test data
    auto testData = Helper::generateTestData(width, height, NumPlanes, pattern);

    // Allocate planar source and packed destination
    PlanarGPUMemoryManager<T, NumPlanes> srcPlanar(width, height);
    GPUMemoryManager<T> dstPacked(width, height, NumPlanes);

    ASSERT_TRUE(srcPlanar.isValid()) << "Failed to allocate planar source memory";
    ASSERT_TRUE(dstPacked.isValid()) << "Failed to allocate packed destination memory";

    // Copy packed data to planar format on GPU
    ASSERT_TRUE(srcPlanar.copyFromHostPacked(testData.src, width, height));

    // Perform planar to packed copy
    NppiSize roi = {width, height};
    NppStatus status;

    if (useContext) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = performPlanarToPackedCopy<T, NumPlanes>(srcPlanar.getPlanes(), srcPlanar.step(), dstPacked.get(),
                                                       dstPacked.step(), roi, ctx);
      cudaStreamSynchronize(ctx.hStream);
    } else {
      status = performPlanarToPackedCopy<T, NumPlanes>(srcPlanar.getPlanes(), srcPlanar.step(), dstPacked.get(),
                                                       dstPacked.step(), roi);
    }

    ASSERT_EQ(status, NPP_SUCCESS) << "Planar to packed copy failed";

    // Copy result back and validate
    std::vector<T> result;
    ASSERT_TRUE(dstPacked.copyToHost(result, width, height, NumPlanes));

    Helper::validateResults(result, testData.expected);
  }

  // Packed to Planar copy test (C3P3R/C4P4R)
  template <typename T, int NumPlanes>
  void testCopyPackedToPlanar(int width, int height, typename CopyTestHelper<T>::DataPattern pattern,
                              bool useContext = false) {
    using Helper = CopyTestHelper<T>;

    // Generate packed test data
    auto testData = Helper::generateTestData(width, height, NumPlanes, pattern);

    // Allocate packed source and planar destination
    GPUMemoryManager<T> srcPacked(width, height, NumPlanes);
    PlanarGPUMemoryManager<T, NumPlanes> dstPlanar(width, height);

    ASSERT_TRUE(srcPacked.isValid()) << "Failed to allocate packed source memory";
    ASSERT_TRUE(dstPlanar.isValid()) << "Failed to allocate planar destination memory";

    // Copy test data to GPU
    ASSERT_TRUE(srcPacked.copyFromHost(testData.src, width, height, NumPlanes));

    // Perform packed to planar copy
    NppiSize roi = {width, height};
    NppStatus status;

    if (useContext) {
      NppStreamContext ctx;
      nppGetStreamContext(&ctx);
      status = performPackedToPlanarCopy<T, NumPlanes>(srcPacked.get(), srcPacked.step(), dstPlanar.getPlanes(),
                                                       dstPlanar.step(), roi, ctx);
      cudaStreamSynchronize(ctx.hStream);
    } else {
      status = performPackedToPlanarCopy<T, NumPlanes>(srcPacked.get(), srcPacked.step(), dstPlanar.getPlanes(),
                                                       dstPlanar.step(), roi);
    }

    ASSERT_EQ(status, NPP_SUCCESS) << "Packed to planar copy failed";

    // Copy result back and validate
    std::vector<T> result;
    ASSERT_TRUE(dstPlanar.copyToHostPacked(result, width, height));

    Helper::validateResults(result, testData.expected);
  }

  // Round-trip test: Packed -> Planar -> Packed
  template <typename T, int NumPlanes>
  void testRoundTripPackedToPlanarToPacked(int width, int height, typename CopyTestHelper<T>::DataPattern pattern) {
    using Helper = CopyTestHelper<T>;

    // Generate original packed data
    auto testData = Helper::generateTestData(width, height, NumPlanes, pattern);

    // Allocate memory for round-trip
    GPUMemoryManager<T> srcPacked(width, height, NumPlanes);
    PlanarGPUMemoryManager<T, NumPlanes> intermediatePlanar(width, height);
    GPUMemoryManager<T> finalPacked(width, height, NumPlanes);

    ASSERT_TRUE(srcPacked.isValid()) << "Failed to allocate source packed memory";
    ASSERT_TRUE(intermediatePlanar.isValid()) << "Failed to allocate intermediate planar memory";
    ASSERT_TRUE(finalPacked.isValid()) << "Failed to allocate final packed memory";

    // Copy original data to GPU
    ASSERT_TRUE(srcPacked.copyFromHost(testData.src, width, height, NumPlanes));

    NppiSize roi = {width, height};

    // Step 1: Packed -> Planar (C3P3R or C4P4R)
    NppStatus status1 = performPackedToPlanarCopy<T, NumPlanes>(srcPacked.get(), srcPacked.step(),
                                                                 intermediatePlanar.getPlanes(),
                                                                 intermediatePlanar.step(), roi);
    ASSERT_EQ(status1, NPP_SUCCESS) << "First conversion (packed to planar) failed";

    // Step 2: Planar -> Packed (P3C3R or P4C4R)
    NppStatus status2 = performPlanarToPackedCopy<T, NumPlanes>(intermediatePlanar.getPlanes(),
                                                                 intermediatePlanar.step(), finalPacked.get(),
                                                                 finalPacked.step(), roi);
    ASSERT_EQ(status2, NPP_SUCCESS) << "Second conversion (planar to packed) failed";

    // Validate final result matches original
    std::vector<T> result;
    ASSERT_TRUE(finalPacked.copyToHost(result, width, height, NumPlanes));

    Helper::validateResults(result, testData.expected);
  }

  // Round-trip test: Planar -> Packed -> Planar
  template <typename T, int NumPlanes>
  void testRoundTripPlanarToPackedToPlanar(int width, int height, typename CopyTestHelper<T>::DataPattern pattern) {
    using Helper = CopyTestHelper<T>;

    // Generate original packed data (will be converted to planar)
    auto testData = Helper::generateTestData(width, height, NumPlanes, pattern);

    // Allocate memory for round-trip
    PlanarGPUMemoryManager<T, NumPlanes> srcPlanar(width, height);
    GPUMemoryManager<T> intermediatePacked(width, height, NumPlanes);
    PlanarGPUMemoryManager<T, NumPlanes> finalPlanar(width, height);

    ASSERT_TRUE(srcPlanar.isValid()) << "Failed to allocate source planar memory";
    ASSERT_TRUE(intermediatePacked.isValid()) << "Failed to allocate intermediate packed memory";
    ASSERT_TRUE(finalPlanar.isValid()) << "Failed to allocate final planar memory";

    // Copy original data to GPU in planar format
    ASSERT_TRUE(srcPlanar.copyFromHostPacked(testData.src, width, height));

    NppiSize roi = {width, height};

    // Step 1: Planar -> Packed (P3C3R or P4C4R)
    NppStatus status1 = performPlanarToPackedCopy<T, NumPlanes>(srcPlanar.getPlanes(), srcPlanar.step(),
                                                                 intermediatePacked.get(), intermediatePacked.step(), roi);
    ASSERT_EQ(status1, NPP_SUCCESS) << "First conversion (planar to packed) failed";

    // Step 2: Packed -> Planar (C3P3R or C4P4R)
    NppStatus status2 = performPackedToPlanarCopy<T, NumPlanes>(intermediatePacked.get(), intermediatePacked.step(),
                                                                 finalPlanar.getPlanes(), finalPlanar.step(), roi);
    ASSERT_EQ(status2, NPP_SUCCESS) << "Second conversion (packed to planar) failed";

    // Validate final result matches original
    std::vector<T> result;
    ASSERT_TRUE(finalPlanar.copyToHostPacked(result, width, height));

    Helper::validateResults(result, testData.expected);
  }

private:
  // Specialized copy function dispatch
  template <typename T, int Channels>
  NppStatus performCopy(const T *src, int srcStep, T *dst, int dstStep, NppiSize roi) {
    if constexpr (std::is_same_v<T, Npp8u>) {
      if constexpr (Channels == 1) {
        return nppiCopy_8u_C1R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 3) {
        return nppiCopy_8u_C3R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 4) {
        return nppiCopy_8u_C4R(src, srcStep, dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      if constexpr (Channels == 1) {
        return nppiCopy_32f_C1R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 3) {
        return nppiCopy_32f_C3R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 4) {
        return nppiCopy_32f_C4R(src, srcStep, dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      if constexpr (Channels == 1) {
        return nppiCopy_16u_C1R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 3) {
        return nppiCopy_16u_C3R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 4) {
        return nppiCopy_16u_C4R(src, srcStep, dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      if constexpr (Channels == 1) {
        return nppiCopy_16s_C1R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 3) {
        return nppiCopy_16s_C3R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 4) {
        return nppiCopy_16s_C4R(src, srcStep, dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp32s>) {
      if constexpr (Channels == 1) {
        return nppiCopy_32s_C1R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 3) {
        return nppiCopy_32s_C3R(src, srcStep, dst, dstStep, roi);
      } else if constexpr (Channels == 4) {
        return nppiCopy_32s_C4R(src, srcStep, dst, dstStep, roi);
      }
    }
    return NPP_NOT_IMPLEMENTED_ERROR;
  }

  template <typename T, int Channels>
  NppStatus performCopyWithContext(const T *src, int srcStep, T *dst, int dstStep, NppiSize roi, NppStreamContext ctx) {
    if constexpr (std::is_same_v<T, Npp8u>) {
      if constexpr (Channels == 1) {
        return nppiCopy_8u_C1R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 3) {
        return nppiCopy_8u_C3R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 4) {
        return nppiCopy_8u_C4R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      if constexpr (Channels == 1) {
        return nppiCopy_32f_C1R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 3) {
        return nppiCopy_32f_C3R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 4) {
        return nppiCopy_32f_C4R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      if constexpr (Channels == 1) {
        return nppiCopy_16u_C1R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 3) {
        return nppiCopy_16u_C3R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 4) {
        return nppiCopy_16u_C4R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      if constexpr (Channels == 1) {
        return nppiCopy_16s_C1R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 3) {
        return nppiCopy_16s_C3R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 4) {
        return nppiCopy_16s_C4R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp32s>) {
      if constexpr (Channels == 1) {
        return nppiCopy_32s_C1R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 3) {
        return nppiCopy_32s_C3R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (Channels == 4) {
        return nppiCopy_32s_C4R_Ctx(src, srcStep, dst, dstStep, roi, ctx);
      }
    }
    return NPP_NOT_IMPLEMENTED_ERROR;
  }

  // Planar to Packed function dispatch (P3C3R/P4C4R)
  template <typename T, int NumPlanes>
  NppStatus performPlanarToPackedCopy(T *const *src, int srcStep, T *dst, int dstStep, NppiSize roi) {
    if constexpr (std::is_same_v<T, Npp8u>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_8u_P3C3R((const Npp8u *const *)src, srcStep, dst, dstStep, roi);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_8u_P4C4R((const Npp8u *const *)src, srcStep, dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_32f_P3C3R((const Npp32f *const *)src, srcStep, dst, dstStep, roi);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_32f_P4C4R((const Npp32f *const *)src, srcStep, dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_16u_P3C3R((const Npp16u *const *)src, srcStep, dst, dstStep, roi);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_16u_P4C4R((const Npp16u *const *)src, srcStep, dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_16s_P3C3R((const Npp16s *const *)src, srcStep, dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp32s>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_32s_P3C3R((const Npp32s *const *)src, srcStep, dst, dstStep, roi);
      }
    }
    return NPP_NOT_IMPLEMENTED_ERROR;
  }

  template <typename T, int NumPlanes>
  NppStatus performPlanarToPackedCopy(T *const *src, int srcStep, T *dst, int dstStep, NppiSize roi,
                                      NppStreamContext ctx) {
    if constexpr (std::is_same_v<T, Npp8u>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_8u_P3C3R_Ctx((const Npp8u *const *)src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_8u_P4C4R_Ctx((const Npp8u *const *)src, srcStep, dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_32f_P3C3R_Ctx((const Npp32f *const *)src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_32f_P4C4R_Ctx((const Npp32f *const *)src, srcStep, dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_16u_P3C3R_Ctx((const Npp16u *const *)src, srcStep, dst, dstStep, roi, ctx);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_16u_P4C4R_Ctx((const Npp16u *const *)src, srcStep, dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_16s_P3C3R_Ctx((const Npp16s *const *)src, srcStep, dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp32s>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_32s_P3C3R_Ctx((const Npp32s *const *)src, srcStep, dst, dstStep, roi, ctx);
      }
    }
    return NPP_NOT_IMPLEMENTED_ERROR;
  }

  // Packed to Planar function dispatch (C3P3R/C4P4R)
  template <typename T, int NumPlanes>
  NppStatus performPackedToPlanarCopy(const T *src, int srcStep, T *const *dst, int dstStep, NppiSize roi) {
    if constexpr (std::is_same_v<T, Npp8u>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_8u_C3P3R(src, srcStep, (Npp8u *const *)dst, dstStep, roi);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_8u_C4P4R(src, srcStep, (Npp8u *const *)dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_32f_C3P3R(src, srcStep, (Npp32f *const *)dst, dstStep, roi);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_32f_C4P4R(src, srcStep, (Npp32f *const *)dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_16u_C3P3R(src, srcStep, (Npp16u *const *)dst, dstStep, roi);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_16u_C4P4R(src, srcStep, (Npp16u *const *)dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_16s_C3P3R(src, srcStep, (Npp16s *const *)dst, dstStep, roi);
      }
    } else if constexpr (std::is_same_v<T, Npp32s>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_32s_C3P3R(src, srcStep, (Npp32s *const *)dst, dstStep, roi);
      }
    }
    return NPP_NOT_IMPLEMENTED_ERROR;
  }

  template <typename T, int NumPlanes>
  NppStatus performPackedToPlanarCopy(const T *src, int srcStep, T *const *dst, int dstStep, NppiSize roi,
                                      NppStreamContext ctx) {
    if constexpr (std::is_same_v<T, Npp8u>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_8u_C3P3R_Ctx(src, srcStep, (Npp8u *const *)dst, dstStep, roi, ctx);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_8u_C4P4R_Ctx(src, srcStep, (Npp8u *const *)dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_32f_C3P3R_Ctx(src, srcStep, (Npp32f *const *)dst, dstStep, roi, ctx);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_32f_C4P4R_Ctx(src, srcStep, (Npp32f *const *)dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_16u_C3P3R_Ctx(src, srcStep, (Npp16u *const *)dst, dstStep, roi, ctx);
      } else if constexpr (NumPlanes == 4) {
        return nppiCopy_16u_C4P4R_Ctx(src, srcStep, (Npp16u *const *)dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_16s_C3P3R_Ctx(src, srcStep, (Npp16s *const *)dst, dstStep, roi, ctx);
      }
    } else if constexpr (std::is_same_v<T, Npp32s>) {
      if constexpr (NumPlanes == 3) {
        return nppiCopy_32s_C3P3R_Ctx(src, srcStep, (Npp32s *const *)dst, dstStep, roi, ctx);
      }
    }
    return NPP_NOT_IMPLEMENTED_ERROR;
  }
};

// ============================================================================
// Simplified Test Cases
// ============================================================================

// Basic functionality tests
TEST_F(CopyTest, Copy_8u_C1R_Sequential) { testCopy<Npp8u, 1>(32, 32, CopyTestHelper<Npp8u>::DataPattern::SEQUENTIAL); }

TEST_F(CopyTest, Copy_8u_C1R_Boundary) { testCopy<Npp8u, 1>(16, 16, CopyTestHelper<Npp8u>::DataPattern::BOUNDARY); }

TEST_F(CopyTest, Copy_8u_C3R_ColorGradient) {
  testCopy<Npp8u, 3>(16, 16, CopyTestHelper<Npp8u>::DataPattern::COLOR_GRADIENT);
}

TEST_F(CopyTest, Copy_8u_C4R_Random) { testCopy<Npp8u, 4>(24, 24, CopyTestHelper<Npp8u>::DataPattern::RANDOM); }

TEST_F(CopyTest, Copy_32f_C1R_Gradient) { testCopy<Npp32f, 1>(64, 48, CopyTestHelper<Npp32f>::DataPattern::GRADIENT); }

TEST_F(CopyTest, Copy_32f_C3R_Coordinate) {
  testCopy<Npp32f, 3>(32, 24, CopyTestHelper<Npp32f>::DataPattern::COORDINATE);
}

TEST_F(CopyTest, Copy_32f_C1R_WithContext) {
  testCopy<Npp32f, 1>(48, 32, CopyTestHelper<Npp32f>::DataPattern::SINE_WAVE, true);
}

// Size variation tests
class CopySizeTest : public CopyTest, public ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

TEST_P(CopySizeTest, Copy_VariousSizes) {
  auto [width, height, channels, dataType] = GetParam();

  if (dataType == 0) { // 8u
    if (channels == 1) {
      testCopy<Npp8u, 1>(width, height, CopyTestHelper<Npp8u>::DataPattern::GRADIENT);
    } else if (channels == 3) {
      testCopy<Npp8u, 3>(width, height, CopyTestHelper<Npp8u>::DataPattern::COLOR_GRADIENT);
    } else if (channels == 4) {
      testCopy<Npp8u, 4>(width, height, CopyTestHelper<Npp8u>::DataPattern::CHECKERBOARD);
    }
  } else if (dataType == 1) { // 32f
    if (channels == 1) {
      testCopy<Npp32f, 1>(width, height, CopyTestHelper<Npp32f>::DataPattern::GRADIENT);
    } else if (channels == 3) {
      testCopy<Npp32f, 3>(width, height, CopyTestHelper<Npp32f>::DataPattern::COORDINATE);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(SizeVariations, CopySizeTest,
                         ::testing::Values(std::make_tuple(1, 1, 1, 0),      // Minimum 8u C1
                                           std::make_tuple(7, 5, 1, 0),      // Odd size 8u C1
                                           std::make_tuple(16, 16, 3, 0),    // Square 8u C3
                                           std::make_tuple(32, 24, 4, 0),    // Rect 8u C4
                                           std::make_tuple(64, 48, 1, 1),    // Medium 32f C1
                                           std::make_tuple(128, 96, 3, 1),   // Large 32f C3
                                           std::make_tuple(1920, 1080, 3, 0) // HD 8u C3
                                           ));

// ROI copy tests
TEST_F(CopyTest, Copy_8u_C1R_ROI) {
  testCopyROI<Npp8u, 1>(64, 48, 16, 12, 32, 24, CopyTestHelper<Npp8u>::DataPattern::SEQUENTIAL);
}

TEST_F(CopyTest, Copy_32f_C3R_ROI) {
  testCopyROI<Npp32f, 3>(128, 96, 32, 24, 64, 48, CopyTestHelper<Npp32f>::DataPattern::COORDINATE);
}

// Tests for 16s (signed 16-bit)
TEST_F(CopyTest, Copy_16s_C1R_Sequential) {
  testCopy<Npp16s, 1>(32, 32, CopyTestHelper<Npp16s>::DataPattern::SEQUENTIAL);
}

TEST_F(CopyTest, Copy_16s_C3R_Gradient) { testCopy<Npp16s, 3>(48, 48, CopyTestHelper<Npp16s>::DataPattern::GRADIENT); }

TEST_F(CopyTest, Copy_16s_C4R_Random) { testCopy<Npp16s, 4>(24, 24, CopyTestHelper<Npp16s>::DataPattern::RANDOM); }

TEST_F(CopyTest, Copy_16s_C1R_WithContext) {
  testCopy<Npp16s, 1>(32, 32, CopyTestHelper<Npp16s>::DataPattern::BOUNDARY, true);
}

// Tests for 32s (signed 32-bit)
TEST_F(CopyTest, Copy_32s_C1R_Gradient) { testCopy<Npp32s, 1>(64, 48, CopyTestHelper<Npp32s>::DataPattern::GRADIENT); }

TEST_F(CopyTest, Copy_32s_C3R_Coordinate) {
  testCopy<Npp32s, 3>(32, 24, CopyTestHelper<Npp32s>::DataPattern::COORDINATE);
}

TEST_F(CopyTest, Copy_32s_C4R_Checkerboard) {
  testCopy<Npp32s, 4>(32, 32, CopyTestHelper<Npp32s>::DataPattern::CHECKERBOARD);
}

TEST_F(CopyTest, Copy_32s_C1R_WithContext) {
  testCopy<Npp32s, 1>(48, 32, CopyTestHelper<Npp32s>::DataPattern::SINE_WAVE, true);
}

// Test for 32f_C4
TEST_F(CopyTest, Copy_32f_C4R_ColorGradient) {
  testCopy<Npp32f, 4>(32, 32, CopyTestHelper<Npp32f>::DataPattern::COLOR_GRADIENT);
}

TEST_F(CopyTest, Copy_32f_C4R_WithContext) {
  testCopy<Npp32f, 4>(48, 32, CopyTestHelper<Npp32f>::DataPattern::RANDOM, true);
}

// ============================================================================
// Planar to Packed (P3C3R/P4C4R) Tests
// ============================================================================

// P3C3R Tests
TEST_F(CopyTest, Copy_32f_P3C3R) {
  testCopyPlanarToPacked<Npp32f, 3>(32, 24, CopyTestHelper<Npp32f>::DataPattern::COLOR_GRADIENT);
}

TEST_F(CopyTest, Copy_8u_P3C3R) {
  testCopyPlanarToPacked<Npp8u, 3>(32, 24, CopyTestHelper<Npp8u>::DataPattern::COORDINATE);
}

TEST_F(CopyTest, Copy_16u_P3C3R) {
  testCopyPlanarToPacked<Npp16u, 3>(32, 32, CopyTestHelper<Npp16u>::DataPattern::SEQUENTIAL);
}

TEST_F(CopyTest, Copy_16s_P3C3R) {
  testCopyPlanarToPacked<Npp16s, 3>(24, 24, CopyTestHelper<Npp16s>::DataPattern::GRADIENT);
}

TEST_F(CopyTest, Copy_32s_P3C3R) {
  testCopyPlanarToPacked<Npp32s, 3>(32, 32, CopyTestHelper<Npp32s>::DataPattern::RANDOM);
}

// P4C4R Tests
TEST_F(CopyTest, Copy_32f_P4C4R) {
  testCopyPlanarToPacked<Npp32f, 4>(32, 24, CopyTestHelper<Npp32f>::DataPattern::COLOR_GRADIENT);
}

TEST_F(CopyTest, Copy_8u_P4C4R) {
  testCopyPlanarToPacked<Npp8u, 4>(32, 32, CopyTestHelper<Npp8u>::DataPattern::CHECKERBOARD);
}

TEST_F(CopyTest, Copy_16u_P4C4R) {
  testCopyPlanarToPacked<Npp16u, 4>(24, 24, CopyTestHelper<Npp16u>::DataPattern::BOUNDARY);
}

// ============================================================================
// Packed to Planar (C3P3R/C4P4R) Tests
// ============================================================================

// C3P3R Tests
TEST_F(CopyTest, Copy_32f_C3P3R) {
  testCopyPackedToPlanar<Npp32f, 3>(32, 24, CopyTestHelper<Npp32f>::DataPattern::COLOR_GRADIENT);
}

TEST_F(CopyTest, Copy_8u_C3P3R) {
  testCopyPackedToPlanar<Npp8u, 3>(32, 24, CopyTestHelper<Npp8u>::DataPattern::COORDINATE);
}

TEST_F(CopyTest, Copy_16u_C3P3R) {
  testCopyPackedToPlanar<Npp16u, 3>(32, 32, CopyTestHelper<Npp16u>::DataPattern::SEQUENTIAL);
}

TEST_F(CopyTest, Copy_16s_C3P3R) {
  testCopyPackedToPlanar<Npp16s, 3>(24, 24, CopyTestHelper<Npp16s>::DataPattern::GRADIENT);
}

TEST_F(CopyTest, Copy_32s_C3P3R) {
  testCopyPackedToPlanar<Npp32s, 3>(32, 32, CopyTestHelper<Npp32s>::DataPattern::RANDOM);
}

// C4P4R Tests
TEST_F(CopyTest, Copy_32f_C4P4R) {
  testCopyPackedToPlanar<Npp32f, 4>(32, 24, CopyTestHelper<Npp32f>::DataPattern::COLOR_GRADIENT);
}

TEST_F(CopyTest, Copy_8u_C4P4R) {
  testCopyPackedToPlanar<Npp8u, 4>(32, 32, CopyTestHelper<Npp8u>::DataPattern::CHECKERBOARD);
}

TEST_F(CopyTest, Copy_16u_C4P4R) {
  testCopyPackedToPlanar<Npp16u, 4>(24, 24, CopyTestHelper<Npp16u>::DataPattern::BOUNDARY);
}

// ============================================================================
// Round-trip Conversion Tests (Packed <-> Planar <-> Packed)
// ============================================================================

// 3-channel round-trip tests: Packed -> Planar -> Packed
TEST_F(CopyTest, RoundTrip_8u_Packed_Planar_Packed_3Channel) {
  testRoundTripPackedToPlanarToPacked<Npp8u, 3>(32, 24, CopyTestHelper<Npp8u>::DataPattern::COLOR_GRADIENT);
}

TEST_F(CopyTest, RoundTrip_16u_Packed_Planar_Packed_3Channel) {
  testRoundTripPackedToPlanarToPacked<Npp16u, 3>(32, 32, CopyTestHelper<Npp16u>::DataPattern::SEQUENTIAL);
}

TEST_F(CopyTest, RoundTrip_16s_Packed_Planar_Packed_3Channel) {
  testRoundTripPackedToPlanarToPacked<Npp16s, 3>(24, 24, CopyTestHelper<Npp16s>::DataPattern::GRADIENT);
}

TEST_F(CopyTest, RoundTrip_32s_Packed_Planar_Packed_3Channel) {
  testRoundTripPackedToPlanarToPacked<Npp32s, 3>(32, 32, CopyTestHelper<Npp32s>::DataPattern::RANDOM);
}

TEST_F(CopyTest, RoundTrip_32f_Packed_Planar_Packed_3Channel) {
  testRoundTripPackedToPlanarToPacked<Npp32f, 3>(32, 24, CopyTestHelper<Npp32f>::DataPattern::COORDINATE);
}

// 4-channel round-trip tests: Packed -> Planar -> Packed
TEST_F(CopyTest, RoundTrip_8u_Packed_Planar_Packed_4Channel) {
  testRoundTripPackedToPlanarToPacked<Npp8u, 4>(32, 32, CopyTestHelper<Npp8u>::DataPattern::CHECKERBOARD);
}

TEST_F(CopyTest, RoundTrip_16u_Packed_Planar_Packed_4Channel) {
  testRoundTripPackedToPlanarToPacked<Npp16u, 4>(24, 24, CopyTestHelper<Npp16u>::DataPattern::BOUNDARY);
}

TEST_F(CopyTest, RoundTrip_32f_Packed_Planar_Packed_4Channel) {
  testRoundTripPackedToPlanarToPacked<Npp32f, 4>(32, 24, CopyTestHelper<Npp32f>::DataPattern::COLOR_GRADIENT);
}

// 3-channel round-trip tests: Planar -> Packed -> Planar
TEST_F(CopyTest, RoundTrip_8u_Planar_Packed_Planar_3Channel) {
  testRoundTripPlanarToPackedToPlanar<Npp8u, 3>(32, 24, CopyTestHelper<Npp8u>::DataPattern::COORDINATE);
}

TEST_F(CopyTest, RoundTrip_16u_Planar_Packed_Planar_3Channel) {
  testRoundTripPlanarToPackedToPlanar<Npp16u, 3>(32, 32, CopyTestHelper<Npp16u>::DataPattern::SEQUENTIAL);
}

TEST_F(CopyTest, RoundTrip_16s_Planar_Packed_Planar_3Channel) {
  testRoundTripPlanarToPackedToPlanar<Npp16s, 3>(24, 24, CopyTestHelper<Npp16s>::DataPattern::RANDOM);
}

TEST_F(CopyTest, RoundTrip_32s_Planar_Packed_Planar_3Channel) {
  testRoundTripPlanarToPackedToPlanar<Npp32s, 3>(32, 32, CopyTestHelper<Npp32s>::DataPattern::GRADIENT);
}

TEST_F(CopyTest, RoundTrip_32f_Planar_Packed_Planar_3Channel) {
  testRoundTripPlanarToPackedToPlanar<Npp32f, 3>(32, 24, CopyTestHelper<Npp32f>::DataPattern::COLOR_GRADIENT);
}

// 4-channel round-trip tests: Planar -> Packed -> Planar
TEST_F(CopyTest, RoundTrip_8u_Planar_Packed_Planar_4Channel) {
  testRoundTripPlanarToPackedToPlanar<Npp8u, 4>(32, 32, CopyTestHelper<Npp8u>::DataPattern::CHECKERBOARD);
}

TEST_F(CopyTest, RoundTrip_16u_Planar_Packed_Planar_4Channel) {
  testRoundTripPlanarToPackedToPlanar<Npp16u, 4>(24, 24, CopyTestHelper<Npp16u>::DataPattern::BOUNDARY);
}

TEST_F(CopyTest, RoundTrip_32f_Planar_Packed_Planar_4Channel) {
  testRoundTripPlanarToPackedToPlanar<Npp32f, 4>(32, 24, CopyTestHelper<Npp32f>::DataPattern::SINE_WAVE);
}
