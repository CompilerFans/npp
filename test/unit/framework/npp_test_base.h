/**
 * @file npp_test_base.h
 * @brief NPP functional test base framework
 *
 * Provides infrastructure for pure functional unit tests, focused on API functionality verification, independent of
 * NVIDIA NPP library
 */

#pragma once

#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

namespace npp_functional_test {

/**
 * @brief NPP functional test base class
 *
 * Provides common functionality for CUDA device management, memory management, data generation, etc.
 */
class NppTestBase : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize CUDA device
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device";

    // Check device availability
    int deviceCount;
    err = cudaGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, cudaSuccess) << "Failed to get device count";
    ASSERT_GT(deviceCount, 0) << "No CUDA devices available";

    // Get device properties
    err = cudaGetDeviceProperties(&deviceProp_, 0);
    ASSERT_EQ(err, cudaSuccess) << "Failed to get device properties";
  }

  void TearDown() override {
    // Synchronize device and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "CUDA error after test: " << cudaGetErrorString(err);
  }

  // Device properties access
  const cudaDeviceProp &getDeviceProperties() const { return deviceProp_; }

  // Check compute capability
  bool hasComputeCapability(int major, int minor) const {
    return deviceProp_.major > major || (deviceProp_.major == major && deviceProp_.minor >= minor);
  }

private:
  cudaDeviceProp deviceProp_;
};

/**
 * @brief Memory management helper class template
 */
template <typename T> class DeviceMemory {
public:
  DeviceMemory() : ptr_(nullptr), size_(0) {}

  explicit DeviceMemory(size_t size) : size_(size) { allocate(size); }

  ~DeviceMemory() { free(); }

  // Disable copy
  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory &operator=(const DeviceMemory &) = delete;

  // Support move
  DeviceMemory(DeviceMemory &&other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  DeviceMemory &operator=(DeviceMemory &&other) noexcept {
    if (this != &other) {
      free();
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void allocate(size_t size) {
    free();
    cudaError_t err = cudaMalloc(&ptr_, size * sizeof(T));
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate device memory: " + std::string(cudaGetErrorString(err)));
    }
    size_ = size;
  }

  void free() {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  T *get() const { return ptr_; }
  size_t size() const { return size_; }

  // Data transfer
  void copyFromHost(const std::vector<T> &hostData) {
    ASSERT_EQ(hostData.size(), size_) << "Size mismatch in copyFromHost";
    cudaError_t err = cudaMemcpy(ptr_, hostData.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy from host";
  }

  void copyToHost(std::vector<T> &hostData) const {
    hostData.resize(size_);
    cudaError_t err = cudaMemcpy(hostData.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy to host";
  }

private:
  T *ptr_;
  size_t size_;
};

/**
 * @brief NPP image memory management helper class
 */
template <typename T> class NppImageMemory {
public:
  NppImageMemory() : ptr_(nullptr), step_(0), width_(0), height_(0), channels_(1) {}

  NppImageMemory(int width, int height, int channels = 1)
      : ptr_(nullptr), step_(0), width_(width), height_(height), channels_(channels) {
    allocate(width, height, channels);
  }

  ~NppImageMemory() { free(); }

  // Disable copy
  NppImageMemory(const NppImageMemory &) = delete;
  NppImageMemory &operator=(const NppImageMemory &) = delete;

  // Support move
  NppImageMemory(NppImageMemory &&other) noexcept
      : ptr_(other.ptr_), step_(other.step_), width_(other.width_), height_(other.height_), channels_(other.channels_) {
    other.ptr_ = nullptr;
    other.step_ = 0;
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 1;
  }

  void allocate(int width, int height, int channels = 1) {
    free();
    width_ = width;
    height_ = height;
    channels_ = channels;

    if constexpr (std::is_same_v<T, Npp8u>) {
      if (channels == 1) {
        ptr_ = nppiMalloc_8u_C1(width, height, &step_);
      } else if (channels == 2) {
        ptr_ = nppiMalloc_8u_C2(width, height, &step_);
      } else if (channels == 3) {
        ptr_ = nppiMalloc_8u_C3(width, height, &step_);
      } else if (channels == 4) {
        ptr_ = nppiMalloc_8u_C4(width, height, &step_);
      } else {
        throw std::runtime_error("Unsupported channel count for Npp8u");
      }
    } else if constexpr (std::is_same_v<T, Npp8s>) {
      static_assert(sizeof(T) == 0, "Unsupported NPP data type");
      ptr_ = reinterpret_cast<T *>(nppiMalloc_8u_C1(width, height * channels, &step_));
    } else if constexpr (std::is_same_v<T, Npp16u>) {
      if (channels == 1) {
        ptr_ = nppiMalloc_16u_C1(width, height, &step_);
      } else if (channels == 3) {
        ptr_ = nppiMalloc_16u_C3(width, height, &step_);
      } else if (channels == 4) {
        ptr_ = nppiMalloc_16u_C4(width, height, &step_);
      } else {
        throw std::runtime_error("Unsupported channel count for Npp16u");
      }
    } else if constexpr (std::is_same_v<T, Npp16s>) {
      if (channels == 1) {
        ptr_ = nppiMalloc_16s_C1(width, height, &step_);
      } else {
        throw std::runtime_error("Unsupported channel count for Npp16s");
      }
    } else if constexpr (std::is_same_v<T, Npp32f>) {
      if (channels == 1) {
        ptr_ = nppiMalloc_32f_C1(width, height, &step_);
      } else if (channels == 3) {
        ptr_ = nppiMalloc_32f_C3(width, height, &step_);
      } else if (channels == 4) {
        ptr_ = nppiMalloc_32f_C4(width, height, &step_);
      } else {
        throw std::runtime_error("Unsupported channel count for Npp32f");
      }
    } else if constexpr (std::is_same_v<T, Npp32fc>) {
      ptr_ = nppiMalloc_32fc_C1(width, height, &step_);
    } else {
      static_assert(sizeof(T) == 0, "Unsupported NPP data type");
    }

    if (!ptr_) {
      throw std::runtime_error("Failed to allocate NPP image memory");
    }
    
    // 初始化分配的内存为0，避免测试中的未定义行为
    cudaError_t err = cudaMemset(ptr_, 0, sizeInBytes());
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to initialize NPP image memory: " + std::string(cudaGetErrorString(err)));
    }
  }

  void free() {
    if (ptr_) {
      nppiFree(ptr_);
      ptr_ = nullptr;
      step_ = 0;
      width_ = 0;
      height_ = 0;
      channels_ = 1;
    }
  }

  T *get() const { return ptr_; }
  int step() const { return step_; }
  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  NppiSize size() const { return {width_, height_}; }
  size_t sizeInBytes() const { return width_ * height_ * channels_ * sizeof(T); }

  // Data transfer
  void copyFromHost(const std::vector<T> &hostData) {
    ASSERT_EQ(hostData.size(), width_ * height_ * channels_) << "Size mismatch in copyFromHost";
    cudaError_t err = cudaMemcpy2D(ptr_, step_, hostData.data(), width_ * channels_ * sizeof(T),
                                   width_ * channels_ * sizeof(T), height_, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy from host";
  }

  void copyToHost(std::vector<T> &hostData) const {
    hostData.resize(width_ * height_ * channels_);
    cudaError_t err = cudaMemcpy2D(hostData.data(), width_ * channels_ * sizeof(T), ptr_, step_,
                                   width_ * channels_ * sizeof(T), height_, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy to host";
  }

  void fill(T value) {
    std::vector<T> hostData(width_ * height_ * channels_, value);
    copyFromHost(hostData);
  }

private:
  T *ptr_;
  int step_;
  int width_;
  int height_;
  int channels_;
};

/**
 * @brief Data generation utility class
 */
class TestDataGenerator {
public:
  // Generate random data
  template <typename T>
  static void generateRandom(std::vector<T> &data, T minVal, T maxVal, unsigned seed = std::random_device{}()) {
    std::mt19937 gen(seed);

    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(minVal, maxVal);
      for (auto &val : data) {
        val = dis(gen);
      }
    } else {
      std::uniform_int_distribution<int> dis(minVal, maxVal);
      for (auto &val : data) {
        val = static_cast<T>(dis(gen));
      }
    }
  }

  // Generate sequential data
  template <typename T> static void generateSequential(std::vector<T> &data, T startVal = T{}, T step = T{1}) {
    T current = startVal;
    for (auto &val : data) {
      val = current;
      current += step;
    }
  }

  // Generate constant data
  template <typename T> static void generateConstant(std::vector<T> &data, T value) {
    std::fill(data.begin(), data.end(), value);
  }

  // Generate test pattern data (checkerboard, stripes, etc.)
  template <typename T>
  static void generateCheckerboard(std::vector<T> &data, int width, int height, T value1, T value2) {
    ASSERT_EQ(data.size(), width * height) << "Data size mismatch";

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = y * width + x;
        data[idx] = ((x + y) % 2 == 0) ? value1 : value2;
      }
    }
  }
};

/**
 * @brief Result validation utility class
 */
class ResultValidator {
public:
  // Verify array equality (integer types)
  template <typename T> static bool arraysEqual(const std::vector<T> &a, const std::vector<T> &b, T tolerance = T{}) {
    if (a.size() != b.size())
      return false;

    if constexpr (std::is_floating_point_v<T>) {
      for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > tolerance) {
          return false;
        }
      }
    } else {
      for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(static_cast<long long>(a[i]) - static_cast<long long>(b[i])) > tolerance) {
          return false;
        }
      }
    }
    return true;
  }

  // Find first mismatch position
  template <typename T>
  static std::pair<bool, size_t> findFirstMismatch(const std::vector<T> &a, const std::vector<T> &b,
                                                   T tolerance = T{}) {
    if (a.size() != b.size()) {
      return {false, 0};
    }

    for (size_t i = 0; i < a.size(); i++) {
      bool match;
      if constexpr (std::is_floating_point_v<T>) {
        match = (std::abs(a[i] - b[i]) <= tolerance);
      } else {
        match = (std::abs(static_cast<long long>(a[i]) - static_cast<long long>(b[i])) <= tolerance);
      }

      if (!match) {
        return {false, i};
      }
    }
    return {true, 0};
  }

  // Compute array statistics
  template <typename T>
  static void computeStats(const std::vector<T> &data, T &minVal, T &maxVal, double &mean, double &stddev) {
    if (data.empty()) {
      minVal = maxVal = T{};
      mean = stddev = 0.0;
      return;
    }

    minVal = *std::min_element(data.begin(), data.end());
    maxVal = *std::max_element(data.begin(), data.end());

    double sum = 0.0;
    for (const auto &val : data) {
      sum += static_cast<double>(val);
    }
    mean = sum / data.size();

    double variance = 0.0;
    for (const auto &val : data) {
      double diff = static_cast<double>(val) - mean;
      variance += diff * diff;
    }
    stddev = std::sqrt(variance / data.size());
  }
};

} // namespace npp_functional_test