#pragma once

#include "npp.h"
#include "npp_version_compat.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

// Helper functions for Npp16f (half-precision float) type
inline unsigned short npp_half_as_ushort_host(__half h) {
  return static_cast<unsigned short>(static_cast<__half_raw>(h).x);
}

inline __half npp_ushort_as_half_host(unsigned short v) {
  __half_raw raw{};
  raw.x = v;
  return __half(raw);
}

inline Npp16f float_to_npp16f_host(float val) {
  Npp16f result;
  __half h = __float2half(val);
  result.fp16 = static_cast<short>(npp_half_as_ushort_host(h));
  return result;
}

inline float npp16f_to_float_host(Npp16f val) {
  __half h = npp_ushort_as_half_host(static_cast<unsigned short>(val.fp16));
  return __half2float(h);
}

// Type trait to check if T is Npp16f
template <typename T> struct is_npp16f : std::false_type {};
template <> struct is_npp16f<Npp16f> : std::true_type {};
template <typename T> constexpr bool is_npp16f_v = is_npp16f<T>::value;

#if CUDA_SDK_AT_LEAST(12, 8)
#define SIZE_TYPE size_t
#else
#define SIZE_TYPE int
#endif

namespace npp_functional_test {

// Implementation file
class NppTestBase : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize device
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
      GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(err);
    }

    // Check device availability
    int deviceCount;
    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
      GTEST_SKIP() << "CUDA device count query failed: " << cudaGetErrorString(err);
    }
    if (deviceCount <= 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }

    // Get device properties
    err = cudaGetDeviceProperties(&deviceProp_, 0);
    ASSERT_EQ(err, cudaSuccess) << "Failed to get device properties";
    device_ready_ = true;
  }

  void TearDown() override {
    if (!device_ready_) {
      return;
    }
    // Synchronize device and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess) << "error after test: " << cudaGetErrorString(err);
  }

  // Device properties access
  const cudaDeviceProp &getDeviceProperties() const { return deviceProp_; }

  // Check compute capability
  bool hasComputeCapability(int major, int minor) const {
    return deviceProp_.major > major || (deviceProp_.major == major && deviceProp_.minor >= minor);
  }

private:
  bool device_ready_ = false;
  cudaDeviceProp deviceProp_;
};

// Implementation file
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

// Implementation file
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
    } else if constexpr (std::is_same_v<T, Npp16f>) {
      // Npp16f has the same size as Npp16u, so we use 16u allocation functions
      if (channels == 1) {
        ptr_ = reinterpret_cast<T *>(nppiMalloc_16u_C1(width, height, &step_));
      } else if (channels == 3) {
        ptr_ = reinterpret_cast<T *>(nppiMalloc_16u_C3(width, height, &step_));
      } else if (channels == 4) {
        ptr_ = reinterpret_cast<T *>(nppiMalloc_16u_C4(width, height, &step_));
      } else {
        throw std::runtime_error("Unsupported channel count for Npp16f");
      }
    } else if constexpr (std::is_same_v<T, Npp32s>) {
      if (channels == 1) {
        ptr_ = nppiMalloc_32s_C1(width, height, &step_);
      } else if (channels == 3) {
        ptr_ = nppiMalloc_32s_C3(width, height, &step_);
      } else if (channels == 4) {
        ptr_ = nppiMalloc_32s_C4(width, height, &step_);
      } else {
        throw std::runtime_error("Unsupported channel count for Npp32s");
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
      if (channels == 1) {
        ptr_ = nppiMalloc_32fc_C1(width, height, &step_);
      } else if (channels == 3) {
        ptr_ = nppiMalloc_32fc_C3(width, height, &step_);
      } else if (channels == 4) {
        ptr_ = nppiMalloc_32fc_C4(width, height, &step_);
      } else {
        throw std::runtime_error("Unsupported channel count for Npp32fc");
      }
    } else if constexpr (std::is_same_v<T, Npp16sc>) {
      // Npp16sc is complex 16-bit signed, use 32s allocation (same size: 4 bytes)
      if (channels == 1) {
        ptr_ = reinterpret_cast<T *>(nppiMalloc_32s_C1(width, height, &step_));
      } else if (channels == 3) {
        ptr_ = reinterpret_cast<T *>(nppiMalloc_32s_C3(width, height, &step_));
      } else if (channels == 4) {
        ptr_ = reinterpret_cast<T *>(nppiMalloc_32s_C4(width, height, &step_));
      } else {
        throw std::runtime_error("Unsupported channel count for Npp16sc");
      }
    } else if constexpr (std::is_same_v<T, Npp32sc>) {
      // Npp32sc is complex 32-bit signed (8 bytes), use 32fc allocation (same size)
      if (channels == 1) {
        ptr_ = reinterpret_cast<T *>(nppiMalloc_32fc_C1(width, height, &step_));
      } else if (channels == 3) {
        ptr_ = reinterpret_cast<T *>(nppiMalloc_32fc_C3(width, height, &step_));
      } else if (channels == 4) {
        ptr_ = reinterpret_cast<T *>(nppiMalloc_32fc_C4(width, height, &step_));
      } else {
        throw std::runtime_error("Unsupported channel count for Npp32sc");
      }
    } else {
      static_assert(sizeof(T) == 0, "Unsupported NPP data type");
    }

    if (!ptr_) {
      throw std::runtime_error("Failed to allocate NPP image memory");
    }

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
  size_t sizeInBytes() const { return step_ * height_; }

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

// Implementation file
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
    } else if constexpr (is_npp16f_v<T>) {
      float minF = npp16f_to_float_host(minVal);
      float maxF = npp16f_to_float_host(maxVal);
      std::uniform_real_distribution<float> dis(minF, maxF);
      for (auto &val : data) {
        val = float_to_npp16f_host(dis(gen));
      }
    } else {
      std::uniform_int_distribution<int> dis(minVal, maxVal);
      for (auto &val : data) {
        val = static_cast<T>(dis(gen));
      }
    }
  }

  // Generate random Npp16f data from float range
  static void generateRandom16f(std::vector<Npp16f> &data, float minVal, float maxVal,
                                unsigned seed = std::random_device{}()) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(minVal, maxVal);
    for (auto &val : data) {
      val = float_to_npp16f_host(dis(gen));
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

// Implementation file
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
    } else if constexpr (is_npp16f_v<T>) {
      float tolF = npp16f_to_float_host(tolerance);
      for (size_t i = 0; i < a.size(); i++) {
        float af = npp16f_to_float_host(a[i]);
        float bf = npp16f_to_float_host(b[i]);
        if (std::abs(af - bf) > tolF) {
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

  // Overload for Npp16f with float tolerance
  static bool arraysEqual16f(const std::vector<Npp16f> &a, const std::vector<Npp16f> &b, float tolerance = 1e-3f) {
    if (a.size() != b.size())
      return false;
    for (size_t i = 0; i < a.size(); i++) {
      float af = npp16f_to_float_host(a[i]);
      float bf = npp16f_to_float_host(b[i]);
      if (std::abs(af - bf) > tolerance) {
        return false;
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
      } else if constexpr (is_npp16f_v<T>) {
        float tolF = npp16f_to_float_host(tolerance);
        float af = npp16f_to_float_host(a[i]);
        float bf = npp16f_to_float_host(b[i]);
        match = (std::abs(af - bf) <= tolF);
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
