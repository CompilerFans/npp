
#pragma once

#include "npp.h"
#include "npp_test_utils.h"
#include "npp_version_compat.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>


namespace npp_functional_test {

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
}