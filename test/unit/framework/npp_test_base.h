#pragma once

#include "npp.h"  
#include "npp_test_memory.h"
#include "npp_test_result_validator.h"
#include "npp_test_rand_gen.h"
#include "npp_test_utils.h"
#include <gtest/gtest.h>

namespace npp_functional_test {

// Implementation file
class NppTestBase : public ::testing::Test {
protected:
  void SetUp() override {
    // Check device availability
    int deviceCount;
    auto err = cudaGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, cudaSuccess) << "Failed to get device count";
    ASSERT_GT(deviceCount, 0) << "No devices available";

    // Get device properties
    err = cudaGetDeviceProperties(&deviceProp_, 0);
    ASSERT_EQ(err, cudaSuccess) << "Failed to get device properties";
  }

  void TearDown() override {
    auto err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "error after test: " << cudaGetErrorString(err);
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

} // namespace npp_functional_test
