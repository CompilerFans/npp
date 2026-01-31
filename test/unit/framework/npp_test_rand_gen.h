#pragma once

#include "npp.h"
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace npp_functional_test {

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
}