#pragma once

#include "npp.h"
#include "npp_test_utils.h"

namespace npp_functional_test {

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
    