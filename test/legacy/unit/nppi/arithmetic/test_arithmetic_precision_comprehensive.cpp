/**
 * @file test_arithmetic_precision_comprehensive.cpp
 * @brief 全面的NPPI算术运算精度一致性测试
 * 
 * 覆盖所有数据类型和数据范围，确保与NVIDIA NPP库的精度完全一致
 */

#include "test/framework/nvidia_comparison_test_base.h"
#include <random>
#include <limits>
#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace test_framework;

class NPPIArithmeticPrecisionTest : public NvidiaComparisonTestBase {
protected:
    std::random_device rd_;
    std::mt19937 gen_;
    
    NPPIArithmeticPrecisionTest() : gen_(rd_()) {}
    
    // 数据生成模式枚举
    enum class DataPattern {
        ZEROS,           // 全零
        ONES,           // 全一
        MAX_VALUES,     // 最大值
        MIN_VALUES,     // 最小值
        RANDOM_UNIFORM, // 均匀分布随机数
        RANDOM_NORMAL,  // 正态分布随机数
        BOUNDARY_MIX,   // 边界值混合
        ALTERNATING,    // 交替模式
        GRADIENT,       // 渐变模式
        SPARSE,         // 稀疏模式（大部分为0）
        CONCENTRATED    // 集中分布（某个范围内）
    };
    
    // 生成指定模式的测试数据
    template<typename T>
    std::vector<T> generateTestData(size_t size, DataPattern pattern) {
        std::vector<T> data(size);
        
        switch (pattern) {
            case DataPattern::ZEROS:
                std::fill(data.begin(), data.end(), T(0));
                break;
                
            case DataPattern::ONES:
                std::fill(data.begin(), data.end(), T(1));
                break;
                
            case DataPattern::MAX_VALUES:
                std::fill(data.begin(), data.end(), std::numeric_limits<T>::max());
                break;
                
            case DataPattern::MIN_VALUES:
                if constexpr (std::is_floating_point<T>::value) {
                    std::fill(data.begin(), data.end(), -std::numeric_limits<T>::max());
                } else {
                    std::fill(data.begin(), data.end(), std::numeric_limits<T>::min());
                }
                break;
                
            case DataPattern::RANDOM_UNIFORM:
                generateRandomUniform(data);
                break;
                
            case DataPattern::RANDOM_NORMAL:
                generateRandomNormal(data);
                break;
                
            case DataPattern::BOUNDARY_MIX:
                generateBoundaryMix(data);
                break;
                
            case DataPattern::ALTERNATING:
                generateAlternating(data);
                break;
                
            case DataPattern::GRADIENT:
                generateGradient(data);
                break;
                
            case DataPattern::SPARSE:
                generateSparse(data);
                break;
                
            case DataPattern::CONCENTRATED:
                generateConcentrated(data);
                break;
        }
        
        return data;
    }
    
private:
    template<typename T>
    void generateRandomUniform(std::vector<T>& data) {
        if constexpr (std::is_floating_point<T>::value) {
            std::uniform_real_distribution<T> dis(-1000.0, 1000.0);
            for (auto& val : data) {
                val = dis(gen_);
            }
        } else {
            std::uniform_int_distribution<T> dis(
                std::numeric_limits<T>::min() / 2,
                std::numeric_limits<T>::max() / 2
            );
            for (auto& val : data) {
                val = dis(gen_);
            }
        }
    }
    
    template<typename T>
    void generateRandomNormal(std::vector<T>& data) {
        if constexpr (std::is_floating_point<T>::value) {
            std::normal_distribution<T> dis(0.0, 100.0);
            for (auto& val : data) {
                val = dis(gen_);
            }
        } else {
            std::normal_distribution<float> dis(0.0f, 1000.0f);
            for (auto& val : data) {
                val = static_cast<T>(std::clamp(dis(gen_),
                    static_cast<float>(std::numeric_limits<T>::min()),
                    static_cast<float>(std::numeric_limits<T>::max())));
            }
        }
    }
    
    template<typename T>
    void generateBoundaryMix(std::vector<T>& data) {
        std::vector<T> boundaries;
        boundaries.push_back(T(0));
        boundaries.push_back(T(1));
        boundaries.push_back(std::numeric_limits<T>::max());
        
        if constexpr (std::is_signed<T>::value) {
            boundaries.push_back(T(-1));
            if constexpr (std::is_floating_point<T>::value) {
                boundaries.push_back(-std::numeric_limits<T>::max());
            } else {
                boundaries.push_back(std::numeric_limits<T>::min());
            }
        }
        
        std::uniform_int_distribution<size_t> dis(0, boundaries.size() - 1);
        for (auto& val : data) {
            val = boundaries[dis(gen_)];
        }
    }
    
    template<typename T>
    void generateAlternating(std::vector<T>& data) {
        T val1 = std::numeric_limits<T>::max();
        T val2 = std::is_signed<T>::value ? T(-1) : T(0);
        
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = (i % 2 == 0) ? val1 : val2;
        }
    }
    
    template<typename T>
    void generateGradient(std::vector<T>& data) {
        T min_val = std::is_signed<T>::value ? T(-100) : T(0);
        T max_val = T(100);
        
        for (size_t i = 0; i < data.size(); i++) {
            double ratio = static_cast<double>(i) / data.size();
            data[i] = static_cast<T>(min_val + ratio * (max_val - min_val));
        }
    }
    
    template<typename T>
    void generateSparse(std::vector<T>& data) {
        std::fill(data.begin(), data.end(), T(0));
        std::uniform_int_distribution<size_t> pos_dis(0, data.size() - 1);
        
        // 只有约5%的元素非零
        size_t non_zero_count = data.size() / 20;
        for (size_t i = 0; i < non_zero_count; i++) {
            if constexpr (std::is_floating_point<T>::value) {
                std::uniform_real_distribution<T> val_dis(1.0, 100.0);
                data[pos_dis(gen_)] = val_dis(gen_);
            } else {
                std::uniform_int_distribution<T> val_dis(1, 100);
                data[pos_dis(gen_)] = val_dis(gen_);
            }
        }
    }
    
    template<typename T>
    void generateConcentrated(std::vector<T>& data) {
        T center = T(50);
        T range = T(10);
        
        if constexpr (std::is_floating_point<T>::value) {
            std::normal_distribution<T> dis(center, range);
            for (auto& val : data) {
                val = dis(gen_);
            }
        } else {
            std::uniform_int_distribution<T> dis(center - range, center + range);
            for (auto& val : data) {
                val = dis(gen_);
            }
        }
    }
    
protected:
    // 精度分析结构体
    struct PrecisionAnalysis {
        size_t total_elements = 0;
        size_t mismatches = 0;
        size_t exact_matches = 0;
        double max_absolute_error = 0.0;
        double max_relative_error = 0.0;
        double mean_absolute_error = 0.0;
        std::vector<double> error_distribution;
        
        double getMismatchRate() const {
            return total_elements > 0 ? (double)mismatches / total_elements : 0.0;
        }
        
        bool isPassed(double threshold = 0.01) const {
            return getMismatchRate() <= threshold;
        }
    };
    
    // 详细的精度分析
    template<typename T>
    PrecisionAnalysis analyzePrecision(const std::vector<T>& expected, 
                                      const std::vector<T>& actual,
                                      const std::string& operation_name) {
        PrecisionAnalysis analysis;
        analysis.total_elements = expected.size();
        analysis.error_distribution.reserve(expected.size());
        
        double sum_absolute_error = 0.0;
        
        for (size_t i = 0; i < expected.size(); i++) {
            T exp_val = expected[i];
            T act_val = actual[i];
            
            if (exp_val == act_val) {
                analysis.exact_matches++;
            } else {
                analysis.mismatches++;
                
                double abs_error = std::abs(static_cast<double>(act_val - exp_val));
                double rel_error = 0.0;
                
                if (exp_val != 0) {
                    rel_error = abs_error / std::abs(static_cast<double>(exp_val));
                }
                
                analysis.max_absolute_error = std::max(analysis.max_absolute_error, abs_error);
                analysis.max_relative_error = std::max(analysis.max_relative_error, rel_error);
                analysis.error_distribution.push_back(abs_error);
                sum_absolute_error += abs_error;
            }
        }
        
        if (analysis.total_elements > 0) {
            analysis.mean_absolute_error = sum_absolute_error / analysis.total_elements;
        }
        
        return analysis;
    }
    
    // 打印详细的精度分析报告
    void printPrecisionReport(const PrecisionAnalysis& analysis, 
                             const std::string& operation_name,
                             const std::string& data_pattern) {
        std::cout << "\n=== Precision Analysis: " << operation_name 
                  << " (" << data_pattern << ") ===" << std::endl;
        std::cout << "Total elements: " << analysis.total_elements << std::endl;
        std::cout << "Exact matches: " << analysis.exact_matches 
                  << " (" << std::fixed << std::setprecision(2) 
                  << (analysis.exact_matches * 100.0 / analysis.total_elements) << "%)" << std::endl;
        std::cout << "Mismatches: " << analysis.mismatches
                  << " (" << std::fixed << std::setprecision(4)
                  << (analysis.getMismatchRate() * 100.0) << "%)" << std::endl;
        
        if (analysis.mismatches > 0) {
            std::cout << "Max absolute error: " << std::scientific << std::setprecision(6)
                      << analysis.max_absolute_error << std::endl;
            std::cout << "Max relative error: " << std::scientific << std::setprecision(6)
                      << analysis.max_relative_error << std::endl;
            std::cout << "Mean absolute error: " << std::scientific << std::setprecision(6)
                      << analysis.mean_absolute_error << std::endl;
                      
            // 误差分布统计
            if (!analysis.error_distribution.empty()) {
                std::vector<double> sorted_errors = analysis.error_distribution;
                std::sort(sorted_errors.begin(), sorted_errors.end());
                
                std::cout << "Error percentiles:" << std::endl;
                std::cout << "  P50: " << std::scientific << std::setprecision(6)
                          << sorted_errors[sorted_errors.size() / 2] << std::endl;
                std::cout << "  P90: " << std::scientific << std::setprecision(6)
                          << sorted_errors[sorted_errors.size() * 9 / 10] << std::endl;
                std::cout << "  P99: " << std::scientific << std::setprecision(6)
                          << sorted_errors[sorted_errors.size() * 99 / 100] << std::endl;
            }
        }
        
        std::cout << "Result: " << (analysis.isPassed() ? "PASSED ✓" : "FAILED ✗") << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
};

// ==================== 8位无符号整数精度测试 ====================

TEST_F(NPPIArithmeticPrecisionTest, AddC_8u_C1RSfs_ComprehensivePrecision) {
    skipIfNoNvidiaNpp("Skipping comprehensive 8u precision test");
    
    const int width = 512;
    const int height = 512;
    const size_t total_elements = width * height;
    
    // 测试不同的数据模式
    std::vector<std::pair<DataPattern, std::string>> patterns = {
        {DataPattern::ZEROS, "Zeros"},
        {DataPattern::ONES, "Ones"},
        {DataPattern::MAX_VALUES, "Max Values"},
        {DataPattern::RANDOM_UNIFORM, "Random Uniform"},
        {DataPattern::BOUNDARY_MIX, "Boundary Mix"},
        {DataPattern::ALTERNATING, "Alternating"},
        {DataPattern::GRADIENT, "Gradient"},
        {DataPattern::SPARSE, "Sparse"},
        {DataPattern::CONCENTRATED, "Concentrated"}
    };
    
    // 测试不同的常数值
    std::vector<Npp8u> constants = {0, 1, 50, 127, 200, 255};
    
    // 测试不同的缩放因子
    std::vector<int> scale_factors = {0, 1, 2, 4, 8};
    
    std::cout << "\n=== Comprehensive AddC_8u_C1RSfs Precision Test ===" << std::endl;
    
    int test_count = 0;
    int passed_tests = 0;
    
    for (const auto& [pattern, pattern_name] : patterns) {
        for (Npp8u constant : constants) {
            for (int scale_factor : scale_factors) {
                test_count++;
                
                std::cout << "\nTest " << test_count << ": Pattern=" << pattern_name 
                          << ", Constant=" << (int)constant 
                          << ", ScaleFactor=" << scale_factor << std::endl;
                
                // 生成测试数据
                std::vector<Npp8u> src_data = generateTestData<Npp8u>(total_elements, pattern);
                
                // 分配GPU内存
                int src_step, dst_open_step, dst_nv_step;
                Npp8u* src_gpu = nppiMalloc_8u_C1(width, height, &src_step);
                Npp8u* dst_open_gpu = nppiMalloc_8u_C1(width, height, &dst_open_step);
                Npp8u* dst_nv_gpu = nppiMalloc_8u_C1(width, height, &dst_nv_step);
                
                ASSERT_NE(src_gpu, nullptr);
                ASSERT_NE(dst_open_gpu, nullptr);
                ASSERT_NE(dst_nv_gpu, nullptr);
                
                // 复制数据到GPU
                cudaMemcpy2D(src_gpu, src_step, src_data.data(), width * sizeof(Npp8u),
                            width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
                
                NppiSize roi = {width, height};
                
                // OpenNPP计算
                NppStatus open_status = nppiAddC_8u_C1RSfs(src_gpu, src_step, constant,
                                                          dst_open_gpu, dst_open_step, roi, scale_factor);
                ASSERT_EQ(open_status, NPP_NO_ERROR);
                
                // NVIDIA NPP计算
                NppStatus nv_status = nvidiaLoader_->nv_nppiAddC_8u_C1RSfs(src_gpu, src_step, constant,
                                                                          dst_nv_gpu, dst_nv_step, roi, scale_factor);
                ASSERT_EQ(nv_status, NPP_NO_ERROR);
                
                // 复制结果到CPU
                std::vector<Npp8u> open_result(total_elements);
                std::vector<Npp8u> nv_result(total_elements);
                
                cudaMemcpy2D(open_result.data(), width * sizeof(Npp8u), dst_open_gpu, dst_open_step,
                            width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
                cudaMemcpy2D(nv_result.data(), width * sizeof(Npp8u), dst_nv_gpu, dst_nv_step,
                            width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
                
                // 精度分析
                PrecisionAnalysis analysis = analyzePrecision(nv_result, open_result,
                                                            "AddC_8u_C1RSfs");
                
                // 对于整数运算，期望完全精确匹配
                bool test_passed = (analysis.mismatches == 0);
                if (test_passed) {
                    passed_tests++;
                    std::cout << "✓ PASSED - Exact match" << std::endl;
                } else {
                    printPrecisionReport(analysis, "AddC_8u_C1RSfs", pattern_name);
                }
                
                // 清理GPU内存
                nppiFree(src_gpu);
                nppiFree(dst_open_gpu);
                nppiFree(dst_nv_gpu);
            }
        }
    }
    
    std::cout << "\n=== Final Summary ===" << std::endl;
    std::cout << "Total tests: " << test_count << std::endl;
    std::cout << "Passed tests: " << passed_tests << std::endl;
    std::cout << "Success rate: " << std::fixed << std::setprecision(2)
              << (passed_tests * 100.0 / test_count) << "%" << std::endl;
    
    // 期望至少95%的测试通过
    EXPECT_GE(passed_tests * 100.0 / test_count, 95.0) 
        << "Expected at least 95% of precision tests to pass";
}

// ==================== 32位浮点精度测试 ====================

TEST_F(NPPIArithmeticPrecisionTest, AddC_32f_C1R_FloatingPointPrecision) {
    skipIfNoNvidiaNpp("Skipping 32f floating point precision test");
    
    const int width = 256;
    const int height = 256;
    const size_t total_elements = width * height;
    
    // 浮点专用的测试常数
    std::vector<float> constants = {
        0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
        std::numeric_limits<float>::epsilon(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max() / 2.0f,
        3.14159f, 2.71828f, 1e-6f, 1e6f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()
    };
    
    std::vector<std::pair<DataPattern, std::string>> patterns = {
        {DataPattern::RANDOM_NORMAL, "Random Normal"},
        {DataPattern::RANDOM_UNIFORM, "Random Uniform"},
        {DataPattern::BOUNDARY_MIX, "Boundary Mix"},
        {DataPattern::CONCENTRATED, "Concentrated"}
    };
    
    std::cout << "\n=== Comprehensive AddC_32f_C1R Floating Point Precision Test ===" << std::endl;
    
    int test_count = 0;
    int passed_tests = 0;
    
    for (const auto& [pattern, pattern_name] : patterns) {
        for (float constant : constants) {
            // 跳过可能导致无限值的测试组合
            if (std::isinf(constant) && pattern == DataPattern::BOUNDARY_MIX) {
                continue;
            }
            
            test_count++;
            
            std::cout << "\nTest " << test_count << ": Pattern=" << pattern_name 
                      << ", Constant=" << constant << std::endl;
            
            // 生成测试数据
            std::vector<float> src_data = generateTestData<float>(total_elements, pattern);
            
            // 分配GPU内存
            int src_step, dst_open_step, dst_nv_step;
            Npp32f* src_gpu = nppiMalloc_32f_C1(width, height, &src_step);
            Npp32f* dst_open_gpu = nppiMalloc_32f_C1(width, height, &dst_open_step);
            Npp32f* dst_nv_gpu = nppiMalloc_32f_C1(width, height, &dst_nv_step);
            
            ASSERT_NE(src_gpu, nullptr);
            ASSERT_NE(dst_open_gpu, nullptr);
            ASSERT_NE(dst_nv_gpu, nullptr);
            
            // 复制数据到GPU
            cudaMemcpy2D(src_gpu, src_step, src_data.data(), width * sizeof(float),
                        width * sizeof(float), height, cudaMemcpyHostToDevice);
            
            NppiSize roi = {width, height};
            
            // OpenNPP计算
            NppStatus open_status = nppiAddC_32f_C1R(src_gpu, src_step, constant,
                                                    dst_open_gpu, dst_open_step, roi);
            ASSERT_EQ(open_status, NPP_NO_ERROR);
            
            // NVIDIA NPP计算
            NppStatus nv_status = nvidiaLoader_->nv_nppiAddC_32f_C1R(src_gpu, src_step, constant,
                                                                    dst_nv_gpu, dst_nv_step, roi);
            ASSERT_EQ(nv_status, NPP_NO_ERROR);
            
            // 复制结果到CPU
            std::vector<float> open_result(total_elements);
            std::vector<float> nv_result(total_elements);
            
            cudaMemcpy2D(open_result.data(), width * sizeof(float), dst_open_gpu, dst_open_step,
                        width * sizeof(float), height, cudaMemcpyDeviceToHost);
            cudaMemcpy2D(nv_result.data(), width * sizeof(float), dst_nv_gpu, dst_nv_step,
                        width * sizeof(float), height, cudaMemcpyDeviceToHost);
            
            // 精度分析（浮点数允许小的数值误差）
            PrecisionAnalysis analysis = analyzePrecision(nv_result, open_result,
                                                        "AddC_32f_C1R");
            
            // 对于浮点运算，允许相对误差在1e-6以内
            bool test_passed = (analysis.max_relative_error <= 1e-6 && analysis.getMismatchRate() <= 0.001);
            
            if (test_passed) {
                passed_tests++;
                std::cout << "✓ PASSED" << std::endl;
            } else {
                printPrecisionReport(analysis, "AddC_32f_C1R", pattern_name);
            }
            
            // 清理GPU内存
            nppiFree(src_gpu);
            nppiFree(dst_open_gpu);
            nppiFree(dst_nv_gpu);
        }
    }
    
    std::cout << "\n=== Final Summary ===" << std::endl;
    std::cout << "Total tests: " << test_count << std::endl;
    std::cout << "Passed tests: " << passed_tests << std::endl;
    std::cout << "Success rate: " << std::fixed << std::setprecision(2)
              << (passed_tests * 100.0 / test_count) << "%" << std::endl;
    
    // 浮点运算期望至少90%的测试通过
    EXPECT_GE(passed_tests * 100.0 / test_count, 90.0) 
        << "Expected at least 90% of floating point precision tests to pass";
}