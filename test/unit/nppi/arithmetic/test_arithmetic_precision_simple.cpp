/**
 * @file test_arithmetic_precision_simple.cpp
 * @brief 简化的NPPI算术运算精度测试
 * 
 * 用于调试和验证基本精度一致性
 */

#include "test/framework/nvidia_comparison_test_base.h"
#include <random>

using namespace test_framework;

class NPPIArithmeticPrecisionSimpleTest : public NvidiaComparisonTestBase {
};

// 简单的8u AddC精度测试
TEST_F(NPPIArithmeticPrecisionSimpleTest, AddC_8u_C1RSfs_BasicPrecision) {
    skipIfNoNvidiaNpp("Skipping basic 8u precision test");
    
    const int width = 64;
    const int height = 64;
    const size_t total_elements = width * height;
    
    // 只测试几种简单情况
    struct TestCase {
        std::string name;
        std::vector<Npp8u> src_data;
        Npp8u constant;
        int scale_factor;
    };
    
    std::vector<TestCase> test_cases = {
        {"Zeros + 50", std::vector<Npp8u>(total_elements, 0), 50, 0},
        {"Ones + 100", std::vector<Npp8u>(total_elements, 1), 100, 0},
        {"128 + 64 >> 1", std::vector<Npp8u>(total_elements, 128), 64, 1},
        {"255 + 1 (saturation)", std::vector<Npp8u>(total_elements, 255), 1, 0}
    };
    
    for (const auto& test_case : test_cases) {
        std::cout << "\nTesting: " << test_case.name << std::endl;
        
        // 分配GPU内存
        int src_step, dst_open_step, dst_nv_step;
        Npp8u* src_gpu = nppiMalloc_8u_C1(width, height, &src_step);
        Npp8u* dst_open_gpu = nppiMalloc_8u_C1(width, height, &dst_open_step);
        Npp8u* dst_nv_gpu = nppiMalloc_8u_C1(width, height, &dst_nv_step);
        
        ASSERT_NE(src_gpu, nullptr);
        ASSERT_NE(dst_open_gpu, nullptr);
        ASSERT_NE(dst_nv_gpu, nullptr);
        
        // 复制数据到GPU
        cudaMemcpy2D(src_gpu, src_step, test_case.src_data.data(), width * sizeof(Npp8u),
                    width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
        
        NppiSize roi = {width, height};
        
        // OpenNPP计算
        NppStatus open_status = nppiAddC_8u_C1RSfs(src_gpu, src_step, test_case.constant,
                                                  dst_open_gpu, dst_open_step, roi, test_case.scale_factor);
        ASSERT_EQ(open_status, NPP_NO_ERROR);
        
        // NVIDIA NPP计算
        NppStatus nv_status = nvidiaLoader_->nv_nppiAddC_8u_C1RSfs(src_gpu, src_step, test_case.constant,
                                                                  dst_nv_gpu, dst_nv_step, roi, test_case.scale_factor);
        ASSERT_EQ(nv_status, NPP_NO_ERROR);
        
        // 复制结果到CPU
        std::vector<Npp8u> open_result(total_elements);
        std::vector<Npp8u> nv_result(total_elements);
        
        cudaMemcpy2D(open_result.data(), width * sizeof(Npp8u), dst_open_gpu, dst_open_step,
                    width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(nv_result.data(), width * sizeof(Npp8u), dst_nv_gpu, dst_nv_step,
                    width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
        
        // 比较结果
        size_t mismatches = 0;
        for (size_t i = 0; i < total_elements; i++) {
            if (open_result[i] != nv_result[i]) {
                mismatches++;
                // 打印前几个不匹配的详细信息
                if (mismatches <= 5) {
                    std::cout << "  Mismatch at [" << i << "]: OpenNPP=" << (int)open_result[i]
                              << ", NVIDIA=" << (int)nv_result[i] << std::endl;
                }
            }
        }
        
        double mismatch_rate = (double)mismatches / total_elements;
        std::cout << "  Total mismatches: " << mismatches << "/" << total_elements 
                  << " (" << (mismatch_rate * 100.0) << "%)" << std::endl;
        
        if (mismatches == 0) {
            std::cout << "  ✓ PERFECT MATCH" << std::endl;
        } else {
            std::cout << "  ✗ MISMATCHES FOUND" << std::endl;
        }
        
        // 清理GPU内存
        nppiFree(src_gpu);
        nppiFree(dst_open_gpu);
        nppiFree(dst_nv_gpu);
        
        // 对于整数运算，期望完全匹配
        EXPECT_EQ(mismatches, 0) << "Expected exact match for integer arithmetic";
    }
}

// 测试随机数据的精度
TEST_F(NPPIArithmeticPrecisionSimpleTest, AddC_8u_C1RSfs_RandomData) {
    skipIfNoNvidiaNpp("Skipping random data precision test");
    
    const int width = 32;
    const int height = 32;
    const size_t total_elements = width * height;
    
    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(42); // 固定种子以便重现
    std::uniform_int_distribution<int> dis(0, 200); // 避免太接近边界
    
    std::vector<Npp8u> src_data(total_elements);
    for (auto& val : src_data) {
        val = static_cast<Npp8u>(dis(gen));
    }
    
    // 测试不同的常数和缩放因子组合
    struct TestParam {
        Npp8u constant;
        int scale_factor;
    };
    
    std::vector<TestParam> params = {
        {10, 0},
        {50, 0}, 
        {32, 1},
        {128, 2},
        {16, 3}
    };
    
    for (const auto& param : params) {
        std::cout << "\nTesting random data: constant=" << (int)param.constant 
                  << ", scale=" << param.scale_factor << std::endl;
        
        // GPU内存分配和处理 (与上面类似)
        int src_step, dst_open_step, dst_nv_step;
        Npp8u* src_gpu = nppiMalloc_8u_C1(width, height, &src_step);
        Npp8u* dst_open_gpu = nppiMalloc_8u_C1(width, height, &dst_open_step);
        Npp8u* dst_nv_gpu = nppiMalloc_8u_C1(width, height, &dst_nv_step);
        
        cudaMemcpy2D(src_gpu, src_step, src_data.data(), width * sizeof(Npp8u),
                    width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
        
        NppiSize roi = {width, height};
        
        // 执行运算
        NppStatus open_status = nppiAddC_8u_C1RSfs(src_gpu, src_step, param.constant,
                                                  dst_open_gpu, dst_open_step, roi, param.scale_factor);
        NppStatus nv_status = nvidiaLoader_->nv_nppiAddC_8u_C1RSfs(src_gpu, src_step, param.constant,
                                                                  dst_nv_gpu, dst_nv_step, roi, param.scale_factor);
        
        ASSERT_EQ(open_status, NPP_NO_ERROR);
        ASSERT_EQ(nv_status, NPP_NO_ERROR);
        
        // 比较结果
        std::vector<Npp8u> open_result(total_elements);
        std::vector<Npp8u> nv_result(total_elements);
        
        cudaMemcpy2D(open_result.data(), width * sizeof(Npp8u), dst_open_gpu, dst_open_step,
                    width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(nv_result.data(), width * sizeof(Npp8u), dst_nv_gpu, dst_nv_step,
                    width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
        
        size_t mismatches = 0;
        for (size_t i = 0; i < total_elements; i++) {
            if (open_result[i] != nv_result[i]) {
                mismatches++;
                if (mismatches <= 3) {
                    // 计算期望值用于调试
                    Npp8u src_val = src_data[i];
                    int expected = (static_cast<int>(src_val) + param.constant) >> param.scale_factor;
                    expected = std::min(255, std::max(0, expected));
                    
                    std::cout << "  Mismatch [" << i << "]: src=" << (int)src_val
                              << ", OpenNPP=" << (int)open_result[i]
                              << ", NVIDIA=" << (int)nv_result[i]
                              << ", expected=" << expected << std::endl;
                }
            }
        }
        
        std::cout << "  Mismatches: " << mismatches << "/" << total_elements << std::endl;
        
        nppiFree(src_gpu);
        nppiFree(dst_open_gpu);
        nppiFree(dst_nv_gpu);
        
        EXPECT_EQ(mismatches, 0) << "Expected exact match for integer arithmetic";
    }
}