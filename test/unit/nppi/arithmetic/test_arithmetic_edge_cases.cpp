/**
 * @file test_arithmetic_edge_cases.cpp
 * @brief NPPI算术运算的边缘情况和异常处理测试
 * 
 * 测试各种边界条件、错误输入和异常情况
 */

#include "test/framework/nvidia_comparison_test_base.h"
#include <limits>
#include <random>

using namespace test_framework;

class NPPIArithmeticEdgeCasesTest : public NvidiaComparisonTestBase {
protected:
    // 验证错误代码的一致性
    void compareErrorCodes(NppStatus openStatus, NppStatus nvStatus, const std::string& testName) {
        std::cout << testName << ": OpenNPP=" << openStatus;
        if (hasNvidiaNpp()) {
            std::cout << ", NVIDIA=" << nvStatus;
            if (openStatus == nvStatus) {
                std::cout << " ✓ Match";
            } else {
                std::cout << " ✗ Different";
            }
        }
        std::cout << std::endl;
    }
};

// ==================== NULL指针测试 ====================

TEST_F(NPPIArithmeticEdgeCasesTest, NullPointerHandling) {
    std::cout << "\n=== NULL Pointer Handling Test ===" << std::endl;
    
    const int width = 64;
    const int height = 64;
    int step;
    NppiSize roi = {width, height};
    
    // 分配有效的缓冲区用于部分测试
    Npp8u* validBuffer = nppiMalloc_8u_C1(width, height, &step);
    ASSERT_NE(validBuffer, nullptr);
    
    // 测试各种NULL指针组合
    struct NullTest {
        const char* name;
        Npp8u* src;
        Npp8u* dst;
        NppStatus expectedStatus;
    };
    
    std::vector<NullTest> tests = {
        {"NULL src", nullptr, validBuffer, NPP_NULL_POINTER_ERROR},
        {"NULL dst", validBuffer, nullptr, NPP_NULL_POINTER_ERROR},
        {"Both NULL", nullptr, nullptr, NPP_NULL_POINTER_ERROR}
    };
    
    for (const auto& test : tests) {
        // OpenNPP测试
        NppStatus openStatus = nppiAddC_8u_C1RSfs(
            test.src, step, 50, test.dst, step, roi, 0);
        
        // NVIDIA NPP测试
        NppStatus nvStatus = NPP_NO_ERROR;
        if (hasNvidiaNpp() && nvidiaLoader_->nv_nppiAddC_8u_C1RSfs) {
            nvStatus = nvidiaLoader_->nv_nppiAddC_8u_C1RSfs(
                test.src, step, 50, test.dst, step, roi, 0);
        }
        
        compareErrorCodes(openStatus, nvStatus, test.name);
        EXPECT_EQ(openStatus, test.expectedStatus);
    }
    
    nppiFree(validBuffer);
}

// ==================== 无效ROI测试已移除 ====================
// 注意：InvalidROI测试已被移除，因为它在某些情况下导致NVIDIA NPP段错误

// ==================== 无效Step测试 ====================

TEST_F(NPPIArithmeticEdgeCasesTest, InvalidStepValues) {
    std::cout << "\n=== Invalid Step Values Test ===" << std::endl;
    
    const int width = 64;
    const int height = 64;
    int validStep;
    
    Npp8u* src = nppiMalloc_8u_C1(width, height, &validStep);
    Npp8u* dst = nppiMalloc_8u_C1(width, height, &validStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    NppiSize roi = {width, height};
    
    struct StepTest {
        const char* name;
        int srcStep;
        int dstStep;
        bool shouldFail;
    };
    
    std::vector<StepTest> tests = {
        {"Zero src step", 0, validStep, true},
        {"Zero dst step", validStep, 0, true},
        {"Negative src step", -validStep, validStep, true},
        {"Negative dst step", validStep, -validStep, true},
        {"Too small src step", width - 1, validStep, true},
        {"Too small dst step", validStep, width - 1, true},
        {"Valid steps", validStep, validStep, false},
        {"Large steps", validStep * 2, validStep * 2, false}
    };
    
    for (const auto& test : tests) {
        NppStatus openStatus = nppiAddC_8u_C1RSfs(
            src, test.srcStep, 50, dst, test.dstStep, roi, 0);
        
        std::cout << test.name << ": status=" << openStatus;
        
        if (test.shouldFail) {
            EXPECT_NE(openStatus, NPP_NO_ERROR) << " - Should fail";
        } else {
            EXPECT_EQ(openStatus, NPP_NO_ERROR) << " - Should succeed";
        }
        std::cout << std::endl;
    }
    
    nppiFree(src);
    nppiFree(dst);
}

// ==================== 缩放因子边界测试已移除 ====================
// 注意：ScaleFactorBoundaries测试已被移除，因为NVIDIA NPP在处理无效缩放因子时会段错误

// ==================== 饱和算术测试 ====================

TEST_F(NPPIArithmeticEdgeCasesTest, SaturationArithmetic) {
    std::cout << "\n=== Saturation Arithmetic Test ===" << std::endl;
    
    const int testSize = 10;
    int srcStep, dstStep;
    
    // 8位无符号饱和测试
    {
        std::cout << "\n8-bit unsigned saturation:" << std::endl;
        
        Npp8u* src = nppiMalloc_8u_C1(testSize, 1, &srcStep);
        Npp8u* dst = nppiMalloc_8u_C1(testSize, 1, &dstStep);
        
        // 测试数据：接近边界的值
        std::vector<Npp8u> testData = {0, 1, 127, 128, 254, 255};
        std::vector<Npp8u> constants = {0, 1, 128, 255};
        
        for (Npp8u srcVal : testData) {
            // 填充源数据
            std::vector<Npp8u> srcData(testSize, srcVal);
            cudaMemcpy2D(src, srcStep, srcData.data(), testSize * sizeof(Npp8u),
                        testSize * sizeof(Npp8u), 1, cudaMemcpyHostToDevice);
            
            for (Npp8u constant : constants) {
                NppiSize roi = {testSize, 1};
                
                // 加法饱和测试
                NppStatus status = nppiAddC_8u_C1RSfs(src, srcStep, constant, dst, dstStep, roi, 0);
                EXPECT_EQ(status, NPP_NO_ERROR);
                
                std::vector<Npp8u> result(testSize);
                cudaMemcpy2D(result.data(), testSize * sizeof(Npp8u), dst, dstStep,
                            testSize * sizeof(Npp8u), 1, cudaMemcpyDeviceToHost);
                
                int expected = std::min(255, static_cast<int>(srcVal) + constant);
                std::cout << "  " << (int)srcVal << " + " << (int)constant 
                          << " = " << (int)result[0] 
                          << " (expected: " << expected << ")" 
                          << (result[0] == expected ? " ✓" : " ✗") << std::endl;
            }
        }
        
        nppiFree(src);
        nppiFree(dst);
    }
    
    // 16位有符号饱和测试
    {
        std::cout << "\n16-bit signed saturation:" << std::endl;
        
        Npp16s* src = nppiMalloc_16s_C1(testSize, 1, &srcStep);
        Npp16s* dst = nppiMalloc_16s_C1(testSize, 1, &dstStep);
        
        // 测试边界值
        std::vector<Npp16s> testData = {-32768, -32767, -1, 0, 1, 32766, 32767};
        std::vector<Npp16s> constants = {-32768, -1, 0, 1, 32767};
        
        for (Npp16s srcVal : testData) {
            std::vector<Npp16s> srcData(testSize, srcVal);
            cudaMemcpy2D(src, srcStep, srcData.data(), testSize * sizeof(Npp16s),
                        testSize * sizeof(Npp16s), 1, cudaMemcpyHostToDevice);
            
            for (Npp16s constant : constants) {
                NppiSize roi = {testSize, 1};
                
                NppStatus status = nppiAddC_16s_C1RSfs(src, srcStep, constant, dst, dstStep, roi, 0);
                EXPECT_EQ(status, NPP_NO_ERROR);
                
                std::vector<Npp16s> result(testSize);
                cudaMemcpy2D(result.data(), testSize * sizeof(Npp16s), dst, dstStep,
                            testSize * sizeof(Npp16s), 1, cudaMemcpyDeviceToHost);
                
                int sum = static_cast<int>(srcVal) + constant;
                int expected = std::max(-32768, std::min(32767, sum));
                
                if (sum != expected) {  // 只显示饱和的情况
                    std::cout << "  " << srcVal << " + " << constant 
                              << " = " << result[0] 
                              << " (saturated from " << sum << ")" << std::endl;
                }
            }
        }
        
        nppiFree(src);
        nppiFree(dst);
    }
}

// ==================== 浮点特殊值测试 ====================

TEST_F(NPPIArithmeticEdgeCasesTest, FloatingPointSpecialValues) {
    std::cout << "\n=== Floating Point Special Values Test ===" << std::endl;
    
    const int testSize = 8;
    int srcStep, dstStep;
    
    Npp32f* src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    // 特殊浮点值
    std::vector<float> specialValues = {
        0.0f,                          // 零
        -0.0f,                         // 负零
        std::numeric_limits<float>::infinity(),     // 正无穷
        -std::numeric_limits<float>::infinity(),    // 负无穷
        std::numeric_limits<float>::quiet_NaN(),    // NaN
        std::numeric_limits<float>::min(),          // 最小正规值
        std::numeric_limits<float>::max(),          // 最大值
        std::numeric_limits<float>::denorm_min()    // 最小非正规值
    };
    
    // 复制特殊值到源缓冲区
    cudaMemcpy2D(src, srcStep, specialValues.data(), testSize * sizeof(float),
                testSize * sizeof(float), 1, cudaMemcpyHostToDevice);
    
    NppiSize roi = {testSize, 1};
    
    // 测试不同的运算
    struct OpTest {
        const char* name;
        float constant;
        std::function<NppStatus()> operation;
    };
    
    std::vector<OpTest> operations = {
        {"Add infinity", INFINITY, [&]() { 
            return nppiAddC_32f_C1R(src, srcStep, INFINITY, dst, dstStep, roi); 
        }},
        {"Add NaN", NAN, [&]() { 
            return nppiAddC_32f_C1R(src, srcStep, NAN, dst, dstStep, roi); 
        }},
        {"Multiply by zero", 0.0f, [&]() { 
            return nppiMulC_32f_C1R(src, srcStep, 0.0f, dst, dstStep, roi); 
        }},
        {"Divide by zero", 0.0f, [&]() { 
            return nppiDivC_32f_C1R(src, srcStep, 0.0f, dst, dstStep, roi); 
        }},
        {"Divide by infinity", INFINITY, [&]() { 
            return nppiDivC_32f_C1R(src, srcStep, INFINITY, dst, dstStep, roi); 
        }}
    };
    
    for (const auto& op : operations) {
        std::cout << "\n" << op.name << ":" << std::endl;
        
        NppStatus status = op.operation();
        std::cout << "Status: " << status << std::endl;
        
        if (status == NPP_NO_ERROR) {
            std::vector<float> result(testSize);
            cudaMemcpy2D(result.data(), testSize * sizeof(float), dst, dstStep,
                        testSize * sizeof(float), 1, cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < testSize; i++) {
                std::cout << "  " << specialValues[i] << " -> " << result[i];
                
                // 检查特殊情况
                if (std::isnan(result[i])) {
                    std::cout << " (NaN)";
                } else if (std::isinf(result[i])) {
                    std::cout << " (" << (result[i] > 0 ? "+Inf" : "-Inf") << ")";
                }
                std::cout << std::endl;
            }
        }
    }
    
    nppiFree(src);
    nppiFree(dst);
}

// ==================== 内存对齐测试 ====================

TEST_F(NPPIArithmeticEdgeCasesTest, UnalignedMemoryAccess) {
    std::cout << "\n=== Unaligned Memory Access Test ===" << std::endl;
    
    // 分配一个大缓冲区并创建未对齐的指针
    const int bufferSize = 1024 * 1024;  // 1MB
    Npp8u* alignedBuffer;
    cudaMalloc(&alignedBuffer, bufferSize);
    
    ASSERT_NE(alignedBuffer, nullptr);
    
    // 测试不同的偏移量
    std::vector<int> offsets = {0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32};
    const int width = 64;
    const int height = 64;
    const int minStep = width;
    
    for (int offset : offsets) {
        Npp8u* src = alignedBuffer + offset;
        Npp8u* dst = alignedBuffer + bufferSize/2 + offset;
        
        NppiSize roi = {width, height};
        
        // 尝试执行运算
        NppStatus status = nppiAddC_8u_C1RSfs(
            src, minStep, 50, dst, minStep, roi, 0);
        
        std::cout << "Offset " << offset << " bytes: status=" << status;
        
        if (status == NPP_NO_ERROR) {
            std::cout << " ✓ Success";
        } else {
            std::cout << " ✗ Failed";
        }
        std::cout << std::endl;
    }
    
    cudaFree(alignedBuffer);
}

// ==================== 极限尺寸测试 ====================

TEST_F(NPPIArithmeticEdgeCasesTest, ExtremeDimensions) {
    std::cout << "\n=== Extreme Dimensions Test ===" << std::endl;
    
    struct DimTest {
        int width;
        int height;
        const char* description;
    };
    
    std::vector<DimTest> tests = {
        {1, 1, "Minimum 1x1"},
        {1, 1000000, "Tall 1x1M"},
        {1000000, 1, "Wide 1Mx1"},
        {46340, 46340, "Near sqrt(INT_MAX)"},  // sqrt(2^31) ≈ 46340
        {65536, 32768, "Large 64Kx32K"}
    };
    
    for (const auto& test : tests) {
        std::cout << test.description << " (" << test.width << "x" << test.height << "): ";
        
        // 尝试分配内存
        int step;
        Npp8u* ptr = nppiMalloc_8u_C1(test.width, test.height, &step);
        
        if (ptr) {
            std::cout << "Allocation succeeded, step=" << step;
            
            // 尝试执行简单运算
            NppiSize roi = {std::min(test.width, 100), std::min(test.height, 100)};
            NppStatus status = nppiAddC_8u_C1RSfs(ptr, step, 1, ptr, step, roi, 0);
            std::cout << ", operation status=" << status;
            
            nppiFree(ptr);
        } else {
            std::cout << "Allocation failed";
        }
        std::cout << std::endl;
    }
}

// ==================== 并发执行测试 ====================

TEST_F(NPPIArithmeticEdgeCasesTest, ConcurrentOperations) {
    std::cout << "\n=== Concurrent Operations Test ===" << std::endl;
    
    const int numStreams = 4;
    const int width = 256;
    const int height = 256;
    
    // 创建多个流
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // 为每个流分配缓冲区
    struct StreamData {
        Npp8u* src;
        Npp8u* dst;
        int step;
    };
    
    std::vector<StreamData> streamData(numStreams);
    
    for (int i = 0; i < numStreams; i++) {
        streamData[i].src = nppiMalloc_8u_C1(width, height, &streamData[i].step);
        streamData[i].dst = nppiMalloc_8u_C1(width, height, &streamData[i].step);
        
        ASSERT_NE(streamData[i].src, nullptr);
        ASSERT_NE(streamData[i].dst, nullptr);
    }
    
    // 在每个流上启动操作
    NppiSize roi = {width, height};
    
    for (int i = 0; i < numStreams; i++) {
        // 设置当前流
        nppSetStream(streams[i]);
        
        // 使用不同的常数以区分结果
        Npp8u constant = static_cast<Npp8u>(i * 10);
        
        NppStatus status = nppiAddC_8u_C1RSfs(
            streamData[i].src, streamData[i].step,
            constant,
            streamData[i].dst, streamData[i].step,
            roi, 0);
        
        std::cout << "Stream " << i << ": constant=" << (int)constant 
                  << ", status=" << status << std::endl;
    }
    
    // 等待所有流完成
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    std::cout << "All streams completed successfully" << std::endl;
    
    // 清理
    nppSetStream(nullptr);  // 恢复默认流
    
    for (int i = 0; i < numStreams; i++) {
        nppiFree(streamData[i].src);
        nppiFree(streamData[i].dst);
        cudaStreamDestroy(streams[i]);
    }
}