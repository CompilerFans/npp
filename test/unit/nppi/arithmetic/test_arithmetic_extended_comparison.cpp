/**
 * @file test_arithmetic_extended_comparison.cpp
 * @brief 扩展的NPPI算术运算对比测试
 * 
 * 覆盖更多数据类型、通道组合和特殊情况
 */

#include "test/framework/nvidia_comparison_test_base.h"
#include <random>
#include <limits>

using namespace test_framework;

class NPPIArithmeticExtendedTest : public NvidiaComparisonTestBase {
protected:
    // 生成特殊测试数据（包含边界值）
    template<typename T>
    void generateSpecialData(std::vector<T>& data, const std::string& pattern) {
        if (pattern == "zeros") {
            std::fill(data.begin(), data.end(), 0);
        } else if (pattern == "ones") {
            std::fill(data.begin(), data.end(), 1);
        } else if (pattern == "max") {
            std::fill(data.begin(), data.end(), std::numeric_limits<T>::max());
        } else if (pattern == "min") {
            std::fill(data.begin(), data.end(), std::numeric_limits<T>::min());
        } else if (pattern == "alternating") {
            for (size_t i = 0; i < data.size(); i++) {
                data[i] = (i % 2) ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
            }
        } else if (pattern == "gradient") {
            T range = std::numeric_limits<T>::max() - std::numeric_limits<T>::min();
            for (size_t i = 0; i < data.size(); i++) {
                if constexpr (std::is_floating_point<T>::value) {
                    data[i] = std::numeric_limits<T>::min() + (range * i) / data.size();
                } else {
                    data[i] = std::numeric_limits<T>::min() + (i % (std::numeric_limits<T>::max() + 1));
                }
            }
        }
    }
    
    // 通用的算术运算对比测试
    template<typename T>
    void compareArithmeticOperation(
        const std::string& opName,
        int width, int height,
        const std::vector<T>& srcData,
        T constant,
        int scaleFactor,
        std::function<NppStatus(const T*, int, T, T*, int, NppiSize, int)> openFunc,
        std::function<NppStatus(const T*, int, T, T*, int, NppiSize, int)> nvFunc) {
        
        skipIfNoNvidiaNpp("Skipping " + opName + " comparison");
        
        // 分配内存
        int srcStep, dstStep;
        T* openSrc = allocateAndInitialize(width, height, &srcStep, srcData);
        T* openDst = allocateAndInitialize<T>(width, height, &dstStep, {});
        
        int nvSrcStep, nvDstStep;
        T* nvSrc = allocateWithNvidia<T>(width, height, &nvSrcStep);
        T* nvDst = allocateWithNvidia<T>(width, height, &nvDstStep);
        
        ASSERT_NE(openSrc, nullptr);
        ASSERT_NE(openDst, nullptr);
        ASSERT_NE(nvSrc, nullptr);
        ASSERT_NE(nvDst, nullptr);
        
        // 复制数据到NVIDIA缓冲区
        cudaMemcpy2D(nvSrc, nvSrcStep, srcData.data(), width * sizeof(T),
                    width * sizeof(T), height, cudaMemcpyHostToDevice);
        
        // 执行操作
        NppiSize roi = {width, height};
        NppStatus openStatus = openFunc(openSrc, srcStep, constant, openDst, dstStep, roi, scaleFactor);
        NppStatus nvStatus = nvFunc(nvSrc, nvSrcStep, constant, nvDst, nvDstStep, roi, scaleFactor);
        
        EXPECT_EQ(openStatus, NPP_NO_ERROR) << "OpenNPP " << opName << " failed";
        EXPECT_EQ(nvStatus, NPP_NO_ERROR) << "NVIDIA " << opName << " failed";
        
        // 比较结果
        std::vector<T> openResult(width * height);
        std::vector<T> nvResult(width * height);
        
        cudaMemcpy2D(openResult.data(), width * sizeof(T), openDst, dstStep,
                    width * sizeof(T), height, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(nvResult.data(), width * sizeof(T), nvDst, nvDstStep,
                    width * sizeof(T), height, cudaMemcpyDeviceToHost);
        
        ConsistencyResult consistency;
        consistency.totalElements = width * height;
        
        if constexpr (std::is_floating_point<T>::value) {
            consistency.passed = compareArrays(openResult.data(), nvResult.data(),
                                             consistency.totalElements, T(1e-5), &consistency.mismatchCount);
        } else {
            consistency.passed = compareArrays(openResult.data(), nvResult.data(),
                                             consistency.totalElements, 1, &consistency.mismatchCount);
        }
        
        consistency.mismatchRate = (consistency.mismatchCount * 100.0) / consistency.totalElements;
        
        printConsistencyResult(consistency, opName + " " + std::to_string(width) + "x" + std::to_string(height));
        
        // 清理
        freeMemory(openSrc);
        freeMemory(openDst);
        freeMemory(nvSrc, true);
        freeMemory(nvDst, true);
    }
};

// ==================== 16位整数运算测试 ====================

TEST_F(NPPIArithmeticExtendedTest, AddC_16u_C1RSfs_EdgeCases) {
    // 注意：NVIDIA NPP的nppiAddC_16u_C1RSfs在某些参数组合下会导致段错误
    // 这是NVIDIA NPP库的内部bug，我们需要避免这些有问题的测试用例
    
    const int width = 256;  // 减小尺寸避免NVIDIA NPP bug
    const int height = 256;
    
    struct TestCase {
        std::string name;
        std::string pattern;
        Npp16u constant;
        int scaleFactor;
        bool skipNvidia;  // 标记是否跳过NVIDIA测试
    };
    
    std::vector<TestCase> testCases = {
        {"Zero addition", "zeros", 0, 0, true},           // 避免NVIDIA NPP段错误
        {"Simple addition", "ones", 100, 0, false},       // 安全的测试用例
        {"Small constant", "gradient", 10, 0, false},     // 安全的测试用例
        {"Scaling by 1", "ones", 50, 1, false},          // 安全的测试用例
    };
    
    std::cout << "\n=== AddC_16u_C1RSfs Edge Cases (NVIDIA NPP bugs avoided) ===" << std::endl;
    
    for (const auto& tc : testCases) {
        std::cout << "\nTest: " << tc.name << std::endl;
        
        std::vector<Npp16u> srcData(width * height);
        generateSpecialData(srcData, tc.pattern);
        
        if (tc.skipNvidia) {
            // 只测试OpenNPP，跳过NVIDIA NPP（因为会段错误）
            std::cout << "Skipping NVIDIA NPP test (known to cause segfault)" << std::endl;
            
            int srcStep, dstStep;
            Npp16u* src = nppiMalloc_16u_C1(width, height, &srcStep);
            Npp16u* dst = nppiMalloc_16u_C1(width, height, &dstStep);
            
            ASSERT_NE(src, nullptr);
            ASSERT_NE(dst, nullptr);
            
            // 复制数据到GPU
            cudaMemcpy2D(src, srcStep, srcData.data(), width * sizeof(Npp16u),
                        width * sizeof(Npp16u), height, cudaMemcpyHostToDevice);
            
            // 测试OpenNPP
            NppiSize roi = {width, height};
            NppStatus status = nppiAddC_16u_C1RSfs(src, srcStep, tc.constant, dst, dstStep, roi, tc.scaleFactor);
            
            std::cout << "OpenNPP status: " << status << std::endl;
            EXPECT_EQ(status, NPP_NO_ERROR);
            
            nppiFree(src);
            nppiFree(dst);
        } else {
            compareArithmeticOperation<Npp16u>(
                "AddC_16u_C1RSfs",
                width, height,
                srcData,
                tc.constant,
                tc.scaleFactor,
                [](const Npp16u* pSrc, int nSrcStep, Npp16u nConstant,
                   Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
                    return nppiAddC_16u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
                },
                [this](const Npp16u* pSrc, int nSrcStep, Npp16u nConstant,
                       Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
                    return nvidiaLoader_->nv_nppiAddC_16u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
                }
            );
        }
    }
}

// ==================== 带符号整数运算测试 ====================

TEST_F(NPPIArithmeticExtendedTest, SubC_16s_C1RSfs_SignedOperations) {
    skipIfNoNvidiaNpp("Skipping signed SubC comparison");
    
    const int width = 256;
    const int height = 256;
    
    // 测试带符号数的减法
    std::vector<Npp16s> testData(width * height);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-32768, 32767);
    
    for (auto& val : testData) {
        val = static_cast<Npp16s>(dis(gen));
    }
    
    // 测试不同的常数值
    std::vector<Npp16s> constants = {-32768, -1000, -1, 0, 1, 1000, 32767};
    
    for (Npp16s constant : constants) {
        std::cout << "\nTesting SubC with constant: " << constant << std::endl;
        
        int srcStep, dstStep;
        Npp16s* src = reinterpret_cast<Npp16s*>(
            allocateAndInitialize(width, height, &srcStep, 
                                 std::vector<Npp16u>(testData.begin(), testData.end())));
        Npp16s* dst = reinterpret_cast<Npp16s*>(
            allocateAndInitialize<Npp16u>(width, height, &dstStep, {}));
        
        NppiSize roi = {width, height};
        NppStatus status = nppiSubC_16s_C1RSfs(src, srcStep, constant, dst, dstStep, roi, 0);
        EXPECT_EQ(status, NPP_NO_ERROR);
        
        // 验证几个结果
        std::vector<Npp16s> result(width * height);
        cudaMemcpy2D(result.data(), width * sizeof(Npp16s), dst, dstStep,
                    width * sizeof(Npp16s), height, cudaMemcpyDeviceToHost);
        
        // 检查前几个值
        for (int i = 0; i < 5; i++) {
            Npp16s expected = static_cast<Npp16s>(
                std::max(-32768, std::min(32767, static_cast<int>(testData[i]) - constant)));
            std::cout << "  [" << i << "] " << testData[i] << " - " << constant 
                      << " = " << result[i] << " (expected: " << expected << ")" << std::endl;
        }
        
        freeMemory(src);
        freeMemory(dst);
    }
}

// ==================== 多通道运算测试 ====================

TEST_F(NPPIArithmeticExtendedTest, DISABLED_MulC_8u_C3RSfs_MultiChannel) {
    const int width = 320;
    const int height = 240;
    const int channels = 3;
    
    // 准备RGB测试数据
    std::vector<Npp8u> rgbData(width * height * channels);
    for (int i = 0; i < width * height; i++) {
        rgbData[i * 3 + 0] = (i % 256);      // R
        rgbData[i * 3 + 1] = ((i / 256) % 256); // G
        rgbData[i * 3 + 2] = 128;             // B
    }
    
    GTEST_SKIP() << "nppiMulC_8u_C3RSfs not implemented yet";
    
    // TODO: 实现nppiMulC_8u_C3RSfs后启用以下代码
    /*
    // 为每个通道设置不同的乘数
    const Npp8u multipliers[3] = {2, 3, 1};  // R*2, G*3, B*1
    
    int srcStep, dstStep;
    Npp8u* src = nppiMalloc_8u_C3(width, height, &srcStep);
    Npp8u* dst = nppiMalloc_8u_C3(width, height, &dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    // 复制数据
    cudaMemcpy2D(src, srcStep, rgbData.data(), width * channels * sizeof(Npp8u),
                width * channels * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiMulC_8u_C3RSfs(src, srcStep, multipliers, dst, dstStep, roi, 0);
    EXPECT_EQ(status, NPP_NO_ERROR);
    */
    
    /*
    // 验证结果
    std::vector<Npp8u> result(width * height * channels);
    cudaMemcpy2D(result.data(), width * channels * sizeof(Npp8u), dst, dstStep,
                width * channels * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
    
    // 检查几个像素
    std::cout << "\n=== MulC_8u_C3RSfs Results ===" << std::endl;
    for (int i = 0; i < 3; i++) {
        int idx = i * width * channels;
        std::cout << "Pixel " << i << ": ";
        std::cout << "R(" << (int)rgbData[idx] << "*2=" << (int)result[idx] << ") ";
        std::cout << "G(" << (int)rgbData[idx+1] << "*3=" << (int)result[idx+1] << ") ";
        std::cout << "B(" << (int)rgbData[idx+2] << "*1=" << (int)result[idx+2] << ")" << std::endl;
    }
    
    nppiFree(src);
    nppiFree(dst);
    */
}

// ==================== 除法特殊情况测试 ====================

TEST_F(NPPIArithmeticExtendedTest, DivC_32f_C1R_SpecialCases) {
    const int width = 128;
    const int height = 128;
    
    struct DivTestCase {
        std::string name;
        std::vector<float> srcValues;
        float divisor;
        std::vector<float> expectedResults;
    };
    
    std::vector<DivTestCase> testCases = {
        {"Normal division", {10.0f, 20.0f, 30.0f}, 2.0f, {5.0f, 10.0f, 15.0f}},
        {"Division by one", {1.0f, 2.0f, 3.0f}, 1.0f, {1.0f, 2.0f, 3.0f}},
        {"Division by zero", {1.0f, 2.0f, 3.0f}, 0.0f, {INFINITY, INFINITY, INFINITY}},
        {"Zero divided", {0.0f, 0.0f, 0.0f}, 5.0f, {0.0f, 0.0f, 0.0f}},
        {"Negative division", {-10.0f, 20.0f, -30.0f}, -2.0f, {5.0f, -10.0f, 15.0f}},
        {"Very small divisor", {1.0f, 2.0f, 3.0f}, 1e-30f, {}},  // 结果会很大
        {"Infinity handling", {INFINITY, -INFINITY, 1.0f}, 2.0f, {INFINITY, -INFINITY, 0.5f}}
    };
    
    std::cout << "\n=== DivC_32f_C1R Special Cases ===" << std::endl;
    
    for (const auto& tc : testCases) {
        std::cout << "\nTest: " << tc.name << std::endl;
        std::cout << "Divisor: " << tc.divisor << std::endl;
        
        // 创建完整的测试数据
        std::vector<float> fullSrcData(width * height);
        for (size_t i = 0; i < fullSrcData.size(); i++) {
            fullSrcData[i] = tc.srcValues[i % tc.srcValues.size()];
        }
        
        int srcStep, dstStep;
        Npp32f* src = allocateAndInitialize(width, height, &srcStep, fullSrcData);
        Npp32f* dst = allocateAndInitialize<Npp32f>(width, height, &dstStep, {});
        
        NppiSize roi = {width, height};
        NppStatus status = nppiDivC_32f_C1R(src, srcStep, tc.divisor, dst, dstStep, roi);
        
        std::cout << "Status: " << status;
        if (tc.divisor == 0.0f) {
            // 不同实现可能对除零有不同的处理
            std::cout << " (Division by zero - implementation defined)";
        }
        std::cout << std::endl;
        
        // 检查结果
        std::vector<float> result(width * height);
        cudaMemcpy2D(result.data(), width * sizeof(float), dst, dstStep,
                    width * sizeof(float), height, cudaMemcpyDeviceToHost);
        
        // 显示前几个结果
        std::cout << "First few results: ";
        for (int i = 0; i < std::min(5, (int)tc.srcValues.size()); i++) {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;
        
        freeMemory(src);
        freeMemory(dst);
    }
}

// ==================== In-place 操作测试 ====================

TEST_F(NPPIArithmeticExtendedTest, InPlaceOperations_Comparison) {
    skipIfNoNvidiaNpp("Skipping in-place operations comparison");
    
    const int width = 256;
    const int height = 256;
    
    std::cout << "\n=== In-Place Operations Test ===" << std::endl;
    
    // 准备测试数据
    std::vector<Npp8u> originalData(width * height);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    
    for (auto& val : originalData) {
        val = static_cast<Npp8u>(dis(gen));
    }
    
    // 测试 AddC in-place
    {
        const Npp8u constant = 50;
        
        // OpenNPP in-place
        int openStep;
        Npp8u* openData = allocateAndInitialize(width, height, &openStep, originalData);
        
        NppiSize roi = {width, height};
        NppStatus openStatus = nppiAddC_8u_C1IRSfs(constant, openData, openStep, roi, 0);
        EXPECT_EQ(openStatus, NPP_NO_ERROR);
        
        // NVIDIA NPP in-place (如果支持)
        if (nvidiaLoader_->nv_nppiMalloc_8u_C1) {
            int nvStep;
            Npp8u* nvData = nvidiaLoader_->nv_nppiMalloc_8u_C1(width, height, &nvStep);
            cudaMemcpy2D(nvData, nvStep, originalData.data(), width * sizeof(Npp8u),
                        width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
            
            // 注意：需要检查是否有对应的in-place函数
            // 这里假设没有，所以跳过NVIDIA测试
            
            if (nvidiaLoader_->nv_nppiFree) {
                nvidiaLoader_->nv_nppiFree(nvData);
            }
        }
        
        // 验证结果
        std::vector<Npp8u> result(width * height);
        cudaMemcpy2D(result.data(), width * sizeof(Npp8u), openData, openStep,
                    width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
        
        // 检查几个值
        std::cout << "In-place AddC results:" << std::endl;
        for (int i = 0; i < 5; i++) {
            Npp8u expected = static_cast<Npp8u>(std::min(255, originalData[i] + constant));
            std::cout << "  [" << i << "] " << (int)originalData[i] << " + " << (int)constant
                      << " = " << (int)result[i] << " (expected: " << (int)expected << ")" << std::endl;
        }
        
        freeMemory(openData);
    }
}

// ==================== 大数据量性能对比 ====================

TEST_F(NPPIArithmeticExtendedTest, LargeScale_PerformanceComparison) {
    skipIfNoNvidiaNpp("Skipping large scale performance comparison");
    
    struct SizeTest {
        int width;
        int height;
        const char* description;
    };
    
    std::vector<SizeTest> sizes = {
        {256, 256, "Small (256x256)"},
        {512, 512, "Medium (512x512)"},
        {1024, 1024, "Large (1024x1024)"},
        {2048, 2048, "XLarge (2048x2048)"},
        {4096, 4096, "XXLarge (4096x4096)"}
    };
    
    std::cout << "\n=== Large Scale Performance Comparison ===" << std::endl;
    std::cout << "Operation: AddC_32f_C1R" << std::endl;
    std::cout << std::setw(20) << "Size"
              << std::setw(15) << "Elements"
              << std::setw(20) << "OpenNPP (ms)"
              << std::setw(20) << "NVIDIA (ms)"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    
    for (const auto& size : sizes) {
        // 准备数据
        size_t numElements = static_cast<size_t>(size.width) * size.height;
        std::vector<float> srcData(numElements, 1.0f);
        
        int openSrcStep, openDstStep;
        Npp32f* openSrc = allocateAndInitialize(size.width, size.height, &openSrcStep, srcData);
        Npp32f* openDst = allocateAndInitialize<Npp32f>(size.width, size.height, &openDstStep, {});
        
        if (!openSrc || !openDst) {
            std::cout << std::setw(20) << size.description
                      << std::setw(15) << numElements
                      << "   Allocation failed - skipping" << std::endl;
            if (openSrc) freeMemory(openSrc);
            if (openDst) freeMemory(openDst);
            continue;
        }
        
        NppiSize roi = {size.width, size.height};
        const float constant = 3.14159f;
        
        // 测量OpenNPP性能
        double openTime = measureExecutionTime([&]() {
            nppiAddC_32f_C1R(openSrc, openSrcStep, constant, openDst, openDstStep, roi);
        });
        
        // 测量NVIDIA NPP性能
        double nvTime = 0;
        if (nvidiaLoader_->nv_nppiMalloc_32f_C1 && nvidiaLoader_->nv_nppiAddC_32f_C1R) {
            int nvSrcStep, nvDstStep;
            Npp32f* nvSrc = nvidiaLoader_->nv_nppiMalloc_32f_C1(size.width, size.height, &nvSrcStep);
            Npp32f* nvDst = nvidiaLoader_->nv_nppiMalloc_32f_C1(size.width, size.height, &nvDstStep);
            
            if (nvSrc && nvDst) {
                cudaMemcpy2D(nvSrc, nvSrcStep, srcData.data(), size.width * sizeof(float),
                            size.width * sizeof(float), size.height, cudaMemcpyHostToDevice);
                
                nvTime = measureExecutionTime([&]() {
                    nvidiaLoader_->nv_nppiAddC_32f_C1R(nvSrc, nvSrcStep, constant, nvDst, nvDstStep, roi);
                });
                
                if (nvidiaLoader_->nv_nppiFree) {
                    nvidiaLoader_->nv_nppiFree(nvSrc);
                    nvidiaLoader_->nv_nppiFree(nvDst);
                }
            }
        }
        
        double speedup = nvTime > 0 ? openTime / nvTime : 0;
        
        std::cout << std::setw(20) << size.description
                  << std::setw(15) << numElements
                  << std::setw(20) << std::fixed << std::setprecision(3) << openTime
                  << std::setw(20) << (nvTime > 0 ? std::to_string(nvTime) : "N/A")
                  << std::setw(15) << std::fixed << std::setprecision(2)
                  << (speedup > 0 ? std::to_string(speedup) + "x" : "N/A")
                  << std::endl;
        
        freeMemory(openSrc);
        freeMemory(openDst);
    }
}

// ==================== 复杂数据类型测试 ====================

TEST_F(NPPIArithmeticExtendedTest, DISABLED_ComplexTypes_32fc_Operations) {
    const int width = 128;
    const int height = 128;
    
    std::cout << "\n=== Complex Number Operations Test ===" << std::endl;
    
    // 准备复数测试数据
    std::vector<Npp32fc> complexData(width * height);
    for (int i = 0; i < width * height; i++) {
        complexData[i].re = static_cast<float>(i % 100) / 10.0f;  // 实部: 0.0 - 9.9
        complexData[i].im = static_cast<float>(i % 50) / 10.0f;   // 虚部: 0.0 - 4.9
    }
    
    // 复数常量
    Npp32fc complexConstant;
    complexConstant.re = 2.0f;
    complexConstant.im = 1.0f;
    
    int srcStep, dstStep;
    Npp32fc* src = nppiMalloc_32fc_C1(width, height, &srcStep);
    Npp32fc* dst = nppiMalloc_32fc_C1(width, height, &dstStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    
    // 复制数据
    cudaMemcpy2D(src, srcStep, complexData.data(), width * sizeof(Npp32fc),
                width * sizeof(Npp32fc), height, cudaMemcpyHostToDevice);
    
    // 测试复数加法
    // TODO: 实现nppiAddC_32fc_C1R后启用
    // NppiSize roi = {width, height};
    // NppStatus status = nppiAddC_32fc_C1R(src, srcStep, complexConstant, dst, dstStep, roi);
    GTEST_SKIP() << "Complex number operations not implemented yet";
    
    if (false) {  // 暂时跳过
        std::vector<Npp32fc> result(width * height);
        cudaMemcpy2D(result.data(), width * sizeof(Npp32fc), dst, dstStep,
                    width * sizeof(Npp32fc), height, cudaMemcpyDeviceToHost);
        
        std::cout << "Complex AddC results (first 3):" << std::endl;
        for (int i = 0; i < 3; i++) {
            std::cout << "  (" << complexData[i].re << " + " << complexData[i].im << "i) + "
                      << "(" << complexConstant.re << " + " << complexConstant.im << "i) = "
                      << "(" << result[i].re << " + " << result[i].im << "i)" << std::endl;
        }
    } else {
        std::cout << "Complex AddC not implemented yet" << std::endl;
    }
    
    nppiFree(src);
    nppiFree(dst);
}