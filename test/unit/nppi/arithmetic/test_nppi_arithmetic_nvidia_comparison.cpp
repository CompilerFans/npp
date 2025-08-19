/**
 * @file test_nppi_arithmetic_nvidia_comparison.cpp
 * @brief NPPI算术运算与NVIDIA NPP的对比测试
 * 
 * 使用统一的测试框架进行OpenNPP与NVIDIA NPP的功能和性能对比
 */

#include "test/framework/nvidia_comparison_test_base.h"
#include <random>
#include <algorithm>

using namespace test_framework;

class NPPIArithmeticComparisonTest : public NvidiaComparisonTestBase {
protected:
    // 生成随机测试数据
    template<typename T>
    void generateRandomData(std::vector<T>& data, T minVal, T maxVal) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if constexpr (std::is_floating_point<T>::value) {
            std::uniform_real_distribution<T> dis(minVal, maxVal);
            for (auto& val : data) {
                val = dis(gen);
            }
        } else {
            std::uniform_int_distribution<int> dis(minVal, maxVal);
            for (auto& val : data) {
                val = static_cast<T>(dis(gen));
            }
        }
    }
};

// ==================== AddC 对比测试 ====================

TEST_F(NPPIArithmeticComparisonTest, AddC_8u_C1RSfs_Comparison) {
    skipIfNoNvidiaNpp("Skipping AddC comparison - NVIDIA NPP not available");
    
    const int width = 1024;
    const int height = 768;
    const Npp8u constant = 50;
    const int scaleFactor = 0;
    
    // 准备测试数据
    std::vector<Npp8u> hostSrc(width * height);
    generateRandomData(hostSrc, static_cast<Npp8u>(0), static_cast<Npp8u>(200));
    
    // OpenNPP执行
    int openSrcStep, openDstStep;
    Npp8u* openSrc = allocateAndInitialize(width, height, &openSrcStep, hostSrc);
    Npp8u* openDst = allocateAndInitialize<Npp8u>(width, height, &openDstStep, {});
    
    NppiSize roi = {width, height};
    NppStatus openStatus = nppiAddC_8u_C1RSfs(openSrc, openSrcStep, constant, 
                                              openDst, openDstStep, roi, scaleFactor);
    ASSERT_EQ(openStatus, NPP_NO_ERROR);
    
    // NVIDIA NPP执行
    int nvSrcStep, nvDstStep;
    Npp8u* nvSrc = allocateWithNvidia<Npp8u>(width, height, &nvSrcStep);
    Npp8u* nvDst = allocateWithNvidia<Npp8u>(width, height, &nvDstStep);
    
    ASSERT_NE(nvSrc, nullptr);
    ASSERT_NE(nvDst, nullptr);
    
    // 复制相同的源数据到NVIDIA缓冲区
    cudaMemcpy2D(nvSrc, nvSrcStep, hostSrc.data(), width * sizeof(Npp8u),
                width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    
    NppStatus nvStatus = nvidiaLoader_->nv_nppiAddC_8u_C1RSfs(
        nvSrc, nvSrcStep, constant, nvDst, nvDstStep, roi, scaleFactor);
    ASSERT_EQ(nvStatus, NPP_NO_ERROR);
    
    // 读回结果并比较
    std::vector<Npp8u> openResult(width * height);
    std::vector<Npp8u> nvResult(width * height);
    
    cudaMemcpy2D(openResult.data(), width * sizeof(Npp8u), openDst, openDstStep,
                width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(nvResult.data(), width * sizeof(Npp8u), nvDst, nvDstStep,
                width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
    
    // 一致性测试
    ConsistencyResult consistency;
    consistency.totalElements = width * height;
    bool match = compareArrays(openResult.data(), nvResult.data(), 
                              consistency.totalElements, 1, &consistency.mismatchCount);
    consistency.passed = match;
    consistency.mismatchRate = (consistency.mismatchCount * 100.0) / consistency.totalElements;
    
    printConsistencyResult(consistency, "AddC_8u_C1RSfs");
    EXPECT_TRUE(match) << "Results don't match between OpenNPP and NVIDIA NPP";
    
    // 性能测试
    PerformanceResult perfResult;
    perfResult.dataSize = width * height;
    
    perfResult.openNppTime = measureExecutionTime([&]() {
        nppiAddC_8u_C1RSfs(openSrc, openSrcStep, constant, 
                          openDst, openDstStep, roi, scaleFactor);
    });
    
    perfResult.nvidiaNppTime = measureExecutionTime([&]() {
        nvidiaLoader_->nv_nppiAddC_8u_C1RSfs(nvSrc, nvSrcStep, constant, 
                                             nvDst, nvDstStep, roi, scaleFactor);
    });
    
    printPerformanceComparison(perfResult, "AddC_8u_C1RSfs");
    
    // 清理
    freeMemory(openSrc);
    freeMemory(openDst);
    freeMemory(nvSrc, true);
    freeMemory(nvDst, true);
}

TEST_F(NPPIArithmeticComparisonTest, AddC_32f_C1R_Comparison) {
    skipIfNoNvidiaNpp("Skipping AddC comparison - NVIDIA NPP not available");
    
    const int width = 640;
    const int height = 480;
    const Npp32f constant = 3.14159f;
    
    // 准备测试数据
    std::vector<Npp32f> hostSrc(width * height);
    generateRandomData(hostSrc, -100.0f, 100.0f);
    
    // OpenNPP执行
    int openSrcStep, openDstStep;
    Npp32f* openSrc = allocateAndInitialize(width, height, &openSrcStep, hostSrc);
    Npp32f* openDst = allocateAndInitialize<Npp32f>(width, height, &openDstStep, {});
    
    NppiSize roi = {width, height};
    NppStatus openStatus = nppiAddC_32f_C1R(openSrc, openSrcStep, constant, 
                                            openDst, openDstStep, roi);
    ASSERT_EQ(openStatus, NPP_NO_ERROR);
    
    // NVIDIA NPP执行
    int nvSrcStep, nvDstStep;
    Npp32f* nvSrc = allocateWithNvidia<Npp32f>(width, height, &nvSrcStep);
    Npp32f* nvDst = allocateWithNvidia<Npp32f>(width, height, &nvDstStep);
    
    ASSERT_NE(nvSrc, nullptr);
    ASSERT_NE(nvDst, nullptr);
    
    cudaMemcpy2D(nvSrc, nvSrcStep, hostSrc.data(), width * sizeof(Npp32f),
                width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    
    NppStatus nvStatus = nvidiaLoader_->nv_nppiAddC_32f_C1R(
        nvSrc, nvSrcStep, constant, nvDst, nvDstStep, roi);
    ASSERT_EQ(nvStatus, NPP_NO_ERROR);
    
    // 读回结果并比较
    std::vector<Npp32f> openResult(width * height);
    std::vector<Npp32f> nvResult(width * height);
    
    cudaMemcpy2D(openResult.data(), width * sizeof(Npp32f), openDst, openDstStep,
                width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(nvResult.data(), width * sizeof(Npp32f), nvDst, nvDstStep,
                width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    
    // 一致性测试
    ConsistencyResult consistency;
    consistency.totalElements = width * height;
    bool match = compareArrays(openResult.data(), nvResult.data(), 
                              consistency.totalElements, 1e-5f, &consistency.mismatchCount);
    consistency.passed = match;
    consistency.mismatchRate = (consistency.mismatchCount * 100.0) / consistency.totalElements;
    
    printConsistencyResult(consistency, "AddC_32f_C1R");
    EXPECT_TRUE(match) << "Results don't match between OpenNPP and NVIDIA NPP";
    
    // 性能测试
    PerformanceResult perfResult;
    perfResult.dataSize = width * height;
    
    perfResult.openNppTime = measureExecutionTime([&]() {
        nppiAddC_32f_C1R(openSrc, openSrcStep, constant, openDst, openDstStep, roi);
    });
    
    perfResult.nvidiaNppTime = measureExecutionTime([&]() {
        nvidiaLoader_->nv_nppiAddC_32f_C1R(nvSrc, nvSrcStep, constant, nvDst, nvDstStep, roi);
    });
    
    printPerformanceComparison(perfResult, "AddC_32f_C1R");
    
    // 清理
    freeMemory(openSrc);
    freeMemory(openDst);
    freeMemory(nvSrc, true);
    freeMemory(nvDst, true);
}

// ==================== SubC 对比测试 ====================

TEST_F(NPPIArithmeticComparisonTest, SubC_8u_C1RSfs_Comparison) {
    skipIfNoNvidiaNpp("Skipping SubC comparison - NVIDIA NPP not available");
    
    const int width = 512;
    const int height = 512;
    const Npp8u constant = 30;
    const int scaleFactor = 0;
    
    // 准备测试数据
    std::vector<Npp8u> hostSrc(width * height);
    generateRandomData(hostSrc, static_cast<Npp8u>(50), static_cast<Npp8u>(255));  // 确保减法后不会下溢
    
    // OpenNPP执行
    int openSrcStep, openDstStep;
    Npp8u* openSrc = allocateAndInitialize(width, height, &openSrcStep, hostSrc);
    Npp8u* openDst = allocateAndInitialize<Npp8u>(width, height, &openDstStep, {});
    
    NppiSize roi = {width, height};
    NppStatus openStatus = nppiSubC_8u_C1RSfs(openSrc, openSrcStep, constant, 
                                              openDst, openDstStep, roi, scaleFactor);
    ASSERT_EQ(openStatus, NPP_NO_ERROR);
    
    // NVIDIA NPP执行
    int nvSrcStep, nvDstStep;
    Npp8u* nvSrc = allocateWithNvidia<Npp8u>(width, height, &nvSrcStep);
    Npp8u* nvDst = allocateWithNvidia<Npp8u>(width, height, &nvDstStep);
    
    ASSERT_NE(nvSrc, nullptr);
    ASSERT_NE(nvDst, nullptr);
    
    cudaMemcpy2D(nvSrc, nvSrcStep, hostSrc.data(), width * sizeof(Npp8u),
                width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    
    NppStatus nvStatus = nvidiaLoader_->nv_nppiSubC_8u_C1RSfs(
        nvSrc, nvSrcStep, constant, nvDst, nvDstStep, roi, scaleFactor);
    ASSERT_EQ(nvStatus, NPP_NO_ERROR);
    
    // 读回结果并比较
    std::vector<Npp8u> openResult(width * height);
    std::vector<Npp8u> nvResult(width * height);
    
    cudaMemcpy2D(openResult.data(), width * sizeof(Npp8u), openDst, openDstStep,
                width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(nvResult.data(), width * sizeof(Npp8u), nvDst, nvDstStep,
                width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
    
    // 一致性测试
    ConsistencyResult consistency;
    consistency.totalElements = width * height;
    bool match = compareArrays(openResult.data(), nvResult.data(), 
                              consistency.totalElements, 1, &consistency.mismatchCount);
    consistency.passed = match;
    consistency.mismatchRate = (consistency.mismatchCount * 100.0) / consistency.totalElements;
    
    printConsistencyResult(consistency, "SubC_8u_C1RSfs");
    EXPECT_TRUE(match) << "Results don't match between OpenNPP and NVIDIA NPP";
    
    // 清理
    freeMemory(openSrc);
    freeMemory(openDst);
    freeMemory(nvSrc, true);
    freeMemory(nvDst, true);
}

// ==================== MulC 对比测试 ====================

TEST_F(NPPIArithmeticComparisonTest, MulC_32f_C1R_Comparison) {
    skipIfNoNvidiaNpp("Skipping MulC comparison - NVIDIA NPP not available");
    
    const int width = 800;
    const int height = 600;
    const Npp32f constant = 2.5f;
    
    // 准备测试数据
    std::vector<Npp32f> hostSrc(width * height);
    generateRandomData(hostSrc, -10.0f, 10.0f);
    
    // OpenNPP执行
    int openSrcStep, openDstStep;
    Npp32f* openSrc = allocateAndInitialize(width, height, &openSrcStep, hostSrc);
    Npp32f* openDst = allocateAndInitialize<Npp32f>(width, height, &openDstStep, {});
    
    NppiSize roi = {width, height};
    NppStatus openStatus = nppiMulC_32f_C1R(openSrc, openSrcStep, constant, 
                                            openDst, openDstStep, roi);
    ASSERT_EQ(openStatus, NPP_NO_ERROR);
    
    // NVIDIA NPP执行
    int nvSrcStep, nvDstStep;
    Npp32f* nvSrc = allocateWithNvidia<Npp32f>(width, height, &nvSrcStep);
    Npp32f* nvDst = allocateWithNvidia<Npp32f>(width, height, &nvDstStep);
    
    ASSERT_NE(nvSrc, nullptr);
    ASSERT_NE(nvDst, nullptr);
    
    cudaMemcpy2D(nvSrc, nvSrcStep, hostSrc.data(), width * sizeof(Npp32f),
                width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    
    NppStatus nvStatus = nvidiaLoader_->nv_nppiMulC_32f_C1R(
        nvSrc, nvSrcStep, constant, nvDst, nvDstStep, roi);
    ASSERT_EQ(nvStatus, NPP_NO_ERROR);
    
    // 读回结果并比较
    std::vector<Npp32f> openResult(width * height);
    std::vector<Npp32f> nvResult(width * height);
    
    cudaMemcpy2D(openResult.data(), width * sizeof(Npp32f), openDst, openDstStep,
                width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(nvResult.data(), width * sizeof(Npp32f), nvDst, nvDstStep,
                width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    
    // 一致性测试
    ConsistencyResult consistency;
    consistency.totalElements = width * height;
    bool match = compareArrays(openResult.data(), nvResult.data(), 
                              consistency.totalElements, 1e-5f, &consistency.mismatchCount);
    consistency.passed = match;
    consistency.mismatchRate = (consistency.mismatchCount * 100.0) / consistency.totalElements;
    
    printConsistencyResult(consistency, "MulC_32f_C1R");
    EXPECT_TRUE(match) << "Results don't match between OpenNPP and NVIDIA NPP";
    
    // 清理
    freeMemory(openSrc);
    freeMemory(openDst);
    freeMemory(nvSrc, true);
    freeMemory(nvDst, true);
}

// ==================== Add (双源) 对比测试 ====================

TEST_F(NPPIArithmeticComparisonTest, Add_32f_C1R_Comparison) {
    skipIfNoNvidiaNpp("Skipping Add comparison - NVIDIA NPP not available");
    
    const int width = 1024;
    const int height = 1024;
    
    // 准备测试数据
    std::vector<Npp32f> hostSrc1(width * height);
    std::vector<Npp32f> hostSrc2(width * height);
    generateRandomData(hostSrc1, -50.0f, 50.0f);
    generateRandomData(hostSrc2, -50.0f, 50.0f);
    
    // OpenNPP执行
    int openSrc1Step, openSrc2Step, openDstStep;
    Npp32f* openSrc1 = allocateAndInitialize(width, height, &openSrc1Step, hostSrc1);
    Npp32f* openSrc2 = allocateAndInitialize(width, height, &openSrc2Step, hostSrc2);
    Npp32f* openDst = allocateAndInitialize<Npp32f>(width, height, &openDstStep, {});
    
    NppiSize roi = {width, height};
    NppStatus openStatus = nppiAdd_32f_C1R(openSrc1, openSrc1Step, openSrc2, openSrc2Step,
                                           openDst, openDstStep, roi);
    ASSERT_EQ(openStatus, NPP_NO_ERROR);
    
    // NVIDIA NPP执行
    int nvSrc1Step, nvSrc2Step, nvDstStep;
    Npp32f* nvSrc1 = allocateWithNvidia<Npp32f>(width, height, &nvSrc1Step);
    Npp32f* nvSrc2 = allocateWithNvidia<Npp32f>(width, height, &nvSrc2Step);
    Npp32f* nvDst = allocateWithNvidia<Npp32f>(width, height, &nvDstStep);
    
    ASSERT_NE(nvSrc1, nullptr);
    ASSERT_NE(nvSrc2, nullptr);
    ASSERT_NE(nvDst, nullptr);
    
    cudaMemcpy2D(nvSrc1, nvSrc1Step, hostSrc1.data(), width * sizeof(Npp32f),
                width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(nvSrc2, nvSrc2Step, hostSrc2.data(), width * sizeof(Npp32f),
                width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    
    NppStatus nvStatus = nvidiaLoader_->nv_nppiAdd_32f_C1R(
        nvSrc1, nvSrc1Step, nvSrc2, nvSrc2Step, nvDst, nvDstStep, roi);
    ASSERT_EQ(nvStatus, NPP_NO_ERROR);
    
    // 读回结果并比较
    std::vector<Npp32f> openResult(width * height);
    std::vector<Npp32f> nvResult(width * height);
    
    cudaMemcpy2D(openResult.data(), width * sizeof(Npp32f), openDst, openDstStep,
                width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(nvResult.data(), width * sizeof(Npp32f), nvDst, nvDstStep,
                width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    
    // 一致性测试
    ConsistencyResult consistency;
    consistency.totalElements = width * height;
    bool match = compareArrays(openResult.data(), nvResult.data(), 
                              consistency.totalElements, 1e-5f, &consistency.mismatchCount);
    consistency.passed = match;
    consistency.mismatchRate = (consistency.mismatchCount * 100.0) / consistency.totalElements;
    
    printConsistencyResult(consistency, "Add_32f_C1R");
    EXPECT_TRUE(match) << "Results don't match between OpenNPP and NVIDIA NPP";
    
    // 性能测试（大数据量）
    PerformanceResult perfResult;
    perfResult.dataSize = width * height;
    
    perfResult.openNppTime = measureExecutionTime([&]() {
        nppiAdd_32f_C1R(openSrc1, openSrc1Step, openSrc2, openSrc2Step,
                       openDst, openDstStep, roi);
    });
    
    perfResult.nvidiaNppTime = measureExecutionTime([&]() {
        nvidiaLoader_->nv_nppiAdd_32f_C1R(nvSrc1, nvSrc1Step, nvSrc2, nvSrc2Step,
                                          nvDst, nvDstStep, roi);
    });
    
    printPerformanceComparison(perfResult, "Add_32f_C1R (1024x1024)");
    
    // 清理
    freeMemory(openSrc1);
    freeMemory(openSrc2);
    freeMemory(openDst);
    freeMemory(nvSrc1, true);
    freeMemory(nvSrc2, true);
    freeMemory(nvDst, true);
}

// ==================== 批量测试多种数据类型 ====================

TEST_F(NPPIArithmeticComparisonTest, BatchComparison_AllTypes) {
    skipIfNoNvidiaNpp("Skipping batch comparison - NVIDIA NPP not available");
    
    struct TestCase {
        std::string name;
        int width;
        int height;
        bool passed;
    };
    
    std::vector<TestCase> testCases = {
        {"Small_64x64", 64, 64, false},
        {"Medium_640x480", 640, 480, false},
        {"Large_1920x1080", 1920, 1080, false},
        {"Square_512x512", 512, 512, false},
        {"Wide_2048x256", 2048, 256, false},
        {"Tall_256x2048", 256, 2048, false}
    };
    
    std::cout << "\n=== Batch Comparison Test Results ===" << std::endl;
    std::cout << std::setw(20) << "Test Case" 
              << std::setw(15) << "Dimensions"
              << std::setw(10) << "Result" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    for (auto& tc : testCases) {
        // 简单的AddC测试
        std::vector<Npp8u> hostSrc(tc.width * tc.height, 100);
        int srcStep, dstStep;
        Npp8u* src = allocateAndInitialize(tc.width, tc.height, &srcStep, hostSrc);
        Npp8u* dst = allocateAndInitialize<Npp8u>(tc.width, tc.height, &dstStep, {});
        
        NppiSize roi = {tc.width, tc.height};
        NppStatus status = nppiAddC_8u_C1RSfs(src, srcStep, 50, dst, dstStep, roi, 0);
        
        tc.passed = (status == NPP_NO_ERROR);
        
        std::cout << std::setw(20) << tc.name
                  << std::setw(7) << tc.width << "x" << std::setw(6) << tc.height
                  << std::setw(10) << (tc.passed ? "PASS ✓" : "FAIL ✗") << std::endl;
        
        freeMemory(src);
        freeMemory(dst);
    }
    
    int passCount = std::count_if(testCases.begin(), testCases.end(), 
                                  [](const TestCase& tc) { return tc.passed; });
    
    std::cout << std::string(45, '-') << std::endl;
    std::cout << "Summary: " << passCount << "/" << testCases.size() << " passed" << std::endl;
}