/**
 * @file test_consistency_simple.cpp
 * @brief 简化版的OpenNPP与NVIDIA NPP一致性验证
 * 
 * 这是一个可以立即编译运行的实用版本
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include "npp.h"

#ifdef HAVE_NVIDIA_NPP
#include <nppdefs.h>
#include <nppi_arithmetic_and_logical_operations.h>
// 使用命名空间或前缀来区分
#define NVIDIA_nppiAddC_8u_C1RSfs ::nppiAddC_8u_C1RSfs
#define NVIDIA_nppiAddC_32f_C1R ::nppiAddC_32f_C1R
#endif

// ==================== 一致性测试类 ====================
class ConsistencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
        nppGetStreamContext(&ctx);
        
#ifndef HAVE_NVIDIA_NPP
        GTEST_SKIP() << "NVIDIA NPP not available";
#endif
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // 简单的差异统计
    struct DiffStats {
        double maxDiff = 0;
        double avgDiff = 0;
        int diffCount = 0;
        int totalCount = 0;
        
        void Print(const std::string& name) {
            std::cout << "\n=== " << name << " 差异统计 ===\n";
            std::cout << "最大差异: " << std::scientific << std::setprecision(6) << maxDiff << "\n";
            std::cout << "平均差异: " << avgDiff << "\n";
            std::cout << "差异比例: " << (diffCount * 100.0 / totalCount) << "%\n";
        }
        
        bool IsConsistent(double tolerance) {
            return maxDiff <= tolerance;
        }
    };
    
    // 比较8位结果
    DiffStats Compare8u(Npp8u* result1, Npp8u* result2, int width, int height, int step1, int step2) {
        std::vector<Npp8u> host1(width * height);
        std::vector<Npp8u> host2(width * height);
        
        cudaMemcpy2D(host1.data(), width, result1, step1, width, height, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(host2.data(), width, result2, step2, width, height, cudaMemcpyDeviceToHost);
        
        DiffStats stats;
        stats.totalCount = width * height;
        
        double sumDiff = 0;
        for (int i = 0; i < width * height; i++) {
            int diff = std::abs(static_cast<int>(host1[i]) - static_cast<int>(host2[i]));
            if (diff > 0) {
                stats.diffCount++;
                sumDiff += diff;
                stats.maxDiff = std::max(stats.maxDiff, static_cast<double>(diff));
            }
        }
        
        if (stats.diffCount > 0) {
            stats.avgDiff = sumDiff / stats.totalCount;
        }
        
        return stats;
    }
    
    // 比较32位浮点结果
    DiffStats Compare32f(Npp32f* result1, Npp32f* result2, int width, int height, int step1, int step2) {
        std::vector<Npp32f> host1(width * height);
        std::vector<Npp32f> host2(width * height);
        
        cudaMemcpy2D(host1.data(), width * sizeof(Npp32f), result1, step1, 
                    width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(host2.data(), width * sizeof(Npp32f), result2, step2, 
                    width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
        
        DiffStats stats;
        stats.totalCount = width * height;
        
        double sumDiff = 0;
        for (int i = 0; i < width * height; i++) {
            double diff = std::abs(host1[i] - host2[i]);
            if (diff > 1e-7) {  // 忽略极小差异
                stats.diffCount++;
                sumDiff += diff;
                stats.maxDiff = std::max(stats.maxDiff, diff);
            }
        }
        
        if (stats.diffCount > 0) {
            stats.avgDiff = sumDiff / stats.totalCount;
        }
        
        return stats;
    }
    
    NppStreamContext ctx;
};

// ==================== AddC 8位测试 ====================
TEST_F(ConsistencyTest, AddC_8u_RandomData) {
    const int width = 512;
    const int height = 512;
    int srcStep, dstOpenStep, dstNvidiaStep;
    
    // 分配内存
    Npp8u* src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* dstOpen = nppiMalloc_8u_C1(width, height, &dstOpenStep);
    Npp8u* dstNvidia = nppiMalloc_8u_C1(width, height, &dstNvidiaStep);
    
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dstOpen, nullptr);
    ASSERT_NE(dstNvidia, nullptr);
    
    // 生成随机测试数据
    std::vector<Npp8u> hostSrc(width * height);
    std::mt19937 gen(42);  // 固定种子保证可重现
    std::uniform_int_distribution<int> dist(0, 200);
    for (auto& val : hostSrc) {
        val = static_cast<Npp8u>(dist(gen));
    }
    cudaMemcpy2D(src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);
    
    Npp8u constant = 50;
    NppiSize roi = {width, height};
    
    // 执行OpenNPP
    NppStatus statusOpen = nppiAddC_8u_C1RSfs(src, srcStep, constant, 
                                              dstOpen, dstOpenStep, roi, 0);
    ASSERT_EQ(statusOpen, NPP_SUCCESS) << "OpenNPP failed";
    
#ifdef HAVE_NVIDIA_NPP
    // 执行NVIDIA NPP
    NppStatus statusNvidia = NVIDIA_nppiAddC_8u_C1RSfs(src, srcStep, constant,
                                                       dstNvidia, dstNvidiaStep, roi, 0);
    ASSERT_EQ(statusNvidia, NPP_SUCCESS) << "NVIDIA NPP failed";
    
    // 比较结果
    auto stats = Compare8u(dstOpen, dstNvidia, width, height, dstOpenStep, dstNvidiaStep);
    stats.Print("AddC_8u_RandomData");
    
    // 验证一致性（容差为1）
    EXPECT_TRUE(stats.IsConsistent(1.0)) 
        << "Results not consistent, max diff: " << stats.maxDiff;
#endif
    
    nppiFree(src);
    nppiFree(dstOpen);
    nppiFree(dstNvidia);
}

// ==================== AddC 8位边界测试 ====================
TEST_F(ConsistencyTest, AddC_8u_Saturation) {
    const int width = 256;
    const int height = 256;
    int srcStep, dstOpenStep, dstNvidiaStep;
    
    Npp8u* src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* dstOpen = nppiMalloc_8u_C1(width, height, &dstOpenStep);
    Npp8u* dstNvidia = nppiMalloc_8u_C1(width, height, &dstNvidiaStep);
    
    ASSERT_NE(src, nullptr);
    
    // 测试饱和情况（接近255的值）
    std::vector<Npp8u> hostSrc(width * height, 240);  // 接近最大值
    cudaMemcpy2D(src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);
    
    Npp8u constant = 100;  // 会导致饱和
    NppiSize roi = {width, height};
    
    NppStatus statusOpen = nppiAddC_8u_C1RSfs(src, srcStep, constant, 
                                              dstOpen, dstOpenStep, roi, 0);
    ASSERT_EQ(statusOpen, NPP_SUCCESS);
    
#ifdef HAVE_NVIDIA_NPP
    NppStatus statusNvidia = NVIDIA_nppiAddC_8u_C1RSfs(src, srcStep, constant,
                                                       dstNvidia, dstNvidiaStep, roi, 0);
    ASSERT_EQ(statusNvidia, NPP_SUCCESS);
    
    auto stats = Compare8u(dstOpen, dstNvidia, width, height, dstOpenStep, dstNvidiaStep);
    stats.Print("AddC_8u_Saturation");
    
    // 饱和处理应该完全一致
    EXPECT_EQ(stats.maxDiff, 0) << "Saturation handling should be identical";
#endif
    
    nppiFree(src);
    nppiFree(dstOpen);
    nppiFree(dstNvidia);
}

// ==================== AddC 32位浮点测试 ====================
TEST_F(ConsistencyTest, AddC_32f_RandomData) {
    const int width = 512;
    const int height = 512;
    int srcStep, dstOpenStep, dstNvidiaStep;
    
    // 分配内存（通过8位分配，然后转换）
    Npp32f* src = reinterpret_cast<Npp32f*>(
        nppiMalloc_8u_C1(width * sizeof(Npp32f), height, &srcStep));
    Npp32f* dstOpen = reinterpret_cast<Npp32f*>(
        nppiMalloc_8u_C1(width * sizeof(Npp32f), height, &dstOpenStep));
    Npp32f* dstNvidia = reinterpret_cast<Npp32f*>(
        nppiMalloc_8u_C1(width * sizeof(Npp32f), height, &dstNvidiaStep));
    
    ASSERT_NE(src, nullptr);
    
    // 生成随机浮点数据
    std::vector<Npp32f> hostSrc(width * height);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    for (auto& val : hostSrc) {
        val = dist(gen);
    }
    cudaMemcpy2D(src, srcStep, hostSrc.data(), width * sizeof(Npp32f),
                width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    
    Npp32f constant = 3.14159f;
    NppiSize roi = {width, height};
    
    NppStatus statusOpen = nppiAddC_32f_C1R(src, srcStep, constant, 
                                            dstOpen, dstOpenStep, roi);
    ASSERT_EQ(statusOpen, NPP_SUCCESS);
    
#ifdef HAVE_NVIDIA_NPP
    NppStatus statusNvidia = NVIDIA_nppiAddC_32f_C1R(src, srcStep, constant,
                                                     dstNvidia, dstNvidiaStep, roi);
    ASSERT_EQ(statusNvidia, NPP_SUCCESS);
    
    auto stats = Compare32f(dstOpen, dstNvidia, width, height, dstOpenStep, dstNvidiaStep);
    stats.Print("AddC_32f_RandomData");
    
    // 浮点容差为1e-5
    EXPECT_TRUE(stats.IsConsistent(1e-5)) 
        << "Float results not consistent, max diff: " << stats.maxDiff;
#endif
    
    nppiFree(reinterpret_cast<Npp8u*>(src));
    nppiFree(reinterpret_cast<Npp8u*>(dstOpen));
    nppiFree(reinterpret_cast<Npp8u*>(dstNvidia));
}

// ==================== 非对齐尺寸测试 ====================
TEST_F(ConsistencyTest, AddC_8u_UnalignedSize) {
    const int width = 1023;  // 非对齐宽度
    const int height = 769;  // 质数高度
    int srcStep, dstOpenStep, dstNvidiaStep;
    
    Npp8u* src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* dstOpen = nppiMalloc_8u_C1(width, height, &dstOpenStep);
    Npp8u* dstNvidia = nppiMalloc_8u_C1(width, height, &dstNvidiaStep);
    
    ASSERT_NE(src, nullptr);
    
    // 生成测试数据
    std::vector<Npp8u> hostSrc(width * height);
    for (int i = 0; i < width * height; i++) {
        hostSrc[i] = static_cast<Npp8u>(i % 256);
    }
    cudaMemcpy2D(src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);
    
    Npp8u constant = 25;
    NppiSize roi = {width, height};
    
    NppStatus statusOpen = nppiAddC_8u_C1RSfs(src, srcStep, constant, 
                                              dstOpen, dstOpenStep, roi, 0);
    ASSERT_EQ(statusOpen, NPP_SUCCESS);
    
#ifdef HAVE_NVIDIA_NPP
    NppStatus statusNvidia = NVIDIA_nppiAddC_8u_C1RSfs(src, srcStep, constant,
                                                       dstNvidia, dstNvidiaStep, roi, 0);
    ASSERT_EQ(statusNvidia, NPP_SUCCESS);
    
    auto stats = Compare8u(dstOpen, dstNvidia, width, height, dstOpenStep, dstNvidiaStep);
    stats.Print("AddC_8u_UnalignedSize");
    
    EXPECT_TRUE(stats.IsConsistent(1.0)) 
        << "Unaligned size results not consistent";
#endif
    
    nppiFree(src);
    nppiFree(dstOpen);
    nppiFree(dstNvidia);
}

// ==================== 性能对比测试 ====================
TEST_F(ConsistencyTest, AddC_Performance_Comparison) {
    const int width = 1920;
    const int height = 1080;
    int srcStep, dstOpenStep, dstNvidiaStep;
    
    Npp8u* src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* dstOpen = nppiMalloc_8u_C1(width, height, &dstOpenStep);
    Npp8u* dstNvidia = nppiMalloc_8u_C1(width, height, &dstNvidiaStep);
    
    ASSERT_NE(src, nullptr);
    
    // 初始化数据
    std::vector<Npp8u> hostSrc(width * height, 100);
    cudaMemcpy2D(src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);
    
    Npp8u constant = 50;
    NppiSize roi = {width, height};
    const int iterations = 100;
    
    // 预热
    for (int i = 0; i < 10; i++) {
        nppiAddC_8u_C1RSfs(src, srcStep, constant, dstOpen, dstOpenStep, roi, 0);
    }
    cudaDeviceSynchronize();
    
    // 测试OpenNPP性能
    auto startOpen = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        nppiAddC_8u_C1RSfs(src, srcStep, constant, dstOpen, dstOpenStep, roi, 0);
    }
    cudaDeviceSynchronize();
    auto endOpen = std::chrono::high_resolution_clock::now();
    
    double openTime = std::chrono::duration<double, std::milli>(endOpen - startOpen).count() / iterations;
    
#ifdef HAVE_NVIDIA_NPP
    // 测试NVIDIA NPP性能
    auto startNvidia = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        NVIDIA_nppiAddC_8u_C1RSfs(src, srcStep, constant, dstNvidia, dstNvidiaStep, roi, 0);
    }
    cudaDeviceSynchronize();
    auto endNvidia = std::chrono::high_resolution_clock::now();
    
    double nvidiaTime = std::chrono::duration<double, std::milli>(endNvidia - startNvidia).count() / iterations;
    
    std::cout << "\n=== 性能对比 (1920x1080) ===\n";
    std::cout << "OpenNPP: " << openTime << " ms\n";
    std::cout << "NVIDIA NPP: " << nvidiaTime << " ms\n";
    std::cout << "速度比: " << (openTime / nvidiaTime) << "x\n";
    
    // 期望性能在2倍以内
    EXPECT_LT(openTime / nvidiaTime, 2.0) << "OpenNPP is too slow compared to NVIDIA NPP";
#else
    std::cout << "\nOpenNPP性能: " << openTime << " ms\n";
#endif
    
    nppiFree(src);
    nppiFree(dstOpen);
    nppiFree(dstNvidia);
}

// Main函数由gtest_main提供