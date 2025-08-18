/**
 * @file test_nppi_support.cpp
 * @brief NPPI支持函数测试套件
 * 
 * 测试nppi_support_functions.cpp中的功能：
 * - 内存分配和释放
 * - Pitch对齐验证
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "npp.h"

class NPPIMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// 测试单通道内存分配
TEST_F(NPPIMemoryTest, Malloc_8u_C1) {
    int width = 1024;
    int height = 768;
    int step;
    
    Npp8u* ptr = nppiMalloc_8u_C1(width, height, &step);
    
    ASSERT_NE(ptr, nullptr) << "Memory allocation failed";
    EXPECT_GE(step, width) << "Step should be at least width";
    
    // 验证对齐（通常是32字节对齐）
    EXPECT_EQ(step % 32, 0) << "Step should be 32-byte aligned";
    
    // 测试内存写入
    std::vector<Npp8u> testData(width * height, 128);
    cudaError_t err = cudaMemcpy2D(ptr, step, testData.data(), width,
                                   width, height, cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess) << "Memory copy failed";
    
    nppiFree(ptr);
}

// 测试三通道内存分配
TEST_F(NPPIMemoryTest, Malloc_8u_C3) {
    int width = 640;
    int height = 480;
    int step;
    
    Npp8u* ptr = nppiMalloc_8u_C3(width, height, &step);
    
    ASSERT_NE(ptr, nullptr) << "Memory allocation failed";
    EXPECT_GE(step, width * 3) << "Step should be at least width * 3";
    
    // 验证对齐
    EXPECT_EQ(step % 32, 0) << "Step should be 32-byte aligned";
    
    nppiFree(ptr);
}

// 测试四通道内存分配
TEST_F(NPPIMemoryTest, Malloc_8u_C4) {
    int width = 800;
    int height = 600;
    int step;
    
    Npp8u* ptr = nppiMalloc_8u_C4(width, height, &step);
    
    ASSERT_NE(ptr, nullptr) << "Memory allocation failed";
    EXPECT_GE(step, width * 4) << "Step should be at least width * 4";
    
    // 验证对齐
    EXPECT_EQ(step % 32, 0) << "Step should be 32-byte aligned";
    
    nppiFree(ptr);
}

// 测试小图像分配
TEST_F(NPPIMemoryTest, SmallImageAllocation) {
    int width = 1;
    int height = 1;
    int step;
    
    Npp8u* ptr = nppiMalloc_8u_C1(width, height, &step);
    
    ASSERT_NE(ptr, nullptr) << "Small image allocation failed";
    EXPECT_GE(step, width) << "Step should be at least width";
    
    // 即使是1x1的图像，step也应该对齐
    EXPECT_GT(step, 1) << "Step should be aligned even for 1x1 image";
    
    nppiFree(ptr);
}

// 测试大图像分配
TEST_F(NPPIMemoryTest, LargeImageAllocation) {
    int width = 4096;
    int height = 2160;  // 4K分辨率
    int step;
    
    Npp8u* ptr = nppiMalloc_8u_C1(width, height, &step);
    
    ASSERT_NE(ptr, nullptr) << "Large image allocation failed";
    EXPECT_GE(step, width) << "Step should be at least width";
    
    nppiFree(ptr);
}

// 测试非对齐宽度
TEST_F(NPPIMemoryTest, NonAlignedWidth) {
    int width = 1023;  // 非对齐宽度
    int height = 769;
    int step;
    
    Npp8u* ptr = nppiMalloc_8u_C1(width, height, &step);
    
    ASSERT_NE(ptr, nullptr) << "Non-aligned width allocation failed";
    EXPECT_GT(step, width) << "Step should be greater than non-aligned width";
    EXPECT_EQ(step % 32, 0) << "Step should still be aligned";
    
    nppiFree(ptr);
}

// 测试空指针释放
TEST_F(NPPIMemoryTest, FreeNullPointer) {
    // 释放空指针不应该崩溃
    nppiFree(nullptr);
    SUCCEED() << "Free nullptr should not crash";
}

// 测试内存泄漏（分配后必须释放）
TEST_F(NPPIMemoryTest, MemoryLeakCheck) {
    const int iterations = 100;
    
    for (int i = 0; i < iterations; i++) {
        int step;
        Npp8u* ptr = nppiMalloc_8u_C1(256, 256, &step);
        ASSERT_NE(ptr, nullptr) << "Allocation failed at iteration " << i;
        nppiFree(ptr);
    }
    
    SUCCEED() << "No memory leak detected after " << iterations << " allocations";
}

// 测试不同数据类型的对齐
TEST_F(NPPIMemoryTest, DataTypeAlignment) {
    int width = 512;
    int height = 512;
    int step8u, step16u, step32f;
    
    // 8-bit
    Npp8u* ptr8u = nppiMalloc_8u_C1(width, height, &step8u);
    ASSERT_NE(ptr8u, nullptr);
    EXPECT_GE(step8u, width * sizeof(Npp8u));
    
    // 16-bit (通过8-bit分配，然后转换)
    Npp16u* ptr16u = reinterpret_cast<Npp16u*>(
        nppiMalloc_8u_C1(width * sizeof(Npp16u), height, &step16u));
    ASSERT_NE(ptr16u, nullptr);
    EXPECT_GE(step16u, width * sizeof(Npp16u));
    
    // 32-bit float
    Npp32f* ptr32f = reinterpret_cast<Npp32f*>(
        nppiMalloc_8u_C1(width * sizeof(Npp32f), height, &step32f));
    ASSERT_NE(ptr32f, nullptr);
    EXPECT_GE(step32f, width * sizeof(Npp32f));
    
    // 所有步长都应该对齐
    EXPECT_EQ(step8u % 32, 0);
    EXPECT_EQ(step16u % 32, 0);
    EXPECT_EQ(step32f % 32, 0);
    
    nppiFree(ptr8u);
    nppiFree(reinterpret_cast<Npp8u*>(ptr16u));
    nppiFree(reinterpret_cast<Npp8u*>(ptr32f));
}

// Main函数由gtest_main提供