/**
 * @file test_memory_allocation.cpp
 * @brief NPPI内存分配功能的综合测试
 * 
 * 合并自原有的多个测试文件，提供完整的内存分配测试覆盖
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include "npp.h"

class NPPIMemoryAllocationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // 通用测试辅助函数
    template<typename T>
    void TestAllocation(T* ptr, int width, int height, int step, int channels, const std::string& typeName) {
        ASSERT_NE(ptr, nullptr) << typeName << " allocation failed";
        
        int elementSize = sizeof(T);
        int minStep = width * elementSize * channels;
        EXPECT_GE(step, minStep) << typeName << " step should be at least width * elementSize * channels";
        
        // 验证32字节对齐
        EXPECT_EQ(step % 32, 0) << typeName << " step should be 32-byte aligned";
        
        // 测试内存访问
        std::vector<T> testData(width * channels);
        for (int i = 0; i < width * channels; i++) {
            testData[i] = static_cast<T>(i % 256);
        }
        
        cudaError_t err = cudaMemcpy2D(ptr, step, testData.data(), width * elementSize * channels,
                                       width * elementSize * channels, 1, cudaMemcpyHostToDevice);
        EXPECT_EQ(err, cudaSuccess) << typeName << " memory copy failed";
        
        // 读回验证
        std::vector<T> readback(width * channels);
        err = cudaMemcpy2D(readback.data(), width * elementSize * channels, ptr, step,
                          width * elementSize * channels, 1, cudaMemcpyDeviceToHost);
        EXPECT_EQ(err, cudaSuccess) << typeName << " memory readback failed";
        
        // 验证数据
        for (int i = 0; i < width * channels; i++) {
            EXPECT_EQ(readback[i], testData[i]) << typeName << " data mismatch at index " << i;
        }
    }
};

// ==================== 8-bit unsigned 测试 ====================

TEST_F(NPPIMemoryAllocationTest, Malloc_8u_C1) {
    const int width = 640;
    const int height = 480;
    int step;
    
    Npp8u* ptr = nppiMalloc_8u_C1(width, height, &step);
    TestAllocation(ptr, width, height, step, 1, "8u_C1");
    nppiFree(ptr);
}

TEST_F(NPPIMemoryAllocationTest, Malloc_8u_C2) {
    const int width = 640;
    const int height = 480;
    int step;
    
    Npp8u* ptr = nppiMalloc_8u_C2(width, height, &step);
    TestAllocation(ptr, width, height, step, 2, "8u_C2");
    nppiFree(ptr);
}

TEST_F(NPPIMemoryAllocationTest, Malloc_8u_C3) {
    const int width = 640;
    const int height = 480;
    int step;
    
    Npp8u* ptr = nppiMalloc_8u_C3(width, height, &step);
    TestAllocation(ptr, width, height, step, 3, "8u_C3");
    nppiFree(ptr);
}

TEST_F(NPPIMemoryAllocationTest, Malloc_8u_C4) {
    const int width = 640;
    const int height = 480;
    int step;
    
    Npp8u* ptr = nppiMalloc_8u_C4(width, height, &step);
    TestAllocation(ptr, width, height, step, 4, "8u_C4");
    nppiFree(ptr);
}

// ==================== 16-bit unsigned 测试 ====================

TEST_F(NPPIMemoryAllocationTest, Malloc_16u_AllChannels) {
    const int width = 512;
    const int height = 384;
    int step;
    
    // C1
    Npp16u* ptr1 = nppiMalloc_16u_C1(width, height, &step);
    TestAllocation(ptr1, width, height, step, 1, "16u_C1");
    nppiFree(ptr1);
    
    // C2
    Npp16u* ptr2 = nppiMalloc_16u_C2(width, height, &step);
    TestAllocation(ptr2, width, height, step, 2, "16u_C2");
    nppiFree(ptr2);
    
    // C3
    Npp16u* ptr3 = nppiMalloc_16u_C3(width, height, &step);
    TestAllocation(ptr3, width, height, step, 3, "16u_C3");
    nppiFree(ptr3);
    
    // C4
    Npp16u* ptr4 = nppiMalloc_16u_C4(width, height, &step);
    TestAllocation(ptr4, width, height, step, 4, "16u_C4");
    nppiFree(ptr4);
}

// ==================== 16-bit signed 测试 ====================

TEST_F(NPPIMemoryAllocationTest, Malloc_16s_AllChannels) {
    const int width = 512;
    const int height = 384;
    int step;
    
    // C1
    Npp16s* ptr1 = nppiMalloc_16s_C1(width, height, &step);
    TestAllocation(ptr1, width, height, step, 1, "16s_C1");
    nppiFree(ptr1);
    
    // C2
    Npp16s* ptr2 = nppiMalloc_16s_C2(width, height, &step);
    TestAllocation(ptr2, width, height, step, 2, "16s_C2");
    nppiFree(ptr2);
    
    // C4
    Npp16s* ptr4 = nppiMalloc_16s_C4(width, height, &step);
    TestAllocation(ptr4, width, height, step, 4, "16s_C4");
    nppiFree(ptr4);
}

// ==================== 32-bit float 测试 ====================

TEST_F(NPPIMemoryAllocationTest, Malloc_32f_AllChannels) {
    const int width = 320;
    const int height = 240;
    int step;
    
    // C1
    Npp32f* ptr1 = nppiMalloc_32f_C1(width, height, &step);
    TestAllocation(ptr1, width, height, step, 1, "32f_C1");
    nppiFree(ptr1);
    
    // C2
    Npp32f* ptr2 = nppiMalloc_32f_C2(width, height, &step);
    TestAllocation(ptr2, width, height, step, 2, "32f_C2");
    nppiFree(ptr2);
    
    // C3
    Npp32f* ptr3 = nppiMalloc_32f_C3(width, height, &step);
    TestAllocation(ptr3, width, height, step, 3, "32f_C3");
    nppiFree(ptr3);
    
    // C4
    Npp32f* ptr4 = nppiMalloc_32f_C4(width, height, &step);
    TestAllocation(ptr4, width, height, step, 4, "32f_C4");
    nppiFree(ptr4);
}

// ==================== 32-bit signed integer 测试 ====================

TEST_F(NPPIMemoryAllocationTest, Malloc_32s_AllChannels) {
    const int width = 256;
    const int height = 256;
    int step;
    
    // C1
    Npp32s* ptr1 = nppiMalloc_32s_C1(width, height, &step);
    TestAllocation(ptr1, width, height, step, 1, "32s_C1");
    nppiFree(ptr1);
    
    // C3
    Npp32s* ptr3 = nppiMalloc_32s_C3(width, height, &step);
    TestAllocation(ptr3, width, height, step, 3, "32s_C3");
    nppiFree(ptr3);
    
    // C4
    Npp32s* ptr4 = nppiMalloc_32s_C4(width, height, &step);
    TestAllocation(ptr4, width, height, step, 4, "32s_C4");
    nppiFree(ptr4);
}

// ==================== 复数类型测试 ====================

TEST_F(NPPIMemoryAllocationTest, Malloc_16sc_AllChannels) {
    const int width = 256;
    const int height = 256;
    int step;
    
    // C1
    Npp16sc* ptr1 = nppiMalloc_16sc_C1(width, height, &step);
    ASSERT_NE(ptr1, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp16sc)));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr1);
    
    // C2
    Npp16sc* ptr2 = nppiMalloc_16sc_C2(width, height, &step);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp16sc) * 2));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr2);
    
    // C3
    Npp16sc* ptr3 = nppiMalloc_16sc_C3(width, height, &step);
    ASSERT_NE(ptr3, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp16sc) * 3));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr3);
    
    // C4
    Npp16sc* ptr4 = nppiMalloc_16sc_C4(width, height, &step);
    ASSERT_NE(ptr4, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp16sc) * 4));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr4);
}

TEST_F(NPPIMemoryAllocationTest, Malloc_32sc_AllChannels) {
    const int width = 128;
    const int height = 128;
    int step;
    
    // C1
    Npp32sc* ptr1 = nppiMalloc_32sc_C1(width, height, &step);
    ASSERT_NE(ptr1, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp32sc)));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr1);
    
    // C2
    Npp32sc* ptr2 = nppiMalloc_32sc_C2(width, height, &step);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp32sc) * 2));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr2);
    
    // C3
    Npp32sc* ptr3 = nppiMalloc_32sc_C3(width, height, &step);
    ASSERT_NE(ptr3, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp32sc) * 3));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr3);
    
    // C4
    Npp32sc* ptr4 = nppiMalloc_32sc_C4(width, height, &step);
    ASSERT_NE(ptr4, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp32sc) * 4));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr4);
}

TEST_F(NPPIMemoryAllocationTest, Malloc_32fc_AllChannels) {
    const int width = 128;
    const int height = 128;
    int step;
    
    // C1
    Npp32fc* ptr1 = nppiMalloc_32fc_C1(width, height, &step);
    ASSERT_NE(ptr1, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp32fc)));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr1);
    
    // C2
    Npp32fc* ptr2 = nppiMalloc_32fc_C2(width, height, &step);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp32fc) * 2));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr2);
    
    // C3
    Npp32fc* ptr3 = nppiMalloc_32fc_C3(width, height, &step);
    ASSERT_NE(ptr3, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp32fc) * 3));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr3);
    
    // C4
    Npp32fc* ptr4 = nppiMalloc_32fc_C4(width, height, &step);
    ASSERT_NE(ptr4, nullptr);
    EXPECT_GE(step, static_cast<int>(width * sizeof(Npp32fc) * 4));
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr4);
}

// ==================== 边界条件测试 ====================

TEST_F(NPPIMemoryAllocationTest, SmallImageAllocation) {
    // 测试非常小的图像
    int step;
    
    // 1x1 图像
    Npp8u* ptr1 = nppiMalloc_8u_C1(1, 1, &step);
    ASSERT_NE(ptr1, nullptr);
    EXPECT_GE(step, 1);
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr1);
    
    // 2x2 图像
    Npp16u* ptr2 = nppiMalloc_16u_C1(2, 2, &step);
    ASSERT_NE(ptr2, nullptr);
    EXPECT_GE(step, 4);
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr2);
    
    // 8x8 图像
    Npp32f* ptr3 = nppiMalloc_32f_C1(8, 8, &step);
    ASSERT_NE(ptr3, nullptr);
    EXPECT_GE(step, 32);
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr3);
}

TEST_F(NPPIMemoryAllocationTest, LargeImageAllocation) {
    // 测试大图像分配
    int step;
    
    // 4K分辨率
    Npp8u* ptr1 = nppiMalloc_8u_C3(3840, 2160, &step);
    ASSERT_NE(ptr1, nullptr);
    EXPECT_GE(step, 3840 * 3);
    EXPECT_EQ(step % 32, 0);
    nppiFree(ptr1);
    
    // 8K分辨率（如果内存足够）
    Npp8u* ptr2 = nppiMalloc_8u_C1(7680, 4320, &step);
    if (ptr2 != nullptr) {
        EXPECT_GE(step, 7680);
        EXPECT_EQ(step % 32, 0);
        nppiFree(ptr2);
    }
}

TEST_F(NPPIMemoryAllocationTest, NonAlignedWidth) {
    // 测试各种非对齐宽度
    const int testWidths[] = {31, 33, 63, 65, 127, 129, 255, 257, 511, 513};
    int step;
    
    for (int width : testWidths) {
        Npp8u* ptr = nppiMalloc_8u_C1(width, 1, &step);
        ASSERT_NE(ptr, nullptr) << "Failed for width " << width;
        EXPECT_GE(step, width) << "Step too small for width " << width;
        EXPECT_EQ(step % 32, 0) << "Step not aligned for width " << width;
        nppiFree(ptr);
    }
}

// ==================== 错误处理测试 ====================

TEST_F(NPPIMemoryAllocationTest, FreeNullPointer) {
    // nppiFree应该能安全处理nullptr
    nppiFree(nullptr);
    // 如果程序没有崩溃，测试通过
    SUCCEED();
}

TEST_F(NPPIMemoryAllocationTest, ZeroSizeAllocation) {
    int step;
    
    // 0宽度
    Npp8u* ptr1 = nppiMalloc_8u_C1(0, 100, &step);
    EXPECT_EQ(ptr1, nullptr);
    
    // 0高度
    Npp8u* ptr2 = nppiMalloc_8u_C1(100, 0, &step);
    EXPECT_EQ(ptr2, nullptr);
    
    // 都是0
    Npp8u* ptr3 = nppiMalloc_8u_C1(0, 0, &step);
    EXPECT_EQ(ptr3, nullptr);
}

TEST_F(NPPIMemoryAllocationTest, NegativeSizeAllocation) {
    int step;
    
    // 负宽度
    Npp8u* ptr1 = nppiMalloc_8u_C1(-100, 100, &step);
    EXPECT_EQ(ptr1, nullptr);
    
    // 负高度
    Npp8u* ptr2 = nppiMalloc_8u_C1(100, -100, &step);
    EXPECT_EQ(ptr2, nullptr);
}

// ==================== 内存泄漏测试 ====================

TEST_F(NPPIMemoryAllocationTest, MemoryLeakCheck) {
    // 进行多次分配和释放，确保没有内存泄漏
    const int iterations = 100;
    
    for (int i = 0; i < iterations; i++) {
        int step;
        Npp8u* ptr = nppiMalloc_8u_C1(640, 480, &step);
        ASSERT_NE(ptr, nullptr) << "Allocation failed at iteration " << i;
        nppiFree(ptr);
    }
    
    // 获取当前内存使用情况
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // 再次进行分配释放
    for (int i = 0; i < iterations; i++) {
        int step;
        Npp8u* ptr = nppiMalloc_8u_C1(640, 480, &step);
        ASSERT_NE(ptr, nullptr);
        nppiFree(ptr);
    }
    
    // 检查内存是否恢复
    size_t free_mem_after, total_mem_after;
    cudaMemGetInfo(&free_mem_after, &total_mem_after);
    
    // 允许一些小的波动（由于CUDA运行时的内部分配）
    const size_t tolerance = 10 * 1024 * 1024; // 10MB容差
    EXPECT_NEAR(free_mem, free_mem_after, tolerance) 
        << "Possible memory leak detected";
}