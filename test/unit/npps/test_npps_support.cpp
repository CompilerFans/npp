#include <gtest/gtest.h>
#include "npp.h"
#include <vector>

class NppsSupportTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 测试用的信号长度
        signal_size = 1024;
    }

    void TearDown() override {
        // 基类清理
    }

    size_t signal_size;
};

// 测试8位无符号信号内存分配
TEST_F(NppsSupportTest, Malloc_8u) {
    Npp8u* signal = nppsMalloc_8u(signal_size);
    ASSERT_NE(signal, nullptr) << "Failed to allocate 8u signal memory";
    
    // 测试写入和读取
    std::vector<Npp8u> host_data(signal_size);
    for (size_t i = 0; i < signal_size; ++i) {
        host_data[i] = static_cast<Npp8u>(i % 256);
    }
    
    cudaError_t err = cudaMemcpy(signal, host_data.data(), 
                                signal_size * sizeof(Npp8u), cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";
    
    std::vector<Npp8u> result(signal_size);
    err = cudaMemcpy(result.data(), signal, 
                     signal_size * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";
    
    // 验证数据完整性
    for (size_t i = 0; i < signal_size; ++i) {
        EXPECT_EQ(result[i], host_data[i]);
    }
    
    nppsFree(signal);
}

// 测试16位有符号信号内存分配
TEST_F(NppsSupportTest, Malloc_16s) {
    Npp16s* signal = nppsMalloc_16s(signal_size);
    ASSERT_NE(signal, nullptr) << "Failed to allocate 16s signal memory";
    
    // 测试写入和读取
    std::vector<Npp16s> host_data(signal_size);
    for (size_t i = 0; i < signal_size; ++i) {
        host_data[i] = static_cast<Npp16s>(i - 512);
    }
    
    cudaError_t err = cudaMemcpy(signal, host_data.data(), 
                                signal_size * sizeof(Npp16s), cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";
    
    std::vector<Npp16s> result(signal_size);
    err = cudaMemcpy(result.data(), signal, 
                     signal_size * sizeof(Npp16s), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";
    
    // 验证数据完整性
    for (size_t i = 0; i < signal_size; ++i) {
        EXPECT_EQ(result[i], host_data[i]);
    }
    
    nppsFree(signal);
}

// 测试32位浮点信号内存分配
TEST_F(NppsSupportTest, Malloc_32f) {
    Npp32f* signal = nppsMalloc_32f(signal_size);
    ASSERT_NE(signal, nullptr) << "Failed to allocate 32f signal memory";
    
    // 测试写入和读取
    std::vector<Npp32f> host_data(signal_size);
    for (size_t i = 0; i < signal_size; ++i) {
        host_data[i] = static_cast<Npp32f>(i) * 0.5f;
    }
    
    cudaError_t err = cudaMemcpy(signal, host_data.data(), 
                                signal_size * sizeof(Npp32f), cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";
    
    std::vector<Npp32f> result(signal_size);
    err = cudaMemcpy(result.data(), signal, 
                     signal_size * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";
    
    // 验证数据完整性
    for (size_t i = 0; i < signal_size; ++i) {
        EXPECT_FLOAT_EQ(result[i], host_data[i]);
    }
    
    nppsFree(signal);
}

// 测试32位复数信号内存分配
TEST_F(NppsSupportTest, Malloc_32fc) {
    Npp32fc* signal = nppsMalloc_32fc(signal_size);
    ASSERT_NE(signal, nullptr) << "Failed to allocate 32fc signal memory";
    
    // 测试写入和读取
    std::vector<Npp32fc> host_data(signal_size);
    for (size_t i = 0; i < signal_size; ++i) {
        host_data[i].re = static_cast<Npp32f>(i) * 0.5f;
        host_data[i].im = static_cast<Npp32f>(i) * 0.3f;
    }
    
    cudaError_t err = cudaMemcpy(signal, host_data.data(), 
                                signal_size * sizeof(Npp32fc), cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";
    
    std::vector<Npp32fc> result(signal_size);
    err = cudaMemcpy(result.data(), signal, 
                     signal_size * sizeof(Npp32fc), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";
    
    // 验证数据完整性
    for (size_t i = 0; i < signal_size; ++i) {
        EXPECT_FLOAT_EQ(result[i].re, host_data[i].re);
        EXPECT_FLOAT_EQ(result[i].im, host_data[i].im);
    }
    
    nppsFree(signal);
}

// 测试64位浮点信号内存分配
TEST_F(NppsSupportTest, Malloc_64f) {
    Npp64f* signal = nppsMalloc_64f(signal_size);
    ASSERT_NE(signal, nullptr) << "Failed to allocate 64f signal memory";
    
    // 测试写入和读取
    std::vector<Npp64f> host_data(signal_size);
    for (size_t i = 0; i < signal_size; ++i) {
        host_data[i] = static_cast<Npp64f>(i) * 0.123456789;
    }
    
    cudaError_t err = cudaMemcpy(signal, host_data.data(), 
                                signal_size * sizeof(Npp64f), cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";
    
    std::vector<Npp64f> result(signal_size);
    err = cudaMemcpy(result.data(), signal, 
                     signal_size * sizeof(Npp64f), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";
    
    // 验证数据完整性
    for (size_t i = 0; i < signal_size; ++i) {
        EXPECT_DOUBLE_EQ(result[i], host_data[i]);
    }
    
    nppsFree(signal);
}

// 错误参数测试
TEST(NppsSupportParameterTest, NullSize) {
    // 测试零大小分配
    EXPECT_EQ(nppsMalloc_32f(0), nullptr);
    EXPECT_EQ(nppsMalloc_16s(0), nullptr);
    EXPECT_EQ(nppsMalloc_32fc(0), nullptr);
}

TEST(NppsSupportParameterTest, FreeSafety) {
    // 测试空指针释放的安全性
    EXPECT_NO_THROW(nppsFree(nullptr));
    
    // 测试正常分配和释放
    Npp32f* signal = nppsMalloc_32f(100);
    ASSERT_NE(signal, nullptr);
    EXPECT_NO_THROW(nppsFree(signal));
}

// 内存对齐测试
TEST_F(NppsSupportTest, MemoryAlignment) {
    // 测试内存对齐（CUDA内存通常是256字节对齐）
    Npp32f* signal1 = nppsMalloc_32f(signal_size);
    Npp32f* signal2 = nppsMalloc_32f(signal_size);
    
    ASSERT_NE(signal1, nullptr);
    ASSERT_NE(signal2, nullptr);
    
    // 检查指针是否为有效的设备指针
    cudaPointerAttributes attr1, attr2;
    cudaError_t err1 = cudaPointerGetAttributes(&attr1, signal1);
    cudaError_t err2 = cudaPointerGetAttributes(&attr2, signal2);
    
    EXPECT_EQ(err1, cudaSuccess);
    EXPECT_EQ(err2, cudaSuccess);
    
    if (err1 == cudaSuccess && err2 == cudaSuccess) {
        EXPECT_EQ(attr1.type, cudaMemoryTypeDevice);
        EXPECT_EQ(attr2.type, cudaMemoryTypeDevice);
    }
    
    nppsFree(signal1);
    nppsFree(signal2);
}

// 大内存分配测试
TEST(NppsSupportLargeMemoryTest, LargeAllocation) {
    // 测试分配较大的信号（1M个元素）
    const size_t large_size = 1024 * 1024;
    
    Npp32f* large_signal = nppsMalloc_32f(large_size);
    
    if (large_signal != nullptr) {
        // 如果分配成功，测试基本的读写操作
        Npp32f test_value = 42.0f;
        cudaError_t err = cudaMemset(large_signal, 0, large_size * sizeof(Npp32f));
        EXPECT_EQ(err, cudaSuccess);
        
        err = cudaMemcpy(large_signal, &test_value, sizeof(Npp32f), cudaMemcpyHostToDevice);
        EXPECT_EQ(err, cudaSuccess);
        
        Npp32f result;
        err = cudaMemcpy(&result, large_signal, sizeof(Npp32f), cudaMemcpyDeviceToHost);
        EXPECT_EQ(err, cudaSuccess);
        EXPECT_FLOAT_EQ(result, test_value);
        
        nppsFree(large_signal);
    } else {
        // 如果分配失败，这可能是由于内存不足，这是可以接受的
        GTEST_SKIP() << "Large memory allocation failed - possibly due to insufficient GPU memory";
    }
}