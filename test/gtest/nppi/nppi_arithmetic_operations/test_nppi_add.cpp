/**
 * @file test_nppi_add.cpp
 * @brief 测试NPPI双源Add函数
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include "npp.h"

class NPPIAddTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    // 辅助函数：生成随机数据
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

// ==================== 8-bit unsigned 测试 ====================

TEST_F(NPPIAddTest, Add_8u_C1RSfs_Basic) {
    const int width = 64;
    const int height = 64;
    const int nScaleFactor = 0;
    
    // 分配设备内存
    int step1, step2, stepDst;
    Npp8u* d_src1 = nppiMalloc_8u_C1(width, height, &step1);
    Npp8u* d_src2 = nppiMalloc_8u_C1(width, height, &step2);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &stepDst);
    
    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 准备主机数据
    std::vector<Npp8u> h_src1(width * height, 50);
    std::vector<Npp8u> h_src2(width * height, 30);
    std::vector<Npp8u> h_dst(width * height);
    
    // 复制到设备
    cudaMemcpy2D(d_src1, step1, h_src1.data(), width * sizeof(Npp8u),
                 width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, step2, h_src2.data(), width * sizeof(Npp8u),
                 width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    
    // 执行Add操作
    NppiSize roi = {width, height};
    NppStatus status = nppiAdd_8u_C1RSfs(d_src1, step1, d_src2, step2,
                                         d_dst, stepDst, roi, nScaleFactor);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 复制回主机
    cudaMemcpy2D(h_dst.data(), width * sizeof(Npp8u), d_dst, stepDst,
                 width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < width * height; i++) {
        EXPECT_EQ(h_dst[i], 80) << "Mismatch at index " << i;
    }
    
    // 清理
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
}

TEST_F(NPPIAddTest, Add_8u_C1RSfs_WithScaling) {
    const int width = 32;
    const int height = 32;
    const int nScaleFactor = 2; // 右移2位，相当于除以4
    
    // 分配设备内存
    int step1, step2, stepDst;
    Npp8u* d_src1 = nppiMalloc_8u_C1(width, height, &step1);
    Npp8u* d_src2 = nppiMalloc_8u_C1(width, height, &step2);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &stepDst);
    
    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 准备主机数据
    std::vector<Npp8u> h_src1(width * height, 100);
    std::vector<Npp8u> h_src2(width * height, 60);
    std::vector<Npp8u> h_dst(width * height);
    
    // 复制到设备
    cudaMemcpy2D(d_src1, step1, h_src1.data(), width * sizeof(Npp8u),
                 width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, step2, h_src2.data(), width * sizeof(Npp8u),
                 width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    
    // 执行Add操作
    NppiSize roi = {width, height};
    NppStatus status = nppiAdd_8u_C1RSfs(d_src1, step1, d_src2, step2,
                                         d_dst, stepDst, roi, nScaleFactor);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 复制回主机
    cudaMemcpy2D(h_dst.data(), width * sizeof(Npp8u), d_dst, stepDst,
                 width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
    
    // 验证结果: (100 + 60 + 2) >> 2 = 162 >> 2 = 40
    for (int i = 0; i < width * height; i++) {
        EXPECT_EQ(h_dst[i], 40) << "Mismatch at index " << i;
    }
    
    // 清理
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
}

TEST_F(NPPIAddTest, Add_8u_C1RSfs_Saturation) {
    const int width = 16;
    const int height = 16;
    const int nScaleFactor = 0;
    
    // 分配设备内存
    int step1, step2, stepDst;
    Npp8u* d_src1 = nppiMalloc_8u_C1(width, height, &step1);
    Npp8u* d_src2 = nppiMalloc_8u_C1(width, height, &step2);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &stepDst);
    
    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 准备主机数据 - 测试饱和
    std::vector<Npp8u> h_src1(width * height, 200);
    std::vector<Npp8u> h_src2(width * height, 100);
    std::vector<Npp8u> h_dst(width * height);
    
    // 复制到设备
    cudaMemcpy2D(d_src1, step1, h_src1.data(), width * sizeof(Npp8u),
                 width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, step2, h_src2.data(), width * sizeof(Npp8u),
                 width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    
    // 执行Add操作
    NppiSize roi = {width, height};
    NppStatus status = nppiAdd_8u_C1RSfs(d_src1, step1, d_src2, step2,
                                         d_dst, stepDst, roi, nScaleFactor);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 复制回主机
    cudaMemcpy2D(h_dst.data(), width * sizeof(Npp8u), d_dst, stepDst,
                 width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
    
    // 验证结果 - 应该饱和到255
    for (int i = 0; i < width * height; i++) {
        EXPECT_EQ(h_dst[i], 255) << "Saturation failed at index " << i;
    }
    
    // 清理
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
}

// ==================== 32-bit float 测试 ====================

TEST_F(NPPIAddTest, Add_32f_C1R_Basic) {
    const int width = 64;
    const int height = 64;
    
    // 分配设备内存
    int step1, step2, stepDst;
    Npp32f* d_src1 = nppiMalloc_32f_C1(width, height, &step1);
    Npp32f* d_src2 = nppiMalloc_32f_C1(width, height, &step2);
    Npp32f* d_dst = nppiMalloc_32f_C1(width, height, &stepDst);
    
    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 准备主机数据
    std::vector<Npp32f> h_src1(width * height, 1.5f);
    std::vector<Npp32f> h_src2(width * height, 2.5f);
    std::vector<Npp32f> h_dst(width * height);
    
    // 复制到设备
    cudaMemcpy2D(d_src1, step1, h_src1.data(), width * sizeof(Npp32f),
                 width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, step2, h_src2.data(), width * sizeof(Npp32f),
                 width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    
    // 执行Add操作
    NppiSize roi = {width, height};
    NppStatus status = nppiAdd_32f_C1R(d_src1, step1, d_src2, step2,
                                       d_dst, stepDst, roi);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 复制回主机
    cudaMemcpy2D(h_dst.data(), width * sizeof(Npp32f), d_dst, stepDst,
                 width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < width * height; i++) {
        EXPECT_FLOAT_EQ(h_dst[i], 4.0f) << "Mismatch at index " << i;
    }
    
    // 清理
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
}

TEST_F(NPPIAddTest, Add_32f_C1R_Random) {
    const int width = 128;
    const int height = 128;
    
    // 分配设备内存
    int step1, step2, stepDst;
    Npp32f* d_src1 = nppiMalloc_32f_C1(width, height, &step1);
    Npp32f* d_src2 = nppiMalloc_32f_C1(width, height, &step2);
    Npp32f* d_dst = nppiMalloc_32f_C1(width, height, &stepDst);
    
    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 准备随机数据
    std::vector<Npp32f> h_src1(width * height);
    std::vector<Npp32f> h_src2(width * height);
    std::vector<Npp32f> h_dst(width * height);
    
    generateRandomData(h_src1, -100.0f, 100.0f);
    generateRandomData(h_src2, -100.0f, 100.0f);
    
    // 复制到设备
    cudaMemcpy2D(d_src1, step1, h_src1.data(), width * sizeof(Npp32f),
                 width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, step2, h_src2.data(), width * sizeof(Npp32f),
                 width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    
    // 执行Add操作
    NppiSize roi = {width, height};
    NppStatus status = nppiAdd_32f_C1R(d_src1, step1, d_src2, step2,
                                       d_dst, stepDst, roi);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 复制回主机
    cudaMemcpy2D(h_dst.data(), width * sizeof(Npp32f), d_dst, stepDst,
                 width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < width * height; i++) {
        float expected = h_src1[i] + h_src2[i];
        EXPECT_NEAR(h_dst[i], expected, 1e-5f) << "Mismatch at index " << i;
    }
    
    // 清理
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
}

// ==================== In-place 测试 ====================

TEST_F(NPPIAddTest, Add_32f_C1IR_InPlace) {
    const int width = 32;
    const int height = 32;
    
    // 分配设备内存
    int step1, step2;
    Npp32f* d_src = nppiMalloc_32f_C1(width, height, &step1);
    Npp32f* d_srcdst = nppiMalloc_32f_C1(width, height, &step2);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_srcdst, nullptr);
    
    // 准备主机数据
    std::vector<Npp32f> h_src(width * height, 3.0f);
    std::vector<Npp32f> h_srcdst(width * height, 2.0f);
    
    // 复制到设备
    cudaMemcpy2D(d_src, step1, h_src.data(), width * sizeof(Npp32f),
                 width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_srcdst, step2, h_srcdst.data(), width * sizeof(Npp32f),
                 width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    
    // 执行In-place Add操作
    NppiSize roi = {width, height};
    NppStatus status = nppiAdd_32f_C1IR(d_src, step1, d_srcdst, step2, roi);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 复制回主机
    cudaMemcpy2D(h_srcdst.data(), width * sizeof(Npp32f), d_srcdst, step2,
                 width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < width * height; i++) {
        EXPECT_FLOAT_EQ(h_srcdst[i], 5.0f) << "Mismatch at index " << i;
    }
    
    // 清理
    nppiFree(d_src);
    nppiFree(d_srcdst);
}

// ==================== Multi-channel 测试 ====================

TEST_F(NPPIAddTest, Add_32f_C3R_ThreeChannel) {
    const int width = 16;
    const int height = 16;
    
    // 分配设备内存
    int step1, step2, stepDst;
    Npp32f* d_src1 = nppiMalloc_32f_C3(width, height, &step1);
    Npp32f* d_src2 = nppiMalloc_32f_C3(width, height, &step2);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &stepDst);
    
    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 准备主机数据
    std::vector<Npp32f> h_src1(width * height * 3);
    std::vector<Npp32f> h_src2(width * height * 3);
    std::vector<Npp32f> h_dst(width * height * 3);
    
    // 填充不同通道的值
    for (int i = 0; i < width * height; i++) {
        h_src1[i*3 + 0] = 1.0f; // R
        h_src1[i*3 + 1] = 2.0f; // G
        h_src1[i*3 + 2] = 3.0f; // B
        
        h_src2[i*3 + 0] = 4.0f; // R
        h_src2[i*3 + 1] = 5.0f; // G
        h_src2[i*3 + 2] = 6.0f; // B
    }
    
    // 复制到设备
    cudaMemcpy2D(d_src1, step1, h_src1.data(), width * 3 * sizeof(Npp32f),
                 width * 3 * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, step2, h_src2.data(), width * 3 * sizeof(Npp32f),
                 width * 3 * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    
    // 执行Add操作
    NppiSize roi = {width, height};
    NppStatus status = nppiAdd_32f_C3R(d_src1, step1, d_src2, step2,
                                       d_dst, stepDst, roi);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 复制回主机
    cudaMemcpy2D(h_dst.data(), width * 3 * sizeof(Npp32f), d_dst, stepDst,
                 width * 3 * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < width * height; i++) {
        EXPECT_FLOAT_EQ(h_dst[i*3 + 0], 5.0f) << "R channel mismatch at pixel " << i;
        EXPECT_FLOAT_EQ(h_dst[i*3 + 1], 7.0f) << "G channel mismatch at pixel " << i;
        EXPECT_FLOAT_EQ(h_dst[i*3 + 2], 9.0f) << "B channel mismatch at pixel " << i;
    }
    
    // 清理
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
}