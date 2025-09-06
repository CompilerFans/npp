#include <gtest/gtest.h>
#include "npp.h"
#include <vector>
#include <cuda_runtime.h>

class LogicalFunctionalTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试数据
        width = 32;
        height = 32;
        size.width = width;
        size.height = height;
        step = width * sizeof(Npp8u);

        // 分配设备内存
        cudaMalloc(&d_src1, height * step);
        cudaMalloc(&d_src2, height * step);
        cudaMalloc(&d_dst, height * step);

        // 准备测试数据
        h_src1.resize(width * height);
        h_src2.resize(width * height);
        h_dst.resize(width * height);
        h_expected.resize(width * height);

        // 初始化测试模式
        for (int i = 0; i < width * height; ++i) {
            h_src1[i] = 0xAA;  // 10101010
            h_src2[i] = 0x55;  // 01010101
        }

        // 上传到设备
        cudaMemcpy(d_src1, h_src1.data(), height * step, cudaMemcpyHostToDevice);
        cudaMemcpy(d_src2, h_src2.data(), height * step, cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_src1);
        cudaFree(d_src2);
        cudaFree(d_dst);
    }

    int width, height, step;
    NppiSize size;
    Npp8u *d_src1, *d_src2, *d_dst;
    std::vector<Npp8u> h_src1, h_src2, h_dst, h_expected;
};

TEST_F(LogicalFunctionalTest, And_8u_C1R) {
    // 执行AND操作
    NppStatus status = nppiAnd_8u_C1R(d_src1, step, d_src2, step, 
                                       d_dst, step, size);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    cudaMemcpy(h_dst.data(), d_dst, height * step, cudaMemcpyDeviceToHost);

    // 验证结果 (0xAA & 0x55 = 0x00)
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(h_dst[i], 0x00);
    }
}

TEST_F(LogicalFunctionalTest, AndC_8u_C1R) {
    const Npp8u constant = 0x0F;  // 00001111
    
    // 执行AND常数操作
    NppStatus status = nppiAndC_8u_C1R(d_src1, step, constant, 
                                       d_dst, step, size);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    cudaMemcpy(h_dst.data(), d_dst, height * step, cudaMemcpyDeviceToHost);

    // 验证结果 (0xAA & 0x0F = 0x0A)
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(h_dst[i], 0x0A);
    }
}

TEST_F(LogicalFunctionalTest, Or_8u_C1R) {
    // 执行OR操作
    NppStatus status = nppiOr_8u_C1R(d_src1, step, d_src2, step, 
                                      d_dst, step, size);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    cudaMemcpy(h_dst.data(), d_dst, height * step, cudaMemcpyDeviceToHost);

    // 验证结果 (0xAA | 0x55 = 0xFF)
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(h_dst[i], 0xFF);
    }
}

TEST_F(LogicalFunctionalTest, OrC_8u_C1R) {
    const Npp8u constant = 0x0F;  // 00001111
    
    // 执行OR常数操作
    NppStatus status = nppiOrC_8u_C1R(d_src1, step, constant, 
                                      d_dst, step, size);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    cudaMemcpy(h_dst.data(), d_dst, height * step, cudaMemcpyDeviceToHost);

    // 验证结果 (0xAA | 0x0F = 0xAF)
    for (int i = 0; i < width * height; ++i) {
        EXPECT_EQ(h_dst[i], 0xAF);
    }
}

// 参数验证测试
TEST(LogicalParameterTest, NullPointerError) {
    NppiSize size = {10, 10};
    int step = 10;
    Npp8u dummy;
    
    // AND测试
    EXPECT_EQ(nppiAnd_8u_C1R(nullptr, step, &dummy, step, &dummy, step, size), 
              NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiAnd_8u_C1R(&dummy, step, nullptr, step, &dummy, step, size), 
              NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiAnd_8u_C1R(&dummy, step, &dummy, step, nullptr, step, size), 
              NPP_NULL_POINTER_ERROR);
    
    // OR测试
    EXPECT_EQ(nppiOr_8u_C1R(nullptr, step, &dummy, step, &dummy, step, size), 
              NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiOr_8u_C1R(&dummy, step, nullptr, step, &dummy, step, size), 
              NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiOr_8u_C1R(&dummy, step, &dummy, step, nullptr, step, size), 
              NPP_NULL_POINTER_ERROR);
}

TEST(LogicalParameterTest, InvalidSize) {
    Npp8u dummy;
    int step = 10;
    
    // 测试无效宽度
    NppiSize size = {0, 10};
    EXPECT_EQ(nppiAnd_8u_C1R(&dummy, step, &dummy, step, &dummy, step, size), 
              NPP_SIZE_ERROR);
    
    // 测试无效高度
    size = {10, 0};
    EXPECT_EQ(nppiOr_8u_C1R(&dummy, step, &dummy, step, &dummy, step, size), 
              NPP_SIZE_ERROR);
}

TEST(LogicalParameterTest, InvalidStep) {
    Npp8u dummy;
    NppiSize size = {10, 10};
    
    // 测试无效步长
    EXPECT_EQ(nppiAnd_8u_C1R(&dummy, 5, &dummy, 10, &dummy, 10, size), 
              NPP_STEP_ERROR);
    EXPECT_EQ(nppiOr_8u_C1R(&dummy, 10, &dummy, 5, &dummy, 10, size), 
              NPP_STEP_ERROR);
}