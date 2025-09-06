#include <gtest/gtest.h>
#include "npp.h"
#include <vector>
#include <cuda_runtime.h>

class GeometryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试数据
        width = 8;
        height = 8;
        size.width = width;
        size.height = height;
        step = width * sizeof(Npp8u);

        // 分配设备内存
        cudaMalloc(&d_src, height * step);
        cudaMalloc(&d_dst, height * step);

        // 准备测试数据
        h_src.resize(width * height);
        h_dst.resize(width * height);
    }

    void TearDown() override {
        cudaFree(d_src);
        cudaFree(d_dst);
    }

    int width, height, step;
    NppiSize size;
    Npp8u *d_src, *d_dst;
    std::vector<Npp8u> h_src, h_dst;
};

TEST_F(GeometryTest, Mirror_HorizontalAxis) {
    // 创建一个简单的渐变图像
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_src[y * width + x] = y * 10 + x;  // 每行递增
        }
    }

    // 上传数据
    cudaMemcpy(d_src, h_src.data(), height * step, cudaMemcpyHostToDevice);

    // 执行水平轴镜像（上下翻转）
    NppStatus status = nppiMirror_8u_C1R(d_src, step, d_dst, step, 
                                         size, NPP_HORIZONTAL_AXIS);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    cudaMemcpy(h_dst.data(), d_dst, height * step, cudaMemcpyDeviceToHost);

    // 验证结果 - 第一行应该变成最后一行
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            EXPECT_EQ(h_dst[y * width + x], 
                      h_src[(height - 1 - y) * width + x]);
        }
    }
}

TEST_F(GeometryTest, Mirror_VerticalAxis) {
    // 创建一个简单的渐变图像
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_src[y * width + x] = y * 10 + x;
        }
    }

    // 上传数据
    cudaMemcpy(d_src, h_src.data(), height * step, cudaMemcpyHostToDevice);

    // 执行垂直轴镜像（左右翻转）
    NppStatus status = nppiMirror_8u_C1R(d_src, step, d_dst, step, 
                                         size, NPP_VERTICAL_AXIS);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    cudaMemcpy(h_dst.data(), d_dst, height * step, cudaMemcpyDeviceToHost);

    // 验证结果 - 第一列应该变成最后一列
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            EXPECT_EQ(h_dst[y * width + x], 
                      h_src[y * width + (width - 1 - x)]);
        }
    }
}

TEST_F(GeometryTest, Mirror_BothAxis) {
    // 创建特定的测试图案
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_src[y * width + x] = y * width + x;
        }
    }

    // 上传数据
    cudaMemcpy(d_src, h_src.data(), height * step, cudaMemcpyHostToDevice);

    // 执行双轴镜像（180度旋转）
    NppStatus status = nppiMirror_8u_C1R(d_src, step, d_dst, step, 
                                         size, NPP_BOTH_AXIS);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    cudaMemcpy(h_dst.data(), d_dst, height * step, cudaMemcpyDeviceToHost);

    // 验证结果 - 应该是180度旋转
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            EXPECT_EQ(h_dst[y * width + x], 
                      h_src[(height - 1 - y) * width + (width - 1 - x)]);
        }
    }
}

TEST_F(GeometryTest, Mirror_InPlace) {
    // 测试原地镜像操作
    for (int i = 0; i < width * height; i++) {
        h_src[i] = i;
    }

    // 上传数据
    cudaMemcpy(d_src, h_src.data(), height * step, cudaMemcpyHostToDevice);

    // 执行原地水平镜像
    NppStatus status = nppiMirror_8u_C1R(d_src, step, d_src, step, 
                                         size, NPP_HORIZONTAL_AXIS);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    cudaMemcpy(h_dst.data(), d_src, height * step, cudaMemcpyDeviceToHost);

    // 验证结果
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            EXPECT_EQ(h_dst[y * width + x], 
                      h_src[(height - 1 - y) * width + x]);
        }
    }
}

// 参数验证测试
TEST(GeometryParameterTest, NullPointerError) {
    NppiSize size = {10, 10};
    int step = 10;
    Npp8u dummy;
    
    EXPECT_EQ(nppiMirror_8u_C1R(nullptr, step, &dummy, step, size, NPP_HORIZONTAL_AXIS), 
              NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiMirror_8u_C1R(&dummy, step, nullptr, step, size, NPP_HORIZONTAL_AXIS), 
              NPP_NULL_POINTER_ERROR);
}

TEST(GeometryParameterTest, InvalidSize) {
    Npp8u dummy;
    int step = 10;
    
    // 测试无效宽度
    NppiSize size = {0, 10};
    EXPECT_EQ(nppiMirror_8u_C1R(&dummy, step, &dummy, step, size, NPP_HORIZONTAL_AXIS), 
              NPP_SIZE_ERROR);
    
    // 测试无效高度
    size = {10, 0};
    EXPECT_EQ(nppiMirror_8u_C1R(&dummy, step, &dummy, step, size, NPP_VERTICAL_AXIS), 
              NPP_SIZE_ERROR);
}

TEST(GeometryParameterTest, InvalidFlipAxis) {
    Npp8u dummy;
    int step = 10;
    NppiSize size = {10, 10};
    
    // 测试无效的flip参数
    EXPECT_EQ(nppiMirror_8u_C1R(&dummy, step, &dummy, step, size, (NppiAxis)999), 
              NPP_MIRROR_FLIP_ERROR);
}