#include <gtest/gtest.h>
#include "npp.h"
#include <vector>

class GeometryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试数据
        width = 8;
        height = 8;
        size.width = width;
        size.height = height;

        // 使用NPP内置内存管理函数分配设备内存
        d_src = nppiMalloc_8u_C1(width, height, &step_src);
        d_dst = nppiMalloc_8u_C1(width, height, &step_dst);

        ASSERT_NE(d_src, nullptr) << "Failed to allocate src memory";
        ASSERT_NE(d_dst, nullptr) << "Failed to allocate dst memory";

        // prepare test data
        h_src.resize(width * height);
        h_dst.resize(width * height);
    }

    void TearDown() override {
        // 使用NPP内置函数释放内存
        if (d_src) nppiFree(d_src);
        if (d_dst) nppiFree(d_dst);
    }

    int width, height;
    int step_src, step_dst;
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
    cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width,
                                   width, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);

    // 执行水平轴镜像（上下翻转）
    NppStatus status = nppiMirror_8u_C1R(d_src, step_src, d_dst, step_dst, 
                                         size, NPP_HORIZONTAL_AXIS);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst,
                       width, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

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
    cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width,
                                   width, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);

    // 执行垂直轴镜像（左右翻转）
    NppStatus status = nppiMirror_8u_C1R(d_src, step_src, d_dst, step_dst, 
                                         size, NPP_VERTICAL_AXIS);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst,
                       width, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

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
    cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width,
                                   width, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);

    // 执行双轴镜像（180度旋转）
    NppStatus status = nppiMirror_8u_C1R(d_src, step_src, d_dst, step_dst, 
                                         size, NPP_BOTH_AXIS);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst,
                       width, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

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
    cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width,
                                   width, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);

    // 执行原地水平镜像
    NppStatus status = nppiMirror_8u_C1R(d_src, step_src, d_src, step_src, 
                                         size, NPP_HORIZONTAL_AXIS);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    err = cudaMemcpy2D(h_dst.data(), width, d_src, step_src,
                       width, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

    // 验证结果
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            EXPECT_EQ(h_dst[y * width + x], 
                      h_src[(height - 1 - y) * width + x]);
        }
    }
}

// 参数验证测试
// NOTE: 测试已被禁用 - NVIDIA NPP对无效参数的处理会污染CUDA上下文
// 传递栈上变量地址(&dummy)作为GPU内存指针会导致CUDA上下文错误
TEST(GeometryParameterTest, DISABLED_NullPointerError) {
    NppiSize size = {10, 10};
    int step = 10;
    Npp8u dummy;
    
    EXPECT_EQ(nppiMirror_8u_C1R(nullptr, step, &dummy, step, size, NPP_HORIZONTAL_AXIS), 
              NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiMirror_8u_C1R(&dummy, step, nullptr, step, size, NPP_HORIZONTAL_AXIS), 
              NPP_NULL_POINTER_ERROR);
}

// NOTE: 测试已被禁用 - NVIDIA NPP对无效参数的处理会污染CUDA上下文
// 传递栈上变量地址(&dummy)作为GPU内存指针会导致CUDA上下文错误
TEST(GeometryParameterTest, DISABLED_InvalidSize) {
    Npp8u dummy = 0;  // 初始化以避免警告
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

// NOTE: 测试已被禁用 - NVIDIA NPP对无效参数的处理会污染CUDA上下文
// 传递栈上变量地址(&dummy)作为GPU内存指针会导致CUDA上下文错误
TEST(GeometryParameterTest, DISABLED_InvalidFlipAxis) {
    Npp8u dummy = 0;  // 初始化以避免警告
    int step = 10;
    NppiSize size = {10, 10};
    
    // 测试无效的flip参数
    EXPECT_EQ(nppiMirror_8u_C1R(&dummy, step, &dummy, step, size, (NppiAxis)999), 
              NPP_MIRROR_FLIP_ERROR);
}