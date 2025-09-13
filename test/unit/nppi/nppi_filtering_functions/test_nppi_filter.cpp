#include <gtest/gtest.h>
#include "npp.h"
#include <vector>

class FilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试数据
        width = 16;
        height = 16;
        size.width = width;
        size.height = height;

        // 使用NPP内置内存管理函数分配设备内存
        d_src = nppiMalloc_8u_C1(width, height, &step_src);
        d_dst = nppiMalloc_8u_C1(width, height, &step_dst);

        ASSERT_NE(d_src, nullptr) << "Failed to allocate src memory";
        ASSERT_NE(d_dst, nullptr) << "Failed to allocate dst memory";

        // 准备测试数据
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

TEST_F(FilterTest, FilterBox_3x3_Uniform) {
    // 创建一个更简单的测试 - 使用全100的图像
    std::fill(h_src.begin(), h_src.end(), 100);

    // 上传数据
    cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width,
                                   width, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "Failed to upload source data";
    
    // 同步确保内存操作完成
    cudaDeviceSynchronize();

    // 设置3x3 box滤波器，锚点在中心
    NppiSize maskSize = {3, 3};
    NppiPoint anchor = {1, 1};

    // 执行滤波
    NppStatus status = nppiFilterBox_8u_C1R(d_src, step_src, d_dst, step_dst, 
                                           size, maskSize, anchor);
    if (status != NPP_SUCCESS) {
        // 获取更多错误信息
        cudaError_t lastError = cudaGetLastError();
        ASSERT_EQ(status, NPP_SUCCESS) << "NPP Status: " << status 
                                       << ", CUDA Error: " << cudaGetErrorString(lastError);
    }

    // 下载结果
    err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst,
                       width, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

    // 验证结果 - 内部区域应该保持100（均匀图像的box滤波结果）
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            EXPECT_EQ(h_dst[y * width + x], 100) << "At position (" << x << ", " << y << ")";
        }
    }
}

TEST_F(FilterTest, FilterBox_5x5_EdgeHandling) {
    // 创建一个全白图像
    std::fill(h_src.begin(), h_src.end(), 100);

    // 上传数据
    cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width,
                                   width, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);
    
    // 设置5x5 box滤波器
    NppiSize maskSize = {5, 5};
    NppiPoint anchor = {2, 2};

    // 执行滤波
    NppStatus status = nppiFilterBox_8u_C1R(d_src, step_src, d_dst, step_dst, 
                                           size, maskSize, anchor);
    ASSERT_EQ(status, NPP_SUCCESS);

    // 下载结果
    err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst,
                       width, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);

    // 验证内部区域 - 应该保持原值
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            EXPECT_EQ(h_dst[y * width + x], 100);
        }
    }

    // 验证边缘处理 - 由于边界效应，边缘的值可能略有不同
    // 角落像素只能访问部分邻域
    EXPECT_GT(h_dst[0], 0);  // 应该有一定的值
    EXPECT_LE(h_dst[0], 100);  // 但不应超过原始值
}

// 参数验证测试
TEST(FilterParameterTest, NullPointerError) {
    NppiSize size = {10, 10};
    NppiSize maskSize = {3, 3};
    NppiPoint anchor = {1, 1};
    int step = 10;
    Npp8u dummy;
    
    EXPECT_EQ(nppiFilterBox_8u_C1R(nullptr, step, &dummy, step, size, maskSize, anchor), 
              NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiFilterBox_8u_C1R(&dummy, step, nullptr, step, size, maskSize, anchor), 
              NPP_NULL_POINTER_ERROR);
}

TEST(FilterParameterTest, InvalidSize) {
    Npp8u dummy = 0;
    int step = 10;
    NppiSize maskSize = {3, 3};
    NppiPoint anchor = {1, 1};
    
    // 测试无效ROI大小
    NppiSize size = {0, 10};
    EXPECT_EQ(nppiFilterBox_8u_C1R(&dummy, step, &dummy, step, size, maskSize, anchor), 
              NPP_SIZE_ERROR);
    
    size = {10, 0};
    EXPECT_EQ(nppiFilterBox_8u_C1R(&dummy, step, &dummy, step, size, maskSize, anchor), 
              NPP_SIZE_ERROR);
}

TEST(FilterParameterTest, InvalidMaskSize) {
    Npp8u dummy = 0;
    int step = 10;
    NppiSize size = {10, 10};
    NppiPoint anchor = {1, 1};
    
    // 测试无效掩码大小
    NppiSize maskSize = {0, 3};
    EXPECT_EQ(nppiFilterBox_8u_C1R(&dummy, step, &dummy, step, size, maskSize, anchor), 
              NPP_MASK_SIZE_ERROR);
    
    // 掩码大于图像
    maskSize = {20, 20};
    EXPECT_EQ(nppiFilterBox_8u_C1R(&dummy, step, &dummy, step, size, maskSize, anchor), 
              NPP_MASK_SIZE_ERROR);
}

TEST(FilterParameterTest, InvalidAnchor) {
    Npp8u dummy = 0;
    int step = 10;
    NppiSize size = {10, 10};
    NppiSize maskSize = {3, 3};
    
    // 测试无效锚点
    NppiPoint anchor = {-1, 1};
    EXPECT_EQ(nppiFilterBox_8u_C1R(&dummy, step, &dummy, step, size, maskSize, anchor), 
              NPP_ANCHOR_ERROR);
    
    anchor = {3, 1};  // 超出掩码范围
    EXPECT_EQ(nppiFilterBox_8u_C1R(&dummy, step, &dummy, step, size, maskSize, anchor), 
              NPP_ANCHOR_ERROR);
}