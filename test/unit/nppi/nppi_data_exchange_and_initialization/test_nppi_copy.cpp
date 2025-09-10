/**
 * @file test_nppi_copy.cpp
 * @brief NPP 图像拷贝函数测试
 */

#include "../../framework/npp_test_base.h"

using namespace npp_functional_test;

class CopyFunctionalTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
    
    // Helper function to verify copy results
    template<typename T>
    bool verifyCopyResult(const std::vector<T>& src, const std::vector<T>& dst) {
        if (src.size() != dst.size()) return false;
        for (size_t i = 0; i < src.size(); i++) {
            if (src[i] != dst[i]) return false;
        }
        return true;
    }
};

// 测试8位单通道拷贝
TEST_F(CopyFunctionalTest, Copy_8u_C1R_Basic) {
    const int width = 32, height = 32;
    
    // 创建测试数据
    std::vector<Npp8u> srcData(width * height);
    for (int i = 0; i < width * height; i++) {
        srcData[i] = (Npp8u)(i % 256);
    }
    
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src.copyFromHost(srcData);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_8u_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_SUCCESS) << "nppiCopy_8u_C1R failed";
    
    // 验证结果
    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    
    EXPECT_TRUE(verifyCopyResult(srcData, resultData)) 
        << "Copy result does not match source data";
}

// 测试8位三通道拷贝
TEST_F(CopyFunctionalTest, Copy_8u_C3R_RGB) {
    const int width = 16, height = 16;
    
    // 创建RGB测试数据
    std::vector<Npp8u> srcData(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        srcData[i * 3 + 0] = (Npp8u)(i % 256);         // Red
        srcData[i * 3 + 1] = (Npp8u)((i * 2) % 256);   // Green
        srcData[i * 3 + 2] = (Npp8u)((i * 3) % 256);   // Blue
    }
    
    // 使用手动内存分配
    int srcStep = width * 3 * sizeof(Npp8u);
    int dstStep = width * 3 * sizeof(Npp8u);
    
    Npp8u* srcPtr = nppsMalloc_8u(width * height * 3);
    Npp8u* dstPtr = nppsMalloc_8u(width * height * 3);
    
    ASSERT_NE(srcPtr, nullptr);
    ASSERT_NE(dstPtr, nullptr);
    
    // 复制数据到GPU
    cudaMemcpy(srcPtr, srcData.data(), width * height * 3 * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_8u_C3R(
        srcPtr, srcStep,
        dstPtr, dstStep,
        roi);
    
    ASSERT_EQ(status, NPP_SUCCESS) << "nppiCopy_8u_C3R failed";
    
    // 验证结果
    std::vector<Npp8u> resultData(width * height * 3);
    cudaMemcpy(resultData.data(), dstPtr, width * height * 3 * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    EXPECT_TRUE(verifyCopyResult(srcData, resultData)) 
        << "RGB copy result does not match source data";
    
    // 清理内存
    nppsFree(srcPtr);
    nppsFree(dstPtr);
}

// 测试8位四通道拷贝
TEST_F(CopyFunctionalTest, Copy_8u_C4R_RGBA) {
    const int width = 8, height = 8;
    
    // 创建RGBA测试数据
    std::vector<Npp8u> srcData(width * height * 4);
    for (int i = 0; i < width * height; i++) {
        srcData[i * 4 + 0] = (Npp8u)(i % 256);         // Red
        srcData[i * 4 + 1] = (Npp8u)((i * 2) % 256);   // Green
        srcData[i * 4 + 2] = (Npp8u)((i * 3) % 256);   // Blue
        srcData[i * 4 + 3] = (Npp8u)((i * 4) % 256);   // Alpha
    }
    
    // 使用手动内存分配
    int srcStep = width * 4 * sizeof(Npp8u);
    int dstStep = width * 4 * sizeof(Npp8u);
    
    Npp8u* srcPtr = nppsMalloc_8u(width * height * 4);
    Npp8u* dstPtr = nppsMalloc_8u(width * height * 4);
    
    ASSERT_NE(srcPtr, nullptr);
    ASSERT_NE(dstPtr, nullptr);
    
    // 复制数据到GPU
    cudaMemcpy(srcPtr, srcData.data(), width * height * 4 * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_8u_C4R(
        srcPtr, srcStep,
        dstPtr, dstStep,
        roi);
    
    ASSERT_EQ(status, NPP_SUCCESS) << "nppiCopy_8u_C4R failed";
    
    // 验证结果
    std::vector<Npp8u> resultData(width * height * 4);
    cudaMemcpy(resultData.data(), dstPtr, width * height * 4 * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    EXPECT_TRUE(verifyCopyResult(srcData, resultData)) 
        << "RGBA copy result does not match source data";
    
    // 清理内存
    nppsFree(srcPtr);
    nppsFree(dstPtr);
}

// 测试32位浮点拷贝
TEST_F(CopyFunctionalTest, Copy_32f_C1R_Float) {
    const int width = 16, height = 16;
    
    // 创建浮点测试数据
    std::vector<Npp32f> srcData(width * height);
    for (int i = 0; i < width * height; i++) {
        srcData[i] = sinf(i * 0.1f) * cosf(i * 0.05f); // 复杂浮点模式
    }
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    src.copyFromHost(srcData);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_32f_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_SUCCESS) << "nppiCopy_32f_C1R failed";
    
    // 验证结果
    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    
    // 检查浮点精度
    bool isEqual = true;
    for (size_t i = 0; i < srcData.size(); i++) {
        if (fabsf(srcData[i] - resultData[i]) > 1e-6f) {
            isEqual = false;
            break;
        }
    }
    
    EXPECT_TRUE(isEqual) << "Float copy result does not match source data";
}

// 测试不同ROI大小
TEST_F(CopyFunctionalTest, Copy_8u_C1R_DifferentROI) {
    const int width = 64, height = 64;
    
    // 创建大图像
    std::vector<Npp8u> srcData(width * height);
    for (int i = 0; i < width * height; i++) {
        srcData[i] = (Npp8u)(i % 256);
    }
    
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src.copyFromHost(srcData);
    
    // 测试不同的ROI大小
    std::vector<NppiSize> roiSizes = {
        {16, 16}, {32, 32}, {48, 48}, {64, 64}
    };
    
    for (const auto& roi : roiSizes) {
        NppStatus status = nppiCopy_8u_C1R(
            src.get(), src.step(),
            dst.get(), dst.step(),
            roi);
        
        ASSERT_EQ(status, NPP_SUCCESS) 
            << "nppiCopy_8u_C1R failed for ROI " << roi.width << "x" << roi.height;
        
        // 验证拷贝区域
        std::vector<Npp8u> resultData(width * height);
        dst.copyToHost(resultData);
        
        bool roiCorrect = true;
        for (int y = 0; y < roi.height; y++) {
            for (int x = 0; x < roi.width; x++) {
                int idx = y * width + x;
                if (srcData[idx] != resultData[idx]) {
                    roiCorrect = false;
                    break;
                }
            }
            if (!roiCorrect) break;
        }
        
        EXPECT_TRUE(roiCorrect) 
            << "ROI copy incorrect for size " << roi.width << "x" << roi.height;
    }
}

// 错误处理测试
TEST_F(CopyFunctionalTest, Copy_ErrorHandling) {
    const int width = 16, height = 16;
    
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    NppiSize roi = {width, height};
    
    // 测试空指针
    NppStatus status = nppiCopy_8u_C1R(
        nullptr, src.step(),
        dst.get(), dst.step(),
        roi);
    EXPECT_NE(status, NPP_SUCCESS) << "Should fail with null source pointer";
    
    status = nppiCopy_8u_C1R(
        src.get(), src.step(),
        nullptr, dst.step(),
        roi);
    EXPECT_NE(status, NPP_SUCCESS) << "Should fail with null destination pointer";
    
    // 测试无效ROI
    NppiSize invalidRoi = {0, 0};
    status = nppiCopy_8u_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        invalidRoi);
    EXPECT_NE(status, NPP_SUCCESS) << "Should fail with invalid ROI";
    
    // 测试负步长
    status = nppiCopy_8u_C1R(
        src.get(), -1,
        dst.get(), dst.step(),
        roi);
    EXPECT_NE(status, NPP_SUCCESS) << "Should fail with negative source step";
    
    status = nppiCopy_8u_C1R(
        src.get(), src.step(),
        dst.get(), -1,
        roi);
    EXPECT_NE(status, NPP_SUCCESS) << "Should fail with negative destination step";
}

// 性能基准测试
TEST_F(CopyFunctionalTest, Copy_Performance_Benchmark) {
    const int width = 1024, height = 1024;
    
    std::vector<Npp8u> srcData(width * height, 128);
    
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src.copyFromHost(srcData);
    
    NppiSize roi = {width, height};
    
    // 预热
    for (int i = 0; i < 5; i++) {
        nppiCopy_8u_C1R(
            src.get(), src.step(),
            dst.get(), dst.step(),
            roi);
    }
    
    // 同步GPU确保准确计时
    cudaDeviceSynchronize();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 执行多次拷贝以获得稳定的测量结果
    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        NppStatus status = nppiCopy_8u_C1R(
            src.get(), src.step(),
            dst.get(), dst.step(),
            roi);
        ASSERT_EQ(status, NPP_SUCCESS) << "Copy failed in performance test";
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avgTime = duration.count() / (double)iterations;
    double throughput = (width * height * sizeof(Npp8u) * 2) / (avgTime * 1e-6) / (1024 * 1024 * 1024); // GB/s
    
    std::cout << "Copy Performance: " << avgTime << " μs per operation, "
              << throughput << " GB/s throughput" << std::endl;
    
    // 验证结果仍然正确
    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    EXPECT_TRUE(verifyCopyResult(srcData, resultData)) 
        << "Performance test result verification failed";
}