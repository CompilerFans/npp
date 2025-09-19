#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

/**
 * @file test_nppi_copy_extended.cpp
 * @brief 测试新增的nppiCopy API功能
 */

class CopyExtendedFunctionalTest : public ::testing::Test {
protected:
    void SetUp() override {
        width = 8;
        height = 6;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

// 测试nppiCopy_32f_C3R函数
TEST_F(CopyExtendedFunctionalTest, Copy_32f_C3R_BasicOperation) {
    const int channels = 3;
    std::vector<Npp32f> srcData(width * height * channels);
    
    // 创建测试数据：RGB渐变模式  
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            srcData[idx + 0] = (float)x / (width - 1);   // R通道: 0.0-1.0渐变
            srcData[idx + 1] = (float)y / (height - 1);  // G通道: 0.0-1.0渐变
            srcData[idx + 2] = 0.5f;                     // B通道: 固定0.5
        }
    }
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制输入数据到GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行复制
    NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 复制结果回主机
    std::vector<Npp32f> resultData(width * height * channels);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * channels,
                   (char*)d_dst + y * dstStep,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 验证结果：应该完全一致
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_FLOAT_EQ(resultData[i], srcData[i]) 
            << "复制失败 at index " << i << ": got " << resultData[i] 
            << ", expected " << srcData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试nppiCopy_32f_C3P3R函数（packed到planar转换）
TEST_F(CopyExtendedFunctionalTest, Copy_32f_C3P3R_BasicOperation) {
    const int channels = 3;
    
    // 创建packed格式的源数据（RGB交错）
    std::vector<Npp32f> srcData(width * height * channels);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            srcData[idx + 0] = (float)(x + y) / (width + height - 2);  // R通道
            srcData[idx + 1] = (float)x / (width - 1);                 // G通道
            srcData[idx + 2] = (float)y / (height - 1);                // B通道
        }
    }
    
    // 分配GPU内存 - 源是packed格式，目标是planar格式
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dstR = nppiMalloc_32f_C1(width, height, &dstStep);
    Npp32f* d_dstG = nppiMalloc_32f_C1(width, height, &dstStep);
    Npp32f* d_dstB = nppiMalloc_32f_C1(width, height, &dstStep);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dstR, nullptr);
    ASSERT_NE(d_dstG, nullptr);
    ASSERT_NE(d_dstB, nullptr);
    
    // 复制packed源数据到GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 设置目标指针数组
    Npp32f* dstPtrs[3] = {d_dstR, d_dstG, d_dstB};
    
    // 执行packed到planar转换
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, dstPtrs, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultR(width * height);
    std::vector<Npp32f> resultG(width * height);
    std::vector<Npp32f> resultB(width * height);
    
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultR.data() + y * width, (char*)d_dstR + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        cudaMemcpy(resultG.data() + y * width, (char*)d_dstG + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        cudaMemcpy(resultB.data() + y * width, (char*)d_dstB + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    }
    
    // 验证每个通道的数据
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int packedIdx = (y * width + x) * channels;
            int planarIdx = y * width + x;
            
            EXPECT_FLOAT_EQ(resultR[planarIdx], srcData[packedIdx + 0]) 
                << "R通道转换失败 at (" << x << "," << y << ")";
            EXPECT_FLOAT_EQ(resultG[planarIdx], srcData[packedIdx + 1]) 
                << "G通道转换失败 at (" << x << "," << y << ")";
            EXPECT_FLOAT_EQ(resultB[planarIdx], srcData[packedIdx + 2]) 
                << "B通道转换失败 at (" << x << "," << y << ")";
        }
    }
    
    // 清理内存
    nppiFree(d_src);
    nppiFree(d_dstR);
    nppiFree(d_dstG);
    nppiFree(d_dstB);
}

// 测试边界条件
TEST_F(CopyExtendedFunctionalTest, Copy_32f_C3R_BoundaryValues) {
    std::vector<Npp32f> srcData = {
        -1.0f, 0.0f, 1.0f,           // 负值，零，正值
        3.14159f, -2.71828f, 0.5f,   // 数学常数
        1e6f, -1e-6f, 1e-38f         // 极值
    };
    
    NppiSize testRoi = {3, 1};
    
    // 分配GPU内存
    int step = 3 * 3 * sizeof(Npp32f);
    Npp32f* d_src = nppsMalloc_32f(3 * 3);
    Npp32f* d_dst = nppsMalloc_32f(3 * 3);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据
    cudaMemcpy(d_src, srcData.data(), srcData.size() * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // 执行复制
    NppStatus status = nppiCopy_32f_C3R(d_src, step, d_dst, step, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(3 * 3);
    cudaMemcpy(resultData.data(), d_dst, resultData.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < srcData.size(); i++) {
        EXPECT_FLOAT_EQ(resultData[i], srcData[i]) 
            << "边界值复制失败 at index " << i;
    }
    
    nppsFree(d_src);
    nppsFree(d_dst);
}

// 测试流上下文版本
TEST_F(CopyExtendedFunctionalTest, Copy_32f_C3R_StreamContext) {
    const int channels = 3;
    std::vector<Npp32f> srcData(width * height * channels, 1.5f); // 所有像素值为1.5f
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 创建流上下文
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    // 执行复制
    NppStatus status = nppiCopy_32f_C3R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(width * height * channels);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * channels,
                   (char*)d_dst + y * dstStep,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 所有值应该都是1.5f
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_FLOAT_EQ(resultData[i], 1.5f) 
            << "流上下文测试失败 at index " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 参数化测试：测试不同的图像尺寸
struct ImageSizeParams {
    int width;
    int height;
    std::string description;
};

class CopyParameterizedTest : public ::testing::TestWithParam<ImageSizeParams> {
protected:
    void SetUp() override {
        auto params = GetParam();
        width = params.width;
        height = params.height;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

// 定义测试参数
INSTANTIATE_TEST_SUITE_P(
    DifferentImageSizes,
    CopyParameterizedTest,
    ::testing::Values(
        ImageSizeParams{1, 1, "SinglePixel"},
        ImageSizeParams{2, 3, "SmallRect"},
        ImageSizeParams{4, 4, "SmallSquare"},
        ImageSizeParams{7, 5, "OddSize"},
        ImageSizeParams{16, 8, "Rect16x8"},
        ImageSizeParams{32, 32, "Square32x32"},
        ImageSizeParams{64, 16, "WideRect"},
        ImageSizeParams{8, 64, "TallRect"},
        ImageSizeParams{128, 96, "MediumSize"},
        ImageSizeParams{8192, 4096, "Large"}
    ),
    [](const ::testing::TestParamInfo<ImageSizeParams>& info) {
        return info.param.description + "_" + 
               std::to_string(info.param.width) + "x" + 
               std::to_string(info.param.height);
    }
);

// 参数化测试nppiCopy_32f_C3R
TEST_P(CopyParameterizedTest, Copy_32f_C3R_ParameterizedSizes) {
    const int channels = 3;
    std::vector<Npp32f> srcData(width * height * channels);
    
    // 创建测试数据：每个像素用坐标编码
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            srcData[idx + 0] = (float)(x + y * 100);        // R: 位置编码
            srcData[idx + 1] = (float)(x * y + 1000);       // G: 乘积编码
            srcData[idx + 2] = (float)(x - y + 2000);       // B: 差值编码
        }
    }
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据到GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行拷贝
    NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(width * height * channels);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * channels,
                   (char*)d_dst + y * dstStep,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_FLOAT_EQ(resultData[i], srcData[i]) 
            << "参数化测试失败 at index " << i << " for size " << width << "x" << height;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 数据模式参数化测试
struct DataPatternParams {
    std::string name;
    std::function<float(int, int, int, int, int)> generator; // x, y, width, height, channel
};

class CopyDataPatternTest : public ::testing::TestWithParam<DataPatternParams> {
protected:
    void SetUp() override {
        width = 16;
        height = 12;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    DifferentDataPatterns,
    CopyDataPatternTest,
    ::testing::Values(
        DataPatternParams{
            "Constant",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)x; (void)y; (void)w; (void)h; (void)c;
                return 42.0f; 
            }
        },
        DataPatternParams{
            "HorizontalGrad",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)y; (void)h; (void)c;
                return (float)x / (w - 1); 
            }
        },
        DataPatternParams{
            "VerticalGrad", 
            [](int x, int y, int w, int h, int c) -> float { 
                (void)x; (void)w; (void)c;
                return (float)y / (h - 1); 
            }
        },
        DataPatternParams{
            "Checkerboard",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)w; (void)h; (void)c;
                return ((x + y) % 2) ? 1.0f : 0.0f; 
            }
        },
        DataPatternParams{
            "DiagonalGrad",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)c;
                return (float)(x + y) / (w + h - 2); 
            }
        },
        DataPatternParams{
            "ChannelRelated",
            [](int x, int y, int w, int h, int c) -> float { 
                return (float)(c + 1) * ((float)x / w + (float)y / h); 
            }
        },
        DataPatternParams{
            "SineWave",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)c;
                return sin((float)x * 3.14159f / w) * cos((float)y * 3.14159f / h); 
            }
        }
    ),
    [](const ::testing::TestParamInfo<DataPatternParams>& info) {
        return info.param.name;
    }
);

// 参数化测试不同数据模式的nppiCopy_32f_C3R
TEST_P(CopyDataPatternTest, Copy_32f_C3R_DataPatterns) {
    auto params = GetParam();
    const int channels = 3;
    std::vector<Npp32f> srcData(width * height * channels);
    
    // 使用参数化的数据生成器
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                srcData[idx + c] = params.generator(x, y, width, height, c);
            }
        }
    }
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据到GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行拷贝
    NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(width * height * channels);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * channels,
                   (char*)d_dst + y * dstStep,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_FLOAT_EQ(resultData[i], srcData[i]) 
            << "数据模式测试失败 at index " << i << " for pattern: " << params.name;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Enhanced parameterized tests for nppiCopy_32f_C3R
class CopyC3REnhancedTest : public ::testing::TestWithParam<ImageSizeParams> {
protected:
    void SetUp() override {
        auto params = GetParam();
        width = params.width;
        height = params.height;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    C3REnhancedSizes,
    CopyC3REnhancedTest,
    ::testing::Values(
        ImageSizeParams{1, 1, "SinglePixel"},
        ImageSizeParams{2, 3, "SmallRect"},
        ImageSizeParams{3, 3, "SmallSquare"},
        ImageSizeParams{5, 7, "OddSize"},
        ImageSizeParams{16, 8, "Rect16x8"},
        ImageSizeParams{32, 32, "Square32x32"},
        ImageSizeParams{64, 16, "WideRect"},
        ImageSizeParams{8, 64, "TallRect"},
        ImageSizeParams{128, 96, "MediumSize"},
        ImageSizeParams{256, 192, "LargeSize"},
        ImageSizeParams{8192, 4096, "Large"}
    ),
    [](const ::testing::TestParamInfo<ImageSizeParams>& info) {
        return info.param.description + "_" + 
               std::to_string(info.param.width) + "x" + 
               std::to_string(info.param.height);
    }
);

TEST_P(CopyC3REnhancedTest, Copy_32f_C3R_EnhancedSizes) {
    const int channels = 3;
    std::vector<Npp32f> srcData(width * height * channels);
    
    // Create RGB channel-specific test data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            srcData[idx + 0] = (float)(x + y * 1000.0f);        // R: position encoding
            srcData[idx + 1] = (float)(x * y + 10000.0f);       // G: product encoding
            srcData[idx + 2] = (float)((x ^ y) + 20000.0f);     // B: XOR encoding
        }
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy data to GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute copy
    NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify result
    std::vector<Npp32f> resultData(width * height * channels);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * channels,
                   (char*)d_dst + y * dstStep,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_FLOAT_EQ(resultData[i], srcData[i]) 
            << "Enhanced C3R copy failed at index " << i << " for size " << width << "x" << height;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Data pattern tests for nppiCopy_32f_C3R
class CopyC3RDataPatternTest : public ::testing::TestWithParam<DataPatternParams> {
protected:
    void SetUp() override {
        width = 16;
        height = 12;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    C3RDataPatterns,
    CopyC3RDataPatternTest,
    ::testing::Values(
        DataPatternParams{
            "Constant",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)x; (void)y; (void)w; (void)h;
                return (float)(c + 1) * 100.0f; 
            }
        },
        DataPatternParams{
            "RGBGradients",
            [](int x, int y, int w, int h, int c) -> float { 
                if (c == 0) return (float)x / (w - 1) * 255.0f;        // R: horizontal gradient
                if (c == 1) return (float)y / (h - 1) * 255.0f;        // G: vertical gradient
                return (float)(x + y) / (w + h - 2) * 255.0f;          // B: diagonal gradient
            }
        },
        DataPatternParams{
            "RGBCheckers",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)w; (void)h;
                bool checker = ((x + y) % 2) == 0;
                if (c == 0) return checker ? 255.0f : 0.0f;            // R: checkerboard
                if (c == 1) return checker ? 0.0f : 255.0f;            // G: inverted checkerboard
                return checker == ((x + y + 1) % 2) ? 128.0f : 64.0f;  // B: offset pattern
            }
        },
        DataPatternParams{
            "RGBSines",
            [](int x, int y, int w, int h, int c) -> float { 
                float pi = 3.14159f;
                if (c == 0) return sin((float)x * pi / w) * 127.5f + 127.5f;           // R: horizontal sine
                if (c == 1) return cos((float)y * pi / h) * 127.5f + 127.5f;           // G: vertical cosine
                return sin((float)(x + y) * pi / (w + h)) * 127.5f + 127.5f;           // B: diagonal sine
            }
        },
        DataPatternParams{
            "RGBCircles",
            [](int x, int y, int w, int h, int c) -> float { 
                float cx = w / 2.0f, cy = h / 2.0f;
                float dx = x - cx, dy = y - cy;
                float dist = sqrt(dx*dx + dy*dy);
                float maxDist = sqrt(cx*cx + cy*cy);
                if (c == 0) return (1.0f - dist / maxDist) * 255.0f;                   // R: radial gradient
                if (c == 1) return sin(dist * 0.5f) * 127.5f + 127.5f;                 // G: concentric circles
                return cos(dist * 0.3f + c) * 127.5f + 127.5f;                         // B: phase-shifted circles
            }
        },
        DataPatternParams{
            "RGBNoise",
            [](int x, int y, int w, int h, int c) -> float { 
                // Deterministic pseudo-random based on position and channel
                unsigned int seed = (unsigned int)(x * 1000 + y * 100 + c * 10 + w + h);
                seed = seed * 1103515245 + 12345;
                return (float)(seed % 256);
            }
        },
        DataPatternParams{
            "RGBBoundaries",
            [](int x, int y, int w, int h, int c) -> float { 
                bool isEdge = (x == 0 || y == 0 || x == w-1 || y == h-1);
                bool isCorner = (x == 0 || x == w-1) && (y == 0 || y == h-1);
                if (c == 0) return isCorner ? 255.0f : (isEdge ? 128.0f : 64.0f);      // R: corner/edge detection
                if (c == 1) return isEdge ? 255.0f : 0.0f;                             // G: edge detection
                return (float)((x + y) % 256);                                         // B: position-based
            }
        }
    ),
    [](const ::testing::TestParamInfo<DataPatternParams>& info) {
        return info.param.name;
    }
);

TEST_P(CopyC3RDataPatternTest, Copy_32f_C3R_DataPatterns) {
    auto params = GetParam();
    const int channels = 3;
    std::vector<Npp32f> srcData(width * height * channels);
    
    // Generate RGB test data using pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                srcData[idx + c] = params.generator(x, y, width, height, c);
            }
        }
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy data to GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute copy
    NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify result
    std::vector<Npp32f> resultData(width * height * channels);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * channels,
                   (char*)d_dst + y * dstStep,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_FLOAT_EQ(resultData[i], srcData[i]) 
            << "C3R pattern test failed at index " << i << " for pattern: " << params.name;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}


// Stream context tests for nppiCopy_32f_C3R
class CopyC3RStreamTest : public ::testing::TestWithParam<ImageSizeParams> {
protected:
    void SetUp() override {
        auto params = GetParam();
        width = params.width;
        height = params.height;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    C3RStreamSizes,
    CopyC3RStreamTest,
    ::testing::Values(
        ImageSizeParams{4, 4, "Small"},
        ImageSizeParams{16, 16, "Medium"},
        ImageSizeParams{64, 64, "Large"},
        ImageSizeParams{4096, 2048, "Large2"}
    ),
    [](const ::testing::TestParamInfo<ImageSizeParams>& info) {
        return info.param.description + "_" + 
               std::to_string(info.param.width) + "x" + 
               std::to_string(info.param.height);
    }
);

TEST_P(CopyC3RStreamTest, Copy_32f_C3R_StreamContext) {
    const int channels = 3;
    std::vector<Npp32f> srcData(width * height * channels);
    
    // Create test data with RGB patterns
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            srcData[idx + 0] = (float)(x % 256);          // R: horizontal pattern
            srcData[idx + 1] = (float)(y % 256);          // G: vertical pattern
            srcData[idx + 2] = (float)((x + y) % 256);    // B: diagonal pattern
        }
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy data to GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // Create stream context
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    // Execute copy with stream context
    NppStatus status = nppiCopy_32f_C3R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify result
    std::vector<Npp32f> resultData(width * height * channels);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * channels,
                   (char*)d_dst + y * dstStep,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_FLOAT_EQ(resultData[i], srcData[i]) 
            << "C3R stream context test failed at index " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Parameterized tests for nppiCopy_32f_C1R
class CopyC1RParameterizedTest : public ::testing::TestWithParam<ImageSizeParams> {
protected:
    void SetUp() override {
        auto params = GetParam();
        width = params.width;
        height = params.height;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    C1RImageSizes,
    CopyC1RParameterizedTest,
    ::testing::Values(
        ImageSizeParams{1, 1, "SinglePixel"},
        ImageSizeParams{2, 3, "SmallRect"},
        ImageSizeParams{4, 4, "SmallSquare"},
        ImageSizeParams{7, 5, "OddSize"},
        ImageSizeParams{16, 8, "Rect16x8"},
        ImageSizeParams{32, 32, "Square32x32"},
        ImageSizeParams{64, 16, "WideRect"},
        ImageSizeParams{8, 64, "TallRect"},
        ImageSizeParams{128, 96, "MediumSize"},
        ImageSizeParams{8192, 4096, "Large"}
    ),
    [](const ::testing::TestParamInfo<ImageSizeParams>& info) {
        return info.param.description + "_" + 
               std::to_string(info.param.width) + "x" + 
               std::to_string(info.param.height);
    }
);

TEST_P(CopyC1RParameterizedTest, Copy_32f_C1R_ParameterizedSizes) {
    std::vector<Npp32f> srcData(width * height);
    
    // Create position-encoded test data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            srcData[idx] = (float)(x + y * 100.0f + 1000.0f);
        }
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy data to GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width,
                   width * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute copy
    NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify result
    std::vector<Npp32f> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < width * height; i++) {
        EXPECT_FLOAT_EQ(resultData[i], srcData[i]) 
            << "C1R copy failed at index " << i << " for size " << width << "x" << height;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Data pattern tests for nppiCopy_32f_C1R
class CopyC1RDataPatternTest : public ::testing::TestWithParam<DataPatternParams> {
protected:
    void SetUp() override {
        width = 16;
        height = 12;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    C1RDataPatterns,
    CopyC1RDataPatternTest,
    ::testing::Values(
        DataPatternParams{
            "Constant",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)x; (void)y; (void)w; (void)h; (void)c;
                return 42.0f; 
            }
        },
        DataPatternParams{
            "HorizontalGrad",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)y; (void)h; (void)c;
                return (float)x / (w - 1); 
            }
        },
        DataPatternParams{
            "VerticalGrad", 
            [](int x, int y, int w, int h, int c) -> float { 
                (void)x; (void)w; (void)c;
                return (float)y / (h - 1); 
            }
        },
        DataPatternParams{
            "Checkerboard",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)w; (void)h; (void)c;
                return ((x + y) % 2) ? 1.0f : 0.0f; 
            }
        },
        DataPatternParams{
            "DiagonalGrad",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)c;
                return (float)(x + y) / (w + h - 2); 
            }
        },
        DataPatternParams{
            "CirclePattern",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)c;
                float cx = w / 2.0f, cy = h / 2.0f;
                float dx = x - cx, dy = y - cy;
                float dist = sqrt(dx*dx + dy*dy);
                return sin(dist * 0.5f); 
            }
        }
    ),
    [](const ::testing::TestParamInfo<DataPatternParams>& info) {
        return info.param.name;
    }
);

TEST_P(CopyC1RDataPatternTest, Copy_32f_C1R_DataPatterns) {
    auto params = GetParam();
    std::vector<Npp32f> srcData(width * height);
    
    // Generate test data using pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            srcData[idx] = params.generator(x, y, width, height, 0);
        }
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy data to GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width,
                   width * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute copy
    NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify result
    std::vector<Npp32f> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < width * height; i++) {
        EXPECT_FLOAT_EQ(resultData[i], srcData[i]) 
            << "C1R pattern test failed at index " << i << " for pattern: " << params.name;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Parameterized tests for nppiCopy_32f_C3P3R  
class CopyC3P3RParameterizedTest : public ::testing::TestWithParam<ImageSizeParams> {
protected:
    void SetUp() override {
        auto params = GetParam();
        width = params.width;
        height = params.height;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    C3P3RImageSizes,
    CopyC3P3RParameterizedTest,
    ::testing::Values(
        ImageSizeParams{1, 1, "SinglePixel"},
        ImageSizeParams{2, 3, "SmallRect"},
        ImageSizeParams{4, 4, "SmallSquare"},
        ImageSizeParams{7, 5, "OddSize"},
        ImageSizeParams{16, 8, "Rect16x8"},
        ImageSizeParams{32, 32, "Square32x32"},
        ImageSizeParams{64, 16, "WideRect"},
        ImageSizeParams{8, 64, "TallRect"},
        ImageSizeParams{128, 96, "MediumSize"},
        ImageSizeParams{8192, 4096, "Large"}
    ),
    [](const ::testing::TestParamInfo<ImageSizeParams>& info) {
        return info.param.description + "_" + 
               std::to_string(info.param.width) + "x" + 
               std::to_string(info.param.height);
    }
);

TEST_P(CopyC3P3RParameterizedTest, Copy_32f_C3P3R_ParameterizedSizes) {
    const int channels = 3;
    std::vector<Npp32f> srcData(width * height * channels);
    
    // Create channel-encoded packed test data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            srcData[idx + 0] = (float)(x + y * 200.0f);       // R channel
            srcData[idx + 1] = (float)(y + x * 300.0f);       // G channel  
            srcData[idx + 2] = (float)(x * y + 1000.0f);      // B channel
        }
    }
    
    // Allocate GPU memory - source is packed, dest is planar
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dstR = nppiMalloc_32f_C1(width, height, &dstStep);
    Npp32f* d_dstG = nppiMalloc_32f_C1(width, height, &dstStep);
    Npp32f* d_dstB = nppiMalloc_32f_C1(width, height, &dstStep);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dstR, nullptr);
    ASSERT_NE(d_dstG, nullptr);
    ASSERT_NE(d_dstB, nullptr);
    
    // Copy packed source data to GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // Setup destination pointer array
    Npp32f* dstPtrs[3] = {d_dstR, d_dstG, d_dstB};
    
    // Execute packed to planar conversion
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, dstPtrs, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify results
    std::vector<Npp32f> resultR(width * height);
    std::vector<Npp32f> resultG(width * height);
    std::vector<Npp32f> resultB(width * height);
    
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultR.data() + y * width, (char*)d_dstR + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        cudaMemcpy(resultG.data() + y * width, (char*)d_dstG + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        cudaMemcpy(resultB.data() + y * width, (char*)d_dstB + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    }
    
    // Verify each channel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int packedIdx = (y * width + x) * channels;
            int planarIdx = y * width + x;
            
            EXPECT_FLOAT_EQ(resultR[planarIdx], srcData[packedIdx + 0]) 
                << "R channel failed at (" << x << "," << y << ") for size " << width << "x" << height;
            EXPECT_FLOAT_EQ(resultG[planarIdx], srcData[packedIdx + 1]) 
                << "G channel failed at (" << x << "," << y << ") for size " << width << "x" << height;
            EXPECT_FLOAT_EQ(resultB[planarIdx], srcData[packedIdx + 2]) 
                << "B channel failed at (" << x << "," << y << ") for size " << width << "x" << height;
        }
    }
    
    nppiFree(d_src);
    nppiFree(d_dstR);
    nppiFree(d_dstG);
    nppiFree(d_dstB);
}

// Data pattern tests for nppiCopy_32f_C3P3R
class CopyC3P3RDataPatternTest : public ::testing::TestWithParam<DataPatternParams> {
protected:
    void SetUp() override {
        width = 16;
        height = 12;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    C3P3RDataPatterns,
    CopyC3P3RDataPatternTest,
    ::testing::Values(
        DataPatternParams{
            "Constant",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)x; (void)y; (void)w; (void)h;
                return (float)(c + 1) * 50.0f; 
            }
        },
        DataPatternParams{
            "ChannelGradients",
            [](int x, int y, int w, int h, int c) -> float { 
                if (c == 0) return (float)x / (w - 1);
                if (c == 1) return (float)y / (h - 1);
                return (float)(x + y) / (w + h - 2);
            }
        },
        DataPatternParams{
            "ChannelCheckers",
            [](int x, int y, int w, int h, int c) -> float { 
                (void)w; (void)h;
                return ((x + y + c) % 2) ? 255.0f : 0.0f; 
            }
        },
        DataPatternParams{
            "ChannelSines",
            [](int x, int y, int w, int h, int c) -> float { 
                float freq = (c + 1) * 0.5f;
                return sin((float)x * freq * 3.14159f / w) * cos((float)y * freq * 3.14159f / h); 
            }
        },
        DataPatternParams{
            "ChannelScaled",
            [](int x, int y, int w, int h, int c) -> float { 
                return (float)(c + 1) * ((float)x / w + (float)y / h) * 100.0f; 
            }
        },
        DataPatternParams{
            "ChannelCircles",
            [](int x, int y, int w, int h, int c) -> float { 
                float cx = w / 2.0f, cy = h / 2.0f;
                float dx = x - cx, dy = y - cy;
                float dist = sqrt(dx*dx + dy*dy);
                return sin(dist * (c + 1) * 0.3f) * 128.0f + 128.0f; 
            }
        }
    ),
    [](const ::testing::TestParamInfo<DataPatternParams>& info) {
        return info.param.name;
    }
);

TEST_P(CopyC3P3RDataPatternTest, Copy_32f_C3P3R_DataPatterns) {
    auto params = GetParam();
    const int channels = 3;
    std::vector<Npp32f> srcData(width * height * channels);
    
    // Generate multi-channel test data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                srcData[idx + c] = params.generator(x, y, width, height, c);
            }
        }
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dstR = nppiMalloc_32f_C1(width, height, &dstStep);
    Npp32f* d_dstG = nppiMalloc_32f_C1(width, height, &dstStep);
    Npp32f* d_dstB = nppiMalloc_32f_C1(width, height, &dstStep);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dstR, nullptr);
    ASSERT_NE(d_dstG, nullptr);
    ASSERT_NE(d_dstB, nullptr);
    
    // Copy data to GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // Setup destination pointer array
    Npp32f* dstPtrs[3] = {d_dstR, d_dstG, d_dstB};
    
    // Execute conversion
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, dstPtrs, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify results
    std::vector<Npp32f> resultR(width * height);
    std::vector<Npp32f> resultG(width * height);
    std::vector<Npp32f> resultB(width * height);
    
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultR.data() + y * width, (char*)d_dstR + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        cudaMemcpy(resultG.data() + y * width, (char*)d_dstG + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        cudaMemcpy(resultB.data() + y * width, (char*)d_dstB + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    }
    
    // Verify each channel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int packedIdx = (y * width + x) * channels;
            int planarIdx = y * width + x;
            
            EXPECT_FLOAT_EQ(resultR[planarIdx], srcData[packedIdx + 0]) 
                << "R channel pattern test failed at (" << x << "," << y << ") for pattern: " << params.name;
            EXPECT_FLOAT_EQ(resultG[planarIdx], srcData[packedIdx + 1]) 
                << "G channel pattern test failed at (" << x << "," << y << ") for pattern: " << params.name;
            EXPECT_FLOAT_EQ(resultB[planarIdx], srcData[packedIdx + 2]) 
                << "B channel pattern test failed at (" << x << "," << y << ") for pattern: " << params.name;
        }
    }
    
    nppiFree(d_src);
    nppiFree(d_dstR);
    nppiFree(d_dstG);
    nppiFree(d_dstB);
}


// 转换和拷贝组合参数化测试
struct ConvertCopyParams {
    std::string name;
    int width;
    int height;
    std::vector<Npp8u> inputPattern;
};

class ConvertCopyChainTest : public ::testing::TestWithParam<ConvertCopyParams> {
protected:
    void SetUp() override {
        auto params = GetParam();
        width = params.width;
        height = params.height;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    ConvertCopyChains,
    ConvertCopyChainTest,
    ::testing::Values(
        ConvertCopyParams{
            "Boundary2x2", 2, 2,
            {0, 128, 255, 64, 192, 32, 96, 160, 224, 16, 48, 80}
        },
        ConvertCopyParams{
            "Gradient4x3", 4, 3,
            {0, 0, 0, 85, 85, 85, 170, 170, 170, 255, 255, 255,
             64, 64, 64, 128, 128, 128, 192, 192, 192, 224, 224, 224,
             32, 96, 160, 48, 112, 176, 64, 128, 192, 80, 144, 208}
        }
    ),
    [](const ::testing::TestParamInfo<ConvertCopyParams>& info) {
        return info.param.name;
    }
);

// 参数化测试转换和拷贝的组合操作
TEST_P(ConvertCopyChainTest, ConvertThenCopy_8u32f_C3R) {
    auto params = GetParam();
    const int channels = 3;
    
    // 分配内存
    int srcStep, tmpStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(width, height, &srcStep);
    Npp32f* d_tmp = nppiMalloc_32f_C3(width, height, &tmpStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_tmp, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制输入数据到GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   params.inputPattern.data() + y * width * channels,
                   width * channels * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 步骤1: 转换 8u -> 32f
    NppStatus status1 = nppiConvert_8u32f_C3R(d_src, srcStep, d_tmp, tmpStep, roi);
    EXPECT_EQ(status1, NPP_SUCCESS);
    
    // 步骤2: 拷贝 32f -> 32f
    NppStatus status2 = nppiCopy_32f_C3R(d_tmp, tmpStep, d_dst, dstStep, roi);
    EXPECT_EQ(status2, NPP_SUCCESS);
    
    // 验证最终结果
    std::vector<Npp32f> resultData(width * height * channels);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * channels,
                   (char*)d_dst + y * dstStep,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 验证转换正确性
    for (int i = 0; i < width * height * channels; i++) {
        float expected = (float)params.inputPattern[i];
        EXPECT_FLOAT_EQ(resultData[i], expected) 
            << "转换拷贝链测试失败 at index " << i << " for " << params.name;
    }
    
    nppiFree(d_src);
    nppiFree(d_tmp);
    nppiFree(d_dst);
}

// ROI尺寸参数化测试
struct ROIParams {
    std::string name;
    int fullWidth;
    int fullHeight;
    int roiWidth;
    int roiHeight;
};

class CopyROITest : public ::testing::TestWithParam<ROIParams> {
protected:
    void SetUp() override {
        auto params = GetParam();
        fullWidth = params.fullWidth;
        fullHeight = params.fullHeight;
        roi.width = params.roiWidth;
        roi.height = params.roiHeight;
    }

    int fullWidth, fullHeight;
    NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    DifferentROISizes,
    CopyROITest,
    ::testing::Values(
        ROIParams{"FullImage", 8, 6, 8, 6},
        ROIParams{"TopLeft", 8, 6, 4, 3},
        ROIParams{"SingleRow", 8, 6, 8, 1},
        ROIParams{"SingleCol", 8, 6, 1, 6},
        ROIParams{"CenterArea", 16, 12, 8, 6},
        ROIParams{"SmallBlock", 32, 32, 4, 4}
    ),
    [](const ::testing::TestParamInfo<ROIParams>& info) {
        return info.param.name + "_" + 
               std::to_string(info.param.roiWidth) + "x" + 
               std::to_string(info.param.roiHeight);
    }
);

// 参数化测试不同ROI尺寸的拷贝
TEST_P(CopyROITest, Copy_32f_C3R_DifferentROI) {
    const int channels = 3;
    std::vector<Npp32f> srcData(fullWidth * fullHeight * channels);
    
    // 创建唯一标识每个像素的数据
    for (int y = 0; y < fullHeight; y++) {
        for (int x = 0; x < fullWidth; x++) {
            int idx = (y * fullWidth + x) * channels;
            srcData[idx + 0] = (float)(y * 1000 + x); // R: 唯一ID
            srcData[idx + 1] = (float)(x);            // G: X坐标
            srcData[idx + 2] = (float)(y);            // B: Y坐标
        }
    }
    
    // 分配GPU内存（使用全图尺寸）
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(fullWidth, fullHeight, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(fullWidth, fullHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据到GPU
    for (int y = 0; y < fullHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * fullWidth * channels,
                   fullWidth * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行ROI拷贝
    NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证ROI区域的结果
    std::vector<Npp32f> resultData(roi.width * roi.height * channels);
    for (int y = 0; y < roi.height; y++) {
        cudaMemcpy(resultData.data() + y * roi.width * channels,
                   (char*)d_dst + y * dstStep,
                   roi.width * channels * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 验证ROI内的数据正确性
    for (int y = 0; y < roi.height; y++) {
        for (int x = 0; x < roi.width; x++) {
            int resultIdx = (y * roi.width + x) * channels;
            int srcIdx = (y * fullWidth + x) * channels;
            
            for (int c = 0; c < channels; c++) {
                EXPECT_FLOAT_EQ(resultData[resultIdx + c], srcData[srcIdx + c]) 
                    << "ROI拷贝失败 at (" << x << "," << y << ") channel " << c;
            }
        }
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}