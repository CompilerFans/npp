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
        ImageSizeParams{128, 96, "MediumSize"}
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