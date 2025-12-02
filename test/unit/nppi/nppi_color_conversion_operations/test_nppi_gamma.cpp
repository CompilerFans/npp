#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

class NppiGammaTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化CUDA
        cudaSetDevice(0);
    }
    
    // NVIDIA NPP使用简单的power-law gamma (不是标准sRGB)
    // 通过反向工程NVIDIA NPP实际输出确定：gamma = 1.87
    static constexpr float NPP_GAMMA = 1.87f;

    // 辅助函数：CPU端Gamma forward (匹配NVIDIA NPP行为)
    Npp8u cpu_srgb_forward(Npp8u input) {
        float normalized = input / 255.0f;
        // NVIDIA使用简单的power law: output = input^(1/gamma)
        float result = powf(normalized, 1.0f / NPP_GAMMA);
        return (Npp8u)(result * 255.0f + 0.5f);
    }

    // 辅助函数：CPU端Gamma inverse (匹配NVIDIA NPP行为)
    Npp8u cpu_srgb_inverse(Npp8u input) {
        float normalized = input / 255.0f;
        // NVIDIA使用简单的power law: output = input^gamma
        float result = powf(normalized, NPP_GAMMA);
        return (Npp8u)(result * 255.0f + 0.5f);
    }
};

// 测试Forward Gamma - C3R
TEST_F(NppiGammaTest, GammaFwd_8u_C3R_Ctx) {
    const int width = 640;
    const int height = 480;
    const int channels = 3;
    
    // 分配主机内存
    std::vector<Npp8u> h_src(width * height * channels);
    std::vector<Npp8u> h_dst(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    // 填充测试数据（渐变）
    for (int i = 0; i < width * height * channels; i++) {
        h_src[i] = i % 256;
    }
    
    // 计算期望结果
    for (int i = 0; i < width * height * channels; i++) {
        h_expected[i] = cpu_srgb_forward(h_src[i]);
    }
    
    // 分配设备内存
    Npp8u *d_src, *d_dst;
    int src_step = width * channels;  // step是每行的字节数
    int dst_step = width * channels;
    
    cudaMalloc(&d_src, width * height * channels);
    cudaMalloc(&d_dst, width * height * channels);
    
    cudaMemcpy(d_src, h_src.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    // 执行NPP函数
    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    NppStatus status = nppiGammaFwd_8u_C3R_Ctx(
        d_src, src_step,
        d_dst, dst_step,
        roi, ctx
    );
    
    ASSERT_EQ(status, NPP_SUCCESS);
    
    // 拷贝结果回主机
    cudaMemcpy(h_dst.data(), d_dst, width * height * channels, cudaMemcpyDeviceToHost);
    
    // 验证结果（允许±1的误差）
    int max_diff = 0;
    for (int i = 0; i < width * height * channels; i++) {
        int diff = abs((int)h_dst[i] - (int)h_expected[i]);
        max_diff = std::max(max_diff, diff);
        EXPECT_LE(diff, 1) << "Pixel " << i << ": got " << (int)h_dst[i] 
                          << ", expected " << (int)h_expected[i];
    }
    
    std::cout << "Max difference: " << max_diff << std::endl;
    
    // 清理
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试Inverse Gamma - C3R
TEST_F(NppiGammaTest, GammaInv_8u_C3R_Ctx) {
    const int width = 640;
    const int height = 480;
    const int channels = 3;
    
    std::vector<Npp8u> h_src(width * height * channels);
    std::vector<Npp8u> h_dst(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height * channels; i++) {
        h_src[i] = i % 256;
        h_expected[i] = cpu_srgb_inverse(h_src[i]);
    }
    
    Npp8u *d_src, *d_dst;
    int src_step = width * channels;
    int dst_step = width * channels;
    
    cudaMalloc(&d_src, width * height * channels);
    cudaMalloc(&d_dst, width * height * channels);
    cudaMemcpy(d_src, h_src.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    NppStatus status = nppiGammaInv_8u_C3R_Ctx(d_src, src_step, d_dst, dst_step, roi, ctx);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_dst.data(), d_dst, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_LE(abs((int)h_dst[i] - (int)h_expected[i]), 1);
    }
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试就地操作 - C3IR
TEST_F(NppiGammaTest, GammaFwd_8u_C3IR_Ctx) {
    const int width = 64;
    const int height = 48;
    const int channels = 3;
    
    std::vector<Npp8u> h_data(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height * channels; i++) {
        h_data[i] = i % 256;
        h_expected[i] = cpu_srgb_forward(h_data[i]);
    }
    
    Npp8u *d_data;
    int step = width * channels;
    
    cudaMalloc(&d_data, width * height * channels);
    cudaMemcpy(d_data, h_data.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    NppStatus status = nppiGammaFwd_8u_C3IR_Ctx(d_data, step, roi, ctx);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_data.data(), d_data, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_LE(abs((int)h_data[i] - (int)h_expected[i]), 1);
    }
    
    cudaFree(d_data);
}

// 测试往返一致性（Forward → Inverse应该恢复原值）
TEST_F(NppiGammaTest, RoundTrip_C3) {
    const int width = 256;
    const int height = 1;
    const int channels = 3;
    
    std::vector<Npp8u> h_original(width * channels);
    std::vector<Npp8u> h_result(width * channels);
    
    // 所有可能的8位值
    for (int i = 0; i < width; i++) {
        for (int c = 0; c < channels; c++) {
            h_original[i * channels + c] = i;
        }
    }
    
    Npp8u *d_src, *d_tmp, *d_dst;
    int step = width * channels;
    
    cudaMalloc(&d_src, width * channels);
    cudaMalloc(&d_tmp, width * channels);
    cudaMalloc(&d_dst, width * channels);
    
    cudaMemcpy(d_src, h_original.data(), width * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    // Forward gamma
    nppiGammaFwd_8u_C3R_Ctx(d_src, step, d_tmp, step, roi, ctx);
    
    // Inverse gamma
    nppiGammaInv_8u_C3R_Ctx(d_tmp, step, d_dst, step, roi, ctx);
    
    cudaMemcpy(h_result.data(), d_dst, width * channels, cudaMemcpyDeviceToHost);
    
    // 验证往返误差（应该≤2）
    for (int i = 0; i < width * channels; i++) {
        int diff = abs((int)h_original[i] - (int)h_result[i]);
        EXPECT_LE(diff, 2) << "Round-trip error at pixel " << i 
                          << ": " << (int)h_original[i] << " → " << (int)h_result[i];
    }
    
    cudaFree(d_src);
    cudaFree(d_tmp);
    cudaFree(d_dst);
}

// 测试AC4格式 - Forward Gamma
TEST_F(NppiGammaTest, GammaFwd_8u_AC4R_Ctx) {
    const int width = 320;
    const int height = 240;
    const int channels = 4;
    
    std::vector<Npp8u> h_src(width * height * channels);
    std::vector<Npp8u> h_dst(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    // 填充测试数据（RGB渐变，Alpha固定）
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            h_src[i * channels + c] = (i + c * 50) % 256;
        }
        h_src[i * channels + 3] = 200;  // Alpha通道
    }
    
    // 计算期望结果（RGB做Gamma，Alpha不变）
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            h_expected[i * channels + c] = cpu_srgb_forward(h_src[i * channels + c]);
        }
        h_expected[i * channels + 3] = h_src[i * channels + 3];  // Alpha不变
    }
    
    Npp8u *d_src, *d_dst;
    int src_step = width * channels;
    int dst_step = width * channels;
    
    cudaMalloc(&d_src, width * height * channels);
    cudaMalloc(&d_dst, width * height * channels);
    cudaMemcpy(d_src, h_src.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    NppStatus status = nppiGammaFwd_8u_AC4R_Ctx(d_src, src_step, d_dst, dst_step, roi, ctx);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_dst.data(), d_dst, width * height * channels, cudaMemcpyDeviceToHost);
    
    // 验证RGB通道
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            int idx = i * channels + c;
            EXPECT_LE(abs((int)h_dst[idx] - (int)h_expected[idx]), 1) 
                << "Pixel " << i << " channel " << c;
        }
        // 验证Alpha通道不变
        EXPECT_EQ(h_dst[i * channels + 3], 200) << "Alpha changed at pixel " << i;
    }
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试AC4格式 - Inverse Gamma
TEST_F(NppiGammaTest, GammaInv_8u_AC4R_Ctx) {
    const int width = 320;
    const int height = 240;
    const int channels = 4;
    
    std::vector<Npp8u> h_src(width * height * channels);
    std::vector<Npp8u> h_dst(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            h_src[i * channels + c] = (i + c * 50) % 256;
            h_expected[i * channels + c] = cpu_srgb_inverse(h_src[i * channels + c]);
        }
        h_src[i * channels + 3] = 150;
        h_expected[i * channels + 3] = 150;
    }
    
    Npp8u *d_src, *d_dst;
    int src_step = width * channels;
    int dst_step = width * channels;
    
    cudaMalloc(&d_src, width * height * channels);
    cudaMalloc(&d_dst, width * height * channels);
    cudaMemcpy(d_src, h_src.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    NppStatus status = nppiGammaInv_8u_AC4R_Ctx(d_src, src_step, d_dst, dst_step, roi, ctx);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_dst.data(), d_dst, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 4; c++) {
            int idx = i * channels + c;
            EXPECT_LE(abs((int)h_dst[idx] - (int)h_expected[idx]), 1);
        }
    }
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试AC4格式 - In-place Forward
TEST_F(NppiGammaTest, GammaFwd_8u_AC4IR_Ctx) {
    const int width = 128;
    const int height = 96;
    const int channels = 4;
    
    std::vector<Npp8u> h_data(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            h_data[i * channels + c] = (i + c * 30) % 256;
            h_expected[i * channels + c] = cpu_srgb_forward(h_data[i * channels + c]);
        }
        h_data[i * channels + 3] = 255;
        h_expected[i * channels + 3] = 255;
    }
    
    Npp8u *d_data;
    int step = width * channels;
    
    cudaMalloc(&d_data, width * height * channels);
    cudaMemcpy(d_data, h_data.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    NppStatus status = nppiGammaFwd_8u_AC4IR_Ctx(d_data, step, roi, ctx);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_data.data(), d_data, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_LE(abs((int)h_data[i] - (int)h_expected[i]), 1);
    }
    
    cudaFree(d_data);
}

// 测试非Ctx版本 - C3R
TEST_F(NppiGammaTest, GammaFwd_8u_C3R_NonCtx) {
    const int width = 160;
    const int height = 120;
    const int channels = 3;
    
    std::vector<Npp8u> h_src(width * height * channels);
    std::vector<Npp8u> h_dst(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height * channels; i++) {
        h_src[i] = i % 256;
        h_expected[i] = cpu_srgb_forward(h_src[i]);
    }
    
    Npp8u *d_src, *d_dst;
    int src_step = width * channels;
    int dst_step = width * channels;
    
    cudaMalloc(&d_src, width * height * channels);
    cudaMalloc(&d_dst, width * height * channels);
    cudaMemcpy(d_src, h_src.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    
    // 调用非Ctx版本（内部使用默认stream）
    NppStatus status = nppiGammaFwd_8u_C3R(d_src, src_step, d_dst, dst_step, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_dst.data(), d_dst, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_LE(abs((int)h_dst[i] - (int)h_expected[i]), 1);
    }
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试非Ctx版本 - AC4R
TEST_F(NppiGammaTest, GammaInv_8u_AC4R_NonCtx) {
    const int width = 160;
    const int height = 120;
    const int channels = 4;
    
    std::vector<Npp8u> h_src(width * height * channels);
    std::vector<Npp8u> h_dst(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            h_src[i * channels + c] = (i + c * 40) % 256;
            h_expected[i * channels + c] = cpu_srgb_inverse(h_src[i * channels + c]);
        }
        h_src[i * channels + 3] = 128;
        h_expected[i * channels + 3] = 128;
    }
    
    Npp8u *d_src, *d_dst;
    int src_step = width * channels;
    int dst_step = width * channels;
    
    cudaMalloc(&d_src, width * height * channels);
    cudaMalloc(&d_dst, width * height * channels);
    cudaMemcpy(d_src, h_src.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    
    NppStatus status = nppiGammaInv_8u_AC4R(d_src, src_step, d_dst, dst_step, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_dst.data(), d_dst, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_LE(abs((int)h_dst[i] - (int)h_expected[i]), 1);
    }
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试往返一致性 - AC4格式
TEST_F(NppiGammaTest, RoundTrip_AC4) {
    const int width = 256;
    const int height = 1;
    const int channels = 4;
    
    std::vector<Npp8u> h_original(width * channels);
    std::vector<Npp8u> h_result(width * channels);
    
    for (int i = 0; i < width; i++) {
        for (int c = 0; c < 3; c++) {
            h_original[i * channels + c] = i;
        }
        h_original[i * channels + 3] = 200;  // Alpha
    }
    
    Npp8u *d_src, *d_tmp, *d_dst;
    int step = width * channels;
    
    cudaMalloc(&d_src, width * channels);
    cudaMalloc(&d_tmp, width * channels);
    cudaMalloc(&d_dst, width * channels);
    cudaMemcpy(d_src, h_original.data(), width * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    nppiGammaFwd_8u_AC4R_Ctx(d_src, step, d_tmp, step, roi, ctx);
    nppiGammaInv_8u_AC4R_Ctx(d_tmp, step, d_dst, step, roi, ctx);
    
    cudaMemcpy(h_result.data(), d_dst, width * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width; i++) {
        // 验证RGB往返误差
        for (int c = 0; c < 3; c++) {
            int idx = i * channels + c;
            int diff = abs((int)h_original[idx] - (int)h_result[idx]);
            EXPECT_LE(diff, 2) << "Round-trip error at pixel " << i << " channel " << c;
        }
        // 验证Alpha完全不变
        EXPECT_EQ(h_result[i * channels + 3], 200) << "Alpha changed at pixel " << i;
    }
    
    cudaFree(d_src);
    cudaFree(d_tmp);
    cudaFree(d_dst);
}

// 测试边界情况 - 最小值和最大值
TEST_F(NppiGammaTest, BoundaryValues_C3) {
    const int width = 4;
    const int height = 1;
    const int channels = 3;
    
    std::vector<Npp8u> h_src = {
        0, 0, 0,        // 黑色
        255, 255, 255,  // 白色
        128, 128, 128,  // 中灰
        1, 254, 127     // 混合
    };
    
    std::vector<Npp8u> h_dst(width * channels);
    
    Npp8u *d_src, *d_dst;
    int step = width * channels;
    
    cudaMalloc(&d_src, width * channels);
    cudaMalloc(&d_dst, width * channels);
    cudaMemcpy(d_src, h_src.data(), width * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    
    // Forward
    nppiGammaFwd_8u_C3R(d_src, step, d_dst, step, roi);
    cudaMemcpy(h_dst.data(), d_dst, width * channels, cudaMemcpyDeviceToHost);
    
    // 验证黑色和白色不变（或几乎不变）
    EXPECT_LE(h_dst[0], 1);       // 黑色应该接近0
    EXPECT_GE(h_dst[3], 254);     // 白色应该接近255
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试参数验证 - 空指针
TEST_F(NppiGammaTest, NullPointerError) {
    NppiSize roi = {100, 100};
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    Npp8u *d_valid;
    cudaMalloc(&d_valid, 100 * 100 * 3);
    
    // 源指针为空
    EXPECT_EQ(nppiGammaFwd_8u_C3R_Ctx(nullptr, 300, d_valid, 300, roi, ctx), 
              NPP_NULL_POINTER_ERROR);
    
    // 目标指针为空
    EXPECT_EQ(nppiGammaFwd_8u_C3R_Ctx(d_valid, 300, nullptr, 300, roi, ctx), 
              NPP_NULL_POINTER_ERROR);
    
    cudaFree(d_valid);
}

// 测试参数验证 - 无效ROI
TEST_F(NppiGammaTest, InvalidROI) {
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    Npp8u *d_src, *d_dst;
    cudaMalloc(&d_src, 1000);
    cudaMalloc(&d_dst, 1000);
    
    // 宽度为0
    NppiSize roi_zero = {0, 100};
    EXPECT_EQ(nppiGammaFwd_8u_C3R_Ctx(d_src, 300, d_dst, 300, roi_zero, ctx), 
              NPP_SIZE_ERROR);
    
    // 高度为负数
    NppiSize roi_neg = {100, -1};
    EXPECT_EQ(nppiGammaFwd_8u_C3R_Ctx(d_src, 300, d_dst, 300, roi_neg, ctx), 
              NPP_SIZE_ERROR);
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

// ============================================================================
// 补充所有缺失的测试，确保16个函数100%覆盖
// ============================================================================

// 测试 nppiGammaFwd_8u_C3IR (非Ctx版本, in-place)
TEST_F(NppiGammaTest, GammaFwd_8u_C3IR_NonCtx) {
    const int width = 128;
    const int height = 96;
    const int channels = 3;
    
    std::vector<Npp8u> h_data(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height * channels; i++) {
        h_data[i] = i % 256;
        h_expected[i] = cpu_srgb_forward(h_data[i]);
    }
    
    Npp8u *d_data;
    int step = width * channels;
    
    cudaMalloc(&d_data, width * height * channels);
    cudaMemcpy(d_data, h_data.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    
    // 非Ctx版本
    NppStatus status = nppiGammaFwd_8u_C3IR(d_data, step, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_data.data(), d_data, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_LE(abs((int)h_data[i] - (int)h_expected[i]), 1);
    }
    
    cudaFree(d_data);
}

// 测试 nppiGammaFwd_8u_AC4R (非Ctx版本, out-of-place)
TEST_F(NppiGammaTest, GammaFwd_8u_AC4R_NonCtx) {
    const int width = 160;
    const int height = 120;
    const int channels = 4;
    
    std::vector<Npp8u> h_src(width * height * channels);
    std::vector<Npp8u> h_dst(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            h_src[i * channels + c] = (i + c * 50) % 256;
            h_expected[i * channels + c] = cpu_srgb_forward(h_src[i * channels + c]);
        }
        h_src[i * channels + 3] = 180;
        h_expected[i * channels + 3] = 180;
    }
    
    Npp8u *d_src, *d_dst;
    int src_step = width * channels;
    int dst_step = width * channels;
    
    cudaMalloc(&d_src, width * height * channels);
    cudaMalloc(&d_dst, width * height * channels);
    cudaMemcpy(d_src, h_src.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    
    NppStatus status = nppiGammaFwd_8u_AC4R(d_src, src_step, d_dst, dst_step, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_dst.data(), d_dst, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            int idx = i * channels + c;
            EXPECT_LE(abs((int)h_dst[idx] - (int)h_expected[idx]), 1);
        }
        EXPECT_EQ(h_dst[i * channels + 3], 180);
    }
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试 nppiGammaFwd_8u_AC4IR (非Ctx版本, in-place)
TEST_F(NppiGammaTest, GammaFwd_8u_AC4IR_NonCtx) {
    const int width = 100;
    const int height = 80;
    const int channels = 4;
    
    std::vector<Npp8u> h_data(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            h_data[i * channels + c] = (i * 3 + c * 20) % 256;
            h_expected[i * channels + c] = cpu_srgb_forward(h_data[i * channels + c]);
        }
        h_data[i * channels + 3] = 220;
        h_expected[i * channels + 3] = 220;
    }
    
    Npp8u *d_data;
    int step = width * channels;
    
    cudaMalloc(&d_data, width * height * channels);
    cudaMemcpy(d_data, h_data.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    
    NppStatus status = nppiGammaFwd_8u_AC4IR(d_data, step, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_data.data(), d_data, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_LE(abs((int)h_data[i] - (int)h_expected[i]), 1);
    }
    
    cudaFree(d_data);
}

// 测试 nppiGammaInv_8u_C3R (非Ctx版本, out-of-place)
TEST_F(NppiGammaTest, GammaInv_8u_C3R_NonCtx) {
    const int width = 200;
    const int height = 150;
    const int channels = 3;
    
    std::vector<Npp8u> h_src(width * height * channels);
    std::vector<Npp8u> h_dst(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height * channels; i++) {
        h_src[i] = (i * 7) % 256;
        h_expected[i] = cpu_srgb_inverse(h_src[i]);
    }
    
    Npp8u *d_src, *d_dst;
    int src_step = width * channels;
    int dst_step = width * channels;
    
    cudaMalloc(&d_src, width * height * channels);
    cudaMalloc(&d_dst, width * height * channels);
    cudaMemcpy(d_src, h_src.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    
    NppStatus status = nppiGammaInv_8u_C3R(d_src, src_step, d_dst, dst_step, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_dst.data(), d_dst, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_LE(abs((int)h_dst[i] - (int)h_expected[i]), 1);
    }
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试 nppiGammaInv_8u_C3IR_Ctx (Ctx版本, in-place)
TEST_F(NppiGammaTest, GammaInv_8u_C3IR_Ctx) {
    const int width = 128;
    const int height = 96;
    const int channels = 3;
    
    std::vector<Npp8u> h_data(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height * channels; i++) {
        h_data[i] = (i * 11) % 256;
        h_expected[i] = cpu_srgb_inverse(h_data[i]);
    }
    
    Npp8u *d_data;
    int step = width * channels;
    
    cudaMalloc(&d_data, width * height * channels);
    cudaMemcpy(d_data, h_data.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    NppStatus status = nppiGammaInv_8u_C3IR_Ctx(d_data, step, roi, ctx);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_data.data(), d_data, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_LE(abs((int)h_data[i] - (int)h_expected[i]), 1);
    }
    
    cudaFree(d_data);
}

// 测试 nppiGammaInv_8u_C3IR (非Ctx版本, in-place)
TEST_F(NppiGammaTest, GammaInv_8u_C3IR_NonCtx) {
    const int width = 96;
    const int height = 64;
    const int channels = 3;
    
    std::vector<Npp8u> h_data(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height * channels; i++) {
        h_data[i] = (i * 13) % 256;
        h_expected[i] = cpu_srgb_inverse(h_data[i]);
    }
    
    Npp8u *d_data;
    int step = width * channels;
    
    cudaMalloc(&d_data, width * height * channels);
    cudaMemcpy(d_data, h_data.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    
    NppStatus status = nppiGammaInv_8u_C3IR(d_data, step, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_data.data(), d_data, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_LE(abs((int)h_data[i] - (int)h_expected[i]), 1);
    }
    
    cudaFree(d_data);
}

// 测试 nppiGammaInv_8u_AC4IR_Ctx (Ctx版本, in-place)
TEST_F(NppiGammaTest, GammaInv_8u_AC4IR_Ctx) {
    const int width = 120;
    const int height = 90;
    const int channels = 4;
    
    std::vector<Npp8u> h_data(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            h_data[i * channels + c] = (i * 5 + c * 30) % 256;
            h_expected[i * channels + c] = cpu_srgb_inverse(h_data[i * channels + c]);
        }
        h_data[i * channels + 3] = 240;
        h_expected[i * channels + 3] = 240;
    }
    
    Npp8u *d_data;
    int step = width * channels;
    
    cudaMalloc(&d_data, width * height * channels);
    cudaMemcpy(d_data, h_data.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;
    
    NppStatus status = nppiGammaInv_8u_AC4IR_Ctx(d_data, step, roi, ctx);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_data.data(), d_data, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            int idx = i * channels + c;
            EXPECT_LE(abs((int)h_data[idx] - (int)h_expected[idx]), 1);
        }
        EXPECT_EQ(h_data[i * channels + 3], 240);
    }
    
    cudaFree(d_data);
}

// 测试 nppiGammaInv_8u_AC4IR (非Ctx版本, in-place)
TEST_F(NppiGammaTest, GammaInv_8u_AC4IR_NonCtx) {
    const int width = 110;
    const int height = 85;
    const int channels = 4;
    
    std::vector<Npp8u> h_data(width * height * channels);
    std::vector<Npp8u> h_expected(width * height * channels);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            h_data[i * channels + c] = (i * 7 + c * 25) % 256;
            h_expected[i * channels + c] = cpu_srgb_inverse(h_data[i * channels + c]);
        }
        h_data[i * channels + 3] = 100;
        h_expected[i * channels + 3] = 100;
    }
    
    Npp8u *d_data;
    int step = width * channels;
    
    cudaMalloc(&d_data, width * height * channels);
    cudaMemcpy(d_data, h_data.data(), width * height * channels, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    
    NppStatus status = nppiGammaInv_8u_AC4IR(d_data, step, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
    
    cudaMemcpy(h_data.data(), d_data, width * height * channels, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < width * height; i++) {
        for (int c = 0; c < 3; c++) {
            int idx = i * channels + c;
            EXPECT_LE(abs((int)h_data[idx] - (int)h_expected[idx]), 1);
        }
        EXPECT_EQ(h_data[i * channels + 3], 100);
    }
    
    cudaFree(d_data);
}