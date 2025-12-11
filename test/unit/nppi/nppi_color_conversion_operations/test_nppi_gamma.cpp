#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

class NppiGammaTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 初始化CUDA
    cudaSetDevice(0);
  }

  // NVIDIA NPP Gamma LUT - Extracted directly from NVIDIA NPP library
  // These provide 100% exact pixel-perfect matching with NVIDIA's implementation

  // Forward Gamma LUT (Linear -> sRGB encoding)
  static constexpr Npp8u NVIDIA_GAMMA_FWD_LUT[256] = {
      0,   4,   9,   13,  18,  22,  26,  30,  33,  36,  40,  42,  45,  48,  50,  53,  55,  57,  59,  61,  63,  65,
      67,  69,  71,  73,  75,  76,  78,  80,  81,  83,  84,  86,  87,  89,  90,  92,  93,  95,  96,  97,  99,  100,
      101, 103, 104, 105, 106, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126,
      127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 142, 143, 144, 145, 146, 147,
      148, 149, 150, 151, 151, 152, 153, 154, 155, 156, 156, 157, 158, 159, 160, 161, 161, 162, 163, 164, 165, 165,
      166, 167, 168, 169, 169, 170, 171, 172, 172, 173, 174, 175, 175, 176, 177, 178, 178, 179, 180, 180, 181, 182,
      183, 183, 184, 185, 185, 186, 187, 188, 188, 189, 190, 190, 191, 192, 192, 193, 194, 194, 195, 196, 196, 197,
      198, 198, 199, 200, 200, 201, 201, 202, 203, 203, 204, 205, 205, 206, 207, 207, 208, 208, 209, 210, 210, 211,
      211, 212, 213, 213, 214, 214, 215, 216, 216, 217, 217, 218, 219, 219, 220, 220, 221, 221, 222, 223, 223, 224,
      224, 225, 225, 226, 227, 227, 228, 228, 229, 229, 230, 231, 231, 232, 232, 233, 233, 234, 234, 235, 235, 236,
      236, 237, 238, 238, 239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247,
      248, 248, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255,
  };

  // Inverse Gamma LUT (sRGB -> Linear decoding)
  static constexpr Npp8u NVIDIA_GAMMA_INV_LUT[256] = {
      0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   2,   2,   2,   2,   3,   3,   3,   3,   3,   4,   4,   4,
      4,   5,   5,   5,   5,   6,   6,   6,   6,   7,   7,   7,   8,   8,   8,   9,   9,   9,   10,  10,  10,  11,
      11,  11,  12,  12,  12,  13,  13,  14,  14,  15,  15,  15,  16,  16,  17,  17,  18,  18,  19,  19,  20,  20,
      21,  21,  22,  22,  23,  23,  24,  24,  25,  26,  26,  27,  27,  28,  28,  29,  30,  30,  31,  32,  32,  33,
      34,  34,  35,  36,  36,  37,  38,  38,  39,  40,  41,  41,  42,  43,  44,  44,  45,  46,  47,  48,  48,  49,
      50,  51,  52,  53,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  62,  63,  64,  65,  66,  67,  68,  69,
      70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  81,  82,  83,  84,  85,  86,  87,  88,  89,  91,  92,  93,
      94,  95,  96,  98,  99,  100, 101, 102, 104, 105, 106, 107, 109, 110, 111, 113, 114, 115, 116, 118, 119, 120,
      122, 123, 124, 126, 127, 129, 130, 131, 133, 134, 136, 137, 139, 140, 141, 143, 144, 146, 147, 149, 150, 152,
      153, 155, 157, 158, 160, 161, 163, 164, 166, 168, 169, 171, 172, 174, 176, 177, 179, 181, 182, 184, 186, 187,
      189, 191, 193, 194, 196, 198, 200, 201, 203, 205, 207, 209, 210, 212, 214, 216, 218, 220, 221, 223, 225, 227,
      229, 231, 233, 235, 237, 239, 241, 243, 245, 246, 248, 250, 252, 255,
  };

  // 辅助函数：CPU端Gamma forward (使用NVIDIA的精确LUT)
  Npp8u cpu_srgb_forward(Npp8u input) { return NVIDIA_GAMMA_FWD_LUT[input]; }

  // 辅助函数：CPU端Gamma inverse (使用NVIDIA的精确LUT)
  Npp8u cpu_srgb_inverse(Npp8u input) { return NVIDIA_GAMMA_INV_LUT[input]; }
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
  int src_step = width * channels; // step是每行的字节数
  int dst_step = width * channels;

  cudaMalloc(&d_src, width * height * channels);
  cudaMalloc(&d_dst, width * height * channels);

  cudaMemcpy(d_src, h_src.data(), width * height * channels, cudaMemcpyHostToDevice);

  // 执行NPP函数
  NppiSize roi = {width, height};
  NppStreamContext ctx;
  ctx.hStream = 0;

  NppStatus status = nppiGammaFwd_8u_C3R_Ctx(d_src, src_step, d_dst, dst_step, roi, ctx);

  ASSERT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(h_dst.data(), d_dst, width * height * channels, cudaMemcpyDeviceToHost);

  // 验证结果（允许±1的误差）
  int max_diff = 0;
  for (int i = 0; i < width * height * channels; i++) {
    int diff = abs((int)h_dst[i] - (int)h_expected[i]);
    max_diff = std::max(max_diff, diff);
    EXPECT_LE(diff, 1) << "Pixel " << i << ": got " << (int)h_dst[i] << ", expected " << (int)h_expected[i];
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
    h_src[i * channels + 3] = 200; // Alpha通道
  }

  // 计算期望结果（RGB做Gamma，Alpha设置为0）
  // NVIDIA NPP的AC4格式会将Alpha通道清零，而不是保持原值
  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      h_expected[i * channels + c] = cpu_srgb_forward(h_src[i * channels + c]);
    }
    h_expected[i * channels + 3] = 0; // NVIDIA NPP clears alpha to 0
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
      EXPECT_LE(abs((int)h_dst[idx] - (int)h_expected[idx]), 1) << "Pixel " << i << " channel " << c;
    }
    // 验证Alpha通道被清零（NVIDIA NPP行为）
    EXPECT_EQ(h_dst[i * channels + 3], 0) << "Alpha not cleared at pixel " << i;
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
    h_expected[i * channels + 3] = 0; // NVIDIA NPP clears alpha to 0
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
    h_expected[i * channels + 3] = 255; // AC4IR: Alpha remains unchanged (in-place)
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

  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      int idx = i * channels + c;
      EXPECT_LE(abs((int)h_data[idx] - (int)h_expected[idx]), 1);
    }
    EXPECT_EQ(h_data[i * channels + 3], 255); // AC4IR: Alpha remains unchanged
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
    h_expected[i * channels + 3] = 0; // NVIDIA NPP clears alpha to 0
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

  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      int idx = i * channels + c;
      EXPECT_LE(abs((int)h_dst[idx] - (int)h_expected[idx]), 1);
    }
    EXPECT_EQ(h_dst[i * channels + 3], 0); // NVIDIA NPP clears alpha
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

// ============================================================================
// 16个Gamma函数的完整测试覆盖
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
    h_expected[i * channels + 3] = 0; // NVIDIA NPP clears alpha to 0
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
    EXPECT_EQ(h_dst[i * channels + 3], 0); // NVIDIA NPP clears alpha
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
    h_expected[i * channels + 3] = 220; // AC4IR: Alpha remains unchanged (in-place)
  }

  Npp8u *d_data;
  int step = width * channels;

  cudaMalloc(&d_data, width * height * channels);
  cudaMemcpy(d_data, h_data.data(), width * height * channels, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};

  NppStatus status = nppiGammaFwd_8u_AC4IR(d_data, step, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  cudaMemcpy(h_data.data(), d_data, width * height * channels, cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; i++) {
    for (int c = 0; c < 3; c++) {
      int idx = i * channels + c;
      EXPECT_LE(abs((int)h_data[idx] - (int)h_expected[idx]), 1);
    }
    EXPECT_EQ(h_data[i * channels + 3], 220); // AC4IR: Alpha remains unchanged
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
    h_expected[i * channels + 3] = 240; // AC4IR: Alpha remains unchanged (in-place)
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
    EXPECT_EQ(h_data[i * channels + 3], 240); // AC4IR: Alpha remains unchanged
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
    h_expected[i * channels + 3] = 100; // AC4IR: Alpha remains unchanged (in-place)
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
    EXPECT_EQ(h_data[i * channels + 3], 100); // AC4IR: Alpha remains unchanged
  }

  cudaFree(d_data);
}