#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <set>

class NPPILabelMarkersUFTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 默认测试图像大小
    width = 8;
    height = 8;
    oSizeROI.width = width;
    oSizeROI.height = height;
  }

  void TearDown() override {
    // 清理资源
  }

  int width, height;
  NppiSize oSizeROI;
};

// 测试 nppiLabelMarkersUFGetBufferSize_32u_C1R
TEST_F(NPPILabelMarkersUFTest, nppiLabelMarkersUFGetBufferSize_32u_C1R_Basic) {
  int bufferSize = 0;

  NppStatus status = nppiLabelMarkersUFGetBufferSize_32u_C1R(oSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);

  // 测试更大的图像
  NppiSize largeROI = {512, 512};
  int largeBufferSize = 0;
  status = nppiLabelMarkersUFGetBufferSize_32u_C1R(largeROI, &largeBufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(largeBufferSize, bufferSize);
}

// 测试 nppiLabelMarkersUF_8u32u_C1R_Ctx - 8 邻域连通性 (nppiNormInf)
TEST_F(NPPILabelMarkersUFTest, nppiLabelMarkersUF_8u32u_C1R_Ctx_8Way) {
  // 创建简单的二值测试图像
  // 0 0 0 0 0 0 0 0
  // 0 1 1 0 0 1 1 0
  // 0 1 1 0 0 1 1 0
  // 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0
  // 0 1 0 0 0 0 1 0
  // 0 1 1 0 0 1 1 0
  // 0 0 0 0 0 0 0 0

  std::vector<Npp8u> srcData(width * height, 0);
  // 第一个区域 (左上)
  srcData[1 * width + 1] = 255; srcData[1 * width + 2] = 255;
  srcData[2 * width + 1] = 255; srcData[2 * width + 2] = 255;
  // 第二个区域 (右上)
  srcData[1 * width + 5] = 255; srcData[1 * width + 6] = 255;
  srcData[2 * width + 5] = 255; srcData[2 * width + 6] = 255;
  // 第三个区域 (左下)
  srcData[5 * width + 1] = 255;
  srcData[6 * width + 1] = 255; srcData[6 * width + 2] = 255;
  // 第四个区域 (右下)
  srcData[5 * width + 6] = 255;
  srcData[6 * width + 5] = 255; srcData[6 * width + 6] = 255;

  // 获取缓冲区大小
  int bufferSize = 0;
  NppStatus status = nppiLabelMarkersUFGetBufferSize_32u_C1R(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  // 分配 GPU 内存
  Npp8u *d_src = nullptr;
  Npp32u *d_labels = nullptr;
  Npp8u *d_buffer = nullptr;
  int srcStep = width * sizeof(Npp8u);
  int labelsStep = width * sizeof(Npp32u);

  cudaMalloc(&d_src, width * height * sizeof(Npp8u));
  cudaMalloc(&d_labels, width * height * sizeof(Npp32u));
  cudaMalloc(&d_buffer, bufferSize);

  // RAII 资源管理
  struct ResourceGuard {
    Npp8u *src;
    Npp32u *labels;
    Npp8u *buffer;
    ~ResourceGuard() {
      if (src) cudaFree(src);
      if (labels) cudaFree(labels);
      if (buffer) cudaFree(buffer);
    }
  } guard{d_src, d_labels, d_buffer};

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_labels, nullptr);
  ASSERT_NE(d_buffer, nullptr);

  // 拷贝数据到 GPU
  cudaMemcpy(d_src, srcData.data(), width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // 获取 NPP stream context
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
  cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
                         cudaDevAttrComputeCapabilityMajor, nppStreamCtx.nCudaDeviceId);
  cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
                         cudaDevAttrComputeCapabilityMinor, nppStreamCtx.nCudaDeviceId);
  cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, nppStreamCtx.nCudaDeviceId);
  nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
  nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  nppStreamCtx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;

  // 执行 8 邻域标记 (nppiNormInf)
  status = nppiLabelMarkersUF_8u32u_C1R_Ctx(d_src, srcStep, d_labels, labelsStep,
                                             oSizeROI, nppiNormInf, d_buffer, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  std::vector<Npp32u> labelData(width * height);
  cudaMemcpy(labelData.data(), d_labels, width * height * sizeof(Npp32u), cudaMemcpyDeviceToHost);

  // 验证结果：应该有 4 个不同的非零标签
  std::set<Npp32u> uniqueLabels;
  for (size_t i = 0; i < labelData.size(); i++) {
    if (labelData[i] > 0) {
      uniqueLabels.insert(labelData[i]);
    }
  }

  // 8 邻域连通性下应该有 4 个独立区域
  EXPECT_EQ(uniqueLabels.size(), 4);

  // 验证同一区域内的像素有相同标签
  // 左上区域
  EXPECT_EQ(labelData[1 * width + 1], labelData[1 * width + 2]);
  EXPECT_EQ(labelData[1 * width + 1], labelData[2 * width + 1]);
  EXPECT_EQ(labelData[1 * width + 1], labelData[2 * width + 2]);

  // 右上区域
  EXPECT_EQ(labelData[1 * width + 5], labelData[1 * width + 6]);
  EXPECT_EQ(labelData[1 * width + 5], labelData[2 * width + 5]);
  EXPECT_EQ(labelData[1 * width + 5], labelData[2 * width + 6]);

  // 背景应该是 0
  EXPECT_EQ(labelData[0], 0);
  EXPECT_EQ(labelData[3 * width + 3], 0);
}

// 测试 nppiLabelMarkersUF_8u32u_C1R_Ctx - 4 邻域连通性 (nppiNormL1)
TEST_F(NPPILabelMarkersUFTest, nppiLabelMarkersUF_8u32u_C1R_Ctx_4Way) {
  // 创建对角连接的测试图像
  // 在 4 邻域下，对角连接的像素应该是不同的区域
  // 0 0 0 0 0 0 0 0
  // 0 1 0 0 0 0 0 0
  // 0 0 1 0 0 0 0 0
  // 0 0 0 1 0 0 0 0
  // 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0

  std::vector<Npp8u> srcData(width * height, 0);
  srcData[1 * width + 1] = 255;
  srcData[2 * width + 2] = 255;
  srcData[3 * width + 3] = 255;

  // 获取缓冲区大小
  int bufferSize = 0;
  NppStatus status = nppiLabelMarkersUFGetBufferSize_32u_C1R(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  // 分配 GPU 内存
  Npp8u *d_src = nullptr;
  Npp32u *d_labels = nullptr;
  Npp8u *d_buffer = nullptr;
  int srcStep = width * sizeof(Npp8u);
  int labelsStep = width * sizeof(Npp32u);

  cudaMalloc(&d_src, width * height * sizeof(Npp8u));
  cudaMalloc(&d_labels, width * height * sizeof(Npp32u));
  cudaMalloc(&d_buffer, bufferSize);

  struct ResourceGuard {
    Npp8u *src;
    Npp32u *labels;
    Npp8u *buffer;
    ~ResourceGuard() {
      if (src) cudaFree(src);
      if (labels) cudaFree(labels);
      if (buffer) cudaFree(buffer);
    }
  } guard{d_src, d_labels, d_buffer};

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_labels, nullptr);
  ASSERT_NE(d_buffer, nullptr);

  cudaMemcpy(d_src, srcData.data(), width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
  cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
                         cudaDevAttrComputeCapabilityMajor, nppStreamCtx.nCudaDeviceId);
  cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
                         cudaDevAttrComputeCapabilityMinor, nppStreamCtx.nCudaDeviceId);
  cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, nppStreamCtx.nCudaDeviceId);
  nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
  nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  nppStreamCtx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;

  // 执行 4 邻域标记 (nppiNormL1)
  status = nppiLabelMarkersUF_8u32u_C1R_Ctx(d_src, srcStep, d_labels, labelsStep,
                                             oSizeROI, nppiNormL1, d_buffer, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32u> labelData(width * height);
  cudaMemcpy(labelData.data(), d_labels, width * height * sizeof(Npp32u), cudaMemcpyDeviceToHost);

  // 在 4 邻域下，对角连接的 3 个像素应该是 3 个不同的区域
  std::set<Npp32u> uniqueLabels;
  for (size_t i = 0; i < labelData.size(); i++) {
    if (labelData[i] > 0) {
      uniqueLabels.insert(labelData[i]);
    }
  }

  EXPECT_EQ(uniqueLabels.size(), 3);

  // 每个像素应该有不同的标签
  EXPECT_NE(labelData[1 * width + 1], labelData[2 * width + 2]);
  EXPECT_NE(labelData[2 * width + 2], labelData[3 * width + 3]);
  EXPECT_NE(labelData[1 * width + 1], labelData[3 * width + 3]);
}

// 测试 nppiLabelMarkersUF_8u32u_C1R_Ctx - 8 邻域下对角连接应该是同一区域
TEST_F(NPPILabelMarkersUFTest, nppiLabelMarkersUF_8u32u_C1R_Ctx_8Way_Diagonal) {
  // 同样的对角连接图像，在 8 邻域下应该是 1 个区域
  std::vector<Npp8u> srcData(width * height, 0);
  srcData[1 * width + 1] = 255;
  srcData[2 * width + 2] = 255;
  srcData[3 * width + 3] = 255;

  int bufferSize = 0;
  NppStatus status = nppiLabelMarkersUFGetBufferSize_32u_C1R(oSizeROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  Npp8u *d_src = nullptr;
  Npp32u *d_labels = nullptr;
  Npp8u *d_buffer = nullptr;
  int srcStep = width * sizeof(Npp8u);
  int labelsStep = width * sizeof(Npp32u);

  cudaMalloc(&d_src, width * height * sizeof(Npp8u));
  cudaMalloc(&d_labels, width * height * sizeof(Npp32u));
  cudaMalloc(&d_buffer, bufferSize);

  struct ResourceGuard {
    Npp8u *src;
    Npp32u *labels;
    Npp8u *buffer;
    ~ResourceGuard() {
      if (src) cudaFree(src);
      if (labels) cudaFree(labels);
      if (buffer) cudaFree(buffer);
    }
  } guard{d_src, d_labels, d_buffer};

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_labels, nullptr);
  ASSERT_NE(d_buffer, nullptr);

  cudaMemcpy(d_src, srcData.data(), width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
  cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
                         cudaDevAttrComputeCapabilityMajor, nppStreamCtx.nCudaDeviceId);
  cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
                         cudaDevAttrComputeCapabilityMinor, nppStreamCtx.nCudaDeviceId);
  cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, nppStreamCtx.nCudaDeviceId);
  nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
  nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  nppStreamCtx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;

  // 执行 8 邻域标记 (nppiNormInf)
  status = nppiLabelMarkersUF_8u32u_C1R_Ctx(d_src, srcStep, d_labels, labelsStep,
                                             oSizeROI, nppiNormInf, d_buffer, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32u> labelData(width * height);
  cudaMemcpy(labelData.data(), d_labels, width * height * sizeof(Npp32u), cudaMemcpyDeviceToHost);

  // 在 8 邻域下，对角连接的 3 个像素应该是 1 个区域
  std::set<Npp32u> uniqueLabels;
  for (size_t i = 0; i < labelData.size(); i++) {
    if (labelData[i] > 0) {
      uniqueLabels.insert(labelData[i]);
    }
  }

  EXPECT_EQ(uniqueLabels.size(), 1);

  // 所有像素应该有相同的标签
  EXPECT_EQ(labelData[1 * width + 1], labelData[2 * width + 2]);
  EXPECT_EQ(labelData[2 * width + 2], labelData[3 * width + 3]);
}

// 测试 nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx
TEST_F(NPPILabelMarkersUFTest, nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx_Basic) {
  const int numImages = 3;
  int imgWidth = 16;
  int imgHeight = 16;

  // 准备多个测试图像
  std::vector<std::vector<Npp8u>> srcImages(numImages);
  for (int i = 0; i < numImages; i++) {
    srcImages[i].resize(imgWidth * imgHeight, 0);
    // 每个图像创建不同数量的区域
    for (int r = 0; r <= i; r++) {
      int startX = r * 4 + 1;
      int startY = 1;
      if (startX + 2 < imgWidth) {
        srcImages[i][startY * imgWidth + startX] = 255;
        srcImages[i][startY * imgWidth + startX + 1] = 255;
        srcImages[i][(startY + 1) * imgWidth + startX] = 255;
        srcImages[i][(startY + 1) * imgWidth + startX + 1] = 255;
      }
    }
  }

  // 获取缓冲区大小
  NppiSize batchROI = {imgWidth, imgHeight};
  int bufferSize = 0;
  NppStatus status = nppiLabelMarkersUFGetBufferSize_32u_C1R(batchROI, &bufferSize);
  ASSERT_EQ(status, NPP_SUCCESS);

  // 分配 GPU 内存
  std::vector<Npp8u*> d_srcList(numImages);
  std::vector<Npp32u*> d_labelsList(numImages);
  Npp8u *d_buffer = nullptr;
  int srcStep = imgWidth * sizeof(Npp8u);
  int labelsStep = imgWidth * sizeof(Npp32u);

  for (int i = 0; i < numImages; i++) {
    cudaMalloc(&d_srcList[i], imgWidth * imgHeight * sizeof(Npp8u));
    cudaMalloc(&d_labelsList[i], imgWidth * imgHeight * sizeof(Npp32u));
    cudaMemcpy(d_srcList[i], srcImages[i].data(), imgWidth * imgHeight * sizeof(Npp8u), cudaMemcpyHostToDevice);
  }
  cudaMalloc(&d_buffer, bufferSize);

  // 准备批量描述符
  std::vector<NppiImageDescriptor> srcBatchList(numImages);
  std::vector<NppiImageDescriptor> dstBatchList(numImages);

  for (int i = 0; i < numImages; i++) {
    srcBatchList[i].pData = d_srcList[i];
    srcBatchList[i].nStep = srcStep;
    srcBatchList[i].oSize = batchROI;

    dstBatchList[i].pData = d_labelsList[i];
    dstBatchList[i].nStep = labelsStep;
    dstBatchList[i].oSize = batchROI;
  }

  // 分配设备端描述符
  NppiImageDescriptor *d_srcBatchList = nullptr;
  NppiImageDescriptor *d_dstBatchList = nullptr;
  cudaMalloc(&d_srcBatchList, numImages * sizeof(NppiImageDescriptor));
  cudaMalloc(&d_dstBatchList, numImages * sizeof(NppiImageDescriptor));
  cudaMemcpy(d_srcBatchList, srcBatchList.data(), numImages * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dstBatchList, dstBatchList.data(), numImages * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice);

  // 清理资源的 guard
  struct ResourceGuard {
    std::vector<Npp8u*> &srcList;
    std::vector<Npp32u*> &labelsList;
    Npp8u *buffer;
    NppiImageDescriptor *srcBatch;
    NppiImageDescriptor *dstBatch;
    int n;
    ~ResourceGuard() {
      for (int i = 0; i < n; i++) {
        if (srcList[i]) cudaFree(srcList[i]);
        if (labelsList[i]) cudaFree(labelsList[i]);
      }
      if (buffer) cudaFree(buffer);
      if (srcBatch) cudaFree(srcBatch);
      if (dstBatch) cudaFree(dstBatch);
    }
  } guard{d_srcList, d_labelsList, d_buffer, d_srcBatchList, d_dstBatchList, numImages};

  // 设置 stream context
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
  cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
                         cudaDevAttrComputeCapabilityMajor, nppStreamCtx.nCudaDeviceId);
  cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
                         cudaDevAttrComputeCapabilityMinor, nppStreamCtx.nCudaDeviceId);
  cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, nppStreamCtx.nCudaDeviceId);
  nppStreamCtx.nMultiProcessorCount = prop.multiProcessorCount;
  nppStreamCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  nppStreamCtx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  nppStreamCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;

  // 执行批量标记
  status = nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(
      d_srcBatchList, d_dstBatchList, numImages, batchROI, nppiNormInf, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 验证每个图像的结果
  for (int i = 0; i < numImages; i++) {
    std::vector<Npp32u> labelData(imgWidth * imgHeight);
    cudaMemcpy(labelData.data(), d_labelsList[i], imgWidth * imgHeight * sizeof(Npp32u), cudaMemcpyDeviceToHost);

    std::set<Npp32u> uniqueLabels;
    for (size_t j = 0; j < labelData.size(); j++) {
      if (labelData[j] > 0) {
        uniqueLabels.insert(labelData[j]);
      }
    }

    // 图像 i 应该有 i+1 个区域
    EXPECT_EQ(uniqueLabels.size(), i + 1) << "Image " << i << " should have " << (i + 1) << " regions";
  }
}
