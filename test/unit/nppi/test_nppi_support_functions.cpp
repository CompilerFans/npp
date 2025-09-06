/**
 * @file test_nppi_support_functions.cpp
 * @brief NPP 支持函数测试
 */

#include "../framework/npp_test_base.h"

using namespace npp_functional_test;

class SupportFunctionsTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
};

TEST_F(SupportFunctionsTest, GetLibVersion_BasicTest) {
    const NppLibraryVersion* version = nppGetLibVersion();
    ASSERT_NE(version, nullptr) << "nppGetLibVersion should not return null";
    
    // 验证版本号格式合理
    EXPECT_GT(version->major, 0) << "Major version should be greater than 0";
    EXPECT_GE(version->minor, 0) << "Minor version should be non-negative";
    EXPECT_GE(version->build, 0) << "Build version should be non-negative";
    
    std::cout << "NPP Library Version: " << version->major << "." 
              << version->minor << "." << version->build << std::endl;
}

TEST_F(SupportFunctionsTest, GetGpuComputeCapability_BasicTest) {
    int major, minor;
    NppStatus status = nppGetGpuComputeCapability(&major, &minor);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppGetGpuComputeCapability should succeed";
    
    // 验证计算能力合理
    EXPECT_GT(major, 0) << "Compute capability major should be positive";
    EXPECT_GE(minor, 0) << "Compute capability minor should be non-negative";
    
    std::cout << "GPU Compute Capability: " << major << "." << minor << std::endl;
}

TEST_F(SupportFunctionsTest, GetGpuName_BasicTest) {
    char name[256];
    NppStatus status = nppGetGpuName(name, sizeof(name));
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppGetGpuName should succeed";
    
    // 验证名称不为空
    EXPECT_GT(strlen(name), 0) << "GPU name should not be empty";
    
    std::cout << "GPU Name: " << name << std::endl;
}

TEST_F(SupportFunctionsTest, GetStreamContext_BasicTest) {
    NppStreamContext streamCtx = nppGetStreamContext();
    
    // 验证默认流上下文
    EXPECT_EQ(streamCtx.hStream, 0) << "Default stream should be 0";
    EXPECT_NE(streamCtx.nCudaDeviceId, -1) << "Device ID should be valid";
}

TEST_F(SupportFunctionsTest, SetGetStreamContext_BasicTest) {
    // 获取原始上下文
    NppStreamContext originalCtx = nppGetStreamContext();
    
    // 创建新的流上下文
    NppStreamContext newCtx = originalCtx;
    // 这里我们不创建新流，只是验证设置/获取功能
    
    nppSetStreamContext(newCtx);
    NppStreamContext retrievedCtx = nppGetStreamContext();
    
    EXPECT_EQ(retrievedCtx.hStream, newCtx.hStream) << "Stream context should match";
    EXPECT_EQ(retrievedCtx.nCudaDeviceId, newCtx.nCudaDeviceId) << "Device ID should match";
    
    // 恢复原始上下文
    nppSetStreamContext(originalCtx);
}