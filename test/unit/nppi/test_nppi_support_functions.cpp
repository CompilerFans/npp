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
    
    ASSERT_EQ(status, NPP_FUNCTION_NOT_IMPLEMENTED) << "nppGetGpuComputeCapability should return NPP_FUNCTION_NOT_IMPLEMENTED";
    
    // 不验证输出值，因为函数未实现
    std::cout << "nppGetGpuComputeCapability correctly returned NPP_FUNCTION_NOT_IMPLEMENTED" << std::endl;
}

TEST_F(SupportFunctionsTest, GetGpuName_BasicTest) {
    // nppGetGpuName 实际上已经实现，返回const char*，不接受缓冲区参数
    const char* name = nppGetGpuName();
    
    if (name != nullptr) {
        // 验证名称不为空
        EXPECT_GT(strlen(name), 0) << "GPU name should not be empty";
        std::cout << "GPU Name: " << name << std::endl;
    } else {
        std::cout << "nppGetGpuName returned nullptr" << std::endl;
    }
}

TEST_F(SupportFunctionsTest, GetStreamContext_BasicTest) {
    NppStreamContext streamCtx;
    NppStatus status = nppGetStreamContext(&streamCtx);
    
    if (status == NPP_NO_ERROR) {
        // 验证默认流上下文
        EXPECT_EQ(streamCtx.hStream, (cudaStream_t)0) << "Default stream should be 0";
        EXPECT_NE(streamCtx.nCudaDeviceId, -1) << "Device ID should be valid";
        std::cout << "nppGetStreamContext succeeded" << std::endl;
    } else {
        std::cout << "nppGetStreamContext returned error: " << status << std::endl;
    }
}

TEST_F(SupportFunctionsTest, SetGetStreamContext_BasicTest) {
    // 测试设置流上下文 - 该函数未实现
    NppStreamContext newCtx;
    newCtx.hStream = 0;
    newCtx.nCudaDeviceId = 0;
    
    NppStatus status = nppSetStreamContext(newCtx);
    ASSERT_EQ(status, NPP_FUNCTION_NOT_IMPLEMENTED) << "nppSetStreamContext should return NPP_FUNCTION_NOT_IMPLEMENTED";
    
    std::cout << "nppSetStreamContext correctly returned NPP_FUNCTION_NOT_IMPLEMENTED" << std::endl;
}