/**
 * @file test_nppi_addc.cpp
 * @brief NPP 加常数函数测试
 */

#include "../../framework/npp_test_base.h"

using namespace npp_functional_test;

class AddcFunctionalTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
};

TEST_F(AddcFunctionalTest, AddC_8u_C1RSfs_BasicOperation) {
    const int width = 32;
    const int height = 32;
    const int scaleFactor = 0;
    const Npp8u nConstant = 50;
    
    std::vector<Npp8u> srcData(width * height);
    std::vector<Npp8u> expectedData(width * height);
    
    TestDataGenerator::generateConstant(srcData, static_cast<Npp8u>(30));
    TestDataGenerator::generateConstant(expectedData, static_cast<Npp8u>(80));
    
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src.copyFromHost(srcData);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiAddC_8u_C1RSfs(
        src.get(), src.step(),
        nConstant,
        dst.get(), dst.step(),
        roi, scaleFactor);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_8u_C1RSfs failed";
    
    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData))
        << "AddC operation produced incorrect results";
}

TEST_F(AddcFunctionalTest, AddC_32f_C1R_BasicOperation) {
    const int width = 32;
    const int height = 32;
    const Npp32f nConstant = 7.5f;
    
    std::vector<Npp32f> srcData(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    TestDataGenerator::generateConstant(srcData, 2.5f);
    TestDataGenerator::generateConstant(expectedData, 10.0f);
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    src.copyFromHost(srcData);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiAddC_32f_C1R(
        src.get(), src.step(),
        nConstant,
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiAddC_32f_C1R failed";
    
    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    
    EXPECT_TRUE(ResultValidator::arraysEqual(resultData, expectedData, 1e-5f))
        << "AddC 32f operation produced incorrect results";
}