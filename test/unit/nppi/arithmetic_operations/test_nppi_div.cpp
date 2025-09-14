/**
 * @file test_nppi_div.cpp
 * @brief NPP 除法函数测试
 */

#include "../../framework/npp_test_base.h"
#include "../../../common/npp_test_utils.h"

using namespace npp_functional_test;

class DivFunctionalTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
};

// NOTE: 此测试被禁用 - NVIDIA NPP的nppiDiv_8u_C1RSfs函数存在严重缺陷
// 该函数总是返回0而非正确的除法结果，这是NVIDIA NPP库的已知问题
TEST_F(DivFunctionalTest, Div_8u_C1RSfs_BasicOperation) {
    const int width = 32;
    const int height = 32;
    const int scaleFactor = 0;
    
    std::vector<Npp8u> srcData1(width * height);
    std::vector<Npp8u> srcData2(width * height);
    std::vector<Npp8u> expectedData(width * height);
    
    // NVIDIA NPP division: pDst = pSrc2 / pSrc1, not pSrc1 / pSrc2
    // So to get 100/5=20, we need src1=5, src2=100
    TestDataGenerator::generateConstant(srcData1, static_cast<Npp8u>(5));   // divisor
    TestDataGenerator::generateConstant(srcData2, static_cast<Npp8u>(100)); // dividend
    TestDataGenerator::generateConstant(expectedData, static_cast<Npp8u>(20)); // 100/5=20
    
    NppImageMemory<Npp8u> src1(width, height);
    NppImageMemory<Npp8u> src2(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src1.copyFromHost(srcData1);
    src2.copyFromHost(srcData2);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiDiv_8u_C1RSfs(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi, scaleFactor);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiDiv_8u_C1RSfs failed";
    
    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    
    // Verify results using precision control system for div operation
    for (size_t i = 0; i < resultData.size(); i++) {
        NPP_EXPECT_ARITHMETIC_EQUAL(resultData[i], expectedData[i], "nppiDiv_8u_C1RSfs")
            << "Div operation mismatch at index " << i 
            << ": got " << (int)resultData[i] << ", expected " << (int)expectedData[i];
    }
}

// NOTE: 此测试被禁用 - NVIDIA NPP的除法函数存在复杂的参数顺序和缩放问题
// 需要进一步研究其确切行为模式  
TEST_F(DivFunctionalTest, Div_32f_C1R_BasicOperation) {
    const int width = 32;
    const int height = 32;
    
    std::vector<Npp32f> srcData1(width * height);
    std::vector<Npp32f> srcData2(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    // NVIDIA NPP division: pDst = pSrc2 / pSrc1, not pSrc1 / pSrc2
    // To get 20/4=5, we need src1=4, src2=20
    TestDataGenerator::generateConstant(srcData1, 4.0f);  // divisor
    TestDataGenerator::generateConstant(srcData2, 20.0f); // dividend  
    TestDataGenerator::generateConstant(expectedData, 5.0f); // 20/4=5
    
    NppImageMemory<Npp32f> src1(width, height);
    NppImageMemory<Npp32f> src2(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    src1.copyFromHost(srcData1);
    src2.copyFromHost(srcData2);
    
    NppiSize roi = {width, height};
    NppStatus status = nppiDiv_32f_C1R(
        src1.get(), src1.step(),
        src2.get(), src2.step(),
        dst.get(), dst.step(),
        roi);
    
    ASSERT_EQ(status, NPP_NO_ERROR) << "nppiDiv_32f_C1R failed";
    
    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    
    // Verify results using precision control system for div operation
    for (size_t i = 0; i < resultData.size(); i++) {
        NPP_EXPECT_ARITHMETIC_EQUAL(resultData[i], expectedData[i], "nppiDiv_32f_C1R")
            << "Div 32f operation mismatch at index " << i 
            << ": got " << resultData[i] << ", expected " << expectedData[i];
    }
}