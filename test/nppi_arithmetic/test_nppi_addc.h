#ifndef TEST_NPPI_ADDC_H
#define TEST_NPPI_ADDC_H

#include "../framework/nppi_arithmetic_test.h"

namespace NPPTest {

// Forward declarations for our implementations
extern "C" {
NppStatus nppiAddC_8u_C1RSfs_Ctx_cuda(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                                      Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                      NppStreamContext nppStreamCtx);

NppStatus nppiAddC_16u_C1RSfs_Ctx_cuda(const Npp16u* pSrc1, int nSrc1Step, const Npp16u nConstant,
                                       Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx);

NppStatus nppiAddC_16s_C1RSfs_Ctx_cuda(const Npp16s* pSrc1, int nSrc1Step, const Npp16s nConstant,
                                       Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor,
                                       NppStreamContext nppStreamCtx);

NppStatus nppiAddC_32f_C1R_Ctx_cuda(const Npp32f* pSrc1, int nSrc1Step, const Npp32f nConstant,
                                    Npp32f* pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
}

/**
 * 8u AddC测试
 */
class AddC_8u_C1RSfs_Test : public AddCTestBase<Npp8u> {
public:
    AddC_8u_C1RSfs_Test() : AddCTestBase<Npp8u>("8u_C1RSfs") {}
    
protected:
    NppStatus runGPUImplementation(const ImageBuffer<Npp8u>& src, ImageBuffer<Npp8u>& dst,
                                 const AddCTestParameters<Npp8u>& params) override {
        return nppiAddC_8u_C1RSfs_Ctx(
            src.deviceData(), src.stepBytes(), params.constant,
            dst.deviceData(), dst.stepBytes(), src.size(), params.scale_factor, stream_ctx_
        );
    }
    
    NppStatus runNVIDIAImplementation(const ImageBuffer<Npp8u>& src, ImageBuffer<Npp8u>& dst,
                                    const AddCTestParameters<Npp8u>& params) override {
#ifdef HAVE_NVIDIA_NPP
        return nppiAddC_8u_C1RSfs_Ctx(
            src.deviceData(), src.stepBytes(), params.constant,
            dst.deviceData(), dst.stepBytes(), src.size(), params.scale_factor, stream_ctx_
        );
#else
        return NPP_NOT_SUPPORTED_MODE_ERROR;
#endif
    }
    
public:
    std::vector<std::unique_ptr<TestParameters>> generateTestCases() const override {
        std::vector<std::unique_ptr<TestParameters>> cases;
        
        // 简化的测试用例以避免过多内存分配
        cases.emplace_back(std::make_unique<AddCTestParameters<Npp8u>>(32, 32, 50, 1, 1, 0, 0.0));
        cases.emplace_back(std::make_unique<AddCTestParameters<Npp8u>>(64, 64, 100, 1, 2, 0, 0.0));
        cases.emplace_back(std::make_unique<AddCTestParameters<Npp8u>>(32, 32, 255, 1, 0, 0, 0.0));
        
        return cases;
    }
};

/**
 * 16u AddC测试
 */
class AddC_16u_C1RSfs_Test : public AddCTestBase<Npp16u> {
public:
    AddC_16u_C1RSfs_Test() : AddCTestBase<Npp16u>("16u_C1RSfs") {}
    
protected:
    NppStatus runGPUImplementation(const ImageBuffer<Npp16u>& src, ImageBuffer<Npp16u>& dst,
                                 const AddCTestParameters<Npp16u>& params) override {
        return nppiAddC_16u_C1RSfs_Ctx(
            src.deviceData(), src.stepBytes(), params.constant,
            dst.deviceData(), dst.stepBytes(), src.size(), params.scale_factor, stream_ctx_
        );
    }
    
    NppStatus runNVIDIAImplementation(const ImageBuffer<Npp16u>& src, ImageBuffer<Npp16u>& dst,
                                    const AddCTestParameters<Npp16u>& params) override {
#ifdef HAVE_NVIDIA_NPP
        return nppiAddC_16u_C1RSfs_Ctx(
            src.deviceData(), src.stepBytes(), params.constant,
            dst.deviceData(), dst.stepBytes(), src.size(), params.scale_factor, stream_ctx_
        );
#else
        return NPP_NOT_SUPPORTED_MODE_ERROR;
#endif
    }
    
public:
    std::vector<std::unique_ptr<TestParameters>> generateTestCases() const override {
        std::vector<std::unique_ptr<TestParameters>> cases;
        
        cases.emplace_back(std::make_unique<AddCTestParameters<Npp16u>>(32, 32, 1000, 1, 1, 0, 0.0));
        cases.emplace_back(std::make_unique<AddCTestParameters<Npp16u>>(64, 64, 10000, 1, 2, 0, 0.0));
        
        return cases;
    }
};

/**
 * 16s AddC测试
 */
class AddC_16s_C1RSfs_Test : public AddCTestBase<Npp16s> {
public:
    AddC_16s_C1RSfs_Test() : AddCTestBase<Npp16s>("16s_C1RSfs") {}
    
protected:
    NppStatus runGPUImplementation(const ImageBuffer<Npp16s>& src, ImageBuffer<Npp16s>& dst,
                                 const AddCTestParameters<Npp16s>& params) override {
        return nppiAddC_16s_C1RSfs_Ctx(
            src.deviceData(), src.stepBytes(), params.constant,
            dst.deviceData(), dst.stepBytes(), src.size(), params.scale_factor, stream_ctx_
        );
    }
    
    NppStatus runNVIDIAImplementation(const ImageBuffer<Npp16s>& src, ImageBuffer<Npp16s>& dst,
                                    const AddCTestParameters<Npp16s>& params) override {
#ifdef HAVE_NVIDIA_NPP
        return nppiAddC_16s_C1RSfs_Ctx(
            src.deviceData(), src.stepBytes(), params.constant,
            dst.deviceData(), dst.stepBytes(), src.size(), params.scale_factor, stream_ctx_
        );
#else
        return NPP_NOT_SUPPORTED_MODE_ERROR;
#endif
    }
    
public:
    std::vector<std::unique_ptr<TestParameters>> generateTestCases() const override {
        std::vector<std::unique_ptr<TestParameters>> cases;
        
        cases.emplace_back(std::make_unique<AddCTestParameters<Npp16s>>(32, 32, 1000, 1, 1, 0, 0.0));
        cases.emplace_back(std::make_unique<AddCTestParameters<Npp16s>>(32, 32, -1000, 1, 1, 0, 0.0));
        
        return cases;
    }
};

/**
 * 32f AddC测试
 */
class AddC_32f_C1R_Test : public AddCTestBase<Npp32f> {
public:
    AddC_32f_C1R_Test() : AddCTestBase<Npp32f>("32f_C1R") {}
    
protected:
    NppStatus runGPUImplementation(const ImageBuffer<Npp32f>& src, ImageBuffer<Npp32f>& dst,
                                 const AddCTestParameters<Npp32f>& params) override {
        return nppiAddC_32f_C1R_Ctx(
            src.deviceData(), src.stepBytes(), params.constant,
            dst.deviceData(), dst.stepBytes(), src.size(), stream_ctx_
        );
    }
    
    NppStatus runNVIDIAImplementation(const ImageBuffer<Npp32f>& src, ImageBuffer<Npp32f>& dst,
                                    const AddCTestParameters<Npp32f>& params) override {
#ifdef HAVE_NVIDIA_NPP
        return nppiAddC_32f_C1R_Ctx(
            src.deviceData(), src.stepBytes(), params.constant,
            dst.deviceData(), dst.stepBytes(), src.size(), stream_ctx_
        );
#else
        return NPP_NOT_SUPPORTED_MODE_ERROR;
#endif
    }
    
public:
    std::vector<std::unique_ptr<TestParameters>> generateTestCases() const override {
        std::vector<std::unique_ptr<TestParameters>> cases;
        
        cases.emplace_back(std::make_unique<AddCTestParameters<Npp32f>>(32, 32, 25.5f, 1, 0, 0, 1e-6));
        cases.emplace_back(std::make_unique<AddCTestParameters<Npp32f>>(32, 32, -15.5f, 1, 0, 0, 1e-6));
        
        return cases;
    }
};

// 注册函数声明
void registerAddCTests();

} // namespace NPPTest

#endif // TEST_NPPI_ADDC_H