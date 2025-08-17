#ifndef NPPI_ARITHMETIC_TEST_H
#define NPPI_ARITHMETIC_TEST_H

#include "npp_test_core.h"
#include "nppi_arithmetic_and_logical_operations.h"

namespace NPPTest {

/**
 * 算术运算测试参数
 */
class ArithmeticTestParameters : public TestParameters {
public:
    int width;
    int height;
    int channels;
    int scale_factor;
    int pattern_type;
    double tolerance;
    
    ArithmeticTestParameters(int w, int h, int ch = 1, int sf = 0, int pt = 0, double tol = 1e-6)
        : width(w), height(h), channels(ch), scale_factor(sf), pattern_type(pt), tolerance(tol) {}
    
    std::string description() const override {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "%dx%dx%d, scale=%d, pattern=%d", 
                width, height, channels, scale_factor, pattern_type);
        return std::string(buffer);
    }
    
    bool isValid() const override {
        return width > 0 && height > 0 && channels > 0 && 
               scale_factor >= 0 && scale_factor <= 31;
    }
};

/**
 * AddC运算测试参数（包含常数）
 */
template<typename T>
class AddCTestParameters : public ArithmeticTestParameters {
public:
    T constant;
    
    AddCTestParameters(int w, int h, T c, int ch = 1, int sf = 0, int pt = 0, double tol = 1e-6)
        : ArithmeticTestParameters(w, h, ch, sf, pt, tol), constant(c) {}
    
    std::string description() const override {
        char buffer[512];
        snprintf(buffer, sizeof(buffer), "%dx%dx%d, const=%.2f, scale=%d, pattern=%d", 
                width, height, channels, static_cast<double>(constant), scale_factor, pattern_type);
        return std::string(buffer);
    }
};

/**
 * CPU参考实现函数
 */
namespace CPUReference {

template<typename T>
void addC_ref(const T* src, int src_step, T constant, T* dst, int dst_step, 
              int width, int height, int scale_factor = 0) {
    for (int y = 0; y < height; y++) {
        const T* src_row = reinterpret_cast<const T*>(
            reinterpret_cast<const char*>(src) + y * src_step);
        T* dst_row = reinterpret_cast<T*>(
            reinterpret_cast<char*>(dst) + y * dst_step);
        
        for (int x = 0; x < width; x++) {
            if (std::is_floating_point<T>::value) {
                // 浮点数不需要缩放
                dst_row[x] = src_row[x] + constant;
            } else {
                // 整数需要缩放和截取
                int result = static_cast<int>(src_row[x]) + static_cast<int>(constant);
                result = result >> scale_factor;
                
                // 截取到数据类型范围
                if (std::is_signed<T>::value) {
                    result = std::max(static_cast<int>(std::numeric_limits<T>::min()),
                                    std::min(static_cast<int>(std::numeric_limits<T>::max()), result));
                } else {
                    result = std::max(0, std::min(static_cast<int>(std::numeric_limits<T>::max()), result));
                }
                dst_row[x] = static_cast<T>(result);
            }
        }
    }
}

template<typename T>
void addC_C3_ref(const T* src, int src_step, const T constants[3], T* dst, int dst_step,
                 int width, int height, int scale_factor = 0) {
    for (int y = 0; y < height; y++) {
        const T* src_row = reinterpret_cast<const T*>(
            reinterpret_cast<const char*>(src) + y * src_step);
        T* dst_row = reinterpret_cast<T*>(
            reinterpret_cast<char*>(dst) + y * dst_step);
        
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                int idx = x * 3 + c;
                if (std::is_floating_point<T>::value) {
                    dst_row[idx] = src_row[idx] + constants[c];
                } else {
                    int result = static_cast<int>(src_row[idx]) + static_cast<int>(constants[c]);
                    result = result >> scale_factor;
                    
                    if (std::is_signed<T>::value) {
                        result = std::max(static_cast<int>(std::numeric_limits<T>::min()),
                                        std::min(static_cast<int>(std::numeric_limits<T>::max()), result));
                    } else {
                        result = std::max(0, std::min(static_cast<int>(std::numeric_limits<T>::max()), result));
                    }
                    dst_row[idx] = static_cast<T>(result);
                }
            }
        }
    }
}

} // namespace CPUReference

/**
 * AddC运算测试基类
 */
template<typename T>
class AddCTestBase : public NPPTestBase {
public:
    AddCTestBase(const std::string& data_type_name) 
        : NPPTestBase("nppiAddC_" + data_type_name, "nppi_arithmetic_and_logical_operations") {}
    
    TestResult runTest(const TestParameters& params) override {
        const auto& add_params = static_cast<const AddCTestParameters<T>&>(params);
        TestResult result;
        
        try {
            // 创建图像缓冲区
            ImageBuffer<T> src(add_params.width, add_params.height, add_params.channels);
            ImageBuffer<T> dst_cpu(add_params.width, add_params.height, add_params.channels);
            ImageBuffer<T> dst_gpu(add_params.width, add_params.height, add_params.channels);
            ImageBuffer<T> dst_nvidia(add_params.width, add_params.height, add_params.channels);
            
            // 填充测试数据
            if (std::is_floating_point<T>::value) {
                src.fillRandom(static_cast<T>(100.0));
            } else {
                src.fillRandom();
            }
            src.copyToDevice();
            
            // 运行CPU参考实现
            Timer cpu_timer(false);
            cpu_timer.start();
            
            if (add_params.channels == 1) {
                CPUReference::addC_ref(src.hostData(), src.stepBytes(), add_params.constant,
                                     dst_cpu.hostData(), dst_cpu.stepBytes(),
                                     add_params.width, add_params.height, add_params.scale_factor);
            } else if (add_params.channels == 3) {
                T constants[3] = {add_params.constant, add_params.constant, add_params.constant};
                CPUReference::addC_C3_ref(src.hostData(), src.stepBytes(), constants,
                                        dst_cpu.hostData(), dst_cpu.stepBytes(),
                                        add_params.width, add_params.height, add_params.scale_factor);
            }
            
            result.cpu_time_ms = cpu_timer.stop();
            
            // 运行GPU实现
            Timer gpu_timer(true);
            gpu_timer.start();
            
            NppStatus gpu_status = runGPUImplementation(src, dst_gpu, add_params);
            
            cudaDeviceSynchronize();
            result.gpu_time_ms = gpu_timer.stop();
            
            if (gpu_status != NPP_NO_ERROR) {
                result.error_message = "GPU implementation failed with status " + std::to_string(gpu_status);
                return result;
            }
            
            dst_gpu.copyFromDevice();
            
            // 比较CPU vs GPU
            result.max_error_vs_cpu = dst_cpu.compareWith(dst_gpu, add_params.tolerance);
            if (result.max_error_vs_cpu > add_params.tolerance) {
                result.error_message = "CPU vs GPU error: " + std::to_string(result.max_error_vs_cpu);
                dst_cpu.printFirstDifferences(dst_gpu, 5);
                return result;
            }
            
            // 运行NVIDIA实现（如果可用）
            if (nvidia_npp_available_) {
                Timer nvidia_timer(true);
                nvidia_timer.start();
                
                NppStatus nvidia_status = runNVIDIAImplementation(src, dst_nvidia, add_params);
                
                cudaDeviceSynchronize();
                result.nvidia_time_ms = nvidia_timer.stop();
                
                if (nvidia_status == NPP_NO_ERROR) {
                    dst_nvidia.copyFromDevice();
                    
                    // 比较GPU vs NVIDIA
                    result.max_error_vs_nvidia = dst_gpu.compareWith(dst_nvidia, add_params.tolerance);
                    if (result.max_error_vs_nvidia > add_params.tolerance) {
                        result.error_message = "GPU vs NVIDIA error: " + std::to_string(result.max_error_vs_nvidia);
                        dst_gpu.printFirstDifferences(dst_nvidia, 5);
                        return result;
                    }
                } else {
                    result.max_error_vs_nvidia = -1.0; // 表示NVIDIA实现不可用
                }
            } else {
                result.max_error_vs_nvidia = -1.0; // 表示NVIDIA实现不可用
            }
            
            result.passed = true;
            
        } catch (const std::exception& e) {
            result.error_message = std::string("Exception: ") + e.what();
        }
        
        return result;
    }
    
protected:
    virtual NppStatus runGPUImplementation(const ImageBuffer<T>& src, ImageBuffer<T>& dst, 
                                         const AddCTestParameters<T>& params) = 0;
    virtual NppStatus runNVIDIAImplementation(const ImageBuffer<T>& src, ImageBuffer<T>& dst,
                                            const AddCTestParameters<T>& params) = 0;
};

} // namespace NPPTest

#endif // NPPI_ARITHMETIC_TEST_H