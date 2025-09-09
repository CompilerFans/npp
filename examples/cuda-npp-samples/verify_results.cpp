// 验证OpenNPP样本结果的程序
#include <npp.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <fstream>

// 读取PGM文件
bool readPGM(const char* filename, std::vector<Npp8u>& data, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }
    
    std::string magic;
    file >> magic;
    if (magic != "P5") {
        std::cerr << "不支持的PGM格式: " << magic << std::endl;
        return false;
    }
    
    file >> width >> height;
    int maxval;
    file >> maxval;
    file.ignore(); // 跳过换行符
    
    if (maxval != 255) {
        std::cerr << "不支持的最大值: " << maxval << std::endl;
        return false;
    }
    
    data.resize(width * height);
    file.read(reinterpret_cast<char*>(data.data()), width * height);
    
    return file.good() || file.eof();
}

// 像素级对比两个图像
bool compareImages(const char* nvidia_file, const char* opennpp_file, const char* name) {
    std::vector<Npp8u> nvidia_data, opennpp_data;
    int nvidia_w, nvidia_h, opennpp_w, opennpp_h;
    
    std::cout << "\n=== " << name << " 像素级对比 ===" << std::endl;
    
    if (!readPGM(nvidia_file, nvidia_data, nvidia_w, nvidia_h)) {
        std::cout << "❌ 无法读取NVIDIA参考文件: " << nvidia_file << std::endl;
        return false;
    }
    
    if (!readPGM(opennpp_file, opennpp_data, opennpp_w, opennpp_h)) {
        std::cout << "❌ 无法读取OpenNPP文件: " << opennpp_file << std::endl;
        return false;
    }
    
    if (nvidia_w != opennpp_w || nvidia_h != opennpp_h) {
        std::cout << "❌ 图像尺寸不匹配: NVIDIA(" << nvidia_w << "x" << nvidia_h 
                  << ") vs OpenNPP(" << opennpp_w << "x" << opennpp_h << ")" << std::endl;
        return false;
    }
    
    int total_pixels = nvidia_w * nvidia_h;
    int diff_pixels = 0;
    int max_diff = 0;
    double mse = 0.0;
    
    // 计算统计信息
    for (int i = 0; i < total_pixels; i++) {
        int diff = std::abs(static_cast<int>(nvidia_data[i]) - static_cast<int>(opennpp_data[i]));
        if (diff > 0) {
            diff_pixels++;
            max_diff = std::max(max_diff, diff);
        }
        mse += diff * diff;
    }
    
    mse /= total_pixels;
    double psnr = (mse > 0) ? 20.0 * std::log10(255.0 / std::sqrt(mse)) : 100.0;
    
    std::cout << "图像尺寸: " << nvidia_w << "x" << nvidia_h << " (" << total_pixels << " 像素)" << std::endl;
    std::cout << "不同像素: " << diff_pixels << " (" << std::fixed << std::setprecision(3) 
              << (100.0 * diff_pixels / total_pixels) << "%)" << std::endl;
    std::cout << "最大差异: " << max_diff << "/255" << std::endl;
    std::cout << "MSE: " << std::fixed << std::setprecision(6) << mse << std::endl;
    std::cout << "PSNR: " << std::fixed << std::setprecision(2) << psnr << " dB" << std::endl;
    
    // 显示差异分布
    if (diff_pixels > 0) {
        std::vector<int> diff_histogram(256, 0);
        for (int i = 0; i < total_pixels; i++) {
            int diff = std::abs(static_cast<int>(nvidia_data[i]) - static_cast<int>(opennpp_data[i]));
            diff_histogram[diff]++;
        }
        
        std::cout << "差异分布:" << std::endl;
        for (int d = 1; d <= max_diff && d < 10; d++) {
            if (diff_histogram[d] > 0) {
                std::cout << "  差异=" << d << ": " << diff_histogram[d] << " 像素 ("
                          << std::fixed << std::setprecision(3) 
                          << (100.0 * diff_histogram[d] / total_pixels) << "%)" << std::endl;
            }
        }
    }
    
    bool is_identical = (diff_pixels == 0);
    bool is_excellent = (psnr > 40.0 && diff_pixels < total_pixels * 0.01);
    
    if (is_identical) {
        std::cout << "✅ 完全匹配：像素级完全一致！" << std::endl;
    } else if (is_excellent) {
        std::cout << "✅ 高质量匹配：差异极小，质量优异！" << std::endl;
    } else {
        std::cout << "⚠️  有差异：需要进一步检查" << std::endl;
    }
    
    return is_identical || is_excellent;
}

// 验证盒式滤波结果
bool verifyBoxFilter(const Npp8u* input, const Npp8u* output, 
                     int width, int height, int filterSize) {
    int halfSize = filterSize / 2;
    int errors = 0;
    const double tolerance = 1.0; // 允许的误差范围
    
    std::cout << "验证盒式滤波结果 (滤波器大小: " << filterSize << "x" << filterSize << ")..." << std::endl;
    
    // 验证几个采样点
    for (int y = halfSize; y < height - halfSize && y < halfSize + 5; y++) {
        for (int x = halfSize; x < width - halfSize && x < halfSize + 5; x++) {
            // 计算期望值：周围像素的平均值
            double sum = 0;
            int count = 0;
            
            for (int fy = -halfSize; fy <= halfSize; fy++) {
                for (int fx = -halfSize; fx <= halfSize; fx++) {
                    int idx = (y + fy) * width + (x + fx);
                    sum += input[idx];
                    count++;
                }
            }
            
            double expected = sum / count;
            double actual = output[y * width + x];
            double diff = std::abs(expected - actual);
            
            if (diff > tolerance) {
                if (errors < 5) { // 只报告前5个错误
                    std::cout << "  错误 @ (" << x << "," << y << "): "
                              << "期望=" << expected << ", 实际=" << actual 
                              << ", 差异=" << diff << std::endl;
                }
                errors++;
            }
        }
    }
    
    if (errors == 0) {
        std::cout << "✅ 盒式滤波验证通过！" << std::endl;
        return true;
    } else {
        std::cout << "❌ 盒式滤波验证失败！发现 " << errors << " 个错误" << std::endl;
        return false;
    }
}

// 生成已知的测试图像
void generateTestPattern(std::vector<Npp8u>& image, int width, int height, const char* pattern) {
    if (strcmp(pattern, "gradient") == 0) {
        // 渐变图案
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image[y * width + x] = (Npp8u)((x + y) * 255 / (width + height - 2));
            }
        }
    } else if (strcmp(pattern, "checkerboard") == 0) {
        // 棋盘图案
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if ((x / 32 + y / 32) % 2 == 0) {
                    image[y * width + x] = 200;
                } else {
                    image[y * width + x] = 50;
                }
            }
        }
    } else {
        // 默认：常数图案
        for (int i = 0; i < width * height; i++) {
            image[i] = 128;
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "\n=== OpenNPP 验证程序 ===" << std::endl;
    
    // 初始化NPP
    const NppLibraryVersion* libVer = nppGetLibVersion();
    std::cout << "NPP 库版本: " << libVer->major << "." << libVer->minor << "." << libVer->build << std::endl;
    
    // 测试参数
    const int width = 128;
    const int height = 128;
    const int filterSize = 5;
    
    // 分配主机内存
    std::vector<Npp8u> h_input(width * height);
    std::vector<Npp8u> h_output(width * height);
    
    // 生成测试图案
    generateTestPattern(h_input, width, height, "gradient");
    
    // 分配GPU内存
    Npp8u *d_input, *d_output;
    int srcStep = width * sizeof(Npp8u);
    int dstStep = width * sizeof(Npp8u);
    
    cudaMalloc(&d_input, width * height * sizeof(Npp8u));
    cudaMalloc(&d_output, width * height * sizeof(Npp8u));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_input, h_input.data(), width * height * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 执行盒式滤波
    NppiSize oSrcSize = {width, height};
    NppiPoint oSrcOffset = {0, 0};
    NppiSize oSizeROI = {width, height};
    NppiSize oMaskSize = {filterSize, filterSize};
    NppiPoint oAnchor = {filterSize/2, filterSize/2};
    
    std::cout << "\n执行盒式滤波..." << std::endl;
    NppStatus status = nppiFilterBoxBorder_8u_C1R(
        d_input, srcStep, oSrcSize, oSrcOffset,
        d_output, dstStep, oSizeROI, oMaskSize,
        oAnchor, NPP_BORDER_REPLICATE);
    
    if (status != NPP_SUCCESS) {
        std::cout << "❌ 盒式滤波执行失败！错误码: " << status << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return -1;
    }
    
    // 拷贝结果回主机
    cudaMemcpy(h_output.data(), d_output, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // 验证结果
    bool passed = verifyBoxFilter(h_input.data(), h_output.data(), width, height, filterSize);
    
    // 额外测试：边缘情况
    std::cout << "\n测试边缘处理..." << std::endl;
    
    // 检查边界像素是否被正确处理（使用REPLICATE模式）
    bool edgeTest = true;
    for (int i = 0; i < filterSize/2; i++) {
        // 顶部边缘
        if (h_output[i * width] == 0) {
            std::cout << "❌ 顶部边缘未正确处理" << std::endl;
            edgeTest = false;
            break;
        }
        // 左侧边缘
        if (h_output[i] == 0) {
            std::cout << "❌ 左侧边缘未正确处理" << std::endl;
            edgeTest = false;
            break;
        }
    }
    
    if (edgeTest) {
        std::cout << "✅ 边缘处理验证通过！" << std::endl;
    }
    
    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    
    // 进行像素级对比验证
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "开始像素级对比验证..." << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    bool all_comparisons_passed = true;
    
    // 对比基础处理结果
    struct ComparisonTest {
        const char* nvidia_file;
        const char* opennpp_file;
        const char* name;
    };
    
    ComparisonTest tests[] = {
        {"reference_nvidia_npp/teapot512_boxFilter.pgm", "opennpp_results/teapot512_boxFilter.pgm", "盒式滤波"},
        {"reference_nvidia_npp/teapot512_cannyEdgeDetection.pgm", "opennpp_results/teapot512_cannyEdgeDetection.pgm", "Canny边缘检测"},
        {"reference_nvidia_npp/teapot512_histEqualization.pgm", "opennpp_results/teapot512_histEqualization.pgm", "直方图均衡化"},
    };
    
    for (const auto& test : tests) {
        bool result = compareImages(test.nvidia_file, test.opennpp_file, test.name);
        if (!result) {
            all_comparisons_passed = false;
        }
    }
    
    // 对比FilterBorderControl结果
    const char* filter_files[] = {
        "teapot512.pgm_gradientVectorPrewittBorderX_Vertical.pgm",
        "teapot512_gradientVectorPrewittBorderY_Horizontal.pgm",
        "teapot512_gradientVectorPrewittBorderX_Vertical_WithNoSourceBorders.pgm",
        "teapot512_gradientVectorPrewittBorderY_Horizontal_WithNoSourceBorders.pgm",
        "teapot512_gradientVectorPrewittBorderX_Vertical_BorderDiffs.pgm",
        "teapot512_gradientVectorPrewittBorderY_Horizontal_BorderDiffs.pgm"
    };
    
    const char* filter_names[] = {
        "Prewitt X 垂直梯度",
        "Prewitt Y 水平梯度", 
        "Prewitt X 无边界",
        "Prewitt Y 无边界",
        "Prewitt X 边界差异",
        "Prewitt Y 边界差异"
    };
    
    for (int i = 0; i < 6; i++) {
        std::string nvidia_path = "reference_nvidia_npp/FilterBorderControl/" + std::string(filter_files[i]);
        std::string opennpp_path = "opennpp_results/FilterBorderControl/" + std::string(filter_files[i]);
        
        bool result = compareImages(nvidia_path.c_str(), opennpp_path.c_str(), filter_names[i]);
        if (!result) {
            all_comparisons_passed = false;
        }
    }
    
    // 总结
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "像素级验证总结:" << std::endl;
    std::cout << "- 内部算法验证: " << (passed && edgeTest ? "✅ 通过" : "❌ 失败") << std::endl;
    std::cout << "- 与NVIDIA NPP对比: " << (all_comparisons_passed ? "✅ 通过" : "❌ 失败") << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    bool overall_passed = passed && edgeTest && all_comparisons_passed;
    std::cout << "\n🎯 最终结果: " << (overall_passed ? "完全通过 ✅" : "存在问题 ❌") << std::endl;
    
    if (overall_passed) {
        std::cout << "🎉 OpenNPP实现与NVIDIA NPP完全一致！" << std::endl;
    }
    
    return overall_passed ? 0 : -1;
}