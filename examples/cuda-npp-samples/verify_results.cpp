// 验证OpenNPP样本结果的程序
#include <npp.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>

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
    
    std::cout << "\n=== 总体结果: " << (passed && edgeTest ? "通过 ✅" : "失败 ❌") << " ===" << std::endl;
    
    return (passed && edgeTest) ? 0 : -1;
}