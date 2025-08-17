#include "npp.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>

#ifdef HAVE_NVIDIA_NPP
#include <npp.h>
#endif

/**
 * 多通道算术运算测试
 * 测试3通道图像的算术运算
 */

bool testAddC_8u_C3_simple() {
    std::cout << "测试 nppiAddC_8u_C3RSfs..." << std::endl;
    
    const int width = 32;
    const int height = 32;
    const int channels = 3;
    const int dataSize = width * height * channels;
    
    // 准备测试数据：RGB图像，每个通道不同的值
    std::vector<Npp8u> srcHost(dataSize);
    std::vector<Npp8u> dstHost(dataSize, 0);
    
    // 填充测试数据：R=50, G=100, B=150 交替模式
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int pixelIdx = (y * width + x) * channels;
            srcHost[pixelIdx + 0] = 50;   // Red
            srcHost[pixelIdx + 1] = 100;  // Green  
            srcHost[pixelIdx + 2] = 150;  // Blue
        }
    }
    
    // 分配设备内存
    Npp8u *srcDev = nullptr, *dstDev = nullptr;
    size_t srcPitch, dstPitch;
    
    cudaError_t cudaResult = cudaMallocPitch((void**)&srcDev, &srcPitch, width * channels * sizeof(Npp8u), height);
    if (cudaResult != cudaSuccess) {
        std::cout << "源内存分配失败" << std::endl;
        return false;
    }
    
    cudaResult = cudaMallocPitch((void**)&dstDev, &dstPitch, width * channels * sizeof(Npp8u), height);
    if (cudaResult != cudaSuccess) {
        std::cout << "目标内存分配失败" << std::endl;
        cudaFree(srcDev);
        return false;
    }
    
    // 复制数据到设备
    cudaResult = cudaMemcpy2D(srcDev, srcPitch, srcHost.data(), width * channels * sizeof(Npp8u),
                             width * channels * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        std::cout << "主机到设备拷贝失败" << std::endl;
        cudaFree(srcDev);
        cudaFree(dstDev);
        return false;
    }
    
    // 定义每个通道的常数
    Npp8u constants[3] = {10, 20, 30}; // R+10, G+20, B+30
    
    // 执行操作
    NppiSize roiSize = {width, height};
    NppStatus status = nppiAddC_8u_C3RSfs(srcDev, (int)srcPitch, constants, dstDev, (int)dstPitch, roiSize, 1);
    
    bool testPassed = false;
    if (status == NPP_NO_ERROR) {
        // 复制结果回主机
        cudaResult = cudaMemcpy2D(dstHost.data(), width * channels * sizeof(Npp8u), dstDev, dstPitch,
                                 width * channels * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
        
        if (cudaResult == cudaSuccess) {
            // 验证结果
            testPassed = true;
            Npp8u expectedR = (50 + 10) >> 1; // 30
            Npp8u expectedG = (100 + 20) >> 1; // 60
            Npp8u expectedB = (150 + 30) >> 1; // 90
            
            for (int y = 0; y < height && testPassed; ++y) {
                for (int x = 0; x < width && testPassed; ++x) {
                    int pixelIdx = (y * width + x) * channels;
                    
                    if (dstHost[pixelIdx + 0] != expectedR || 
                        dstHost[pixelIdx + 1] != expectedG || 
                        dstHost[pixelIdx + 2] != expectedB) {
                        std::cout << "结果不匹配在像素 (" << x << "," << y << "): "
                                  << "得到 RGB(" << (int)dstHost[pixelIdx + 0] << "," 
                                  << (int)dstHost[pixelIdx + 1] << "," 
                                  << (int)dstHost[pixelIdx + 2] << "), "
                                  << "期望 RGB(" << (int)expectedR << "," 
                                  << (int)expectedG << "," << (int)expectedB << ")" << std::endl;
                        testPassed = false;
                    }
                }
            }
            
            if (testPassed) {
                std::cout << "nppiAddC_8u_C3RSfs: 通过 ✓" << std::endl;
                
                // 显示一些示例结果
                std::cout << "示例结果 - 前3个像素:" << std::endl;
                for (int i = 0; i < 3; ++i) {
                    int pixelIdx = i * channels;
                    std::cout << "  像素" << i << " RGB: (" 
                              << (int)dstHost[pixelIdx + 0] << "," 
                              << (int)dstHost[pixelIdx + 1] << "," 
                              << (int)dstHost[pixelIdx + 2] << ")" << std::endl;
                }
            }
        } else {
            std::cout << "设备到主机拷贝失败" << std::endl;
        }
    } else {
        std::cout << "nppiAddC_8u_C3RSfs 执行失败，状态码: " << status << std::endl;
    }
    
#ifdef HAVE_NVIDIA_NPP
    // 与NVIDIA NPP对比测试
    if (testPassed) {
        std::vector<Npp8u> nvidiaDstHost(dataSize, 0);
        Npp8u* nvidiaDstDev = nullptr;
        size_t nvidiaDstPitch;
        
        cudaResult = cudaMallocPitch((void**)&nvidiaDstDev, &nvidiaDstPitch, width * channels * sizeof(Npp8u), height);
        if (cudaResult == cudaSuccess) {
            NppStatus nvidiaStatus = ::nppiAddC_8u_C3RSfs(srcDev, (int)srcPitch, constants, nvidiaDstDev, (int)nvidiaDstPitch, roiSize, 1);
            
            if (nvidiaStatus == NPP_NO_ERROR) {
                cudaResult = cudaMemcpy2D(nvidiaDstHost.data(), width * channels * sizeof(Npp8u), nvidiaDstDev, nvidiaDstPitch,
                                         width * channels * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
                
                if (cudaResult == cudaSuccess) {
                    // 比较结果
                    bool nvidiaMatch = true;
                    for (int i = 0; i < dataSize && nvidiaMatch; ++i) {
                        if (dstHost[i] != nvidiaDstHost[i]) {
                            std::cout << "NVIDIA对比不匹配在索引 " << i << ": 我们的 " << (int)dstHost[i] 
                                      << ", NVIDIA " << (int)nvidiaDstHost[i] << std::endl;
                            nvidiaMatch = false;
                        }
                    }
                    
                    if (nvidiaMatch) {
                        std::cout << "NVIDIA NPP对比: 通过 ✓" << std::endl;
                    } else {
                        std::cout << "NVIDIA NPP对比: 失败 ✗" << std::endl;
                        testPassed = false;
                    }
                }
            } else {
                std::cout << "NVIDIA NPP执行失败，状态码: " << nvidiaStatus << std::endl;
            }
            
            cudaFree(nvidiaDstDev);
        }
    }
#endif
    
    // 清理
    cudaFree(srcDev);
    cudaFree(dstDev);
    
    return testPassed;
}

bool testInPlaceAddC_8u_simple() {
    std::cout << "测试 nppiAddC_8u_C1IRSfs (in-place)..." << std::endl;
    
    const int width = 32;
    const int height = 32;
    const int dataSize = width * height;
    
    // 准备测试数据
    std::vector<Npp8u> srcHost(dataSize, 100);
    std::vector<Npp8u> originalData(srcHost); // 保存原始数据用于对比
    
    // 分配设备内存
    Npp8u *srcDstDev = nullptr;
    size_t srcDstPitch;
    
    cudaError_t cudaResult = cudaMallocPitch((void**)&srcDstDev, &srcDstPitch, width * sizeof(Npp8u), height);
    if (cudaResult != cudaSuccess) {
        std::cout << "内存分配失败" << std::endl;
        return false;
    }
    
    // 复制数据到设备
    cudaResult = cudaMemcpy2D(srcDstDev, srcDstPitch, srcHost.data(), width * sizeof(Npp8u),
                             width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        std::cout << "主机到设备拷贝失败" << std::endl;
        cudaFree(srcDstDev);
        return false;
    }
    
    // 执行in-place操作
    NppiSize roiSize = {width, height};
    Npp8u constant = 50;
    NppStatus status = nppiAddC_8u_C1IRSfs(constant, srcDstDev, (int)srcDstPitch, roiSize, 1);
    
    bool testPassed = false;
    if (status == NPP_NO_ERROR) {
        // 复制结果回主机
        cudaResult = cudaMemcpy2D(srcHost.data(), width * sizeof(Npp8u), srcDstDev, srcDstPitch,
                                 width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
        
        if (cudaResult == cudaSuccess) {
            // 验证结果
            Npp8u expected = (100 + 50) >> 1; // 75
            testPassed = true;
            
            for (int i = 0; i < dataSize && testPassed; ++i) {
                if (srcHost[i] != expected) {
                    std::cout << "结果不匹配在索引 " << i << ": 得到 " << (int)srcHost[i] 
                              << ", 期望 " << (int)expected << std::endl;
                    testPassed = false;
                }
            }
            
            if (testPassed) {
                std::cout << "nppiAddC_8u_C1IRSfs: 通过 ✓" << std::endl;
                std::cout << "原始值: " << (int)originalData[0] << " -> 结果值: " << (int)srcHost[0] << std::endl;
            }
        }
    } else {
        std::cout << "nppiAddC_8u_C1IRSfs 执行失败，状态码: " << status << std::endl;
    }
    
    cudaFree(srcDstDev);
    return testPassed;
}

int main() {
    std::cout << "=== NPP 多通道算术运算测试 ===" << std::endl;
    
    // 初始化CUDA
    cudaError_t cudaResult = cudaSetDevice(0);
    if (cudaResult != cudaSuccess) {
        std::cout << "CUDA设备初始化失败" << std::endl;
        return 1;
    }
    
    bool allPassed = true;
    
    // 测试3通道算术运算
    std::cout << "\n--- 3通道图像算术运算 ---" << std::endl;
    allPassed &= testAddC_8u_C3_simple();
    
    // 测试in-place算术运算
    std::cout << "\n--- In-Place算术运算 ---" << std::endl;
    allPassed &= testInPlaceAddC_8u_simple();
    
    // 总结
    std::cout << "\n=== 测试总结 ===" << std::endl;
    std::cout << "总体结果: " << (allPassed ? "全部通过 ✓" : "有测试失败 ✗") << std::endl;
    
    return allPassed ? 0 : 1;
}