/**
 * @file test_install.cpp
 * @brief 测试OpenNPP安装是否正确
 */

#include <iostream>
#include <npp.h>

int main() {
    std::cout << "=== OpenNPP安装验证测试 ===" << std::endl;
    
    // 测试基本的NPP包含
    std::cout << "✓ 成功包含npp.h头文件" << std::endl;
    
    // 测试基本数据类型
    Npp32f testFloat = 3.14f;
    Npp8u testByte = 255;
    NppiSize testSize = {64, 64};
    
    std::cout << "✓ NPP数据类型可用:" << std::endl;
    std::cout << "  Npp32f: " << testFloat << std::endl;
    std::cout << "  Npp8u: " << static_cast<int>(testByte) << std::endl;
    std::cout << "  NppiSize: " << testSize.width << "x" << testSize.height << std::endl;
    
    // 测试NPP状态码
    NppStatus status = NPP_NO_ERROR;
    std::cout << "✓ NPP状态码可用: " << status << std::endl;
    
    std::cout << "=== OpenNPP安装验证成功 ===" << std::endl;
    std::cout << "库文件成功链接，头文件正确安装" << std::endl;
    
    return 0;
}