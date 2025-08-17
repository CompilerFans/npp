#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>

#include "framework/npp_test_core.h"
#include "nppi_arithmetic/test_nppi_addc.h"

// 声明测试注册函数
namespace NPPTest {
    void registerAddCTests();
}

/**
 * NPP通用测试运行器
 * 
 * 用法:
 *   ./npp_test_runner                           # 运行所有测试
 *   ./npp_test_runner [category]                # 运行指定类别的测试
 *   ./npp_test_runner nppi_arithmetic_and_logical_operations  # 运行算术运算测试
 */

void printUsage(const char* program_name) {
    printf("Usage: %s [category]\n", program_name);
    printf("\n");
    printf("Available categories:\n");
    printf("  nppi_arithmetic_and_logical_operations - Image arithmetic operations\n");
    printf("  nppi_support_functions                 - Memory allocation and support\n");
    printf("  nppcore                               - Core NPP functions\n");
    printf("\n");
    printf("If no category is specified, all tests will be run.\n");
}

int main(int argc, char* argv[]) {
    // 初始化随机数生成器
    srand(static_cast<unsigned int>(time(nullptr)));
    
    // 检查CUDA设备
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        printf("Error: No CUDA devices found!\n");
        return 1;
    }
    
    // 设置CUDA设备
    cudaSetDevice(0);
    
    // 打印设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using CUDA device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("\n");
    
    // 检查NPP库版本
    const NppLibraryVersion* npp_version = nppGetLibVersion();
    if (npp_version) {
        printf("NPP Library version: %d.%d.%d\n", 
               npp_version->major, npp_version->minor, npp_version->build);
    }
    printf("\n");
    
    // 手动注册测试
    printf("Registering tests...\n");
    NPPTest::registerAddCTests();
    printf("Tests registered successfully.\n\n");
    
    try {
        if (argc == 1) {
            // 运行所有测试
            printf("Running all NPP tests...\n");
            printf("==========================================\n\n");
            NPPTest::TestRegistry::runAllTests();
        } else if (argc == 2) {
            std::string category = argv[1];
            
            if (category == "--help" || category == "-h") {
                printUsage(argv[0]);
                return 0;
            }
            
            // 运行指定类别的测试
            printf("Running NPP tests for category: %s\n", category.c_str());
            printf("==========================================\n\n");
            NPPTest::TestRegistry::runCategoryTests(category);
        } else {
            printUsage(argv[0]);
            return 1;
        }
        
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
        return 1;
    }
    
    return 0;
}