#include "framework/npp_test_framework.h"
#include "framework/test_nppi_addc_8u_c1rsfs.h"
#include "framework/test_report_simple.h"
#include <iostream>
#include <getopt.h>

/**
 * @brief Main test runner for OpenNPP
 * 
 * This executable runs comprehensive tests comparing:
 * 1. CPU implementation
 * 2. CUDA implementation  
 * 3. NVIDIA NPP implementation (when available)
 */

void printUsage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nOptions:\n");
    printf("  -f, --function FUNCTION   Test specific function (default: all)\n");
    printf("  -r, --report NAME         Report name (default: npp_test_report)\n");
    printf("  -o, --output DIR          Output directory (default: test_reports)\n");
    printf("  -v, --verbose             Verbose output\n");
    printf("  -h, --help                Show this help\n");
    printf("\nExamples:\n");
    printf("  %s                          # Run all tests\n", program_name);
    printf("  %s -f nppiAddC_8u_C1RSfs_Ctx # Test specific function\n", program_name);
    printf("  %s -r my_report -o results   # Custom report name and directory\n", program_name);
}

int main(int argc, char* argv[]) {
    // Default options
    std::string function_name = "";
    std::string report_name = "npp_test_report";
    std::string output_dir = "test_reports";
    bool verbose = false;
    
    // Parse command line arguments
    int opt;
    static struct option long_options[] = {
        {"function", required_argument, 0, 'f'},
        {"report", required_argument, 0, 'r'},
        {"output", required_argument, 0, 'o'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    while ((opt = getopt_long(argc, argv, "f:r:o:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'f':
                function_name = optarg;
                break;
            case 'r':
                report_name = optarg;
                break;
            case 'o':
                output_dir = optarg;
                break;
            case 'v':
                verbose = true;
                break;
            case 'h':
                printUsage(argv[0]);
                return 0;
            default:
                printUsage(argv[0]);
                return 1;
        }
    }
    
    printf("=== OpenNPP Test Runner ===\n");
    printf("CUDA Version: %d\n", CUDART_VERSION);
    printf("Report: %s\n", report_name.c_str());
    printf("Output: %s\n", output_dir.c_str());
    
    // Initialize CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("Error: No CUDA devices found\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    
    // Create report
    TestReport report(report_name, output_dir);
    report.startReport();
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Run tests
    const auto& tests = NPPTest::TestRegistry::getAllTests();
    
    printf("\nFound %zu test functions:\n", tests.size());
    for (const auto& [func_name, test] : tests) {
        printf("  - %s\n", func_name.c_str());
    }
    
    printf("\nRunning tests...\n");
    
    for (const auto& [func_name, test] : tests) {
        if (!function_name.empty() && function_name != func_name) {
            continue; // Skip if specific function requested
        }
        
        printf("\n--- %s ---\n", func_name.c_str());
        printf("NVIDIA NPP Available: %s\n", test->isNVIDIAAvailable() ? "Yes" : "No");
        
        auto test_cases = test->getTestCases();
        int func_tests = 0;
        int func_passed = 0;
        
        for (const auto& params : test_cases) {
            if (!params->isValid()) {
                if (verbose) {
                    printf("  Skipping invalid: %s\n", params->toString().c_str());
                }
                continue;
            }
            
            func_tests++;
            auto result = test->run(*params);
            
            report.addTestResult(func_name, *params, result);
            
            if (result.passed) {
                func_passed++;
                printf("  ✓ PASS: %s\n", params->toString().c_str());
            } else {
                printf("  ✗ FAIL: %s - %s\n", params->toString().c_str(), result.error_message.c_str());
            }
            
            if (verbose) {
                printf("    CPU: %.2f ms, GPU: %.2f ms, NVIDIA: %.2f ms\n",
                       result.cpu_time_ms, result.gpu_time_ms, result.nvidia_time_ms);
                printf("    Max error: %.4f\n", result.max_error);
            }
        }
        
        printf("  %s: %d/%d passed\n", func_name.c_str(), func_passed, func_tests);
        total_tests += func_tests;
        passed_tests += func_passed;
    }
    
    report.finalizeReport();
    
    printf("\n=== Final Results ===\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    printf("Success rate: %.2f%%\n", 100.0 * passed_tests / total_tests);
    
    if (passed_tests == total_tests) {
        printf("\n🎉 All tests passed!\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed. Check the reports for details.\n");
        return 1;
    }
}
