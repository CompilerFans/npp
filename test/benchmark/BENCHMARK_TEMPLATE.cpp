// NPP Benchmark Template
// 复制此文件并替换占位符来快速创建新的 benchmark
//
// 占位符说明：
//   {{FUNCTION_NAME}}    - NPP 函数名，如 nppiSub, nppiMul
//   {{DATA_TYPE}}        - 数据类型，如 8u, 16u, 32f
//   {{CHANNELS}}         - 通道数，如 C1, C3, C4
//   {{OPERATION_TYPE}}   - 操作类型，如 R (两输入), I (in-place)
//   {{EXTRA_PARAMS}}     - 额外参数，如 Sfs (scale factor)

#include "benchmark_base.h"
#include <nppi_arithmetic_and_logical_operations.h>

using namespace npp_benchmark;

// ============================================================
// 基准测试：nppi{{FUNCTION_NAME}}_{{DATA_TYPE}}_{{CHANNELS}}{{OPERATION_TYPE}}{{EXTRA_PARAMS}}
// ============================================================

// 固定参数版本 - 用于快速测试
static void BM_nppi{{FUNCTION_NAME}}_{{DATA_TYPE}}_{{CHANNELS}}{{OPERATION_TYPE}}{{EXTRA_PARAMS}}_Fixed(benchmark::State& state) {
    const int width = 1920;
    const int height = 1080;
    // 根据具体函数添加额外参数
    // const int scaleFactor = 0;  // 对于 Sfs 函数
    
    // 选择合适的数据类型
    // ImageBenchmarkBase<Npp8u> base;   // 8u
    // ImageBenchmarkBase<Npp16u> base;  // 16u
    // ImageBenchmarkBase<Npp32f> base;  // 32f
    ImageBenchmarkBase<Npp{{DATA_TYPE}}> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {width, height};
    
    // 性能测试主循环
    for (auto _ : state) {
        // 替换为实际的 NPP 函数调用
        NppStatus status = nppi{{FUNCTION_NAME}}_{{DATA_TYPE}}_{{CHANNELS}}{{OPERATION_TYPE}}{{EXTRA_PARAMS}}(
            base.d_src1_, base.step_,
            base.d_src2_, base.step_,
            base.d_dst_, base.step_,
            roi
            // scaleFactor  // 如果需要
        );
        
        // 确保 GPU 完成
        base.SyncAndCheckError();
        
        if (status != NPP_NO_ERROR) {
            state.SkipWithError("Function failed");
            break;
        }
    }
    
    // 报告性能指标
    size_t bytesProcessed = base.ComputeImageBytes(2, 1);  // 根据实际输入/输出数量调整
    REPORT_THROUGHPUT(state, bytesProcessed);
    
    // 自定义指标
    REPORT_CUSTOM_METRIC(state, "Width", width);
    REPORT_CUSTOM_METRIC(state, "Height", height);
    REPORT_CUSTOM_METRIC(state, "Megapixels", (width * height) / 1e6);
    
    base.TeardownImageMemory();
}

BENCHMARK(BM_nppi{{FUNCTION_NAME}}_{{DATA_TYPE}}_{{CHANNELS}}{{OPERATION_TYPE}}{{EXTRA_PARAMS}}_Fixed)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// 参数化版本 - 测试不同图像尺寸
static void BM_nppi{{FUNCTION_NAME}}_{{DATA_TYPE}}_{{CHANNELS}}{{OPERATION_TYPE}}{{EXTRA_PARAMS}}_Sizes(benchmark::State& state) {
    int width = state.range(0);
    int height = state.range(1);
    
    ImageBenchmarkBase<Npp{{DATA_TYPE}}> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {width, height};
    
    for (auto _ : state) {
        NppStatus status = nppi{{FUNCTION_NAME}}_{{DATA_TYPE}}_{{CHANNELS}}{{OPERATION_TYPE}}{{EXTRA_PARAMS}}(
            base.d_src1_, base.step_,
            base.d_src2_, base.step_,
            base.d_dst_, base.step_,
            roi
        );
        
        base.SyncAndCheckError();
        
        if (status != NPP_NO_ERROR) {
            state.SkipWithError("Function failed");
            break;
        }
    }
    
    size_t bytesProcessed = base.ComputeImageBytes(2, 1);
    REPORT_THROUGHPUT(state, bytesProcessed);
    REPORT_CUSTOM_METRIC(state, "Megapixels", (width * height) / 1e6);
    
    base.TeardownImageMemory();
}

// 注册不同尺寸的测试
BENCHMARK(BM_nppi{{FUNCTION_NAME}}_{{DATA_TYPE}}_{{CHANNELS}}{{OPERATION_TYPE}}{{EXTRA_PARAMS}}_Sizes)
    ->Args({256, 256})        // 65K pixels
    ->Args({512, 512})        // 262K pixels
    ->Args({1024, 1024})      // 1M pixels
    ->Args({1920, 1080})      // 2M pixels (Full HD)
    ->Args({3840, 2160})      // 8M pixels (4K UHD)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// 注册主函数（如果这是独立的 benchmark 文件）
BENCHMARK_MAIN();
