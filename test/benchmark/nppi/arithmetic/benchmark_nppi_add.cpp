#include "benchmark_base.h"
#include <nppi_arithmetic_and_logical_operations.h>

using namespace npp_benchmark;

// ============================================================
// 基准测试：nppiAdd_8u_C1RSfs
// ============================================================

// 固定参数版本 - 用于快速测试
static void BM_nppiAdd_8u_C1RSfs_Fixed(benchmark::State& state) {
    const int width = 1920;
    const int height = 1080;
    const int scaleFactor = 0;
    
    ImageBenchmarkBase<Npp8u> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {width, height};
    
    // 性能测试主循环
    for (auto _ : state) {
        NppStatus status = nppiAdd_8u_C1RSfs(
            base.d_src1_, base.step_,
            base.d_src2_, base.step_,
            base.d_dst_, base.step_,
            roi, scaleFactor);
        
        // 确保 GPU 完成
        base.SyncAndCheckError();
        
        if (status != NPP_NO_ERROR) {
            state.SkipWithError("nppiAdd_8u_C1RSfs failed");
            break;
        }
    }
    
    // 报告性能指标
    size_t bytesProcessed = base.ComputeImageBytes(2, 1);  // 2 inputs, 1 output
    REPORT_THROUGHPUT(state, bytesProcessed);
    
    // 自定义指标
    REPORT_CUSTOM_METRIC(state, "Width", width);
    REPORT_CUSTOM_METRIC(state, "Height", height);
    REPORT_CUSTOM_METRIC(state, "Megapixels", (width * height) / 1e6);
    
    base.TeardownImageMemory();
}
BENCHMARK(BM_nppiAdd_8u_C1RSfs_Fixed)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// 参数化版本 - 测试不同图像尺寸
static void BM_nppiAdd_8u_C1RSfs_Sizes(benchmark::State& state) {
    int width = state.range(0);
    int height = state.range(1);
    int scaleFactor = 0;
    
    ImageBenchmarkBase<Npp8u> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {width, height};
    
    for (auto _ : state) {
        NppStatus status = nppiAdd_8u_C1RSfs(
            base.d_src1_, base.step_,
            base.d_src2_, base.step_,
            base.d_dst_, base.step_,
            roi, scaleFactor);
        
        base.SyncAndCheckError();
        
        if (status != NPP_NO_ERROR) {
            state.SkipWithError("nppiAdd_8u_C1RSfs failed");
            break;
        }
    }
    
    size_t bytesProcessed = base.ComputeImageBytes(2, 1);
    REPORT_THROUGHPUT(state, bytesProcessed);
    REPORT_CUSTOM_METRIC(state, "Megapixels", (width * height) / 1e6);
    
    base.TeardownImageMemory();
}

// 注册不同尺寸的测试
BENCHMARK(BM_nppiAdd_8u_C1RSfs_Sizes)
    ->Args({64, 64})          // 4K pixels
    ->Args({256, 256})        // 65K pixels
    ->Args({512, 512})        // 262K pixels
    ->Args({1024, 1024})      // 1M pixels
    ->Args({1920, 1080})      // 2M pixels (Full HD)
    ->Args({3840, 2160})      // 8M pixels (4K UHD)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// ============================================================
// 基准测试：nppiAdd_32f_C1R
// ============================================================

static void BM_nppiAdd_32f_C1R(benchmark::State& state) {
    int width = state.range(0);
    int height = state.range(1);
    
    ImageBenchmarkBase<Npp32f> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {width, height};
    
    for (auto _ : state) {
        NppStatus status = nppiAdd_32f_C1R(
            base.d_src1_, base.step_,
            base.d_src2_, base.step_,
            base.d_dst_, base.step_,
            roi);
        
        base.SyncAndCheckError();
        
        if (status != NPP_NO_ERROR) {
            state.SkipWithError("nppiAdd_32f_C1R failed");
            break;
        }
    }
    
    size_t bytesProcessed = base.ComputeImageBytes(2, 1);
    REPORT_THROUGHPUT(state, bytesProcessed);
    REPORT_CUSTOM_METRIC(state, "Megapixels", (width * height) / 1e6);
    
    base.TeardownImageMemory();
}

BENCHMARK(BM_nppiAdd_32f_C1R)
    ->Args({1920, 1080})
    ->Args({3840, 2160})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// ============================================================
// 对比测试：不同 Scale Factor 的影响
// ============================================================

static void BM_nppiAdd_8u_C1RSfs_ScaleFactors(benchmark::State& state) {
    int width = 1920;
    int height = 1080;
    int scaleFactor = state.range(0);
    
    ImageBenchmarkBase<Npp8u> base;
    base.SetupImageMemory(width, height);
    
    NppiSize roi = {width, height};
    
    for (auto _ : state) {
        NppStatus status = nppiAdd_8u_C1RSfs(
            base.d_src1_, base.step_,
            base.d_src2_, base.step_,
            base.d_dst_, base.step_,
            roi, scaleFactor);
        
        base.SyncAndCheckError();
        
        if (status != NPP_NO_ERROR) {
            state.SkipWithError("nppiAdd_8u_C1RSfs failed");
            break;
        }
    }
    
    size_t bytesProcessed = base.ComputeImageBytes(2, 1);
    REPORT_THROUGHPUT(state, bytesProcessed);
    REPORT_CUSTOM_METRIC(state, "ScaleFactor", scaleFactor);
    
    base.TeardownImageMemory();
}

BENCHMARK(BM_nppiAdd_8u_C1RSfs_ScaleFactors)
    ->DenseRange(0, 7, 1)  // 测试 scaleFactor 从 0 到 7
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

// Google Benchmark main() 由 benchmark::benchmark_main 提供
