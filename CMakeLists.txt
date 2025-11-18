cmake_minimum_required(VERSION 3.10...3.28)
project(MPP VERSION 1.0.0 LANGUAGES CXX CUDA)

# CMake 版本兼容性说明：
# - 最低支持：3.10 (Ubuntu 18.04 默认版本)
# - 最高测试：3.28 (最新稳定版)
# - 3.10-3.16: 使用传统 CUDA 检测
# - 3.17+: 使用 CUDAToolkit (更好的 CUDA 支持)

# 包含模块化cmake配置
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CommonConfig)
include(CudaConfig)
include(FindSources)
include(TestConfig)
include(InstallConfig)
include(BenchmarkConfig)

# 需要的cmake模块
include(CMakePackageConfigHelpers)

# 显示当前 CMake 版本和策略
message(STATUS "CMake version: ${CMAKE_VERSION}")
if(CMAKE_VERSION VERSION_LESS "3.17")
    message(STATUS "Using legacy CUDA detection (CMake < 3.17)")
else()
    message(STATUS "Using modern CUDAToolkit detection (CMake >= 3.17)")
endif()

# 构建选项 - moved before subdirectories to control build flow
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_SHARED_LIBS "Build shared libraries instead of static libraries" OFF)
option(NPP_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(USE_NVIDIA_NPP "Use NVIDIA NPP library instead of MPP" OFF)
option(NPP_STRICT_TESTING "Use strict precision testing (exact match)" OFF)
option(NPP_REGISTER_BENCHMARKS_TO_CTEST "Register benchmarks to CTest" OFF)

# 配置CUDA环境
npp_configure_cuda()

# 构建src目录 - 仅在不使用NVIDIA NPP时构建
if(NOT USE_NVIDIA_NPP)
    add_subdirectory(src)
else()
    message(STATUS "Skipping src directory build (using NVIDIA NPP library)")
endif()

# 显示配置选项
if(BUILD_SHARED_LIBS)
    message(STATUS "Building shared libraries (.so/.dll)")
else()
    message(STATUS "Building static libraries (.a/.lib)")
endif()

if(NPP_WARNINGS_AS_ERRORS)
    message(STATUS "Warnings will be treated as errors")
else()
    message(STATUS "Warnings will not be treated as errors")
endif()

if(NPP_STRICT_TESTING)
    message(STATUS "Using strict precision testing (exact match)")
else()
    message(STATUS "Using tolerant precision testing (with allowance for minor differences)")
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(BUILD_EXAMPLES)
    if(NOT USE_NVIDIA_NPP)
        add_subdirectory(examples)
    else()
        message(STATUS "Skipping examples build (requires MPP library)")
    endif()
endif()

# 配置安装 - 仅在构建MPP库时需要
if(NOT USE_NVIDIA_NPP)
    npp_configure_install()
endif()
