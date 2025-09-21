cmake_minimum_required(VERSION 3.18)
project(MPP VERSION 1.0.0 LANGUAGES CXX CUDA)

# 包含模块化cmake配置
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CommonConfig)
include(CudaConfig)
include(FindSources)
include(TestConfig)
include(InstallConfig)

# 需要的cmake模块
include(CMakePackageConfigHelpers)

# 配置CUDA环境
npp_configure_cuda()

# 构建src目录 - 核心库
add_subdirectory(src)

# 构建选项
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_SHARED_LIBS "Build shared libraries instead of static libraries" OFF)
option(NPP_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(USE_NVIDIA_NPP "Use NVIDIA NPP library instead of MPP" OFF)
option(NPP_STRICT_TESTING "Use strict precision testing (exact match)" OFF)

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
    add_subdirectory(examples)
endif()

# 配置安装
npp_configure_install()