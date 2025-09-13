cmake_minimum_required(VERSION 3.18)
project(OpenNPP VERSION 1.0.0 LANGUAGES CXX CUDA)

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
option(NPP_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
option(USE_NVIDIA_NPP "Use NVIDIA NPP library instead of OpenNPP" OFF)

# 显示配置选项
if(NPP_WARNINGS_AS_ERRORS)
    message(STATUS "Warnings will be treated as errors")
else()
    message(STATUS "Warnings will not be treated as errors")
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