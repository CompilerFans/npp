# NPP测试套件CMake配置

# 设置GoogleTest
npp_setup_gtest()

# 测试构建选项
option(BUILD_FUNCTIONAL_TESTS "Build functional tests" ON)
option(BUILD_COMPARISON_TESTS "Build comparison tests" OFF)
option(BUILD_LEGACY_TESTS "Build legacy tests" OFF)

# 功能测试（独立单元测试）
if(BUILD_FUNCTIONAL_TESTS)
    add_subdirectory(functional)
endif()

# 对比测试（与NVIDIA NPP对比）
if(BUILD_COMPARISON_TESTS)
    add_subdirectory(comparison)
endif()

# 遗留测试（向后兼容）
if(BUILD_LEGACY_TESTS)
    add_subdirectory(legacy)
endif()