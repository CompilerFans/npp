# NPP单元测试CMake配置

# 设置GoogleTest
npp_setup_gtest()

# 单元测试（纯功能测试，无外部依赖）
add_subdirectory(unit)