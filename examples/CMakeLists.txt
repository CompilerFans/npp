# NPP示例程序CMake配置

# 基础示例：简单的加法运算
add_executable(basic_add_example basic_add_example.cpp)
target_link_libraries(basic_add_example PRIVATE npp)
npp_setup_target(basic_add_example)
