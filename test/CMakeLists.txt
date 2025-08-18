# OpenNPP测试模块CMakeLists.txt
# 组织所有测试的构建配置

# 核心测试
add_executable(test_nppcore 
    test/nppcore/test_nppcore.cpp 
    ${NPP_CORE_SOURCES}
)
target_link_libraries(test_nppcore PRIVATE CUDA::cudart)
target_compile_features(test_nppcore PRIVATE cxx_std_17)

# NPPI支持函数测试
add_executable(test_nppi_support 
    test/nppi/test_nppi_support_functions.cpp 
    ${NPPI_SUPPORT_SOURCES}
)
target_link_libraries(test_nppi_support PRIVATE CUDA::cudart)
target_compile_features(test_nppi_support PRIVATE cxx_std_17)

# 算术运算综合测试（新框架）
add_executable(test_arithmetic_comprehensive 
    test/test_arithmetic_comprehensive.cpp 
    ${CPU_SOURCES} 
    ${CUDA_SOURCES}
)
target_link_libraries(test_arithmetic_comprehensive PRIVATE 
    CUDA::cudart
    ${NPP_LIBRARY_IAL} 
    ${NPP_LIBRARY_CORE}
)
target_compile_definitions(test_arithmetic_comprehensive PRIVATE HAVE_NVIDIA_NPP=1)
target_compile_features(test_arithmetic_comprehensive PRIVATE cxx_std_17)

# 三方验证测试V2（主要验证测试）
add_executable(test_validation 
    test/test_three_way_validation_v2.cpp 
    ${CPU_SOURCES} 
    ${CUDA_SOURCES}
)
target_link_libraries(test_validation PRIVATE 
    CUDA::cudart
    ${NPP_LIBRARY_IAL} 
    ${NPP_LIBRARY_CORE}
)
target_compile_definitions(test_validation PRIVATE HAVE_NVIDIA_NPP=1)
target_compile_features(test_validation PRIVATE cxx_std_17)

# NPPI内存分配测试
add_executable(test_nppi_memory 
    test/test_comprehensive_arithmetic_nppi.cpp 
    ${CPU_SOURCES} 
    ${CUDA_SOURCES}
)
target_link_libraries(test_nppi_memory PRIVATE 
    CUDA::cudart
    ${NPP_LIBRARY_IAL} 
    ${NPP_LIBRARY_CORE}
)
target_compile_definitions(test_nppi_memory PRIVATE HAVE_NVIDIA_NPP=1)
target_compile_features(test_nppi_memory PRIVATE cxx_std_17)

# 添加CTest测试
enable_testing()
add_test(NAME NppCore COMMAND test_nppcore)
add_test(NAME NppiSupport COMMAND test_nppi_support)
add_test(NAME Arithmetic COMMAND test_arithmetic_comprehensive)
add_test(NAME Validation COMMAND test_validation)
add_test(NAME Memory COMMAND test_nppi_memory)