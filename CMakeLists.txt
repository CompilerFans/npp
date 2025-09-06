cmake_minimum_required(VERSION 3.18)
project(OpenNPP VERSION 1.0.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用测试
enable_testing()

# 添加Google Test
add_subdirectory(third_party/googletest)

# 查找必需包
find_package(CUDAToolkit REQUIRED)

# 启用CUDA语言
enable_language(CUDA)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 80 89)

# 编译选项
# 对C++文件设置合理的警告级别，并将警告视为错误
option(WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
if(WARNINGS_AS_ERRORS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unused-parameter")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-parameter")
endif()
add_compile_options(-Wno-gnu-line-marker)

# CUDA编译选项
# 抑制行指令警告，并传递Werror选项
if(WARNINGS_AS_ERRORS)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-gnu-line-marker -Xcompiler -Werror")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-gnu-line-marker")
endif()

# 输出目录设置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/api)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test/framework)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})  # 添加根目录以支持 "test/framework/..." 包含路径
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# 递归查找所有源文件
file(GLOB_RECURSE CPU_SOURCES 
    "src/**/*.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES
    "src/**/*.cu"
)

# 创建OpenNPP静态库
add_library(opennpp STATIC
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# 创建OpenNPP共享库（用于动态加载测试）
add_library(opennpp_shared SHARED
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# 链接CUDA库（静态库）
target_link_libraries(opennpp 
    PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)

# 链接CUDA库（共享库）
target_link_libraries(opennpp_shared 
    PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)

# 设置编译标志（静态库）
target_compile_features(opennpp PUBLIC cxx_std_17)
target_compile_options(opennpp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>
)

# 设置编译标志（共享库）
target_compile_features(opennpp_shared PUBLIC cxx_std_17)
target_compile_options(opennpp_shared PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>
)

# 设置共享库输出名称
set_target_properties(opennpp_shared PROPERTIES OUTPUT_NAME "opennpp")

# 查找NVIDIA NPP库（可选，用于验证测试）
find_library(NPP_LIBRARY_IAL nppial HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
find_library(NPP_LIBRARY_CORE nppc HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)

if(NPP_LIBRARY_IAL AND NPP_LIBRARY_CORE AND ENABLE_NVIDIA_NPP_TESTS)
    message(STATUS "Found NVIDIA NPP libraries: ${NPP_LIBRARY_IAL}, ${NPP_LIBRARY_CORE}")
    set(HAVE_NVIDIA_NPP TRUE)
else()
    if(NOT ENABLE_NVIDIA_NPP_TESTS)
        message(STATUS "NVIDIA NPP tests disabled by user configuration")
    else()
        message(WARNING "NVIDIA NPP library not found - validation tests will skip NVIDIA comparison")
    endif()
    set(HAVE_NVIDIA_NPP FALSE)
endif()

# 测试选项
option(BUILD_TESTS "Build tests" ON)
option(BUILD_FUNCTIONAL_TESTS "Build functional tests" ON)
option(BUILD_COMPARISON_TESTS "Build comparison tests" ON)
option(BUILD_LEGACY_TESTS "Build legacy tests" OFF)
option(ENABLE_NVIDIA_NPP_TESTS "Enable NVIDIA NPP comparison tests (requires NVIDIA NPP libraries)" ON)

if(BUILD_TESTS)
    enable_testing()
    
    # ========== 新测试结构 ==========
    
    # 功能测试（纯单元测试，无外部依赖）
    if(BUILD_FUNCTIONAL_TESTS)
        add_subdirectory(test/functional)
    endif()
    
    # 对比测试（与NVIDIA NPP对比）
    if(BUILD_COMPARISON_TESTS AND HAVE_NVIDIA_NPP)
        add_subdirectory(test/comparison)
    endif()
    
    # 遗留测试（向后兼容）
    if(BUILD_LEGACY_TESTS)
        # 递归查找遗留测试文件
        file(GLOB_RECURSE LEGACY_TEST_SOURCES
            "test/legacy/**/*.cpp"
        )
        
        # 创建遗留测试可执行程序
        add_executable(opennpp_legacy_test 
            ${LEGACY_TEST_SOURCES}
            ${CPU_SOURCES}
            ${CUDA_SOURCES}
        )
        
        # 链接库
        target_link_libraries(opennpp_legacy_test PRIVATE 
            gtest 
            gtest_main 
            CUDA::cudart
            dl
        )
        
        # 包含目录
        target_include_directories(opennpp_legacy_test PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest/include
        )
        
        # 如果有NVIDIA NPP，添加编译定义
        if(HAVE_NVIDIA_NPP)
            target_compile_definitions(opennpp_legacy_test PRIVATE HAVE_NVIDIA_NPP=1)
            target_compile_definitions(opennpp_legacy_test PRIVATE 
                NVIDIA_NPP_IAL_PATH="${NPP_LIBRARY_IAL}"
                NVIDIA_NPP_CORE_PATH="${NPP_LIBRARY_CORE}"
            )
        endif()
        
        # 添加遗留测试
        add_test(NAME legacy_tests COMMAND opennpp_legacy_test)
    endif()
    
    # ========== 遗留统一测试（保留但不推荐使用） ==========
    if(FALSE) # 暂时禁用旧的统一测试结构
        # 创建统一的测试可执行程序
        add_executable(opennpp_test 
            ${LEGACY_TEST_SOURCES}
            ${CPU_SOURCES}
            ${CUDA_SOURCES}
        )
        
        # 链接库
        target_link_libraries(opennpp_test PRIVATE 
            gtest 
            gtest_main 
            CUDA::cudart
            dl  # 添加动态链接库支持
        )
    
    # 包含目录
    target_include_directories(opennpp_test PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/googletest/include
    )
    
    # 如果有NVIDIA NPP，只添加编译定义，不链接库（避免符号冲突）
    if(HAVE_NVIDIA_NPP)
        # 不再直接链接NVIDIA NPP库，改用dlopen动态加载
        # 这样可以避免符号冲突，真正测试两个不同的实现
        target_compile_definitions(opennpp_test PRIVATE HAVE_NVIDIA_NPP=1)
        # 传递NVIDIA NPP库路径供dlopen使用
        target_compile_definitions(opennpp_test PRIVATE 
            NVIDIA_NPP_IAL_PATH="${NPP_LIBRARY_IAL}"
            NVIDIA_NPP_CORE_PATH="${NPP_LIBRARY_CORE}"
        )
    endif()
    
    # 添加不同的测试过滤器
    add_test(NAME all_tests COMMAND opennpp_test)
    add_test(NAME core_tests COMMAND opennpp_test --gtest_filter=NPP*:*Properties*:*Stream*:*Error*:*Device*)
    add_test(NAME arithmetic_tests COMMAND opennpp_test --gtest_filter=*AddC*:*SubC*:*MulC*:*DivC*)
    add_test(NAME support_tests COMMAND opennpp_test --gtest_filter=*Memory*:*Support*)
    add_test(NAME quick_tests COMMAND opennpp_test --gtest_filter=-*Performance*:-*Stress*:-*Consistency*)
    
    if(HAVE_NVIDIA_NPP)
        add_test(NAME validation_tests COMMAND opennpp_test --gtest_filter=*Consistency*)
    endif()
    
    # ========== 统一测试目标 ==========
    # 设置测试目标列表
    set(ALL_TEST_TARGETS opennpp_test)
    
    # 添加test_all目标来运行所有测试
    add_custom_target(test_all
        COMMAND ${CMAKE_CTEST_COMMAND} -R all_tests --output-on-failure --verbose
        DEPENDS ${ALL_TEST_TARGETS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running all OpenNPP tests..."
    )
    
    # 添加快速测试目标（不运行性能和压力测试）
    add_custom_target(test_quick
        COMMAND ${CMAKE_CTEST_COMMAND} -R quick_tests --output-on-failure
        DEPENDS ${ALL_TEST_TARGETS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running quick OpenNPP tests..."
    )
    
    # 添加特定模块的测试目标
    add_custom_target(test_core
        COMMAND ${CMAKE_CTEST_COMMAND} -R core_tests --output-on-failure
        DEPENDS ${ALL_TEST_TARGETS}
        COMMENT "Running NPP Core tests..."
    )
    
    add_custom_target(test_arithmetic
        COMMAND ${CMAKE_CTEST_COMMAND} -R arithmetic_tests --output-on-failure
        DEPENDS ${ALL_TEST_TARGETS}
        COMMENT "Running NPPI Arithmetic tests..."
    )
    
    add_custom_target(test_support
        COMMAND ${CMAKE_CTEST_COMMAND} -R support_tests --output-on-failure
        DEPENDS ${ALL_TEST_TARGETS}
        COMMENT "Running NPPI Support Function tests..."
    )
    
    if(HAVE_NVIDIA_NPP)
        add_custom_target(test_validation
            COMMAND ${CMAKE_CTEST_COMMAND} -R validation_tests --output-on-failure
            DEPENDS ${ALL_TEST_TARGETS}
            COMMENT "Running OpenNPP vs NVIDIA NPP consistency tests..."
        )
    endif()
    
    # 生成测试报告的目标
    add_custom_target(test_report
        COMMAND opennpp_test --gtest_output=xml:test_results.xml
        COMMAND ${CMAKE_COMMAND} -E echo "Test report generated: test_results.xml"
        DEPENDS ${ALL_TEST_TARGETS}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating test report..."
    )
    
        # 运行特定测试的辅助目标（使用GTest过滤器）
        add_custom_target(test_filter
            COMMAND opennpp_test --gtest_filter="${GTEST_FILTER}"
            DEPENDS ${ALL_TEST_TARGETS}
            COMMENT "Running tests with filter: ${GTEST_FILTER}"
        )
    endif() # 结束旧的统一测试结构
    
    # ========== 新测试结构目标 ==========
    
    # 添加新的综合测试目标
    add_custom_target(test_new_all
        COMMENT "Running all new structure tests..."
    )
    
    if(BUILD_FUNCTIONAL_TESTS)
        add_custom_target(test_functional
            COMMAND ${CMAKE_CTEST_COMMAND} -R "NPP_Functional_Tests" --output-on-failure
            COMMENT "Running functional unit tests..."
        )
    endif()
    
    if(BUILD_COMPARISON_TESTS AND HAVE_NVIDIA_NPP)
        add_custom_target(test_comparison
            COMMAND ${CMAKE_CTEST_COMMAND} -R "NPP_Comparison_Tests" --output-on-failure
            COMMENT "Running comparison tests vs NVIDIA NPP..."
        )
    endif()
    
    if(BUILD_LEGACY_TESTS)
        add_custom_target(test_legacy
            COMMAND ${CMAKE_CTEST_COMMAND} -R "legacy_tests" --output-on-failure
            COMMENT "Running legacy tests..."
        )
    endif()
    
endif() # BUILD_TESTS

# 编译器特定选项
if(MSVC)
    target_compile_options(opennpp PRIVATE /W4)
else()
    target_compile_options(opennpp PRIVATE -Wall -Wextra -Wno-gnu-line-marker)
endif()
