# NPP 性能测试配置

# 查找或下载 Google Benchmark
function(npp_setup_benchmark)
    # 方案 1：使用源码版本（推荐）
    if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/benchmark/CMakeLists.txt)
        message(STATUS "Using in-tree Google Benchmark source")
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/benchmark benchmark EXCLUDE_FROM_ALL)
        return()
    endif()
    
    # 方案 2：查找系统安装的 Benchmark
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
        message(STATUS "Using system Google Benchmark")
        return()
    endif()
    
    # 方案 3：使用 FetchContent 自动下载
    include(FetchContent)
    message(STATUS "Downloading Google Benchmark...")
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endfunction()

# 创建性能测试目标
# target_name: 测试目标名称
# sources: 源文件列表
# library_target: MPP 库目标（当 USE_NVIDIA_NPP=OFF 时）
function(npp_create_benchmark_target target_name sources library_target)
    add_executable(${target_name} ${sources})
    
    # 链接 Google Benchmark
    target_link_libraries(${target_name}
        PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
    )
    
    # 根据选项决定链接哪个 NPP 库
    if(USE_NVIDIA_NPP)
        # 查找 NVIDIA NPP 库 - 只需要检查一个核心库的存在
        find_library(NVIDIA_NPPC_LIB
            NAMES nppc
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib /usr/local/cuda/lib64
        )
        
        if(NOT NVIDIA_NPPC_LIB)
            message(FATAL_ERROR "NVIDIA NPP libraries not found. Please check your CUDA installation.")
        endif()
        
        # 链接 NVIDIA NPP 库（第一次调用：基础 CUDA 库）
        target_link_libraries(${target_name}
            PRIVATE
            ${CUDA_LIBRARIES}
            ${CUDA_cudart_LIBRARY}
        )
        
        # 添加库搜索路径和链接 NPP 库（第二次调用：NPP 库）
        target_link_directories(${target_name} PRIVATE /usr/local/cuda/lib64)
        target_link_libraries(${target_name} PRIVATE
            nppial nppicc nppidei nppif nppig nppim nppist nppisu nppitc npps nppc)
        
        # 使用 NVIDIA NPP 头文件
        target_include_directories(${target_name}
            PRIVATE
            ${CUDA_TOOLKIT_ROOT_DIR}/include
        )
        
        # 定义宏标识使用 NVIDIA NPP
        target_compile_definitions(${target_name} PRIVATE USE_NVIDIA_NPP_BENCHMARK)
        message(STATUS "Benchmark ${target_name} will use NVIDIA NPP")
    else()
        # 链接 MPP 库
        if(NOT library_target)
            message(FATAL_ERROR "library_target required when USE_NVIDIA_NPP=OFF")
        endif()
        
        target_link_libraries(${target_name}
            PRIVATE
            ${library_target}
        )
        
        message(STATUS "Benchmark ${target_name} will use MPP library")
    endif()
    
    # 设置通用配置
    npp_setup_target(${target_name})
    
    # 添加包含目录
    if(NOT USE_NVIDIA_NPP)
        target_include_directories(${target_name}
            PRIVATE
            ${CMAKE_SOURCE_DIR}
        )
    endif()
    
    # 设置输出目录
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark
    )
endfunction()

# 注册性能测试（可选，用于 CTest 集成）
function(npp_register_benchmark benchmark_name target_name)
    # 性能测试不自动运行，需要手动执行
    # 但可以注册为 CTest 用于 CI
    add_test(NAME ${benchmark_name}_benchmark 
             COMMAND ${target_name} --benchmark_min_time=0.1)
    set_tests_properties(${benchmark_name}_benchmark PROPERTIES
        LABELS "benchmark"
        TIMEOUT 600
    )
endfunction()
