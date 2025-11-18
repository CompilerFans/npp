# NPP测试配置功能

# 查找并配置GoogleTest
function(npp_setup_gtest)
    # 首先检查是否有源码版本的GoogleTest
    if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/googletest/CMakeLists.txt)
        message(STATUS "Using in-tree GoogleTest source")
        # 配置GoogleTest选项
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(BUILD_GTEST ON CACHE BOOL "" FORCE)
        
        # 添加GoogleTest源码子目录
        add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest googletest EXCLUDE_FROM_ALL)
        return()
    endif()
    
    # 尝试查找系统安装的GoogleTest
    find_package(GTest QUIET)
    
    if(NOT GTest_FOUND)
        message(STATUS "System GoogleTest not found. Using in-tree GoogleTest.")
    else()
        message(STATUS "Using system GoogleTest")
    endif()
endfunction()

# 创建NPP测试目标
# library_target: 当USE_NVIDIA_NPP=OFF时使用的MPP库target，USE_NVIDIA_NPP=ON时可为空
function(npp_create_test_target target_name sources library_target)
    add_executable(${target_name} ${sources})

    # 根据选项决定链接哪个NPP库
    if(USE_NVIDIA_NPP)
        # 查找NVIDIA NPP库 - 只需要检查一个核心库的存在
        # 尝试多个可能的路径
        find_library(NVIDIA_NPPC_LIB
            NAMES nppc
            PATHS 
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                ${CUDA_TOOLKIT_ROOT_DIR}/lib
                ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib
                /usr/local/cuda/lib64
                /usr/local/cuda/targets/x86_64-linux/lib
                /usr/local/cuda-12.6/lib64
                /usr/local/cuda-12.6/targets/x86_64-linux/lib
        )

        if(NOT NVIDIA_NPPC_LIB)
            message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
            message(STATUS "Searched paths:")
            message(STATUS "  - ${CUDA_TOOLKIT_ROOT_DIR}/lib64")
            message(STATUS "  - ${CUDA_TOOLKIT_ROOT_DIR}/lib")
            message(STATUS "  - /usr/local/cuda/lib64")
            message(FATAL_ERROR "NVIDIA NPP libraries not found. Please check your CUDA installation.")
        else()
            message(STATUS "Found NVIDIA NPP library: ${NVIDIA_NPPC_LIB}")
        endif()

        # 链接NVIDIA NPP库
        target_link_libraries(${target_name}
            PRIVATE
            ${CUDA_LIBRARIES}
            ${CUDA_cudart_LIBRARY}
            gtest
            gtest_main
        )

        # 查找并链接 NPP 库 - 使用完整路径更可靠
        set(NPP_SEARCH_PATHS
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
            ${CUDA_TOOLKIT_ROOT_DIR}/lib
            ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib
            /usr/local/cuda/lib64
            /usr/local/cuda/lib
            /usr/local/cuda/targets/x86_64-linux/lib
            /usr/local/cuda-12.6/lib64
            /usr/local/cuda-12.6/targets/x86_64-linux/lib
            /usr/local/cuda-12.5/lib64
            /usr/local/cuda-12.5/targets/x86_64-linux/lib
            /usr/local/cuda-12.4/lib64
            /usr/local/cuda-12.4/targets/x86_64-linux/lib
            /usr/local/cuda-12.3/lib64
            /usr/local/cuda-12.3/targets/x86_64-linux/lib
            /usr/local/cuda-12.2/lib64
            /usr/local/cuda-12.2/targets/x86_64-linux/lib
            /usr/local/cuda-12.1/lib64
            /usr/local/cuda-12.1/targets/x86_64-linux/lib
            /usr/local/cuda-12.0/lib64
            /usr/local/cuda-12.0/targets/x86_64-linux/lib
            /usr/local/cuda-11.8/lib64
            /usr/local/cuda-11.8/targets/x86_64-linux/lib
            /usr/local/cuda-11.7/lib64
            /usr/local/cuda-11.7/targets/x86_64-linux/lib
            /opt/cuda/lib64
            /opt/cuda/targets/x86_64-linux/lib
        )
        
        # 找到每个 NPP 库的完整路径
        set(NPP_LIBS nppial nppicc nppidei nppif nppig nppim nppist nppisu nppitc npps nppc)
        foreach(npp_lib ${NPP_LIBS})
            find_library(${npp_lib}_TEST_LIBRARY
                NAMES ${npp_lib}
                PATHS ${NPP_SEARCH_PATHS}
                NO_DEFAULT_PATH
            )
            if(${npp_lib}_TEST_LIBRARY)
                target_link_libraries(${target_name} PRIVATE ${${npp_lib}_TEST_LIBRARY})
                message(STATUS "Found NPP library for tests: ${${npp_lib}_TEST_LIBRARY}")
            else()
                message(WARNING "NPP library not found for tests: ${npp_lib}")
            endif()
        endforeach()

        # 使用NVIDIA NPP头文件
        target_include_directories(${target_name}
            PRIVATE
            ${CUDA_TOOLKIT_ROOT_DIR}/include
        )

        # 定义宏标识使用NVIDIA NPP
        target_compile_definitions(${target_name} PRIVATE USE_NVIDIA_NPP_TESTS)
        message(STATUS "Test ${target_name} will use NVIDIA NPP libraries")
    else()
        # 链接MPP库 - library_target must be provided
        if(NOT library_target)
            message(FATAL_ERROR "library_target must be provided when USE_NVIDIA_NPP=OFF")
        endif()

        target_link_libraries(${target_name}
            PRIVATE
            ${library_target}
            gtest
            gtest_main
        )

        message(STATUS "Test ${target_name} will use MPP library")
    endif()
    
    # 设置通用配置
    npp_setup_target(${target_name})
    
    # 配置精度测试选项
    if(NPP_STRICT_TESTING)
        target_compile_definitions(${target_name} PRIVATE NPP_STRICT_TESTING)
        message(STATUS "Test ${target_name} will use strict precision testing")
    else()
        message(STATUS "Test ${target_name} will use tolerant precision testing")
    endif()
    
    # 添加包含目录
    if(NOT USE_NVIDIA_NPP)
        target_include_directories(${target_name} 
            PRIVATE
            ${CMAKE_SOURCE_DIR}     # 项目根目录，仅用于MPP
        )
    endif()
    
    # 设置输出目录到build根目录
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endfunction()

# 注册CTest测试
function(npp_register_test test_name target_name)
    add_test(NAME ${test_name} COMMAND ${target_name})
endfunction()

# 创建完整的测试套件
function(npp_create_test_suite suite_name test_sources library_target)
    set(target_name "${suite_name}_tests")
    
    npp_create_test_target(${target_name} "${test_sources}" ${library_target})
    npp_register_test(${suite_name} ${target_name})
    
    message(STATUS "Created test suite: ${suite_name}")
endfunction()