# NPP核心库CMake配置

# 查找源文件
npp_find_all_sources(CPU_SOURCES CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR})

# 创建MPP库(根据BUILD_SHARED_LIBS选项决定是静态还是动态)
add_library(npp
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# 设置NPP目标配置
npp_setup_target(npp)

# 添加NPP特有包含目录
npp_add_common_includes(npp)

# 设置库的导出名称和版本信息
set_target_properties(npp PROPERTIES
    OUTPUT_NAME "mpp"
    EXPORT_NAME "npp"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# 如果是动态库，添加动态库特有的配置
if(BUILD_SHARED_LIBS)
    # Windows DLL导出符号
    target_compile_definitions(npp PRIVATE MPP_EXPORTS)
    target_compile_definitions(npp INTERFACE MPP_IMPORTS)
    
    # 设置动态库的运行时路径
    if(NOT WIN32)
        set_target_properties(npp PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()
endif()

# 安装库文件
npp_install_library(npp)