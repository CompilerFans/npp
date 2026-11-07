# NPP项目公共配置

include(GNUInstallDirs)

# 检测当前选中的 API 头文件版本（switch_cuda_version.sh 切换 API 软链接）。
# API-unified 是 12.1.1 基线并合入了 13.4 声明，版本号保持 12.1.1，因此
# 不能只用 NPP_VER_MAJOR 判断是否具备 13.4 接口，需要额外标记 unified 契约。
file(READ "${CMAKE_SOURCE_DIR}/API/npp.h" _npp_h)
string(REGEX MATCH "#define NPP_VER_MAJOR[ \t]+([0-9]+)" _npp_major_match "${_npp_h}")
set(NPP_HEADER_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define NPP_VER_MINOR[ \t]+([0-9]+)" _npp_minor_match "${_npp_h}")
set(NPP_HEADER_MINOR "${CMAKE_MATCH_1}")
if(NOT NPP_HEADER_MAJOR MATCHES "^[0-9]+$")
    message(FATAL_ERROR "无法从 ${CMAKE_SOURCE_DIR}/API/npp.h 解析 NPP_VER_MAJOR")
endif()
file(READ "${CMAKE_SOURCE_DIR}/API/nppdefs.h" _npp_defs)
string(FIND "${_npp_defs}" "NppiColorSpace" _npp_colorspace_pos)
if(_npp_colorspace_pos GREATER -1 AND NPP_HEADER_MAJOR LESS 13)
    set(NPP_HEADER_UNIFIED TRUE)
else()
    set(NPP_HEADER_UNIFIED FALSE)
endif()
if(NPP_HEADER_UNIFIED OR NPP_HEADER_MAJOR GREATER_EQUAL 13)
    set(NPP_HAS_13_4 TRUE)
else()
    set(NPP_HAS_13_4 FALSE)
endif()
if(NPP_HEADER_UNIFIED)
    set(NPP_HEADER_UNIFIED_DEF 1)
else()
    set(NPP_HEADER_UNIFIED_DEF 0)
endif()
if(NPP_HAS_13_4)
    set(NPP_HAS_13_4_DEF 1)
else()
    set(NPP_HAS_13_4_DEF 0)
endif()
if(NPP_HEADER_UNIFIED)
    set(NPP_API_EXTRA_HEADER "")
elseif(NPP_HEADER_MAJOR EQUAL 11)
    set(NPP_API_EXTRA_HEADER "${CMAKE_SOURCE_DIR}/test/unit/framework/npp_api_extra_11_4.h")
elseif(NPP_HEADER_MAJOR EQUAL 12 AND NPP_HEADER_MINOR LESS 3)
    set(NPP_API_EXTRA_HEADER "${CMAKE_SOURCE_DIR}/test/unit/framework/npp_api_extra_12_2.h")
elseif(NPP_HEADER_MAJOR EQUAL 12)
    set(NPP_API_EXTRA_HEADER "${CMAKE_SOURCE_DIR}/test/unit/framework/npp_api_extra_12_8.h")
else()
    set(NPP_API_EXTRA_HEADER "${CMAKE_SOURCE_DIR}/test/unit/framework/npp_api_extra_13_4.h")
endif()
message(STATUS "API headers: NPP ${NPP_HEADER_MAJOR}.${NPP_HEADER_MINOR}, unified=${NPP_HEADER_UNIFIED}, 13.4 additions=${NPP_HAS_13_4}")
add_compile_definitions(
    NPP_HEADER_MAJOR=${NPP_HEADER_MAJOR}
    NPP_HEADER_MINOR=${NPP_HEADER_MINOR}
    NPP_HEADER_UNIFIED=${NPP_HEADER_UNIFIED_DEF}
    NPP_HAS_13_4=${NPP_HAS_13_4_DEF}
)

# 设置通用编译标准
function(npp_set_common_standards target)
    # MACA and not use cu-bridge, use maca c directly
    if(NOT WITHOUT_CU_BRIDGE)
        set_target_properties(${target} PROPERTIES
	     CUDA_SEPARABLE_COMPILATION OFF
        )
    endif()
endfunction()

# 添加NPP通用包含目录
function(npp_add_common_includes target)

    # NPP 源码库头文件include
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/API>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    # SDK 头文件include
    target_include_directories(${target} PRIVATE ${NPP_SDK_LIB_DIR})

    # 统一注入版本兼容层（旧头文件回填 13.4 枚举，13.4 头文件补齐被移除的流 API）
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-include${CMAKE_SOURCE_DIR}/src/include/npp_api_compat.h>
        $<$<COMPILE_LANGUAGE:CUDA>:--pre-include=${CMAKE_SOURCE_DIR}/src/include/npp_api_compat.h>
    )
    # 超集实现：为当前头文件版本缺失、但库仍然实现的入口补 C 链接声明。
    if(NPP_API_EXTRA_HEADER)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-include${NPP_API_EXTRA_HEADER}>
            $<$<COMPILE_LANGUAGE:CUDA>:--pre-include=${NPP_API_EXTRA_HEADER}>
        )
    endif()
endfunction()

function(npp_link_libs target)
  target_link_directories(${target} PUBLIC ${NPP_SDK_LIB_DIR})
  target_link_libraries(${target} PUBLIC ${NPP_SDK_LINK_LIBS})
endfunction()

# 设置完整的NPP目标配置
function(npp_setup_target target)
    npp_set_common_standards(${target})
    npp_link_libs(${target})
endfunction()
