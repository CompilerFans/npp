
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was OpenNPPConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

# OpenNPP CMake配置文件

set(OpenNPP_VERSION "1.0.0")

# 检查依赖项
include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit REQUIRED)

# 包含导出的目标
include("${CMAKE_CURRENT_LIST_DIR}/OpenNPPTargets.cmake")

# 设置变量
set_and_check(OpenNPP_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include/opennpp")
set_and_check(OpenNPP_LIB_DIR "${PACKAGE_PREFIX_DIR}/lib")

# 验证所有组件都找到了
check_required_components(OpenNPP)

# 提供便捷的目标别名
if(TARGET OpenNPP::npp AND NOT TARGET OpenNPP::OpenNPP)
    add_library(OpenNPP::OpenNPP ALIAS OpenNPP::npp)
endif()

# 输出找到的信息
if(NOT OpenNPP_FIND_QUIETLY)
    message(STATUS "Found OpenNPP: ${OpenNPP_VERSION}")
    message(STATUS "  Include dir: ${OpenNPP_INCLUDE_DIR}")
    message(STATUS "  Library dir: ${OpenNPP_LIB_DIR}")
endif()
