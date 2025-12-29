# CPack configuration for MPP packaging (headers + built libs only).

# Package generators: DEB, RPM, or "DEB;RPM"
set(PACKAGE_GENERATOR "DEB" CACHE STRING "Package types to build")
set(CPACK_GENERATOR "${PACKAGE_GENERATOR}")

# Components: headers + libs in a single package
set(CPACK_COMPONENTS_ALL headers libs)
set(CPACK_COMPONENTS_ALL_IN_ONE_PACKAGE ON)
if(CPACK_GENERATOR MATCHES "DEB")
    set(CPACK_DEB_COMPONENT_INSTALL ON)
endif()
if(CPACK_GENERATOR MATCHES "RPM")
    set(CPACK_RPM_COMPONENT_INSTALL ON)
endif()
set(CPACK_INSTALL_LOCAL_ONLY OFF)

set ( LIBRARY_NAME "mcmpp" )
set ( CPACK_PACKAGE_NAME ${LIBRARY_NAME}_${PACK_VERSION} )
set ( CPACK_PACKAGE_VERSION "${MACA_VERSION}" )
set ( CPACK_PACKAGE_CONTACT "https://www.metax-tech.com" )
set ( CPACK_DEBIAN_PACKAGE_MAINTAINER "mxmaca@metax-tech.com" )
set ( CPACK_PACKAGE_VENDOR "MetaX Integrated Circuits (Shanghai) Co., Ltd" )
# set ( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" )

# Where to place generated packages
if ( DEFINED CMAKE_PACKAGE_DIR AND NOT CMAKE_PACKAGE_DIR STREQUAL "" )
    set ( CPACK_PACKAGE_DIRECTORY "${CMAKE_PACKAGE_DIR}" )
elseif ( DEFINED PACK_LOCATION AND NOT PACK_LOCATION STREQUAL "" )
    set ( CPACK_PACKAGE_DIRECTORY "${PACK_LOCATION}" )
else ()
    set ( CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}/packages" )
endif ()

# Use the same prefix as the install tree
set ( CPACK_SET_DESTDIR ON )
set ( CPACK_INSTALL_PREFIX /opt/maca-${PACK_VERSION}/ )
set ( CPACK_OUTPUT_FILE_PREFIX "${CMAKE_INSTALL_PREFIX}/${PACKAGE_LOCATION}" CACHE STRING "Default output file prefix.")

if(CPACK_GENERATOR MATCHES "DEB")
    set ( CPACK_DEBIAN_FILE_NAME "${LIBRARY_NAME}_${MACA_VERSION}.${DEB_PACKAGE_ARCH}.deb" )
    set ( CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${DEB_PACKAGE_ARCH} )
    set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://www.metax-tech.com" )
    set ( CPACK_DEBIAN_PACKAGE_MAINTAINER "mxmaca@metax-tech.com" )
    set ( CPACK_DEBIAN_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}" )
endif()

if(CPACK_GENERATOR MATCHES "RPM")
    set(_mpp_arch "${RPM_PACKAGE_ARCH}")
    if(NOT _mpp_arch)
        set(_mpp_arch "${PACKAGE_ARCH}")
    endif()
    if(NOT _mpp_arch)
        set(_mpp_arch "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "${_mpp_arch}")

    execute_process(COMMAND rpm --eval %{?dist}
        RESULT_VARIABLE PROC_RESULT
        OUTPUT_VARIABLE EVAL_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(PROC_RESULT EQUAL "0" AND NOT EVAL_RESULT STREQUAL "")
        string(APPEND CPACK_RPM_PACKAGE_RELEASE "%{?dist}")
    endif()
endif()

# Only package selected components (headers + libs).
# Let CPack run the top-level install script with component filtering.

include(CPack)

message(STATUS "[CPack] cmake/Cpack.cmake loaded")
