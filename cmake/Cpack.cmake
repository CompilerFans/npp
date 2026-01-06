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
set ( CPACK_DEBIAN_PACKAGE_MAINTAINER "Metax Developers <support-sw@metax-tech.com>" )
set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "maca ${LIBRARY_NAME} lib package." )
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

set ( CPACK_DEBIAN_FILE_NAME "${LIBRARY_NAME}_${MACA_VERSION}.${DEB_PACKAGE_ARCH}.deb" )
set ( CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${DEB_PACKAGE_ARCH} )
set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://www.metax-tech.com" )
set ( CPACK_DEBIAN_PACKAGE_MAINTAINER "Metax Developers <support-sw@metax-tech.com>" )
set ( CPACK_DEBIAN_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}" )


set ( CPACK_RPM_PACKAGE_RELEASE ${BUILD_NUMBER})
set ( CPACK_RPM_PACKAGE_RELEASE_DIST "")
set ( CPACK_RPM_PACKAGE_AUTOREQ no)
set ( CPACK_RPM_PACKAGE_GROUP "MetaX")
set ( CPACK_RPM_PACKAGE_LICENSE "MetaX License")
set ( CPACK_RPM_PACKAGE_DESCRIPTION "${CPACK_PACKAGE_DESCRIPTION_SUMMARY}" )
set ( CPACK_RPM_FILE_NAME "${LIBRARY_NAME}_${MACA_VERSION}.${RPM_PACKAGE_ARCH}.rpm" )
set ( CPACK_RPM_PACKAGE_ARCHITECTURE ${RPM_PACKAGE_ARCH})

# Only package selected components (headers + libs).
# Let CPack run the top-level install script with component filtering.

include(CPack)

message(STATUS "[CPack] cmake/Cpack.cmake loaded")
