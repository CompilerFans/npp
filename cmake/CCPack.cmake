#Generate by autoTools, you can not modify it by yourself!!!

include(CPackComponent)

set ( CPACK_GENERATOR "DEB" )
set ( CPACK_PACKAGE_VERSION "0.dev" )
set ( CPACK_PACKAGE_CONTACT "https://developer.metax-tech.com" )
set ( CPACK_PACKAGE_VENDOR "MetaX Integrated Circuits (Shanghai) Co., Ltd" )
set ( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set ( CPACK_PACKAGE_DIRECTORY  "${CMAKE_PACKAGE_DIR}" )
set ( CPACK_SET_DESTDIR ON )
set (CPACK_OUTPUT_FILE_PREFIX ${CMAKE_INSTALL_PREFIX}/${PACKAGE_LOCATION})
set( CPACK_RPM_COMPONENT_INSTALL ON )
set( CPACK_DEB_COMPONENT_INSTALL ON )
set ( CPACK_INSTALL_PREFIX /opt/maca-0.dev/ )
set ( CPACK_COMPONENTS_ALL_IN_ONE_PACKAGE 1 )
set ( CPACK_COMPONENTS_ALL headers libs )

set ( CPACK_DEBIAN_PACKAGE_NAME "mcmpp_0.dev")
set ( CPACK_DEBIAN_PACKAGE_DESCRIPTION "maca mcmpp lib package.")
set ( CPACK_DEBIAN_FILE_NAME "mcmpp_0.dev.${DEB_PACKAGE_ARCH}.deb" )
set ( CPACK_DEBIAN_PACKAGE_MAINTAINER "Metax Developers <support-sw@metax-tech.com>" )
set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://developer.metax-tech.com" )
set ( CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${DEB_PACKAGE_ARCH} )
set ( CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "DEBIAN/postinst;DEBIAN/prerm" )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/DEBIAN/post_install DEBIAN/postinst @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/DEBIAN/pre_remove DEBIAN/prerm @ONLY)


include ( CPack )