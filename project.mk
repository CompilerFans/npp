# Minimal one-step build/package for mcMPP under maca SDK.

BUILD_TYPE ?= Release
BUILD_SHARED_LIBS ?= ON
BUILD_EXAMPLES ?= OFF
USE_MACA_SDK="ON"
# default use Ninja, otherwise CMake can choose (usually cmake).
CMAKE_GENERATOR ?= Ninja

.PHONY: convertMpp
convertMpp: PRIVATE_PROJ_DIR ?= $(CURDIR)/library/mcMPP
convertMpp:
	@echo
	@echo "Converting sources to MPP (if needed)..."
	@if [ -d "$(PRIVATE_PROJ_DIR)/src/mppi" ]; then \
	  echo "  Detected converted sources (src/mppi exists), skipping."; \
	else \
	  bash "$(PRIVATE_PROJ_DIR)/mpp_scripts/setup_mpp.sh" --skip-build --skip-deploy; \
	fi

# -DCMAKE_CXX_COMPILER=$(MACA_PATH)/mxgpu_llvm/bin/mxcc \

.PHONY: mcMPP
mcMPP: convertMpp
mcMPP: PRIVATE_BUILD_DIR ?= $(BUILDDIR)/library/mcMPP
mcMPP: PRIVATE_MK_FILE := $(abspath $(lastword $(MAKEFILE_LIST)))
mcMPP: PRIVATE_MK_PATH := $(patsubst %/, %, $(dir $(PRIVATE_MK_FILE)))
mcMPP:
	@echo "Configuring + building + installing + packaging mcMPP..."
	cmake -S $(PRIVATE_MK_PATH) \
	  -B $(PRIVATE_BUILD_DIR) \
	  $(if $(CMAKE_GENERATOR),-G "$(CMAKE_GENERATOR)") \
	  -DCMAKE_CXX_COMPILER=$(MACA_CLANG_PATH)/mxcc \
	  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	  -DUSE_MACA_SDK=$(USE_MACA_SDK) \
	  -DBUILD_TESTS=OFF \
	  -DBUILD_EXAMPLES=$(BUILD_EXAMPLES) \
	  -DBUILD_SHARED_LIBS=$(BUILD_SHARED_LIBS) \
	  -DCMAKE_INSTALL_PREFIX=$(BUILDROOT) \
	  -DBUILD_NUMBER=$(BUILD_NUMBER) \
	  -DCMAKE_PACKAGE_DIR=$(PRIVATE_BUILD_DIR) -DPACKAGE_ARCH=$(PACK_ARCH) -DPACKAGE_EXT=$(PACK_EXT) \
	  -DCMAKE_TARGET_ARCH=$(ARCH) -DMACA_VERSION=$(MACA_VERSION) -DPACK_VERSION=$(PACK_VERSION) \
	  -DDISTRO_CODE=$(DISTRO_CODE) -DPACKAGE_GENERATOR="$(PACKAGE_GENERATOR)" \
	  -DDEB_PACKAGE_ARCH=$(DEB_PACK_ARCH) -DRPM_PACKAGE_ARCH=$(RPM_PACK_ARCH) -DPACKAGE_LOCATION=$(PACK_LOCATION)
	ninja -C $(PRIVATE_BUILD_DIR)
	ninja -C $(PRIVATE_BUILD_DIR) install
	ninja -C $(PRIVATE_BUILD_DIR) package

.PHONY: clean
clean: PRIVATE_BUILD_DIR ?= $(BUILDDIR)/library/mcMPP
clean:
	rm -rf $(PRIVATE_BUILD_DIR)
