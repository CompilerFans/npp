PROJECT_DEPS 	 	:=
PROJECT_NAME 	 	:= mcMPP
PROJECT_BUILDDIR 	:= $(BUILDDIR)/library
PROJECT_MAKEFILE 	:= $(abspath $(lastword $(MAKEFILE_LIST)))
PROJECT_MK_PATH		:= $(patsubst %/, %, $(dir $(PROJECT_MAKEFILE)))

MPP_SETUP_STAMP := $(PROJECT_BUILDDIR)/$(PROJECT_NAME)/.mpp_setup.stamp

.PHONY: $(PROJECT_NAME)_PRE_CONFIGURE
$(PROJECT_NAME)_PRE_CONFIGURE: PRIVATE_PROJ_DIR  := $(PROJECT_MK_PATH)
$(PROJECT_NAME)_PRE_CONFIGURE:
$(MPP_SETUP_STAMP):
	@echo "[PRE] run setup_mpp.sh (first time only)"
	@mkdir -p "$(dir $@)"
	cd $(PRIVATE_PROJ_DIR) && bash "./mpp_scripts/setup_mpp.sh" --skip-build --skip-deploy;
	@touch "$@"

# pre config: convert to mpp
$(PROJECT_NAME)_PRE_CONFIGURE: $(MPP_SETUP_STAMP)

PROJECT_EXTERND_OPS	:= \
		-DWITHOUT_CU_BRIDGE=ON \
		-DBUILD_TESTS=OFF \
		-DBUILD_EXAMPLES=OFF \
		-DBUILD_SHARED_LIBS=ON

$(PROJECT_NAME)_CONFIGURE: $(PROJECT_NAME)_PRE_CONFIGURE

include $(BUILD_COMMON)