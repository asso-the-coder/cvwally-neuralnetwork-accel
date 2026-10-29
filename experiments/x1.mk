CONFIG_DESC = B1c + qmac8 in 1x1 conv and FC
CONFIG_CXXFLAGS = -DNDEBUG -DCVW_DW_B1 -DCVW_OFFSET_FOLD -DCVW_XQMAC
CONFIG_SPIKE_ARGS = --isa=rv64imc --extlib=$(TOOLS_DIR)/spike/lib/libxqmac.so --extension=xqmac
