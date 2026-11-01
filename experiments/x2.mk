CONFIG_DESC = X1 + rqnt in 1x1 conv, FC and depthwise
CONFIG_CXXFLAGS = -DNDEBUG -DCVW_DW_B1 -DCVW_OFFSET_FOLD -DCVW_XQMAC -DCVW_RQNT
CONFIG_SPIKE_ARGS = --isa=rv64imc --extlib=$(TOOLS_DIR)/spike/lib/libxqmac.so --extension=xqmac
