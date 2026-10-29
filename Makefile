CONFIG ?= b0
ifeq ($(shell printf '%s\n' '$(CONFIG)' | LC_ALL=C grep -Ex '[a-z0-9]+'),)
$(error CONFIG must match [a-z0-9]+)
endif
TOOLS_DIR ?= $(HOME)/opt/cvwally
include experiments/$(CONFIG).mk
CONFIG_SPIKE_ARGS ?= --isa=rv64imc
BUILD = build/$(CONFIG)

MAKEFLAGS += --no-builtin-rules
CROSS ?= riscv-none-elf-
SPIKE ?= spike
CC = $(CROSS)gcc
CFLAGS = -std=c11 -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany -O2 -Wall -Wextra -ffunction-sections -fdata-sections
LDFLAGS = --specs=nano.specs -nostartfiles -Wl,--gc-sections -T src/runtime/link.ld
RUNTIME = src/runtime/crt0.S src/runtime/htif.c src/runtime/syscalls.c
PROGRAMS = hello exitcode illegal_fp
ELFS = $(addprefix $(BUILD)/baremetal/,$(addsuffix .elf,$(PROGRAMS)))

.PHONY: all run-hello clean
all: $(ELFS)

$(BUILD)/baremetal/%.elf: src/tests/baremetal/%.c $(RUNTIME) src/runtime/htif.h src/runtime/link.ld Makefile experiments/$(CONFIG).mk
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) $(RUNTIME) $< -o $@

run-hello: $(BUILD)/baremetal/hello.elf
	$(SPIKE) $(CONFIG_SPIKE_ARGS) $<

clean:
	rm -rf $(addprefix build/,$(basename $(notdir $(wildcard experiments/*.mk)))) $(patsubst %/libtflm.a,%,$(wildcard build/*/libtflm.a))
	rm -rf build/baremetal build/app build/runtime build/tflm build/libtflm.a
	rm -f build/model/*.o build/model/*.d build/model/*.elf build/model/*.map

.PHONY: config-desc
config-desc:
	@printf '%s\n' '$(CONFIG_DESC)'

# The generated standalone tree contains upstream library sources only.
CXX = $(CROSS)g++
AR = $(CROSS)ar
TFLM = third_party/tflm_tree
CXXFLAGS = $(filter-out -std=c11,$(CFLAGS)) -std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics -DTF_LITE_STATIC_MEMORY $(CONFIG_CXXFLAGS)
INCLUDES = -I. -Ibuild/model -isystem $(TFLM) -isystem $(TFLM)/third_party/flatbuffers/include -isystem $(TFLM)/third_party/gemmlowp -isystem $(TFLM)/third_party/ruy
TFLM_CORE = $(shell find $(TFLM)/tensorflow -name '*.cc' ! -path '*/kernels/*' ! -name debug_log.cc 2>/dev/null)
KERNEL_NAMES = add add_common conv conv_common depthwise_conv depthwise_conv_common fully_connected fully_connected_common pooling pooling_common quantize quantize_common reshape reshape_common softmax softmax_common kernel_util
TFLM_KERNELS = $(addprefix $(TFLM)/tensorflow/lite/micro/kernels/,$(addsuffix .cc,$(KERNEL_NAMES)))
TFLM_INTERNAL = $(wildcard $(TFLM)/tensorflow/lite/micro/kernels/internal/*.cc) $(wildcard $(TFLM)/tensorflow/lite/kernels/*.cc)
TFLM_SOURCES = $(sort $(TFLM_CORE) $(TFLM_KERNELS) $(TFLM_INTERNAL))
TFLM_OBJECTS = $(patsubst $(TFLM)/%.cc,$(BUILD)/tflm/%.o,$(TFLM_SOURCES))
KERNEL_OBJECTS = $(patsubst src/%.cc,$(BUILD)/%.o,$(wildcard src/kernels/*.cc))
MODEL_OBJECTS = $(KERNEL_OBJECTS) $(BUILD)/model/model_data.o $(BUILD)/model/inputs.o $(BUILD)/app/debug_log.o $(BUILD)/runtime/cxx_support.o $(BUILD)/runtime/crt0.o $(BUILD)/runtime/htif.o $(BUILD)/runtime/syscalls.o
MODEL_ELFS = $(addprefix $(BUILD)/model_,$(addsuffix .elf,0 1 2))
all: $(MODEL_ELFS) $(BUILD)/model_prof_0.elf

$(BUILD)/tflm/%.o: $(TFLM)/%.cc Makefile experiments/$(CONFIG).mk
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -w $(INCLUDES) -MMD -MP -c $< -o $@

# Recreate the archive so member order is the sorted source order, not rebuild history.
$(BUILD)/libtflm.a: $(TFLM_OBJECTS)
	rm -f $@
	$(AR) rcs $@ $^

$(BUILD)/model/%.o: build/model/%.cc Makefile experiments/$(CONFIG).mk
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(addprefix $(BUILD)/app/main_,$(addsuffix .o,0 1 2)): $(BUILD)/app/main_%.o: src/app/main.cc Makefile experiments/$(CONFIG).mk
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DINPUT_INDEX=$* -MMD -MP -c $< -o $@

$(BUILD)/%.o: src/%.cc Makefile experiments/$(CONFIG).mk
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(BUILD)/%.o: src/%.c Makefile experiments/$(CONFIG).mk
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/%.o: src/%.S Makefile experiments/$(CONFIG).mk
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/model_%.elf: $(BUILD)/app/main_%.o $(MODEL_OBJECTS) $(BUILD)/libtflm.a src/runtime/link.ld
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(filter %.o %.a,$^) -Wl,-Map,$(@:.elf=.map) -lm -o $@

.SECONDARY: $(MODEL_OBJECTS) $(addprefix $(BUILD)/app/main_,$(addsuffix .o,0 1 2))
-include $(TFLM_OBJECTS:.o=.d) $(MODEL_OBJECTS:.o=.d) $(wildcard $(BUILD)/app/main_*.d)

# Profiling changes only the application object; reference kernels are shared.
$(BUILD)/app/main_prof_0.o: src/app/main.cc src/app/op_profiler.h Makefile experiments/$(CONFIG).mk
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DINPUT_INDEX=0 -DPROFILE_OPS -MMD -MP -c $< -o $@

$(BUILD)/model_prof_0.elf: $(BUILD)/app/main_prof_0.o $(MODEL_OBJECTS) $(BUILD)/libtflm.a src/runtime/link.ld
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(filter %.o %.a,$^) -Wl,-Map,$(@:.elf=.map) -lm -o $@

.PHONY: kernel-tests
kernel-tests: $(BUILD)/kernel_tests.elf
$(BUILD)/kernel_tests.elf: $(BUILD)/tests/kernels.o $(KERNEL_OBJECTS) $(BUILD)/app/debug_log.o $(BUILD)/runtime/cxx_support.o $(BUILD)/runtime/crt0.o $(BUILD)/runtime/htif.o $(BUILD)/runtime/syscalls.o $(BUILD)/libtflm.a src/runtime/link.ld
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(filter %.o %.a,$^) -Wl,-Map,$(@:.elf=.map) -lm -o $@
-include $(BUILD)/tests/kernels.d

.PHONY: config-spike-args xqmac-tests microbench
config-spike-args:
	@printf '%s\n' '$(CONFIG_SPIKE_ARGS)'

build/xqmac/vectors.h: scripts/xqmac_golden.py
	python3 $< --output $@

$(BUILD)/tests/xqmac.o: build/xqmac/vectors.h
xqmac-tests: $(BUILD)/xqmac_tests.elf
microbench: $(BUILD)/microbench.elf
$(BUILD)/xqmac_tests.elf: $(BUILD)/tests/xqmac.o $(BUILD)/runtime/crt0.o $(BUILD)/runtime/htif.o $(BUILD)/runtime/syscalls.o src/runtime/link.ld
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(filter %.o,$^) -o $@
$(BUILD)/microbench.elf: $(BUILD)/tests/microbench.o $(BUILD)/runtime/crt0.o $(BUILD)/runtime/htif.o $(BUILD)/runtime/syscalls.o src/runtime/link.ld
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(filter %.o,$^) -o $@
-include $(BUILD)/tests/xqmac.d $(BUILD)/tests/microbench.d
