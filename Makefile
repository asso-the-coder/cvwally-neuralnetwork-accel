MAKEFLAGS += --no-builtin-rules
CROSS ?= riscv-none-elf-
SPIKE ?= spike
CC = $(CROSS)gcc
CFLAGS = -std=c11 -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany -O2 -Wall -Wextra -ffunction-sections -fdata-sections
LDFLAGS = --specs=nano.specs -nostartfiles -Wl,--gc-sections -T src/runtime/link.ld
RUNTIME = src/runtime/crt0.S src/runtime/htif.c src/runtime/syscalls.c
PROGRAMS = hello exitcode illegal_fp
ELFS = $(addprefix build/baremetal/,$(addsuffix .elf,$(PROGRAMS)))

.PHONY: all run-hello clean
all: $(ELFS)

build/baremetal/%.elf: src/tests/baremetal/%.c $(RUNTIME) src/runtime/htif.h src/runtime/link.ld Makefile
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(LDFLAGS) $(RUNTIME) $< -o $@

run-hello: build/baremetal/hello.elf
	$(SPIKE) --isa=rv64imc $<

clean:
	rm -rf build/baremetal build/app build/runtime build/tflm build/libtflm.a
	rm -f build/model/*.o build/model/*.d build/model/*.elf build/model/*.map

# The generated standalone tree contains upstream library sources only.
CXX = $(CROSS)g++
AR = $(CROSS)ar
TFLM = third_party/tflm_tree
CXXFLAGS = $(filter-out -std=c11,$(CFLAGS)) -std=c++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics -DTF_LITE_STATIC_MEMORY
INCLUDES = -I. -Ibuild/model -isystem $(TFLM) -isystem $(TFLM)/third_party/flatbuffers/include -isystem $(TFLM)/third_party/gemmlowp -isystem $(TFLM)/third_party/ruy
TFLM_CORE = $(shell find $(TFLM)/tensorflow -name '*.cc' ! -path '*/kernels/*' ! -name debug_log.cc 2>/dev/null)
KERNEL_NAMES = add add_common conv conv_common depthwise_conv depthwise_conv_common fully_connected fully_connected_common pooling pooling_common quantize quantize_common reshape reshape_common softmax softmax_common kernel_util
TFLM_KERNELS = $(addprefix $(TFLM)/tensorflow/lite/micro/kernels/,$(addsuffix .cc,$(KERNEL_NAMES)))
TFLM_INTERNAL = $(wildcard $(TFLM)/tensorflow/lite/micro/kernels/internal/*.cc) $(wildcard $(TFLM)/tensorflow/lite/kernels/*.cc)
TFLM_SOURCES = $(sort $(TFLM_CORE) $(TFLM_KERNELS) $(TFLM_INTERNAL))
TFLM_OBJECTS = $(patsubst $(TFLM)/%.cc,build/tflm/%.o,$(TFLM_SOURCES))
MODEL_OBJECTS = build/model/model_data.o build/model/inputs.o build/app/debug_log.o build/runtime/cxx_support.o build/runtime/crt0.o build/runtime/htif.o build/runtime/syscalls.o
MODEL_ELFS = $(addprefix build/model/b0_,$(addsuffix .elf,0 1 2))
all: $(MODEL_ELFS)

build/tflm/%.o: $(TFLM)/%.cc Makefile
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -w $(INCLUDES) -MMD -MP -c $< -o $@

build/libtflm.a: $(TFLM_OBJECTS)
	$(AR) rcs $@ $^

build/model/%.o: build/model/%.cc Makefile
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(addprefix build/app/main_,$(addsuffix .o,0 1 2)): build/app/main_%.o: src/app/main.cc Makefile
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DINPUT_INDEX=$* -MMD -MP -c $< -o $@

build/%.o: src/%.cc Makefile
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

build/%.o: src/%.c Makefile
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

build/%.o: src/%.S Makefile
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

build/model/b0_%.elf: build/app/main_%.o $(MODEL_OBJECTS) build/libtflm.a src/runtime/link.ld
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(filter %.o %.a,$^) -Wl,-Map,$(@:.elf=.map) -lm -o $@

.SECONDARY: $(MODEL_OBJECTS) $(addprefix build/app/main_,$(addsuffix .o,0 1 2))
-include $(TFLM_OBJECTS:.o=.d) $(MODEL_OBJECTS:.o=.d) $(wildcard build/app/main_*.d)
