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
	rm -rf build/baremetal
