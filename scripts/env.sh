#!/usr/bin/env bash
# Source this file to select the pinned tools.
export TOOLS_DIR="${TOOLS_DIR:-$HOME/opt/cvwally}"
export PATH="$TOOLS_DIR/xpack-riscv-none-elf-gcc/bin:$TOOLS_DIR/spike/bin:$TOOLS_DIR/venv/bin:$PATH"
