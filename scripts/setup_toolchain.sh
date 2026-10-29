#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/versions.env"
TOOLS_DIR=${TOOLS_DIR:-$HOME/opt/cvwally}
missing=0
for command in gcc g++ make dtc git python3 tar sha256sum nproc; do
    command -v "$command" >/dev/null 2>&1 || missing=1
done
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    missing=1
fi
python3 -c 'import venv, ensurepip' >/dev/null 2>&1 || missing=1
if (( missing )); then
    echo 'Missing host dependencies. Install with:' >&2
    echo 'sudo apt-get install build-essential device-tree-compiler curl git python3 python3-venv' >&2
    exit 1
fi
mkdir -p "$TOOLS_DIR"
TOOLS_DIR=$(cd -- "$TOOLS_DIR" && pwd)
mkdir -p "$TOOLS_DIR/src" "$TOOLS_DIR/tmp" "$TOOLS_DIR/cache"
# Keep temporary files and package caches inside the installation tree.
export TMPDIR="$TOOLS_DIR/tmp" XDG_CACHE_HOME="$TOOLS_DIR/cache"
export PIP_CACHE_DIR="$TOOLS_DIR/cache/pip"
match_stamp() { [[ -f "$1" && "$(cat "$1")" == "$2" ]]; }
gcc_pin="$XPACK_VERSION $XPACK_SHA256"
if ! match_stamp "$TOOLS_DIR/.gcc-stamp" "$gcc_pin"; then
    archive="$TOOLS_DIR/src/$(basename "$XPACK_URL")"
    if [[ ! -f "$archive" ]]; then
        if command -v curl >/dev/null 2>&1; then
            curl -fL "$XPACK_URL" -o "$archive.part"
        else
            wget "$XPACK_URL" -O "$archive.part"
        fi
        mv "$archive.part" "$archive"
    fi
    actual_sha256=$(sha256sum "$archive")
    actual_sha256=${actual_sha256%% *}
    if [[ "$actual_sha256" != "$XPACK_SHA256" ]]; then
        rm -f -- "$archive"
        printf 'SHA256 mismatch for %s: expected %s, actual %s. Deleted archive; rerun to download again.\n' \
            "$archive" "$XPACK_SHA256" "$actual_sha256" >&2
        exit 1
    fi
    staging="$TOOLS_DIR/src/gcc-extract"
    rm -rf "$staging"
    mkdir -p "$staging"
    tar -xzf "$archive" -C "$staging" --strip-components=1
    rm -rf "$TOOLS_DIR/xpack-riscv-none-elf-gcc"
    mv "$staging" "$TOOLS_DIR/xpack-riscv-none-elf-gcc"
    printf '%s\n' "$gcc_pin" > "$TOOLS_DIR/.gcc-stamp"
fi
if ! match_stamp "$TOOLS_DIR/.spike-stamp" "$SPIKE_COMMIT"; then
    source_dir="$TOOLS_DIR/src/riscv-isa-sim"
    if [[ ! -d "$source_dir/.git" ]]; then
        git clone https://github.com/riscv-software-src/riscv-isa-sim.git "$source_dir"
    fi
    if ! git -C "$source_dir" cat-file -e "$SPIKE_COMMIT^{commit}" 2>/dev/null; then
        git -C "$source_dir" fetch origin "$SPIKE_COMMIT"
    fi
    git -C "$source_dir" checkout --detach "$SPIKE_COMMIT"
    build_dir="$TOOLS_DIR/src/spike-build"
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    (
        cd "$build_dir"
        "$source_dir/configure" --prefix="$TOOLS_DIR/spike"
        make -j"$(nproc)"
        make install
    )
    printf '%s\n' "$SPIKE_COMMIT" > "$TOOLS_DIR/.spike-stamp"
fi
extension_source="$SCRIPT_DIR/../src/spike_ext/xqmac.cc"
extension_hash=$(sha256sum "$extension_source")
extension_hash=${extension_hash%% *}
extension_pin="$SPIKE_COMMIT $extension_hash"
if ! match_stamp "$TOOLS_DIR/.xqmac-stamp" "$extension_pin" ||
    [[ ! -f "$TOOLS_DIR/spike/lib/libxqmac.so" ]]; then
    mkdir -p "$TOOLS_DIR/spike/lib"
    g++ -std=c++17 -O2 -Wall -Wextra -fPIC -shared \
        -isystem "$TOOLS_DIR/spike/include" -isystem "$TOOLS_DIR/spike/include/riscv" \
        "$extension_source" -o "$TOOLS_DIR/spike/lib/libxqmac.so"
    printf '%s\n' "$extension_pin" > "$TOOLS_DIR/.xqmac-stamp"
fi
python_pin=$(sha256sum "$SCRIPT_DIR/requirements.txt")
python_pin=${python_pin%% *}
if ! match_stamp "$TOOLS_DIR/.python-stamp" "$python_pin"; then
    python3 -m venv "$TOOLS_DIR/venv"
    "$TOOLS_DIR/venv/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"
    printf '%s\n' "$python_pin" > "$TOOLS_DIR/.python-stamp"
fi
printf 'GCC: xPack %s\nSpike: %s\n' "$XPACK_VERSION" "$SPIKE_COMMIT"
"$TOOLS_DIR/xpack-riscv-none-elf-gcc/bin/riscv-none-elf-gcc" --version | sed -n '1p'
"$TOOLS_DIR/venv/bin/python" -c 'import numpy; print("numpy: " + numpy.__version__)'
printf 'Installed in %s\nActivate with: source scripts/env.sh\n' "$TOOLS_DIR"
