#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
failed=0
pass() { printf 'PASS: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; failed=1; }
config_list=$(python3 -c 'import sys; sys.path.insert(0, "scripts"); from configs import config_order; print("\n".join(config_order()))')
mapfile -t configs <<< "$config_list"
[[ ${#configs[@]} -gt 0 ]] || exit 1
for config in "${configs[@]}"; do
    if ! make CONFIG="$config"; then
        echo "FAIL: build $config"
        exit 1
    fi
done
SPIKE=${SPIKE:-spike}
CROSS=${CROSS:-riscv-none-elf-}
args_text=$(python3 scripts/configs.py --spike-args b0)
mapfile -t plain_args <<< "$args_text"
args_text=$(python3 scripts/configs.py --spike-args x1)
mapfile -t x1_args <<< "$args_text"
run() {
    status=0
    timeout -k 2s 30s "$SPIKE" "${plain_args[@]}" "build/b0/baremetal/$1.elf" > "build/b0/baremetal/$1.log" 2>&1 || status=$?
    cat "build/b0/baremetal/$1.log"
}
run hello
if [[ $status == 0 ]] && grep -Fxq 'hello from rv64imc' build/b0/baremetal/hello.log &&
    grep -Fxq 'int = 42' build/b0/baremetal/hello.log; then
    pass 'hello (exit 0, printf string and integer)'
else
    fail "hello (exit $status)"
fi
run exitcode
if [[ $status == 3 ]]; then pass 'exitcode (exit 3)'; else fail "exitcode (exit $status)"; fi
run illegal_fp
if [[ $status -gt 0 && $status -lt 124 ]] &&
    grep -Eq '^trap: mcause=0x0*2 mepc=0x[0-9a-f]+ mtval=0x[0-9a-f]+$' build/b0/baremetal/illegal_fp.log; then
    pass 'illegal_fp (illegal-instruction trap, nonzero exit)'
else
    fail "illegal_fp (exit $status)"
fi
for config in b1c x1; do
    args_text=$(python3 scripts/configs.py --spike-args "$config")
    mapfile -t kernel_args <<< "$args_text"
    if make CONFIG="$config" kernel-tests &&
        timeout -k 2s 600s "$SPIKE" "${kernel_args[@]}" "build/$config/kernel_tests.elf" > "build/$config/kernel_tests.log" 2>&1; then
        cat "build/$config/kernel_tests.log"
        pass "$config optimized kernels against TFLM reference"
    else
        cat "build/$config/kernel_tests.log" 2>/dev/null || true
        fail "$config optimized kernels against TFLM reference"
    fi
done
if python3 scripts/xqmac_golden.py && make CONFIG=x1 xqmac-tests &&
    timeout -k 2s 60s "$SPIKE" "${x1_args[@]}" build/x1/xqmac_tests.elf > build/x1/xqmac_tests.log 2>&1; then
    cat build/x1/xqmac_tests.log
    pass 'qmac8 instruction and C fallback'
else
    cat build/x1/xqmac_tests.log 2>/dev/null || true
    fail 'qmac8 instruction and C fallback'
fi
status=0
timeout -k 2s 30s "$SPIKE" "${plain_args[@]}" build/x1/xqmac_tests.elf > build/x1/xqmac_negative.log 2>&1 || status=$?
if [[ $status -gt 0 && $status -lt 124 ]] &&
    grep -Eq '^trap: mcause=0x0*2 mepc=0x[0-9a-f]+ mtval=0x[0-9a-f]+$' build/x1/xqmac_negative.log; then
    pass 'qmac8 illegal on plain RV64IMC'
else
    fail "qmac8 negative test (exit $status)"
fi
elfs=(build/b0/baremetal/hello.elf build/b1c/kernel_tests.elf build/x1/kernel_tests.elf build/x1/xqmac_tests.elf)
for config in "${configs[@]}"; do
    elfs+=(build/"$config"/model_{0,1,2}.elf build/"$config"/model_prof_0.elf)
done
for elf in "${elfs[@]}"; do
    if "${CROSS}objdump" -d -M max --no-show-raw-insn "$elf" > "${elf%.elf}.dis"; then
        if python3 - "${elf%.elf}.dis" <<'PYSCAN'
import re
import sys
from pathlib import Path
instruction = re.compile(r'^\s*[0-9a-f]+:\s+(\S+)', re.MULTILINE)
forbidden = re.compile(r'^(?:f(?!ence(?:\.|$))|lr\.|sc\.|amo|v|c\.(?:fl|fs))')
mnemonics = instruction.findall(Path(sys.argv[1]).read_text())
bad = sorted({m for m in mnemonics if forbidden.match(m)})
if not mnemonics or bad:
    print('Missing instructions or forbidden mnemonics:', ', '.join(bad))
    raise SystemExit(1)
PYSCAN
        then
            pass "$elf instruction scan (no F/D/A/V/C-float)"
        else
            fail "$elf instruction scan"
        fi
    else
        fail "$elf objdump"
    fi
done
if python3 scripts/count_macs.py --check; then
    pass 'analytical model MAC count'
else
    fail 'analytical model MAC count'
fi
if python3 scripts/profile_mix.py; then
    pass 'instruction classifier'
else
    fail 'instruction classifier'
fi
for config in "${configs[@]}"; do
    model_args=()
    if [[ $config == x1 ]]; then model_args+=(--all); fi
    if python3 scripts/check_model.py --config "$config" "${model_args[@]}" --output "build/$config/test_results.csv"; then
        pass "$config full model (real image, bit-exact BUILTIN_REF)"
    else
        fail "$config full model (real image)"
    fi
done
exit "$failed"
