#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
failed=0
pass() { printf 'PASS: %s\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1"; failed=1; }
if ! make; then
    echo 'FAIL: build'
    exit 1
fi
SPIKE=${SPIKE:-spike}
CROSS=${CROSS:-riscv-none-elf-}
run() {
    status=0
    timeout -k 2s 30s "$SPIKE" --isa=rv64imc "build/baremetal/$1.elf" > "build/baremetal/$1.log" 2>&1 || status=$?
    cat "build/baremetal/$1.log"
}
run hello
if [[ $status == 0 ]] && grep -Fxq 'hello from rv64imc' build/baremetal/hello.log &&
    grep -Fxq 'int = 42' build/baremetal/hello.log; then
    pass 'hello (exit 0, printf string and integer)'
else
    fail "hello (exit $status)"
fi
run exitcode
if [[ $status == 3 ]]; then pass 'exitcode (exit 3)'; else fail "exitcode (exit $status)"; fi
run illegal_fp
if [[ $status -gt 0 && $status -lt 124 ]] &&
    grep -Eq '^trap: mcause=0x0*2 mepc=0x[0-9a-f]+ mtval=0x[0-9a-f]+$' build/baremetal/illegal_fp.log; then
    pass 'illegal_fp (illegal-instruction trap, nonzero exit)'
else
    fail "illegal_fp (exit $status)"
fi
if "${CROSS}objdump" -d -M max --no-show-raw-insn build/baremetal/hello.elf > build/baremetal/hello.dis; then
    # Only instruction lines have a hex address followed by a colon; labels do not.
    if python3 - <<'PY'
import re
from pathlib import Path
instruction = re.compile(r'^\s*[0-9a-f]+:\s+(\S+)', re.MULTILINE)
forbidden = re.compile(r'^(?:f(?!ence(?:\.|$))|lr\.|sc\.|amo|v|c\.(?:fl|fs))')
mnemonics = instruction.findall(Path('build/baremetal/hello.dis').read_text())
bad = sorted({m for m in mnemonics if forbidden.match(m)})
if not mnemonics or bad:
    print('Missing instructions or forbidden mnemonics:', ', '.join(bad))
    raise SystemExit(1)
PY
    then
        pass 'hello instruction scan (no F/D/A/V/C-float)'
    else
        fail 'hello instruction scan'
    fi
else
    fail 'hello objdump'
fi
exit "$failed"
