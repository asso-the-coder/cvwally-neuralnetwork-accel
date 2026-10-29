#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
JOBS=${JOBS:-4}
if [[ ! $JOBS =~ ^[1-9][0-9]*$ ]]; then
    echo "JOBS must be a positive integer" >&2
    exit 2
fi
config_list=$(python3 -c 'import sys; sys.path.insert(0, "scripts"); from configs import config_order; print("\n".join(config_order()))')
mapfile -t configs <<< "$config_list"
[[ ${#configs[@]} -gt 0 ]] || exit 1
for config in "${configs[@]}"; do
    make CONFIG="$config"
done
python3 scripts/golden.py

pids=()
names=()
failed=0
wait_jobs() {
    local i
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            printf 'PASS: %s (build/%s/run_all.log)\n' "${names[$i]}" "${names[$i]}"
        else
            printf 'FAIL: %s (see build/%s/run_all.log)\n' "${names[$i]}" "${names[$i]}" >&2
            failed=1
        fi
    done
    pids=()
    names=()
}
for config in "${configs[@]}"; do
    (
        python3 scripts/check_model.py --config "$config" --all --skip-golden &&
            python3 scripts/profile.py --config "$config" --skip-golden
    ) > "build/$config/run_all.log" 2>&1 &
    pids+=("$!")
    names+=("$config")
    if (( ${#pids[@]} >= JOBS )); then
        wait_jobs
    fi
done
if (( ${#pids[@]} )); then
    wait_jobs
fi
if (( failed )); then
    exit 1
fi
python3 scripts/microbench.py
python3 scripts/compare_configs.py
cat results/configs.md
