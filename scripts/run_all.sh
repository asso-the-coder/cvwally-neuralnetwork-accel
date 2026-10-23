#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
configs=$(python3 -c 'import sys; sys.path.insert(0, "scripts"); from configs import config_order; print("\n".join(config_order()))')
while IFS= read -r config; do
    make CONFIG="$config"
    python3 scripts/check_model.py --config "$config" --all
    python3 scripts/profile.py --config "$config"
done <<< "$configs"
python3 scripts/compare_configs.py
cat results/configs.md
