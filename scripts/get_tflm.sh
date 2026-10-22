#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/scripts/versions.env"
source "$ROOT/scripts/env.sh"
mkdir -p "$ROOT/third_party"
source_dir="$ROOT/third_party/tflite-micro"
tree="$ROOT/third_party/tflm_tree"
if [[ ! -d "$source_dir/.git" ]]; then
    git clone https://github.com/tensorflow/tflite-micro.git "$source_dir"
fi
if ! git -C "$source_dir" cat-file -e "$TFLM_COMMIT^{commit}" 2>/dev/null; then
    git -C "$source_dir" fetch origin "$TFLM_COMMIT"
fi
git -C "$source_dir" checkout --detach "$TFLM_COMMIT"
if [[ -f "$tree/.pin" && $(cat "$tree/.pin") == "$TFLM_COMMIT" ]]; then
    echo "TFLM tree already generated at $TFLM_COMMIT"
    exit 0
fi
rm -rf "$tree"
cd "$source_dir"
python tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py "$tree"
printf '%s\n' "$TFLM_COMMIT" > "$tree/.pin"
