#!/usr/bin/env python3
"""Fetch the unmodified model and generate deterministic embedded inputs."""
import argparse
import hashlib
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def pins():
    return dict(line.split('=', 1) for line in
                (ROOT / 'scripts/versions.env').read_text().splitlines()
                if line and not line.startswith('#'))


def verify(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        path.unlink()
        raise ValueError(f'SHA256 mismatch: {path}: {actual}, expected {expected}; deleted')


def download(url, path, expected):
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    verify(path, expected)


def emit(directory, name, arrays):
    declarations, definitions = [], []
    for symbol, data in arrays:
        raw = data if isinstance(data, bytes) else data.tobytes()
        declarations.extend([f'extern const unsigned char {symbol}[{len(raw)}];',
                             f'extern const unsigned int {symbol}_len;'])
        rows = [','.join(str(b) for b in raw[i:i+24]) for i in range(0, len(raw), 24)]
        definitions.append(f'alignas(16) const unsigned char {symbol}[{len(raw)}] = {{\n' +
                           ',\n'.join(rows) + f'\n}};\nconst unsigned int {symbol}_len = {len(raw)};')
    (directory / f'{name}.h').write_text('#pragma once\n' + '\n'.join(declarations) + '\n')
    (directory / f'{name}.cc').write_text(f'#include "{name}.h"\n' + '\n'.join(definitions) + '\n')


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    directory = ROOT / 'build/model'
    directory.mkdir(parents=True, exist_ok=True)
    p = pins()
    archive = directory / 'efficientnet-lite0.tar.gz'
    download(p['MODEL_URL'], archive, p['MODEL_ARCHIVE_SHA256'])
    model = directory / 'efficientnet-lite0-int8.tflite'
    if not model.exists():
        with tarfile.open(archive) as tar:
            model.write_bytes(tar.extractfile(p['MODEL_MEMBER']).read())
    verify(model, p['MODEL_SHA256'])
    photo = directory / 'grace_hopper.jpg'
    download(p['IMAGE_URL'], photo, p['IMAGE_SHA256'])
    # RGB; shorter side exactly 256, longer side floor-scaled; bilinear resize;
    # centered 224 square with floor-rounded left/top coordinates; raw uint8 HWC.
    with Image.open(photo) as image:
        image = image.convert('RGB')
        w, h = image.size
        image = image.resize((w * 256 // min(w, h), h * 256 // min(w, h)), Image.Resampling.BILINEAR)
        w, h = image.size
        x, y = (w - 224) // 2, (h - 224) // 2
        inputs = [np.asarray(image.crop((x, y, x + 224, y + 224)), dtype=np.uint8)]
    inputs += [np.random.default_rng(seed).integers(0, 256, (224, 224, 3), dtype=np.uint8)
               for seed in (1, 2)]
    for k, data in enumerate(inputs):
        np.save(directory / f'input_{k}.npy', data)
    emit(directory, 'model_data', [('model_data', model.read_bytes())])
    emit(directory, 'inputs', [(f'input_{k}', data) for k, data in enumerate(inputs)])
    print(f'Model and 3 inputs ready: {directory}')


if __name__ == '__main__':
    main()
