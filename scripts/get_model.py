#!/usr/bin/env python3
"""Fetch and align model buffers; verify unchanged tensors and inference outputs."""
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


def repack_model(raw, inputs):
    import flatbuffers
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    class AlignedBuffer(schema.BufferT):
        def Pack(self, builder):
            if self.data is not None:
                # The generated object API omits the schema's force_align on data.
                builder.Prep(16, len(self.data))
            return super().Pack(builder)

    original = schema.Model.GetRootAsModel(raw, 0)
    model = schema.ModelT.InitFromObj(original)
    for index, buffer in enumerate(model.buffers):
        if buffer.offset or buffer.size:
            raise ValueError('External buffers are unsupported')
        aligned = AlignedBuffer()
        aligned.__dict__.update(buffer.__dict__)
        model.buffers[index] = aligned
    builder = flatbuffers.Builder(len(raw))
    root = model.Pack(builder)
    builder.Finish(root, file_identifier=b'TFL3')
    packed = bytes(builder.Output())
    result = schema.Model.GetRootAsModel(packed, 0)
    if original.BuffersLength() != result.BuffersLength():
        raise ValueError('Buffer count changed')
    for k in range(result.BuffersLength()):
        before, after = original.Buffers(k), result.Buffers(k)
        if before.DataLength() != after.DataLength():
            raise ValueError(f'Buffer {k} length changed')
        if after.DataLength():
            if not np.array_equal(before.DataAsNumpy(), after.DataAsNumpy()):
                raise ValueError(f'Buffer {k} bytes changed')
            offset = after._tab.Vector(after._tab.Offset(4))
            if offset % 16:
                raise ValueError(f'Buffer {k} is not 16-byte aligned: {offset}')
    for g in range(original.SubgraphsLength()):
        a, b = original.Subgraphs(g), result.Subgraphs(g)
        if a.TensorsLength() != b.TensorsLength():
            raise ValueError('Tensor count changed')
        for k in range(a.TensorsLength()):
            if a.Tensors(k).Buffer() != b.Tensors(k).Buffer():
                raise ValueError(f'Tensor {g}:{k} buffer changed')
    outputs = []
    for content in (raw, packed):
        interpreter = Interpreter(model_content=content, num_threads=1,
                                  experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        source = interpreter.get_input_details()[0]['index']
        target = interpreter.get_output_details()[0]['index']
        values = []
        for data in inputs:
            interpreter.set_tensor(source, data[np.newaxis])
            interpreter.invoke()
            values.append(interpreter.get_tensor(target))
        outputs.append(values)
    if any(not np.array_equal(a, b) for a, b in zip(*outputs)):
        raise ValueError('Re-packed model inference differs from original')
    print('PASS: re-pack buffer bytes, 16-byte alignment, tensor mappings, and 3 reference outputs')
    return packed


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
    packed = repack_model(model.read_bytes(), inputs)
    (directory / 'efficientnet-lite0-int8-aligned.tflite').write_bytes(packed)
    print('Re-packed sha256: ' + hashlib.sha256(packed).hexdigest())
    emit(directory, 'model_data', [('model_data', packed)])
    emit(directory, 'inputs', [(f'input_{k}', data) for k, data in enumerate(inputs)])
    print(f'Model and 3 inputs ready: {directory}')


if __name__ == '__main__':
    main()
