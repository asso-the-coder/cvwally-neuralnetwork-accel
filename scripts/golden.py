#!/usr/bin/env python3
"""Generate uint8 golden outputs using LiteRT reference kernels only."""
import argparse
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from get_model import ROOT


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    # Optimized and XNNPACK kernels round differently from TFLM's reference kernels.
    interpreter = Interpreter(model_path=str(ROOT / 'build/model/efficientnet-lite0-int8.tflite'),
                              experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                              num_threads=1)
    interpreter.allocate_tensors()
    assert all(op['op_name'] != 'DELEGATE' for op in interpreter._get_ops_details())
    source, target = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    assert source['dtype'] == np.uint8 and tuple(source['shape']) == (1, 224, 224, 3)
    assert target['dtype'] == np.uint8 and tuple(target['shape']) == (1, 1000)
    directory = ROOT / 'build/golden'
    directory.mkdir(parents=True, exist_ok=True)
    for k in range(3):
        data = np.load(ROOT / f'build/model/input_{k}.npy')
        interpreter.set_tensor(source['index'], data[np.newaxis])
        interpreter.invoke()
        output = interpreter.get_tensor(target['index']).reshape(1000)
        np.save(directory / f'expected_{k}.npy', output)
        print(f'input={k} resolver=BUILTIN_REF top1={int(output.argmax())}')


if __name__ == '__main__':
    main()
