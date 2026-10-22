#!/usr/bin/env python3
"""Count MACs analytically in graph order, using LiteRT tensor shapes."""
import argparse
import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from get_model import ROOT


def count_macs(model):
    interpreter = Interpreter(model_path=str(model), num_threads=1,
                              experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    tensors = {t['index']: t for t in interpreter.get_tensor_details()}
    rows = []
    for op in interpreter._get_ops_details():
        kind = op['op_name']
        shape = [int(x) for x in tensors[int(op['outputs'][0])]['shape']]
        elements = int(np.prod(shape))
        macs, conv_kind = 0, '-'
        weights = []
        if kind in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED'):
            weights = [int(x) for x in tensors[int(op['inputs'][1])]['shape']]
            if kind == 'CONV_2D':
                _, kh, kw, cin = weights
                macs = elements * kh * kw * cin
                conv_kind = '1x1' if kh == kw == 1 else 'kxk'
            elif kind == 'DEPTHWISE_CONV_2D':
                macs = elements * weights[1] * weights[2]
            else:
                macs = elements * weights[-1]
        elif kind not in ('ADD', 'AVERAGE_POOL_2D', 'MAX_POOL_2D', 'QUANTIZE', 'RESHAPE', 'SOFTMAX'):
            raise ValueError(f'Unsupported operator: {kind}')
        rows.append(dict(index=int(op['index']), type=kind, conv_kind=conv_kind,
                         macs=macs, elements=elements, output_shape=shape, weight_shape=weights))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'build/model/efficientnet-lite0-int8.tflite')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    rows = count_macs(args.model)
    total = sum(r['macs'] for r in rows)
    if args.check:
        if not 380_000_000 <= total <= 410_000_000:
            raise SystemExit(f'FAIL: unexpected MAC total {total}')
        print(f'PASS: {len(rows)} operators, {total} MACs')
    else:
        print(json.dumps(dict(ops=rows, total_macs=total), indent=2))


if __name__ == '__main__':
    main()
