#!/usr/bin/env python3
"""Generate rqnt vectors and cross-check two independent integer formulations."""
import argparse
import math
from pathlib import Path
import random


def signed(x, bits):
    x &= (1 << bits) - 1
    return x - (1 << bits) if x & (1 << (bits - 1)) else x


def pack(m, s, zp, lo, hi):
    return (m & 0xffffffff) | sum((v & 255) << b for v, b in
                                 zip((s, zp, lo, hi), (32, 40, 48, 56)))


def rqnt(acc, params):
    m = signed(params, 32)
    s, zp, lo, hi = (signed(params >> b, 8) for b in (32, 40, 48, 56))
    x = signed(signed(acc, 32) << max(s, 0), 32)
    p = x * m
    n = p + ((1 << 30) if p >= 0 else 1 - (1 << 30))
    h = (abs(n) // (1 << 31)) * (-1 if n < 0 else 1)
    if x == m == -(1 << 31):
        h = (1 << 31) - 1
    r = max(-s, 0)
    mask = (1 << r) - 1
    q = (h >> r) + ((h & mask) > (mask >> 1) + (h < 0))
    return min(max(signed(q + zp, 32), lo), hi)


def direct(acc, params):
    m = signed(params, 32)
    s, zp, lo, hi = (signed(params >> b, 8) for b in (32, 40, 48, 56))
    x = signed(acc * (2 ** max(s, 0)), 32)
    h = min((x * m + 2**30) // 2**31, 2**31 - 1)
    divisor = 2 ** max(-s, 0)
    q = (2 * abs(h) + divisor) // (2 * divisor)
    if h < 0:
        q = -q
    return min(max(signed(q + zp, 32), lo), hi)


def model_shifts(model):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    it = Interpreter(model_path=str(model), num_threads=1,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    ts = {t['index']: t for t in it.get_tensor_details()}
    shifts = []
    for op in it._get_ops_details():
        if op['op_name'] not in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED'):
            continue
        x, w = (ts[int(i)]['quantization_parameters']['scales'] for i in op['inputs'][:2])
        y = ts[int(op['outputs'][0])]['quantization_parameters']['scales']
        for weight in w:
            fraction, shift = math.frexp(float(x[0]) * float(weight) / float(y[0]))
            if round(fraction * 2**31) == 2**31:
                shift += 1
            shifts.append(shift)
    if not shifts or min(shifts) < -31 or max(shifts) > 30:
        raise ValueError('Unexpected model shift range')
    return min(shifts), max(shifts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('build/xqmac/rqnt_vectors.h'))
    ap.add_argument('--model', type=Path, default=Path('build/model/efficientnet-lite0-int8.tflite'))
    args = ap.parse_args()
    low, high = model_shifts(args.model)
    pairs = []
    def add(a, m, s=0, zp=0, lo=-128, hi=127):
        pairs.append((a & (2**64-1), pack(m, s, zp, lo, hi)))
    for s in sorted({0, -1, -31, 1, 30, high}):
        for a in (-2**31, 2**31-1, -3, -1, 0, 1, 3):
            for m in (0, 2**30, 2**31-1):
                add(a, m, s)
    add(-2**31, -2**31)  # Saturating high multiply.
    for a in (-2**31, 2**31-1):
        for zp in (-128, 127):
            add(a, 2**31-1, zp=zp)  # Offset wrap at either int32 limit.
    for a in (-6, -2, 2, 6):
        add(a, 2**30, -1)  # h is odd: exact positive/negative rounding ties.
    for a in (-100000, 0, 100000):
        for lo, hi in ((-128, 127), (7, 7), (-128, 24), (10, -10)):
            add(a, 2**30, zp=-128 if (lo, hi) == (-128, 24) else 0, lo=lo, hi=hi)
    add(0xdeadbeef00000003, 2**30, -1)
    directed = len(pairs)
    rng = random.Random(0x52514e54)
    for _ in range(5000):
        add(rng.getrandbits(64), rng.randrange(2**30, 2**31), rng.randint(low, high),
            rng.randint(-128, 127), -128, 127)
    for _ in range(5000):
        word = rng.getrandbits(64)
        word = (word & ~(255 << 32)) | ((rng.randint(-31, 30) & 255) << 32)
        pairs.append((rng.getrandbits(64), word))
    rows = []
    for a, b in pairs:
        expected = rqnt(a, b)
        if expected != direct(a, b):
            raise ValueError(f'Golden self-check failed: {a:#x} {b:#x}')
        rows.append(f'{{UINT64_C(0x{a:016x}), UINT64_C(0x{b:016x}), {expected}}},')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('#pragma once\n#include <stdint.h>\n'
        'struct RqntVector { uint64_t acc, params; int64_t expected; };\n'
        f'static const unsigned rqnt_directed_count = {directed};\n'
        'static const RqntVector rqnt_vectors[] = {\n' + '\n'.join(rows) + '\n};\n')
    print(f'PASS: rqnt golden vs big-integer formula, {directed} directed + 10000 random; '
          f'model shifts {low}..{high}')


if __name__ == '__main__':
    main()
