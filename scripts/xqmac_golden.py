#!/usr/bin/env python3
"""Generate deterministic packed signed-byte dot-product instruction tests."""
import argparse
from pathlib import Path
import random
import numpy as np


def qmac8(a, b):
    def lane(value, i):
        byte = (value >> (8 * i)) & 255
        return byte if byte < 128 else byte - 256
    return sum(lane(a, i) * lane(b, i) for i in range(8))


def pack(lanes):
    return sum((v & 255) << (8 * i) for i, v in enumerate(lanes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/xqmac/vectors.h'))
    args = parser.parse_args()
    pairs = [(pack([a] * 8), pack([b] * 8)) for a, b in
             [(-128, -128), (-128, 127), (127, 127), (0, 0), (0, -128)]]
    pairs.append((pack([-128, 127, -1, 1, -32, 64, 0, 7]),
                  pack([127, -128, 1, -1, 64, -32, 7, 0])))
    for i in range(8):
        lanes = [0] * 8
        lanes[i] = -128
        pairs.append((pack(lanes), pack([127] * 8)))
    directed = len(pairs)
    rng = random.Random(0x514d4143)
    pairs.extend((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(10000))
    rows = []
    for a, b in pairs:
        expected = qmac8(a, b)
        av = np.frombuffer(a.to_bytes(8, 'little'), dtype=np.int8).astype(np.int64)
        bv = np.frombuffer(b.to_bytes(8, 'little'), dtype=np.int8).astype(np.int64)
        if expected != int(np.dot(av, bv)):
            raise ValueError('Golden model differs from numpy dot')
        rows.append(f'{{UINT64_C(0x{a:016x}), UINT64_C(0x{b:016x}), {expected}, {qmac8(a, a)}}},')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('#pragma once\n#include <stdint.h>\n'
                          'struct QmacVector { uint64_t a, b; int64_t expected, square; };\n'
                          f'static const unsigned directed_count = {directed};\n'
                          'static const QmacVector vectors[] = {\n' + '\n'.join(rows) + '\n};\n')
    print(f'PASS: golden vs numpy, {directed} directed + 10000 random pairs')


if __name__ == '__main__':
    main()
