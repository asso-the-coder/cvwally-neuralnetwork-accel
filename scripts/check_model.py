#!/usr/bin/env python3
"""Run B0 on RV64IMC Spike and require byte-exact reference outputs."""
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from get_model import ROOT, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='check photo and both seeded random inputs')
    parser.add_argument("--output", type=Path, default=ROOT / "results/b0_fullmodel.csv",
                        help="CSV destination (replaced for this run)")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / 'scripts/golden.py')], check=True)
    p = pins()
    # Source archives (e.g. a GitHub ZIP) have no .git; record that instead of failing.
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = 'unknown', 'unknown'
    results = args.output
    results.parent.mkdir(parents=True, exist_ok=True)
    fields = ['config', 'input', 'instret_invoke', 'top1', 'golden_top1', 'max_abs_diff',
              'bit_exact', 'spike_wall_seconds', 'git_commit', 'git_dirty', 'gcc_pin',
              'spike_pin', 'tflm_pin', 'model_sha256']
    # Replace rather than mix measurements from different builds or invocations.
    with results.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        csv_file.flush()
        for k in (range(3) if args.all else [0]):
            elf = ROOT / f'build/model/b0_{k}.elf'
            print(f'Running input={k} on spike --isa=rv64imc ...', flush=True)
            start = time.monotonic()
            run = subprocess.run([os.environ.get('SPIKE', 'spike'), '--isa=rv64imc', str(elf)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            wall = time.monotonic() - start
            (ROOT / f'build/model/spike_{k}.log').write_text(run.stdout)
            if run.returncode:
                print(run.stdout)
                raise SystemExit(f'Spike failed: input={k}, exit={run.returncode}')
            def field(pattern):
                matches = re.findall(pattern, run.stdout, re.MULTILINE)
                if len(matches) != 1:
                    raise ValueError(f'Expected exactly one field matching {pattern!r}')
                return matches[0]
            assert int(field(r'^input=(\d+)$')) == k
            output = np.frombuffer(bytes.fromhex(field(r'^output_hex=([0-9a-f]{2000})$')), dtype=np.uint8)
            expected = np.load(ROOT / f'build/golden/expected_{k}.npy')
            assert expected.dtype == np.uint8 and expected.shape == (1000,)
            top1 = int(field(r'^top1=(\d+)$'))
            assert top1 == int(output.argmax())
            exact = np.array_equal(output, expected)
            diff = int(np.abs(output.astype(np.int16) - expected.astype(np.int16)).max())
            count = int(field(r'^instret_invoke=(\d+)$'))
            print(f'{"PASS" if exact else "FAIL"}: input={k} max_abs_diff={diff} '
                  f'top1_spike={top1} top1_golden={expected.argmax()} '
                  f'instret_invoke={count} spike_wall_seconds={wall:.3f}', flush=True)
            print(field(r'^(arena_size=\d+ arena_used=\d+)$'), flush=True)
            writer.writerow(dict(config='B0', input=k, instret_invoke=count, top1=top1,
                                 golden_top1=int(expected.argmax()), max_abs_diff=diff,
                                 bit_exact=exact, spike_wall_seconds=f'{wall:.3f}',
                                 git_commit=commit, git_dirty=dirty, gcc_pin=p['XPACK_VERSION'],
                                 spike_pin=p['SPIKE_COMMIT'], tflm_pin=p['TFLM_COMMIT'],
                                 model_sha256=p['MODEL_SHA256']))
            csv_file.flush()
            if not exact:
                indices = np.flatnonzero(output != expected)
                print(f'First differing output index={indices[0]}, '
                      f'Spike={output[indices[0]]}, golden={expected[indices[0]]}')
                raise SystemExit('Bit-exact mismatch: stopped; remaining inputs were not run.')


if __name__ == '__main__':
    main()
