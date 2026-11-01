#!/usr/bin/env python3
"""Run and record the verified dot-product and requantization microbenchmark."""
import argparse
import csv
import os
import re
import subprocess
import time
from configs import execution_metadata, git_state, spike_args
from get_model import ROOT, pins


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    rows = []
    commit, dirty = git_state()
    p = pins()
    for config, variants in [('x1', {'scalar', 'qmac8'}), ('x2', {'requant_scalar', 'rqnt'})]:
        subprocess.run(['make', f'CONFIG={config}', 'microbench'], cwd=ROOT, check=True)
        start = time.monotonic()
        result = subprocess.run([os.environ.get('SPIKE', 'spike'), *spike_args(config),
                                 str(ROOT / f'build/{config}/microbench.elf')],
                                capture_output=True, text=True, check=True)
        wall = time.monotonic() - start
        (ROOT / f'build/{config}/microbench.log').write_text(result.stdout + result.stderr)
        matches = [r for r in re.findall(r'^microbench,(\w+),(\d+),(\d+)$', result.stdout, re.M)
                   if r[0] in variants]
        if len(matches) != 2 or {r[0] for r in matches} != variants or 'PASS: microbench' not in result.stdout:
            raise ValueError('Missing verified microbenchmark results')
        for v, n, c in matches:
            requant = v in {'requant_scalar', 'rqnt'}
            rows.append(dict(variant=v, length=int(n), instret=int(c),
                instr_per_mac='' if requant else int(c)/int(n),
                instr_per_output=int(c)/int(n) if requant else '',
                git_commit=commit, git_dirty=dirty, gcc_pin=p['XPACK_VERSION'],
                spike_pin=p['SPIKE_COMMIT'], spike_wall_seconds=wall,
                **execution_metadata(config)))
        print(result.stdout, end='')
    with (ROOT / 'results/microbench.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
