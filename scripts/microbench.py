#!/usr/bin/env python3
"""Run and record the verified scalar/qmac8 dot-product microbenchmark."""
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
    subprocess.run(['make', 'CONFIG=x1', 'microbench'], cwd=ROOT, check=True)
    start = time.monotonic()
    result = subprocess.run([os.environ.get('SPIKE', 'spike'), *spike_args('x1'),
                             str(ROOT / 'build/x1/microbench.elf')],
                            capture_output=True, text=True, check=True)
    wall = time.monotonic() - start
    (ROOT / 'build/x1/microbench.log').write_text(result.stdout + result.stderr)
    matches = re.findall(r'^microbench,(scalar|qmac8),(\d+),(\d+)$', result.stdout, re.M)
    if len(matches) != 2 or {r[0] for r in matches} != {'scalar', 'qmac8'} or 'PASS: microbench' not in result.stdout:
        raise ValueError('Missing verified microbenchmark results')
    commit, dirty = git_state()
    p = pins()
    rows = [dict(variant=v, length=int(n), instret=int(c), instr_per_mac=int(c)/int(n),
                 git_commit=commit, git_dirty=dirty, gcc_pin=p['XPACK_VERSION'],
                 spike_pin=p['SPIKE_COMMIT'], spike_wall_seconds=wall,
                 **execution_metadata('x1')) for v, n, c in matches]
    with (ROOT / 'results/microbench.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(result.stdout, end='')


if __name__ == '__main__':
    main()
