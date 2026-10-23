#!/usr/bin/env python3
"""Compare measured configurations in experiments/order.txt order.

Mix shares and memory instructions/MAC use the whole-program histogram.
A positive delta_instr_vs_previous means fewer Invoke instructions.
"""
import argparse
import csv
from pathlib import Path

from configs import config_desc, config_order
from get_model import ROOT
from profile import table


SOURCES = ('fullmodel', 'summary', 'mix', 'optype')


def read_rows(path, config):
    with path.open(newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows or any(r['config'] != config.upper() for r in rows):
        raise ValueError(f'{path}: empty or mismatched config')
    return rows


def single(rows, **keys):
    matches = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    if len(matches) != 1:
        raise ValueError(f'Expected one row for {keys}, got {len(matches)}')
    return matches[0]


def compare(directory):
    output = []
    baseline = previous = None
    for config in config_order():
        paths = {kind: directory / f'{config}_{kind}.csv' for kind in SOURCES}
        if not any(p.exists() for p in paths.values()):
            continue
        sources = {kind: read_rows(path, config) for kind, path in paths.items()}
        if any(r['bit_exact'] != 'True' for r in sources['fullmodel']):
            raise ValueError(f'{config}: fullmodel results are not all bit-exact')
        count = int(single(sources['fullmodel'], input='0')['instret_invoke'])

        def metric(name, scope='inference / validation'):
            return single(sources['summary'], scope=scope, metric=name)['value']

        if count != int(metric('instret_invoke')):
            raise ValueError(f'{config}: fullmodel and profiling Invoke counts differ')
        if baseline is None:
            if config != 'b0':
                raise ValueError('B0 results must precede measured configurations')
            baseline = count
        row = dict(config=config.upper(), config_desc=config_desc(config), instret_invoke=count,
                   instr_per_mac=float(metric('instr_per_mac')),
                   reduction_vs_b0=baseline / count,
                   reduction_vs_previous=previous / count if previous is not None else '',
                   delta_instr_vs_previous=previous-count if previous is not None else '',
                   mix_scope='whole program')
        for category in ('load', 'store', 'mul', 'branch', 'alu', 'jump'):
            row[f'{category}_share'] = float(single(sources['mix'], scope='whole program',
                                                   category=category)['share'])
        row['loads_stores_per_mac'] = float(metric('loads_stores_per_mac', 'whole program'))
        for label, kind, conv in [('conv_1x1', 'CONV_2D', '1x1'), ('conv_kxk', 'CONV_2D', 'kxk'),
                                 ('depthwise', 'DEPTHWISE_CONV_2D', '-'),
                                 ('fc', 'FULLY_CONNECTED', '-')]:
            row[f'{label}_instr_per_mac'] = float(single(sources['optype'], type=kind,
                                                        conv_kind=conv)['instr_per_mac'])
        row['linked_text_size'] = int(metric('linked_text_size'))
        for kind, rows in sources.items():
            provenance = {(r['git_commit'], r['git_dirty']) for r in rows}
            if len(provenance) != 1:
                raise ValueError(f'{paths[kind]}: inconsistent provenance')
            commit, dirty = provenance.pop()
            row[f'{kind}_git_commit'] = commit
            row[f'{kind}_git_dirty'] = dirty
        output.append(row)
        previous = count
    if not output:
        raise ValueError('No measured configurations')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    rows = compare(args.results_dir)
    with (args.results_dir / 'configs.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.results_dir / 'configs.md').write_text(table(rows, list(rows[0])) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    bars = ax.bar([r['config'] for r in rows], [r['instret_invoke'] for r in rows])
    ax.bar_label(bars, labels=[f"{r['reduction_vs_b0']:.3f}×" for r in rows], padding=3)
    ax.set(xlabel='Configuration', ylabel='Invoke retired instructions',
           title='Full-model instructions (labels: reduction vs B0)')
    ax.margins(y=.15)
    plots = args.results_dir / 'plots'
    plots.mkdir(exist_ok=True)
    fig.savefig(plots / 'configs_instret.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
