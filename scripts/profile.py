#!/usr/bin/env python3
"""Profile a bit-exact model ELF and its separately instrumented counterpart.

The whole-program histogram includes boot, setup, Invoke, output and exit.
The kernel-functions view is a conservative symbol filter: reference_integer_ops
operator Eval functions, and the integer requantization helper. It is NOT a
temporal Invoke histogram: other shared
helpers and interpreter dispatch remain in the complementary view. Every
executed function and its scope is listed in *_function_scopes.csv.

The application reads minstret after printing its output. The histogram
includes the instructions after that read through program exit. The pinned
Spike records the PC_SERIALIZE_BEFORE attempt as well as the retired CSR
instruction; these extra histogram entries are counted explicitly. No
instruction trace is enabled.
Linked code ownership comes from live .text input sections in the ELF's linker
map, not unlinked object sizes; alignment padding is counted under rest.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from count_macs import count_macs
from get_model import ROOT, pins
from configs import config_id
from profile_mix import CATEGORIES, disassembly, histogram


def command(args):
    return subprocess.check_output([str(a) for a in args], cwd=ROOT, text=True)


def field(text, pattern):
    matches = re.findall(pattern, text, re.M)
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {pattern!r}, found {len(matches)}')
    return matches[0]


def check_output(text):
    k = int(field(text, r'^input=(\d+)$'))
    output = np.frombuffer(bytes.fromhex(field(text, r'^output_hex=([0-9a-f]{2000})$')), dtype=np.uint8)
    expected = np.load(ROOT / f'build/golden/expected_{k}.npy')
    if expected.dtype != np.uint8 or expected.shape != (1000,) or not np.array_equal(output, expected):
        raise ValueError('Bit-exact output check FAILED; no results written')
    if int(field(text, r'^top1=(\d+)$')) != int(output.argmax()):
        raise ValueError('Top-1 mismatch')
    print(f'PASS: input={k}, all 1000 output bytes bit-exact', flush=True)
    return k, int(field(text, r'^instret_invoke=(\d+)$'))


def run_spike(elf, log, hist=False):
    args = [os.environ.get('SPIKE', 'spike'), '--isa=rv64imc']
    if hist:
        args += ['-g']
    print(f'Running {elf.name}, histogram={hist} ...', flush=True)
    start = time.monotonic()
    # Spike prints the PC histogram on stderr; keep it apart from HTIF output on stdout.
    errors = log.with_suffix('.stderr')
    with log.open('w') as out, errors.open('w') as err:
        subprocess.run(args + [str(elf)], stdout=out, stderr=err, check=True)
    wall = time.monotonic() - start
    output = log.read_text()
    print(f'Spike wall time: {wall:.3f} s', flush=True)
    return output, wall, check_output(output)


def symbols(elf, cross):
    rows = []
    for line in command([cross + 'nm', '-S', '--defined-only', elf]).splitlines():
        m = re.match(r'([0-9a-f]+) (?:([0-9a-f]+) )?([tTwW]) (.+)$', line)
        if m:
            addr, size, _, name = m.groups()
            rows.append((int(addr, 16), int(size, 16) if size else 0, name))
    # Weak aliases at the same address describe one instruction range.
    unique = {}
    for addr, size, name in sorted(rows):
        if addr not in unique or size > unique[addr][0]:
            unique[addr] = (size, name)
    rows = [(a, size, name) for a, (size, name) in sorted(unique.items())]
    names = command([cross + 'c++filt', *[n for _, _, n in rows]]).splitlines()
    result = []
    for i, ((addr, size, _), name) in enumerate(zip(rows, names)):
        # Assembly labels without .size extend to the next text symbol.
        inferred = not size
        if inferred and i + 1 < len(rows):
            size = rows[i + 1][0] - addr
        result.append((addr, size, name))
    return result


def kernel_function(name):
    return ('reference_integer_ops::' in name or
            name.startswith('tflite::micro::MultiplyByQuantizedMultiplier(') or
            bool(re.search(r'::(?:Eval\w*|\w+Eval)\(', name)))


def linked_sizes(elf):
    text = elf.with_suffix('.map').read_text()
    live = text.split('Linker script and memory map', 1)[1]
    # GNU ld wraps long input section names onto the previous line.
    live = re.sub(r'( \.text\S*)\n\s+', r'\1 ', live)
    kernel_objects = {p.name for p in (elf.parent / 'tflm').rglob('*.o') if 'kernels' in p.parts}
    groups = Counter()
    for m in re.finditer(r'^ (\.text\S*)\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(\S+)', live, re.M):
        _, _, size, owner = m.groups()
        member = re.search(r'libtflm\.a\(([^)]+)\)', owner)
        if member and member[1] in kernel_objects:
            group = 'kernels'
        elif '/runtime/' in owner or ('libtflm.a' not in owner and '.a(' in owner):
            group = 'runtime'
        else:
            group = 'rest'
        groups[group] += int(size, 16)
    if not groups:
        raise ValueError('No linked code sections found in map')
    return groups


def write_csv(path, rows, metadata):
    rows = [{**metadata, **r} for r in rows]
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows, columns):
    def value(v):
        return str(v).replace('|', '\\|')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |'] +
                     ['| ' + ' | '.join(value(r[c]) for c in columns) + ' |' for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=config_id, default='b0')
    parser.add_argument('--elf', type=Path)
    parser.add_argument('--prof-elf', type=Path)
    parser.add_argument('--model', type=Path, default=ROOT / 'build/model/efficientnet-lite0-int8.tflite')
    args = parser.parse_args()
    elf = (args.elf or ROOT / f'build/{args.config}/model_0.elf').resolve()
    prof = (args.prof_elf or ROOT / f'build/{args.config}/model_prof_0.elf').resolve()
    cross = os.environ.get('CROSS', 'riscv-none-elf-')
    work = ROOT / 'build/profile' / args.config.lower()
    work.mkdir(parents=True, exist_ok=True)
    command([sys.executable, ROOT / 'scripts/golden.py'])
    syms = symbols(elf, cross)
    prof_text, prof_wall, prof_result = run_spike(prof, work / 'profiled.log')
    # Read the actual Spike boot ROM; these PCs do not belong to the ELF.
    debug = work / 'histogram.cmd'
    debug.write_text(''.join(f'mem {pc:x}\n' for pc in range(0x1000, 0x1020, 8)) + 'quit\n')
    boot_text = subprocess.check_output(
        [os.environ.get('SPIKE', 'spike'), '--isa=rv64imc', '-d',
         f'--debug-cmd={debug}', str(elf)], stderr=subprocess.STDOUT, text=True)
    text, wall, result = run_spike(elf, work / 'histogram.log', hist=True)
    if result[0] != prof_result[0]:
        raise ValueError('ELFs use different inputs')
    hist = histogram((work / 'histogram.stderr').read_text())
    total = sum(hist.values())
    instret_end = int(field(text, r'^instret_end=(\d+)$'))
    boot_values = re.findall(r'^0x([0-9a-f]{16})$', boot_text, re.M)
    if len(boot_values) != 4:
        raise ValueError('Missing boot ROM words')
    boot = work / 'boot.bin'
    boot.write_bytes(b''.join(int(v, 16).to_bytes(8, 'little') for v in boot_values))
    decoded = disassembly(command([cross + 'objdump', '-d', '-M', 'max,no-aliases', elf]))
    decoded.update(disassembly(command([cross + 'objdump', '-D', '-b', 'binary', '-m', 'riscv:rv64',
                                        '-M', 'max,no-aliases', '--adjust-vma=0x1000', boot])))
    # validate_csr in the pinned Spike returns PC_SERIALIZE_BEFORE once;
    # execute_insn_logged counts both attempts, but minstret counts only one.
    csr_retries = 0
    for pc, n in hist.items():
        if pc not in decoded:
            raise ValueError(f'Undecoded executed PC: {pc:#x}')
        if decoded[pc][1].startswith('csr'):
            if n % 2:
                raise ValueError(f'Odd CSR histogram count at {pc:#x}')
            csr_retries += n // 2
    tail = total - csr_retries - instret_end
    if not 0 <= tail < 100000:
        raise ValueError(f'Histogram/counter mismatch: total={total}, instret_end={instret_end}, '
                         f'post-counter tail={tail}, CSR retries={csr_retries}')
    print(f'PASS: histogram total={total}, instret_end={instret_end}, '
          f'post-counter tail={tail}, CSR retries={csr_retries}', flush=True)
    starts = [s[0] for s in syms]
    whole, kernels, functions = Counter(), Counter(), defaultdict(Counter)
    compressed = Counter()
    sizes, unknown = {}, Counter()
    for pc, n in hist.items():
        if pc not in decoded:
            raise ValueError(f'Undecoded executed PC: {pc:#x}')
        raw, mnemonic, cat = decoded[pc]
        i = bisect_right(starts, pc) - 1
        if 0x1000 <= pc < 0x1020:
            name, size = '[Spike boot ROM]', 32
        elif i >= 0 and syms[i][0] <= pc < syms[i][0] + syms[i][1]:
            _, size, name = syms[i]
        else:
            raise ValueError(f'Executed PC outside function symbols: {pc:#x}')
        sizes[name] = size
        whole[cat] += n
        functions[name][cat] += n
        if cat == 'other':
            unknown[mnemonic] += n
        if len(raw) == 4:
            compressed['whole program'] += n
        if kernel_function(name):
            kernels[cat] += n
            if len(raw) == 4:
                compressed['kernel functions'] += n
    print(f'Other instructions: {dict(unknown)}', flush=True)
    if unknown:
        raise ValueError('Unclassified executed instructions; no results written')
    if sum(whole.values()) != total:
        raise ValueError('Category sum mismatch')
    graph = count_macs(args.model)
    events = re.findall(r'^op=(\d+) type=(\S+) instret=(\d+)$', prof_text, re.M)
    if len(events) != len(graph):
        raise ValueError('Profiler event count differs from graph')
    ops = []
    for row, (index, tag, count) in zip(graph, events):
        if row['index'] != int(index) or row['type'] != tag:
            raise ValueError(f'Profiler/graph operator mismatch: {row}, {index}, {tag}')
        n, macs = int(count), row['macs']
        ops.append({**row, 'instret': n, 'instr_per_mac': n / macs if macs else '',
                    'share': n / prof_result[1]})
    op_sum = sum(r['instret'] for r in ops)
    if abs(op_sum / prof_result[1] - 1) > .01:
        raise ValueError('Per-op sum differs from profiled Invoke by more than 1%')
    aggregates = {}
    for r in ops:
        key = (r['type'], r['conv_kind'])
        a = aggregates.setdefault(key, dict(type=key[0], conv_kind=key[1], count=0, macs=0, elements=0, instret=0))
        for k in ('macs', 'elements', 'instret'):
            a[k] += r[k]
        a['count'] += 1
    optypes = sorted(aggregates.values(), key=lambda r: -r['instret'])
    for a in optypes:
        a.update(share=a['instret'] / prof_result[1], instr_per_mac=a['instret'] / a['macs'] if a['macs'] else '')
    mix = []
    for scope, counts in [('whole program', whole), ('kernel functions', kernels)]:
        for cat in CATEGORIES:
            mix.append(dict(scope=scope, category=cat, count=counts[cat], share=counts[cat] / sum(counts.values())))
    funcs = []
    for name, counts in sorted(functions.items(), key=lambda item: -sum(item[1].values())):
        n = sum(counts.values())
        funcs.append(dict(name=name, scope='kernel functions' if kernel_function(name) else 'outside kernel filter',
                          dyn_instr=n, share=n / total, size_bytes=sizes[name],
                          **{cat: counts[cat] for cat in CATEGORIES}))
    size_values = command([cross + 'size', elf]).splitlines()[1].split()
    text_size, data_size, bss_size = map(int, size_values[:3])
    sections = command([cross + 'size', '-A', elf])
    code_size = int(field(sections, r'^\.text\s+(\d+)\s+\d+$'))
    groups = linked_sizes(elf)
    padding = code_size - sum(groups.values())
    if padding < 0:
        raise ValueError('Linked section ownership exceeds .text')
    groups['rest'] += padding
    codes = [dict(scope='elf', name=elf.name, text=text_size, data=data_size, bss=bss_size),
             *[dict(scope='linked .text', name=k, text=groups[k], data='', bss='') for k in ('kernels', 'runtime', 'rest')],
             *[dict(scope='function', name=f['name'], text=f['size_bytes'], data='', bss='') for f in funcs[:25]]]
    macs = sum(r['macs'] for r in ops)
    metrics = dict(instret_invoke=result[1], total_macs=macs, mac_per_instr=macs / result[1],
                   instr_per_mac=result[1] / macs, histogram_total=total,
                   instret_end=instret_end, post_counter_tail=tail,
                   csr_serialization_retries=csr_retries,
                   whole_program_instret=total-csr_retries,
                   text_size=text_size, linked_text_size=code_size,
                   data_size=data_size, bss_size=bss_size, profiled_instret_invoke=prof_result[1],
                   per_op_instret_sum=op_sum, profiled_minus_op_sum=prof_result[1]-op_sum,
                   profiled_minus_headline=prof_result[1]-result[1], op_sum_minus_headline=op_sum-result[1],
                   profiler_overhead_fraction=(prof_result[1]-result[1])/result[1],
                   spike_histogram_wall_seconds=wall,
                   spike_profiled_wall_seconds=prof_wall)
    summary = [dict(scope='inference / validation', metric=k, value=v) for k, v in metrics.items()]
    for scope, counts in [('whole program', whole), ('kernel functions', kernels)]:
        n = sum(counts.values())
        values = dict(dyn_instr=n, dyn_loads=counts['load'], dyn_stores=counts['store'],
                      dyn_memory_instructions=counts['load']+counts['store'],
                      mem_instr_share=(counts['load']+counts['store'])/n,
                      loads_stores_per_mac=(counts['load']+counts['store'])/macs,
                      mul_share=counts['mul']/n, branch_share=counts['branch']/n,
                      custom_count=counts['custom'], compressed_count=compressed[scope],
                      compressed_fraction=compressed[scope]/n)
        summary += [dict(scope=scope, metric=k, value=v) for k, v in values.items()]
    p = pins()
    metadata = dict(config=args.config.upper(), git_commit=command(['git', 'rev-parse', 'HEAD']).strip(),
                    git_dirty=bool(command(['git', 'status', '--porcelain'])), gcc_pin=p['XPACK_VERSION'],
                    spike_pin=p['SPIKE_COMMIT'], tflm_pin=p['TFLM_COMMIT'], model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
                    prof_elf_sha256=hashlib.sha256(prof.read_bytes()).hexdigest())
    # All correctness and consistency checks complete before publishing any result.
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    stem = args.config.lower()
    for suffix, rows in [('ops', ops), ('optype', optypes), ('mix', mix), ('functions', funcs[:25]),
                         ('function_scopes', funcs), ('summary', summary), ('codesize', codes)]:
        write_csv(out / f'{stem}_{suffix}.csv', rows, metadata)
    md = '\n\n'.join([table([metadata], ['config', 'git_commit', 'git_dirty']),
                         table(summary, ['scope', 'metric', 'value']),
                         table(optypes, ['type', 'conv_kind', 'count', 'macs', 'instret', 'share', 'instr_per_mac']),
                         table(mix, ['scope', 'category', 'count', 'share']),
                         table(funcs[:10], ['name', 'dyn_instr', 'share', 'size_bytes']),
                         table([dict(rank=i+1, **r) for i, r in enumerate(optypes[:3])],
                               ['rank', 'type', 'conv_kind', 'instret', 'share']),
                         table([dict(rank=i+1, **r) for i, r in enumerate(funcs[:3])],
                               ['rank', 'name', 'dyn_instr', 'share'])])
    (out / f'{stem}_profile_summary.md').write_text(md + '\n')
    plots(out / 'plots', stem, ops, optypes, mix)
    print(f'PASS: wrote {stem} profiling results; Invoke={result[1]}, MACs={macs}', flush=True)


def plots(directory, stem, ops, optypes, mix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    labels = [r['type'] + (' ' + r['conv_kind'] if r['conv_kind'] != '-' else '') for r in optypes]
    ax.bar(labels, [100*r['share'] for r in optypes])
    ax.set(ylabel='Profiled Invoke instructions (%)', xlabel='Operator type', title=f'{stem.upper()} operator contribution')
    ax.tick_params(axis='x', labelrotation=45)
    fig.savefig(directory / f'{stem}_optype.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    x = np.arange(len(CATEGORIES))
    for offset, scope in [(-.2, 'whole program'), (.2, 'kernel functions')]:
        ax.bar(x+offset, [100*r['share'] for r in mix if r['scope']==scope], .4, label=scope)
    ax.set(xticks=x, xticklabels=CATEGORIES, ylabel='Dynamic instructions (%)', xlabel='Instruction category', title=f'{stem.upper()} instruction mix')
    ax.legend()
    fig.savefig(directory / f'{stem}_mix.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    for i, kind in enumerate(sorted({r['type'] for r in ops})):
        rows = [r for r in ops if r['type']==kind and r['macs']]
        if rows:
            ax.bar([r['index'] for r in rows], [r['instr_per_mac'] for r in rows], color=f'C{i}', label=kind)
    zero = [r['index'] for r in ops if not r['macs']]
    ax.scatter(zero, [0]*len(zero), marker='x', color='gray', label='No MACs (ratio undefined)')
    ax.set(xlabel='Operator index', ylabel='Instructions / MAC', title=f'{stem.upper()} per-operator cost', xlim=(-1, len(ops)))
    ax.legend()
    fig.savefig(directory / f'{stem}_ops_instr_per_mac.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
